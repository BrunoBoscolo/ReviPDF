import fitz  # PyMuPDF
import spacy
from langdetect import detect, DetectorFactory

# Set seed for deterministic language detection
DetectorFactory.seed = 0

def detect_language_and_load_model(extracted_data):
    """
    Detects language of the extracted text (Portuguese or English)
    and dynamically loads the correct large spaCy model.
    """
    if not extracted_data:
        raise ValueError("No text extracted from PDF to detect language.")

    # Combine a few blocks of text to get a better language detection sample
    sample_text = " ".join([block["text"] for block in extracted_data[:5]])

    lang = detect(sample_text)

    if lang == 'pt':
        print("Detected language: Portuguese")
        nlp = spacy.load("pt_core_news_lg")
    elif lang == 'en':
        print("Detected language: English")
        nlp = spacy.load("en_core_web_lg")
    else:
        # Fallback to English or raise an error depending on requirement
        print(f"Detected language '{lang}', which is unsupported. Defaulting to English.")
        nlp = spacy.load("en_core_web_lg")

    return nlp


def extract_text_from_pdf(filepath):
    """
    Reads a PDF file and extracts text block by block, preserving page numbers.
    Returns a list of dictionaries, each containing 'page' and 'text'.
    """
    document = fitz.open(filepath)
    extracted_data = []

    for page_num, page in enumerate(document, start=1):
        # text, blocks, dict, words, html, xhtml, xml, rawdict
        blocks = page.get_text("blocks")

        for block in blocks:
            # block format: (x0, y0, x1, y1, "lines in block", block_no, block_type)
            # block_type 0 corresponds to text
            if block[6] == 0:
                text = block[4].strip()
                if text: # keep only non-empty paragraphs
                    extracted_data.append({
                        "page": page_num,
                        "text": text
                    })

    return extracted_data


def extract_named_entities(extracted_data, nlp):
    """
    Extracts named entities (Persons, Locations, Dates) from the text using the provided spaCy model.
    Maps English and Portuguese entity labels for consistency.
    """
    entities = {
        "PERSON": set(),
        "LOCATION": set(),
        "DATE": set()
    }

    # Label mappings between Portuguese and English
    person_labels = {"PER", "PERSON"}
    location_labels = {"LOC", "GPE"}
    date_labels = {"DATE"}

    for item in extracted_data:
        doc = nlp(item["text"])
        for ent in doc.ents:
            if ent.label_ in person_labels:
                entities["PERSON"].add(ent.text)
            elif ent.label_ in location_labels:
                entities["LOCATION"].add(ent.text)
            elif ent.label_ in date_labels:
                entities["DATE"].add(ent.text)

    # Convert sets back to lists for easier consumption (like JSON serialization)
    return {k: list(v) for k, v in entities.items()}


def detect_semantic_redundancy(extracted_data, nlp, threshold=0.85):
    """
    Cross-checks all paragraphs to find semantic redundancies using spaCy embeddings.
    Returns a list of tuples containing redundant paragraph pairs and their similarity score.
    """
    redundancies = []

    # Process all texts to get their spaCy document representations (which contain embeddings)
    docs = []
    for item in extracted_data:
        doc = nlp(item["text"])
        # We only consider paragraphs that have vectors and aren't empty
        if doc.has_vector and len(doc.text.strip()) > 10:
            docs.append({
                "page": item["page"],
                "text": item["text"],
                "doc": doc
            })

    num_docs = len(docs)
    for i in range(num_docs):
        for j in range(i + 1, num_docs):
            doc1 = docs[i]["doc"]
            doc2 = docs[j]["doc"]

            # Compute cosine similarity between the document vectors
            similarity = doc1.similarity(doc2)

            if similarity >= threshold:
                redundancies.append({
                    "score": round(similarity, 4),
                    "para1": {
                        "page": docs[i]["page"],
                        "text": docs[i]["text"]
                    },
                    "para2": {
                        "page": docs[j]["page"],
                        "text": docs[j]["text"]
                    }
                })

    return redundancies
