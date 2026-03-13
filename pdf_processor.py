import fitz  # PyMuPDF
import spacy
from langdetect import detect, DetectorFactory

# Set seed for deterministic language detection
DetectorFactory.seed = 0

def load_spacy_model(model_name):
    """
    Helper function to load a spaCy model. If it's not installed,
    it will attempt to download it programmatically before loading.
    """
    try:
        return spacy.load(model_name)
    except OSError:
        print(f"Model '{model_name}' not found. Downloading it now...")
        spacy.cli.download(model_name)
        return spacy.load(model_name)

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
        nlp = load_spacy_model("pt_core_news_lg")
    elif lang == 'en':
        print("Detected language: English")
        nlp = load_spacy_model("en_core_web_lg")
    else:
        # Fallback to English or raise an error depending on requirement
        print(f"Detected language '{lang}', which is unsupported. Defaulting to English.")
        nlp = load_spacy_model("en_core_web_lg")

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


def evaluate_vocabulary(data, nlp):
    """
    Evaluates vocabulary using wordfreq.
    Provides:
    - Vocabulary Frequency Counter (Term Frequency)
    - Rare Word & Jargon Flagger (Corpus Mapping)
    - Lexical Richness & Readability Scorer
    """
    from collections import Counter
    import wordfreq

    lang = nlp.meta['lang'] # 'en' or 'pt'

    words = []
    lemmas = []

    for item in data:
        doc = nlp(item["text"])
        for token in doc:
            if token.is_alpha and not token.is_stop and not token.is_punct:
                words.append(token.text.lower())
                lemmas.append(token.lemma_.lower())

    # Vocabulary Frequency Counter (Term Frequency)
    term_frequency = dict(Counter(lemmas))

    # Rare Word & Jargon Flagger (Corpus Mapping)
    # Using Zipf frequency: 1.0 to 8.0, lower means rarer.
    # threshold for rare words: < 3.0
    rare_words = {}
    total_zipf = 0
    valid_words_count = 0

    for word in set(words):
        zipf = wordfreq.zipf_frequency(word, lang)
        # Include words with 0.0 frequency as they are the rarest (out of vocabulary)
        if zipf < 3.5:
            rare_words[word] = zipf

        # Only use valid words (>0) for average zipf to avoid skewing readability too much with OCR errors
        if zipf > 0:
            total_zipf += zipf
            valid_words_count += 1

    # Lexical Richness & Readability Scorer
    # Type-Token Ratio
    ttr = len(set(lemmas)) / len(lemmas) if len(lemmas) > 0 else 0.0

    # Average Zipf frequency acts as a proxy for readability (higher = easier words used)
    avg_zipf = (total_zipf / valid_words_count) if valid_words_count > 0 else 0.0

    return {
        "term_frequency": term_frequency,
        "rare_words_and_jargon": rare_words,
        "lexical_richness_ttr": round(ttr, 4),
        "readability_avg_zipf": round(avg_zipf, 4)
    }

def validate_sense(teacher_data, student_data, nlp, threshold=0.85):
    """
    Validates sense between teacher and student materials by calculating vector similarity
    between their paragraphs. Checks if student topics convey the same meaning as teacher topics.
    Returns a list of matched topics with their similarity scores.
    """
    matches = []

    # Get doc embeddings
    teacher_docs = []
    for item in teacher_data:
        doc = nlp(item["text"])
        if doc.has_vector and len(doc.text.strip()) > 10:
            teacher_docs.append({
                "page": item["page"],
                "text": item["text"],
                "doc": doc
            })

    student_docs = []
    for item in student_data:
        doc = nlp(item["text"])
        if doc.has_vector and len(doc.text.strip()) > 10:
            student_docs.append({
                "page": item["page"],
                "text": item["text"],
                "doc": doc
            })

    # Compare each student paragraph against teacher paragraphs to find matches
    for s_doc in student_docs:
        best_match = None
        best_score = -1

        for t_doc in teacher_docs:
            score = s_doc["doc"].similarity(t_doc["doc"])
            if score > best_score:
                best_score = score
                best_match = t_doc

        if best_match and best_score >= threshold:
            matches.append({
                "score": round(best_score, 4),
                "student_topic": {
                    "page": s_doc["page"],
                    "text": s_doc["text"]
                },
                "teacher_topic": {
                    "page": best_match["page"],
                    "text": best_match["text"]
                }
            })

    return matches


import difflib

def analyze_topic_order(teacher_data, student_data, nlp):
    """
    Extracts sequential key nouns from teacher and student materials
    to check if the topic order matches.
    Returns the sequence of key nouns and a match ratio.
    """
    def extract_key_nouns(data):
        nouns = []
        for item in data:
            doc = nlp(item["text"])
            for token in doc:
                if token.pos_ in ("NOUN", "PROPN") and not token.is_stop and len(token.text) > 2:
                    nouns.append(token.lemma_.lower())
        return nouns

    teacher_nouns = extract_key_nouns(teacher_data)
    student_nouns = extract_key_nouns(student_data)

    matcher = difflib.SequenceMatcher(None, teacher_nouns, student_nouns)
    match_ratio = matcher.ratio()

    return {
        "match_ratio": round(match_ratio, 4),
        "teacher_key_topics": teacher_nouns,
        "student_key_topics": student_nouns
    }


def check_ner_consistency(teacher_entities, student_entities):
    """
    Cross-checks named entities (Persons, Locations, Dates) from both documents.
    Returns entities present in both, missing in student, and missing in teacher.
    """
    consistency = {}

    for category in ["PERSON", "LOCATION", "DATE"]:
        teacher_set = set(teacher_entities.get(category, []))
        student_set = set(student_entities.get(category, []))

        consistency[category] = {
            "in_both": list(teacher_set.intersection(student_set)),
            "missing_in_student": list(teacher_set - student_set),
            "missing_in_teacher": list(student_set - teacher_set)
        }

    return consistency


def compare_pedagogic_materials(teacher_pdf_path, student_pdf_path, output_json="comparison_response.json"):
    """
    Main orchestration function for the new pedagogic mode.
    Extracts text from both PDFs, runs semantic redundancy checks individually,
    and then compares them for sense validity, topic order, and NER consistency.
    """
    import json

    # 1. Extract text and pages from both PDFs
    teacher_data = extract_text_from_pdf(teacher_pdf_path)
    student_data = extract_text_from_pdf(student_pdf_path)

    # We combine both initially just to ensure language is detected accurately across the board
    # Or, preferably, detect based on teacher material.
    nlp = detect_language_and_load_model(teacher_data)

    # 2. Extract Named Entities individually
    teacher_entities = extract_named_entities(teacher_data, nlp)
    student_entities = extract_named_entities(student_data, nlp)

    # 3. Detect internal semantic redundancies individually
    teacher_redundancies = detect_semantic_redundancy(teacher_data, nlp)
    student_redundancies = detect_semantic_redundancy(student_data, nlp)

    # 4. Compare mode: NER Consistency
    ner_consistency = check_ner_consistency(teacher_entities, student_entities)

    # 5. Compare mode: Validate Sense (Professor vs. Aluno)
    sense_validation = validate_sense(teacher_data, student_data, nlp)

    # 6. Compare mode: Analyze Topic Order
    topic_order = analyze_topic_order(teacher_data, student_data, nlp)

    # 7. Compare mode: Evaluate Vocabulary
    teacher_vocabulary = evaluate_vocabulary(teacher_data, nlp)
    student_vocabulary = evaluate_vocabulary(student_data, nlp)

    response_data = {
        "metadata": {
            "teacher_source": teacher_pdf_path,
            "student_source": student_pdf_path,
            "teacher_total_paragraphs": len(teacher_data),
            "student_total_paragraphs": len(student_data)
        },
        "teacher_redundancies": teacher_redundancies,
        "student_redundancies": student_redundancies,
        "ner_consistency": ner_consistency,
        "sense_validation": sense_validation,
        "topic_order": topic_order,
        "teacher_vocabulary": teacher_vocabulary,
        "student_vocabulary": student_vocabulary
    }

    # Export to JSON
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(response_data, f, indent=4, ensure_ascii=False)

    print(f"Pedagogic comparison data successfully exported to {output_json}")
    return response_data


def process_pdf_and_export_json(filepath, output_json="response.json"):
    """
    Main pipeline function that processes a PDF and exports the extracted data,
    named entities, and semantic redundancies into a JSON file.
    """
    import json

    # 1. Extract text and pages
    extracted_data = extract_text_from_pdf(filepath)

    # 2. Detect language and load appropriate spaCy model
    nlp = detect_language_and_load_model(extracted_data)

    # 3. Extract Named Entities
    entities = extract_named_entities(extracted_data, nlp)

    # 4. Detect Semantic Redundancies
    redundancies = detect_semantic_redundancy(extracted_data, nlp)

    # Compile the final response payload
    response_data = {
        "metadata": {
            "source_file": filepath,
            "total_paragraphs": len(extracted_data)
        },
        "extracted_text": extracted_data,
        "named_entities": entities,
        "redundancies": redundancies
    }

    # Export to JSON
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(response_data, f, indent=4, ensure_ascii=False)

    print(f"Data successfully exported to {output_json}")
    return response_data
