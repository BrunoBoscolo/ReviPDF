import fitz  # PyMuPDF
import spacy
from langdetect import detect, DetectorFactory
import re
import os
import hashlib

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
                text = block[4]
                # Filter out designer instructions to the end of the block (which acts as a paragraph)
                text = re.sub(r"(?i)(Instrução Visual|\[INSTRUÇÃO PARA O DIAGRAMADOR\]).*", "", text, flags=re.DOTALL)
                # Replace newlines and any other whitespace sequences with a single space
                text = re.sub(r"\s+", " ", text)
                text = text.strip()
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


def parse_aulas(extracted_data):
    """
    Parses extracted text data into chapters, 'aulas', and their internal sections:
    - GUIA DO PROFESSOR
    - CONTEÚDO DO LIVRO DO ALUNO
    - ATIVIDADES DO ALUNO
    """
    # Tolerant regexes to handle formatting, typos, and numbering variations
    chapter_re = re.compile(r"(?i)^\s*(?:C[Aa][Pp][ÍiI][Tt][Uu][Ll][Oo]|C[Hh][Aa][Pp][Tt][Ee][Rr])\s+(\d{1,2})\s*[-–—]?\s*(.*)$")
    aula_re = re.compile(r"(?i)^\s*A[uU][lL][aA]\s+(\d{1,2})\s*[-–—]?\s*(.*)$")
    sec1_re = re.compile(r"(?i)^\s*(?:1\.?\s*)?GUIA\s+D[OE]\s+PROFESS[OÓ]R\s*$")
    sec2_re = re.compile(r"(?i)^\s*(?:2\.?\s*)?CONTE[ÚU]DO\s+D[OE]\s+LIVRO\s+D[OE]\s+ALUNO(?:\s*\(Exposi[çc][ãa]o\))?\s*$")
    sec3_re = re.compile(r"(?i)^\s*(?:3\.?\s*)?ATIVIDADES?\s+D[OE]\s+ALUNO\s*$")

    chapters = []
    current_chapter = None
    current_aula = None
    current_section = None

    for item in extracted_data:
        text = item["text"].strip()

        # Check for Chapter header
        chapter_match = chapter_re.match(text)
        if chapter_match:
            number = chapter_match.group(1).zfill(2)
            theme = chapter_match.group(2).strip()
            current_chapter = {
                "number": number,
                "theme": theme,
                "aulas": []
            }
            chapters.append(current_chapter)
            current_aula = None
            current_section = None
            continue

        # Check for Aula header
        aula_match = aula_re.match(text)
        if aula_match:
            number = aula_match.group(1).zfill(2) # normalize to 01, 02...
            theme = aula_match.group(2).strip()
            current_aula = {
                "number": number,
                "theme": theme,
                "guia_do_professor": [],
                "conteudo_do_aluno": [],
                "atividades_do_aluno": []
            }

            # If an Aula is found before any Chapter, put it in a default Chapter 00
            if not current_chapter:
                current_chapter = {
                    "number": "00",
                    "theme": "Default Chapter",
                    "aulas": []
                }
                chapters.append(current_chapter)

            current_chapter["aulas"].append(current_aula)
            current_section = None
            continue

        if not current_aula:
            continue # ignore text before the first Aula

        # Check for Section headers
        if sec1_re.match(text):
            current_section = "guia_do_professor"
            continue
        elif sec2_re.match(text):
            current_section = "conteudo_do_aluno"
            continue
        elif sec3_re.match(text):
            current_section = "atividades_do_aluno"
            continue

        # Add content to current section
        if current_section:
            current_aula[current_section].append(item)

    # Assign unique IDs
    for chapter in chapters:
        c_num = int(chapter["number"])
        for aula in chapter["aulas"]:
            a_num = int(aula["number"])

            for i, item in enumerate(aula["guia_do_professor"], 1):
                item["id"] = f"C{c_num}A{a_num}P{i}"

            for i, item in enumerate(aula["conteudo_do_aluno"], 1):
                item["id"] = f"C{c_num}A{a_num}S{i}"

            for i, item in enumerate(aula["atividades_do_aluno"], 1):
                item["id"] = f"C{c_num}A{a_num}A{i}"

    return chapters


def compare_aula_sections(aula, nlp):
    """
    Compares the three internal sections of an Aula:
    - Guia do Professor vs Conteúdo do Aluno
    - Guia do Professor vs Atividades do Aluno
    - Conteúdo do Aluno vs Atividades do Aluno
    """
    guia = aula["guia_do_professor"]
    conteudo = aula["conteudo_do_aluno"]
    atividades = aula["atividades_do_aluno"]

    report = {
        "aula_info": f"Aula {aula['number']} - {aula['theme']}",
        "guia_vs_conteudo": {},
        "guia_vs_atividades": {},
        "conteudo_vs_atividades": {},
        "section_metrics": {
            "guia_do_professor": {},
            "conteudo_do_aluno": {},
            "atividades_do_aluno": {}
        }
    }

    # Evaluate individual sections
    if guia:
        report["section_metrics"]["guia_do_professor"] = {
            "entities": extract_named_entities(guia, nlp),
            "redundancies": detect_semantic_redundancy(guia, nlp),
            "vocabulary": evaluate_vocabulary(guia, nlp)
        }
    if conteudo:
        report["section_metrics"]["conteudo_do_aluno"] = {
            "entities": extract_named_entities(conteudo, nlp),
            "redundancies": detect_semantic_redundancy(conteudo, nlp),
            "vocabulary": evaluate_vocabulary(conteudo, nlp)
        }
    if atividades:
        report["section_metrics"]["atividades_do_aluno"] = {
            "entities": extract_named_entities(atividades, nlp),
            "redundancies": [],
            "vocabulary": evaluate_vocabulary(atividades, nlp)
        }

    # Compare Guia vs Conteudo
    if guia and conteudo:
        guia_ents = report["section_metrics"]["guia_do_professor"]["entities"]
        conteudo_ents = report["section_metrics"]["conteudo_do_aluno"]["entities"]

        report["guia_vs_conteudo"] = {
            "topic_order": analyze_topic_order(guia, conteudo, nlp),
            "ner_consistency": check_ner_consistency(guia_ents, conteudo_ents)
        }

    # Compare Guia vs Atividades
    if guia and atividades:
        guia_ents = report["section_metrics"]["guia_do_professor"]["entities"]
        atividades_ents = report["section_metrics"]["atividades_do_aluno"]["entities"]

        report["guia_vs_atividades"] = {
            "topic_order": analyze_topic_order(guia, atividades, nlp),
            "ner_consistency": check_ner_consistency(guia_ents, atividades_ents)
        }

    # Compare Conteudo vs Atividades
    if conteudo and atividades:
        conteudo_ents = report["section_metrics"]["conteudo_do_aluno"]["entities"]
        atividades_ents = report["section_metrics"]["atividades_do_aluno"]["entities"]

        report["conteudo_vs_atividades"] = {
            "topic_order": analyze_topic_order(conteudo, atividades, nlp),
            "ner_consistency": check_ner_consistency(conteudo_ents, atividades_ents)
        }

    return report


def extract_and_cache_pdf(filepath):
    """
    Computes MD5 hash of PDF, creates caching directory structure,
    extracts/parses texts if not cached, or loads from cache if it exists.
    """
    import json

    # 1. Compute MD5 hash of the file
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    pdf_hash = hasher.hexdigest()

    base_dir = os.path.join("processed_pdfs", pdf_hash)

    # If cache exists and looks populated (has chapters.json), just load it
    chapters_json_path = os.path.join(base_dir, "chapters.json")
    if os.path.exists(chapters_json_path):
        print(f"Loading cached extracted data for {filepath} from {base_dir}")
        with open(chapters_json_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    # Cache doesn't exist or is incomplete, extract from scratch
    print(f"Extracting data for {filepath} (Hash: {pdf_hash})")
    extracted_data = extract_text_from_pdf(filepath)
    chapters = parse_aulas(extracted_data)

    # Create directory structure and save individual files
    os.makedirs(base_dir, exist_ok=True)

    for chapter in chapters:
        chapter_dir = os.path.join(base_dir, f"Chapter_{chapter['number']}")
        os.makedirs(chapter_dir, exist_ok=True)

        for aula in chapter["aulas"]:
            aula_dir = os.path.join(chapter_dir, f"Aula_{aula['number']}")
            os.makedirs(aula_dir, exist_ok=True)

            # Save sections to separate JSON files
            with open(os.path.join(aula_dir, "teacher.json"), 'w', encoding='utf-8') as f:
                json.dump(aula["guia_do_professor"], f, indent=4, ensure_ascii=False)

            with open(os.path.join(aula_dir, "student.json"), 'w', encoding='utf-8') as f:
                json.dump(aula["conteudo_do_aluno"], f, indent=4, ensure_ascii=False)

            with open(os.path.join(aula_dir, "activities.json"), 'w', encoding='utf-8') as f:
                json.dump(aula["atividades_do_aluno"], f, indent=4, ensure_ascii=False)

    # Save the master chapters structure
    with open(chapters_json_path, 'w', encoding='utf-8') as f:
        json.dump(chapters, f, indent=4, ensure_ascii=False)

    return chapters


def process_aulas_from_pdf(filepath, output_json="aulas_report.json"):
    """
    Main orchestration function for Aula processing.
    Uses cached structure or extracts texts, parses chapters/aulas and their sections,
    compares them, and exports a JSON report.
    """
    import json
    import hashlib

    # 1. Get parsed Chapters and Aulas (either extracted fresh or from cache)
    chapters = extract_and_cache_pdf(filepath)

    # Calculate hash to find the root cache directory
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    pdf_hash = hasher.hexdigest()

    # 2. Collect all extracted text to detect language
    # We flatten all sections from all aulas from all chapters to pass to language detection
    all_extracted_texts = []
    for chapter in chapters:
        for aula in chapter["aulas"]:
            all_extracted_texts.extend(aula.get("guia_do_professor", []))
            all_extracted_texts.extend(aula.get("conteudo_do_aluno", []))
            all_extracted_texts.extend(aula.get("atividades_do_aluno", []))

    # 3. Detect language and load appropriate spaCy model
    if all_extracted_texts:
        nlp = detect_language_and_load_model(all_extracted_texts)
    else:
        raise ValueError("No text extracted from PDF to detect language.")

    # 4. Process and compare sections for each aula within each chapter
    aulas_reports = []
    total_aulas = 0
    for chapter in chapters:
        chapter_report = {
            "chapter_info": f"Chapter {chapter['number']} - {chapter['theme']}",
            "aulas": []
        }
        for aula in chapter["aulas"]:
            aula_report = compare_aula_sections(aula, nlp)
            chapter_report["aulas"].append(aula_report)
            total_aulas += 1

            # Save separate JSONs for the NLP processes inside the chapter/aula directory
            aula_dir = os.path.join("processed_pdfs", pdf_hash, f"Chapter_{chapter['number']}", f"Aula_{aula['number']}")
            os.makedirs(aula_dir, exist_ok=True)

            topic_order = {
                "guia_vs_conteudo": aula_report.get("guia_vs_conteudo", {}).get("topic_order", {}),
                "guia_vs_atividades": aula_report.get("guia_vs_atividades", {}).get("topic_order", {}),
                "conteudo_vs_atividades": aula_report.get("conteudo_vs_atividades", {}).get("topic_order", {})
            }
            with open(os.path.join(aula_dir, "topic_order.json"), 'w', encoding='utf-8') as f:
                json.dump(topic_order, f, indent=4, ensure_ascii=False)

            ner_consistency = {
                "guia_vs_conteudo": aula_report.get("guia_vs_conteudo", {}).get("ner_consistency", {}),
                "guia_vs_atividades": aula_report.get("guia_vs_atividades", {}).get("ner_consistency", {}),
                "conteudo_vs_atividades": aula_report.get("conteudo_vs_atividades", {}).get("ner_consistency", {})
            }
            with open(os.path.join(aula_dir, "ner_consistency.json"), 'w', encoding='utf-8') as f:
                json.dump(ner_consistency, f, indent=4, ensure_ascii=False)

            redundancies = {
                "guia_do_professor": aula_report.get("section_metrics", {}).get("guia_do_professor", {}).get("redundancies", []),
                "conteudo_do_aluno": aula_report.get("section_metrics", {}).get("conteudo_do_aluno", {}).get("redundancies", []),
                "atividades_do_aluno": aula_report.get("section_metrics", {}).get("atividades_do_aluno", {}).get("redundancies", [])
            }
            with open(os.path.join(aula_dir, "redundancies.json"), 'w', encoding='utf-8') as f:
                json.dump(redundancies, f, indent=4, ensure_ascii=False)

            vocabulary = {
                "guia_do_professor": aula_report.get("section_metrics", {}).get("guia_do_professor", {}).get("vocabulary", {}),
                "conteudo_do_aluno": aula_report.get("section_metrics", {}).get("conteudo_do_aluno", {}).get("vocabulary", {}),
                "atividades_do_aluno": aula_report.get("section_metrics", {}).get("atividades_do_aluno", {}).get("vocabulary", {})
            }
            with open(os.path.join(aula_dir, "vocabulary.json"), 'w', encoding='utf-8') as f:
                json.dump(vocabulary, f, indent=4, ensure_ascii=False)

        aulas_reports.append(chapter_report)

    # Compile the final response payload
    response_data = {
        "metadata": {
            "source_file": filepath,
            "total_chapters_parsed": len(chapters),
            "total_aulas_parsed": total_aulas
        },
        "chapters_analysis": aulas_reports
    }

    # Export to JSON
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(response_data, f, indent=4, ensure_ascii=False)

    print(f"Aula comparison data successfully exported to {output_json}")
    return response_data
