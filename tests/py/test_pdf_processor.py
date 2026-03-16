import unittest
import os
import json
import sys

# Add parent directory to path so we can import pdf_processor
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from pdf_processor import process_pdf_and_export_json, compare_pedagogic_materials, process_aulas_from_pdf

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

class TestPDFProcessor(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
        Creates temporary PDF files for testing before any tests are run.
        """
        cls.test_pdf_pt = "test_document_pt.pdf"
        cls.test_json_pt = "response_pt.json"

        cls.test_pdf_en = "test_document_en.pdf"
        cls.test_json_en = "response_en.json"

        cls.test_teacher_pdf = "teacher_document.pdf"
        cls.test_student_pdf = "student_document.pdf"
        cls.test_comparison_json = "comparison_response.json"

        cls.test_aula_pdf = "aula_document.pdf"
        cls.test_aula_json = "aulas_report.json"

        cls.test_designer_pdf = "test_designer.pdf"
        cls.test_json_designer = "response_designer.json"

        cls.test_newline_pdf = "test_newline.pdf"

        paragraphs_pt = [
            "O presidente do Brasil, Luiz Inácio Lula da Silva, visitou Brasília em 15 de novembro de 2023.",
            "Nesta reunião, foram discutidos os novos avanços na área da tecnologia, inteligência artificial e os impactos ambientais na Amazônia.",
            "Luiz Inácio Lula da Silva esteve em Brasília no dia 15 de novembro de 2023 para uma visita oficial." # Semantic redundancy
        ]
        cls.create_sample_pdf(cls.test_pdf_pt, paragraphs_pt)

        paragraphs_en = [
            "The President of the United States, Joe Biden, visited Washington D.C. on November 15, 2023.",
            "During this meeting, new advances in technology, artificial intelligence, and environmental impacts were discussed.",
            "Joe Biden was in Washington D.C. on the 15th of November 2023 for an official visit." # Semantic redundancy
        ]
        cls.create_sample_pdf(cls.test_pdf_en, paragraphs_en)

        teacher_paragraphs = [
            "Photosynthesis is a process used by plants and other organisms to convert light energy into chemical energy.",
            "This process was first extensively studied by Jan Ingenhousz in 1779.",
            "Jan Ingenhousz studied this process in 1779." # redundancy
        ]
        cls.create_sample_pdf(cls.test_teacher_pdf, teacher_paragraphs)

        student_paragraphs = [
            "Plants use photosynthesis to transform sunlight into chemical energy.",
            "It was discovered by Jan Ingenhousz.",
            "Jan Ingenhousz found it out." # redundancy
        ]
        cls.create_sample_pdf(cls.test_student_pdf, student_paragraphs)

        aula_paragraphs = [
            "Aula 01 - Introdução à Fotossíntese",
            "1. GUIA DO PROFESSOR",
            "Photosynthesis is a process used by plants and other organisms to convert light energy into chemical energy.",
            "This process was first extensively studied by Jan Ingenhousz in 1779.",
            "2. CONTEÚDO DO LIVRO DO ALUNO",
            "Plants use photosynthesis to transform sunlight into chemical energy.",
            "It was discovered by Jan Ingenhousz.",
            "3. ATIVIDADES DO ALUNO",
            "What is photosynthesis? Who discovered it?",
            "Explain the process of photosynthesis."
        ]
        cls.create_sample_pdf(cls.test_aula_pdf, aula_paragraphs)

        designer_paragraphs = [
            "This is normal text.",
            "Instrução Visual: Make this bold\nAnd this is some other instruction.",
            "More normal text.",
            "[INSTRUÇÃO PARA O DIAGRAMADOR] Place an image here.",
            "The final normal text."
        ]
        cls.create_sample_pdf(cls.test_designer_pdf, designer_paragraphs)

        newline_paragraphs = [
            "This paragraph has\na newline character.",
            "This one\nalso has\nnewlines."
        ]
        cls.create_sample_pdf_with_newlines(cls.test_newline_pdf, newline_paragraphs)

    @classmethod
    def create_sample_pdf_with_newlines(cls, filename, paragraphs):
        c = canvas.Canvas(filename, pagesize=letter)
        width, height = letter
        y_position = height - 50
        for text_content in paragraphs:
            text_obj = c.beginText(50, y_position)
            text_obj.textLines(text_content)
            c.drawText(text_obj)
            # Adjust y_position based on number of lines
            lines = text_content.count('\n') + 1
            y_position -= 40 * lines
        c.save()

    @classmethod
    def create_sample_pdf(cls, filename, paragraphs):
        c = canvas.Canvas(filename, pagesize=letter)
        width, height = letter
        y_position = height - 50
        for text in paragraphs:
            c.drawString(50, y_position, text)
            y_position -= 40
        c.save()

    @classmethod
    def tearDownClass(cls):
        """
        Cleans up the generated PDF files after all tests finish, but keeps the responses.
        """
        files_to_remove = [
            cls.test_pdf_pt,
            cls.test_pdf_en,
            cls.test_teacher_pdf, cls.test_student_pdf,
            cls.test_aula_pdf,
            cls.test_designer_pdf,
            cls.test_json_designer,
            cls.test_newline_pdf
        ]
        for f in files_to_remove:
            if os.path.exists(f):
                os.remove(f)

    def test_pipeline_portuguese(self):
        # Run the processor
        result = process_pdf_and_export_json(self.test_pdf_pt, self.test_json_pt)

        # Verify JSON file was created
        self.assertTrue(os.path.exists(self.test_json_pt))

        with open(self.test_json_pt, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Verify metadata
        self.assertEqual(data["metadata"]["source_file"], self.test_pdf_pt)
        self.assertEqual(data["metadata"]["total_paragraphs"], 3)

        # Verify entities
        self.assertIn("Luiz Inácio Lula da Silva", data["named_entities"]["PERSON"])
        self.assertIn("Brasília", data["named_entities"]["LOCATION"])

        # Verify redundancies (should be exactly 1)
        self.assertEqual(len(data["redundancies"]), 1)
        redundancy = data["redundancies"][0]
        self.assertGreater(redundancy["score"], 0.85)

    def test_pipeline_english(self):
        # Run the processor
        result = process_pdf_and_export_json(self.test_pdf_en, self.test_json_en)

        # Verify JSON file was created
        self.assertTrue(os.path.exists(self.test_json_en))

        with open(self.test_json_en, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Verify metadata
        self.assertEqual(data["metadata"]["source_file"], self.test_pdf_en)
        self.assertEqual(data["metadata"]["total_paragraphs"], 3)

        # Verify entities
        self.assertIn("Joe Biden", data["named_entities"]["PERSON"])
        self.assertIn("Washington D.C.", data["named_entities"]["LOCATION"])

        # Verify redundancies (should be exactly 1)
        self.assertEqual(len(data["redundancies"]), 1)
        redundancy = data["redundancies"][0]
        self.assertGreater(redundancy["score"], 0.85)

    def test_compare_pedagogic_materials(self):
        result = compare_pedagogic_materials(self.test_teacher_pdf, self.test_student_pdf, self.test_comparison_json)

        self.assertTrue(os.path.exists(self.test_comparison_json))

        with open(self.test_comparison_json, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.assertEqual(data["metadata"]["teacher_source"], self.test_teacher_pdf)
        self.assertEqual(data["metadata"]["student_source"], self.test_student_pdf)
        self.assertEqual(data["metadata"]["teacher_total_paragraphs"], 3)
        self.assertEqual(data["metadata"]["student_total_paragraphs"], 3)

        # redundancies isolated
        self.assertEqual(len(data["teacher_redundancies"]), 1)
        self.assertEqual(len(data["student_redundancies"]), 1)

        # NER consistency
        ner = data["ner_consistency"]
        self.assertIn("Jan Ingenhousz", ner["PERSON"]["in_both"])
        self.assertIn("1779", ner["DATE"]["missing_in_student"]) # Present in teacher but not student

        # Sense Validation
        sense = data["sense_validation"]
        self.assertTrue(len(sense) > 0)
        # Assuming the first paragraphs will have a good match score:
        # "Photosynthesis is a process..." vs "Plants use photosynthesis..."
        match_scores = [m["score"] for m in sense]
        self.assertTrue(any(score > 0.85 for score in match_scores))

        # Topic Order
        order = data["topic_order"]
        self.assertGreater(order["match_ratio"], 0.0)
        self.assertTrue(len(order["teacher_key_topics"]) > 0)
        self.assertTrue(len(order["student_key_topics"]) > 0)

        # Vocabulary Evaluation
        self.assertIn("teacher_vocabulary", data)
        self.assertIn("student_vocabulary", data)
        self.assertIn("term_frequency", data["teacher_vocabulary"])
        self.assertIn("rare_words_and_jargon", data["teacher_vocabulary"])
        self.assertIn("lexical_richness_ttr", data["teacher_vocabulary"])
        self.assertIn("readability_avg_zipf", data["teacher_vocabulary"])

    def test_process_aulas_from_pdf(self):
        result = process_aulas_from_pdf(self.test_aula_pdf, self.test_aula_json)

        self.assertTrue(os.path.exists(self.test_aula_json))

        with open(self.test_aula_json, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.assertEqual(data["metadata"]["source_file"], self.test_aula_pdf)
        self.assertEqual(data["metadata"]["total_aulas_parsed"], 1)

        self.assertEqual(len(data["aulas_analysis"]), 1)
        aula_report = data["aulas_analysis"][0]

        self.assertEqual(aula_report["aula_info"], "Aula 01 - Introdução à Fotossíntese")

        # Check section metrics exist
        self.assertIn("guia_do_professor", aula_report["section_metrics"])
        self.assertIn("conteudo_do_aluno", aula_report["section_metrics"])
        self.assertIn("atividades_do_aluno", aula_report["section_metrics"])

        # Check that comparison objects are populated
        self.assertIn("guia_vs_conteudo", aula_report)
        self.assertIn("guia_vs_atividades", aula_report)
        self.assertIn("conteudo_vs_atividades", aula_report)

        # Basic spot check of contents
        guia_vs_conteudo = aula_report["guia_vs_conteudo"]
        self.assertNotIn("sense_validation", guia_vs_conteudo) # No longer validating paragraphs across sections
        self.assertIn("topic_order", guia_vs_conteudo)
        self.assertIn("ner_consistency", guia_vs_conteudo)

    def test_pipeline_designer_instructions(self):
        # Run the processor
        result = process_pdf_and_export_json(self.test_designer_pdf, self.test_json_designer)

        # Verify JSON file was created
        self.assertTrue(os.path.exists(self.test_json_designer))

        with open(self.test_json_designer, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Verify metadata (total_paragraphs should be 3, since 2 were entirely filtered out)
        self.assertEqual(data["metadata"]["source_file"], self.test_designer_pdf)
        self.assertEqual(data["metadata"]["total_paragraphs"], 3)

        # Verify the content
        extracted_texts = [item["text"] for item in data["extracted_text"]]
        self.assertIn("This is normal text.", extracted_texts)
        self.assertIn("More normal text.", extracted_texts)
        self.assertIn("The final normal text.", extracted_texts)

        # Verify instructions were completely removed
        for text in extracted_texts:
            self.assertNotIn("Instrução Visual", text)
            self.assertNotIn("DIAGRAMADOR", text)

    def test_pipeline_newline_handling(self):
        # Extract text directly
        from pdf_processor import extract_text_from_pdf

        extracted_data = extract_text_from_pdf(self.test_newline_pdf)

        extracted_texts = [item["text"] for item in extracted_data]
        self.assertEqual(len(extracted_texts), 2)

        # Verify newlines are replaced with space
        self.assertEqual(extracted_texts[0], "This paragraph has a newline character.")
        self.assertEqual(extracted_texts[1], "This one also has newlines.")

if __name__ == "__main__":
    unittest.main()
