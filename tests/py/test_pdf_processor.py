import unittest
import os
import json
import sys

# Add parent directory to path so we can import pdf_processor
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from pdf_processor import process_pdf_and_export_json

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
        Cleans up the generated files after all tests finish.
        """
        files_to_remove = [
            cls.test_pdf_pt, cls.test_json_pt,
            cls.test_pdf_en, cls.test_json_en
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

if __name__ == "__main__":
    unittest.main()
