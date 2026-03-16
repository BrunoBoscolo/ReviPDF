import sys
import os
import json
from pdf_processor import process_pdf_and_export_json

def main():
    if len(sys.argv) < 2:
        print("Usage: python test_custom_pdf.py <path_to_pdf>")
        sys.exit(1)

    pdf_path = sys.argv[1]

    if not os.path.isfile(pdf_path):
        print(f"Error: File '{pdf_path}' does not exist.")
        sys.exit(1)

    # By default, process_pdf_and_export_json outputs to response.json
    # but let's make it output to a specific name based on the input PDF
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    output_json = f"{base_name}_output.json"

    print(f"Processing '{pdf_path}'...")
    try:
        result = process_pdf_and_export_json(pdf_path, output_json)
        print(f"\nSuccess! Processed {result['metadata']['total_paragraphs']} paragraphs.")
        print(f"Data exported to '{output_json}'.\n")

        # Optional: Print a small sample of the extracted text for quick visual verification
        print("Sample of extracted text (first 3 paragraphs):")
        for i, item in enumerate(result['extracted_text'][:3], 1):
            print(f"  [{i}] {item['text']}")

    except Exception as e:
        print(f"An error occurred while processing the PDF: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
