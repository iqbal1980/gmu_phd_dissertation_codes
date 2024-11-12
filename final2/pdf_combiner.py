import os
from PyPDF2 import PdfMerger
from tqdm import tqdm

def merge_pdfs(input_folder, output_filename):
    # Create a PdfMerger object
    merger = PdfMerger()

    # Get a list of all PDF files in the input folder
    pdf_files = [f for f in os.listdir(input_folder) if f.endswith('.pdf')]

    # Sort the files to ensure a consistent order
    pdf_files.sort()

    # Merge PDFs
    for pdf in tqdm(pdf_files, desc="Merging PDFs"):
        file_path = os.path.join(input_folder, pdf)
        merger.append(file_path)

    # Write the merged PDF to the output file
    with open(output_filename, 'wb') as output_file:
        merger.write(output_file)

    print(f"Merged {len(pdf_files)} PDFs into {output_filename}")

# Specify the input folder and output filename
input_folder = "C:\\research_pdfs"
output_filename = 'merged_output.pdf'

# Call the function to merge PDFs
merge_pdfs(input_folder, output_filename)