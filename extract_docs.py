import sys
from docx import Document
import PyPDF2

def extract_pdf(filepath):
    try:
        with open(filepath, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
    except Exception as e:
        return f"Error reading PDF: {e}"

def extract_docx(filepath):
    try:
        doc = Document(filepath)
        text = ""
        for para in doc.paragraphs:
            text += para.text + "\n"
        return text
    except Exception as e:
        return f"Error reading DOCX: {e}"

def extract_doc(filepath):
    try:
        import subprocess
        result = subprocess.run(['antiword', filepath], capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout
        else:
            # Try alternative method
            import docx2txt
            return docx2txt.process(filepath)
    except Exception as e:
        return f"Error reading DOC: {e}"

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python extract_docs.py <filepath>")
        sys.exit(1)
    
    filepath = sys.argv[1]
    
    if filepath.endswith('.pdf'):
        content = extract_pdf(filepath)
    elif filepath.endswith('.docx'):
        content = extract_docx(filepath)
    elif filepath.endswith('.doc'):
        content = extract_doc(filepath)
    else:
        content = "Unsupported file format"
    
    print(content)
