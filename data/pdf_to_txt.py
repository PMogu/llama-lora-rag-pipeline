import pdfplumber
from pathlib import Path
import re

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

PDF_DIR = BASE_DIR / "data" / "raw" / "pdf"
OUT_DIR = BASE_DIR / "data" / "raw" / "pdf2txt"

OUT_DIR.mkdir(exist_ok=True)

def extract_pdf_text(pdf_path: Path) -> str:
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
    return "\n\n".join(pages)

def clean_text(text: str) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def main():
    for pdf_file in PDF_DIR.glob("*.pdf"):
        raw_text = extract_pdf_text(pdf_file)
        clean = clean_text(raw_text)

        out_path = OUT_DIR / (pdf_file.stem + ".txt")
        out_path.write_text(clean, encoding="utf-8")

        print(f"[OK] {pdf_file.name} → {out_path.name} ({len(clean)} chars)")

if __name__ == "__main__":
    main()
