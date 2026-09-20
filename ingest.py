from pathlib import Path
import pickle
import re

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------
# PROJECT PATHS
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
VECTOR_DIR = BASE_DIR / "vector_store"
VECTOR_DIR.mkdir(exist_ok=True)

MODEL_NAME = "all-MiniLM-L6-v2"


# ---------------------------------------------------------
# CHAPTER / UNIT INFORMATION
# ---------------------------------------------------------
# We use fixed unit names instead of trying to guess them
# from the scanned PDF.
# ---------------------------------------------------------

FIRST_YEAR_UNITS = {
    1: "Physical Quantities and Measurements",
    2: "Vectors",
    3: "Translatory Motion",
    4: "Rotational and Circular Motion",
    5: "Work and Kinetic Energy",
    6: "Fluid Dynamics",
    7: "Deformation of Solids",
    8: "Heat and Thermodynamics",
    9: "Waves",
    10: "Electrostatics",
    11: "Electricity",
    12: "Magnetism",
    13: "Special Theory of Relativity",
    14: "Particle Physics",
}

SECOND_YEAR_UNITS = {
    15: "Gravitation",
    16: "Statistical Mechanics and Thermodynamics",
    17: "Simple Harmonic Motion",
    18: "Diffraction and Interference",
    19: "Electric Potential and Capacitor",
    20: "Alternating Current",
    21: "Quantum Physics",
    22: "Nuclear Physics",
    23: "Cosmology",
    24: "Earth's Climate",
    25: "Medical Imaging",
    26: "Nature of Science",
}


# ---------------------------------------------------------
# TEXT CLEANING
# ---------------------------------------------------------

def clean_text(text):
    if not text:
        return ""

    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove common website watermarks from our scanned books
    text = re.sub(
        r"(www\.)?(studyplusplus|taleem360)\.com",
        " ",
        text,
        flags=re.IGNORECASE,
    )

    return text.strip()


# ---------------------------------------------------------
# OCR
# ---------------------------------------------------------

def extract_scanned_pdf(pdf_path):
    """
    Convert scanned PDF pages to images and OCR them.

    Returns:
        [
            {
                "pdf_page": 1,
                "text": "..."
            },
            ...
        ]
    """

    try:
        import fitz
        import pytesseract
        from PIL import Image

        pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        
    except ImportError:
        raise RuntimeError(
            "OCR packages are missing. Install PyMuPDF and pytesseract first."
        )

    document = fitz.open(pdf_path)
    pages = []

    print(f"\nReading scanned book: {pdf_path.name}")
    print(f"Total PDF pages: {len(document)}")

    for page_number, page in enumerate(document, start=1):

        # Render at roughly 200 DPI.
        matrix = fitz.Matrix(2.0, 2.0)
        pix = page.get_pixmap(matrix=matrix, alpha=False)

        image = Image.frombytes(
            "RGB",
            [pix.width, pix.height],
            pix.samples,
        )

        text = pytesseract.image_to_string(
            image,
            lang="eng",
            config="--psm 6",
        )

        text = clean_text(text)

        pages.append(
            {
                "pdf_page": page_number,
                "text": text,
            }
        )

        print(
            f"\rOCR page {page_number}/{len(document)}",
            end="",
            flush=True,
        )

    print("\nOCR finished.")

    document.close()
    return pages


# ---------------------------------------------------------
# DETECT UNIT FROM OCR TEXT
# ---------------------------------------------------------

def find_unit_in_text(text, units):
    normalized = text.lower()

    # First try unit/chapter number
    for number, title in units.items():

        patterns = [
            rf"\bunit\s*[-:]?\s*{number}\b",
            rf"\bchapter\s*[-:]?\s*{number}\b",
        ]

        for pattern in patterns:
            if re.search(pattern, normalized, flags=re.IGNORECASE):
                return number

    # Then try exact-ish unit title
    for number, title in units.items():
        if title.lower() in normalized:
            return number

    return None


def assign_units(pages, units):
    """
    Detect unit-opening pages and carry the detected unit forward
    until another unit is found.
    """

    current_unit = None
    found_units = set()

    for page in pages:

        detected = find_unit_in_text(page["text"], units)

        if detected is not None:
            current_unit = detected
            found_units.add(detected)

        page["unit"] = current_unit

        if current_unit:
            page["unit_title"] = units[current_unit]
        else:
            page["unit_title"] = "Front Matter"

    print("\nUnits detected:", sorted(found_units))

    missing = sorted(set(units.keys()) - found_units)

    if missing:
        print("Units not automatically detected:", missing)

    return pages


# ---------------------------------------------------------
# CHUNKING
# ---------------------------------------------------------

def split_text(text, chunk_size=1200, overlap=200):

    text = text.strip()

    if not text:
        return []

    chunks = []

    start = 0

    while start < len(text):

        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = max(0, end - overlap)

    return chunks


def create_chunks(pages, book_name):

    chunks = []

    for page in pages:

        # Don't index front matter as textbook chapter content.
        if page["unit"] is None:
            continue

        page_chunks = split_text(page["text"])

        for chunk_number, text in enumerate(page_chunks):

            chunks.append(
                {
                    "text": text,
                    "book": book_name,
                    "pdf_page": page["pdf_page"],
                    "unit": page["unit"],
                    "unit_title": page["unit_title"],
                    "chunk_number": chunk_number,
                }
            )

    return chunks


# ---------------------------------------------------------
# CREATE FAISS DATABASE
# ---------------------------------------------------------

def build_vector_store(chunks, output_dir, model):

    if not chunks:
        raise RuntimeError("No textbook chunks were created.")

    output_dir.mkdir(parents=True, exist_ok=True)

    texts = [item["text"] for item in chunks]

    print(f"\nCreating embeddings for {len(texts)} chunks...")

    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    embeddings = np.asarray(embeddings, dtype="float32")

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    faiss.write_index(
        index,
        str(output_dir / "index.faiss"),
    )

    with open(output_dir / "metadata.pkl", "wb") as file:
        pickle.dump(chunks, file)

    print(f"Saved {len(chunks)} chunks -> {output_dir}")


# ---------------------------------------------------------
# PROCESS ONE BOOK
# ---------------------------------------------------------

def process_book(pdf_path, book_name, units, output_name, model):

    if not pdf_path.exists():
        raise FileNotFoundError(
            f"\nCould not find:\n{pdf_path}\n"
        )

    pages = extract_scanned_pdf(pdf_path)

    pages = assign_units(
        pages,
        units,
    )

    chunks = create_chunks(
        pages,
        book_name,
    )

    build_vector_store(
        chunks,
        VECTOR_DIR / output_name,
        model,
    )


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():

    print("=" * 60)
    print("FBISE PHYSICS SCANNED-TEXTBOOK INGESTION")
    print("=" * 60)

    print("\nLoading embedding model...")

    model = SentenceTransformer(MODEL_NAME)

    first_year_pdf = DATA_DIR / "first_year_physics.pdf"
    second_year_pdf = DATA_DIR / "second_year_physics.pdf"

    process_book(
        first_year_pdf,
        "FSC Part-I / HSSC-I / First Year",
        FIRST_YEAR_UNITS,
        "first_year",
        model,
    )

    process_book(
        second_year_pdf,
        "FSC Part-II / HSSC-II / Second Year",
        SECOND_YEAR_UNITS,
        "second_year",
        model,
    )

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)

    print(
        "\nBoth scanned textbooks have been OCR'd and indexed."
    )


if __name__ == "__main__":
    main()