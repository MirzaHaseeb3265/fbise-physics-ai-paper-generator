# FBISE Physics AI Paper Generator

A Streamlit + RAG application for generating HSSC-I and HSSC-II Physics papers grounded in the supplied textbooks.

## Paper patterns calibrated from the supplied references

### Simple Test
Matches the compact `Chapter 1,2,3,4,5` reference: FSC heading, topics, compact MCQs, Short Questions and Long Questions. The teacher can independently turn OR alternatives on/off.

### FLP Paper
Matches the supplied FLP structure: `Physics HSC-I/HSC-II - FLP`, `SECTION-A (OBJECTIVE)`, `SECTION-B (SHORT QUESTIONS)`, and `SECTION-C (LONG QUESTIONS)`, with configurable attempt counts and OR alternatives.

### FBISE Standard difficulty
The prompt is calibrated to the supplied examples: concise 3-mark conceptual/reasoning questions, dimensional homogeneity, conditions/comparisons, everyday applications, derivations, and solvable numericals; MCQs use plausible distractors. The samples are style references only. Textbook retrieval remains the knowledge source.

## Project structure

```text
fbise-paper-generator/
  app.py
  ingest.py
  requirements.txt
  README.md
  data/
    first_year_physics.pdf
    second_year_physics.pdf
  templates/
    normal_sample.docx
    flp_sample.docx
  vector_store/
    first_year/
    second_year/
  utils/
```

## One-time setup on your computer

1. Install Python 3.11 (recommended).
2. Open the project folder in Terminal/PowerShell.
3. Create a virtual environment:

```bash
python -m venv .venv
```

Windows activation:

```powershell
.venv\Scripts\activate
```

4. Install dependencies:

```bash
pip install -r requirements.txt
```

5. Download the two textbook PDFs from your Google Drive folder and rename/copy them exactly to:

```text
data/first_year_physics.pdf
data/second_year_physics.pdf
```

6. Build embeddings **once**:

```bash
python ingest.py
```

Expected output:

```text
vector_store/first_year/index.faiss
vector_store/first_year/metadata.pkl
vector_store/second_year/index.faiss
vector_store/second_year/metadata.pkl
```

7. Create `.streamlit/secrets.toml` locally:

```toml
GROQ_API_KEY = "your_key_here"
LLM_MODEL = "llama-3.3-70b-versatile"
```

Do not commit this secrets file.

8. Test locally:

```bash
streamlit run app.py
```

## Deploy using only GitHub + Streamlit Community Cloud

### A. Push to GitHub

Create a new GitHub repository, for example `fbise-physics-paper-generator`. In this project folder run:

```bash
git init
git add .
git commit -m "Initial FBISE Physics paper generator"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/fbise-physics-paper-generator.git
git push -u origin main
```

**Important:** commit the generated `vector_store/first_year` and `vector_store/second_year` files. Streamlit should load the saved indexes, not regenerate embeddings at startup. You may also commit the textbook PDFs only if you have permission to redistribute them. The deployed app does not require PDFs once the indexes are present.

### B. Deploy on Streamlit Community Cloud

1. Sign in to Streamlit Community Cloud with GitHub.
2. Click **Create app** / **New app**.
3. Select your GitHub repository.
4. Branch: `main`.
5. Main file path: `app.py`.
6. Open **Advanced settings / Secrets** and add:

```toml
GROQ_API_KEY = "your_real_key"
LLM_MODEL = "llama-3.3-70b-versatile"
```

7. Click **Deploy**.
8. Wait while Streamlit installs `requirements.txt`.
9. Test both First Year and Second Year, Simple and FLP patterns, and all OR modes.

## How the RAG pipeline works

```text
ONE TIME LOCALLY
PDF -> ingest.py -> chapter-aware chunks -> SentenceTransformer -> FAISS -> saved index

NORMAL STREAMLIT USE
teacher settings -> selected FAISS index -> chapter-filtered retrieval -> textbook chunks -> LLM -> validation -> preview -> DOCX/PDF
```

The embedding model is cached with `st.cache_resource`. The app does not embed the textbooks during normal generation.

## Recommended test after deployment

Select one chapter only, enter a very specific topic from that chapter, generate a paper, and expand **Teacher grounding / textbook sources**. Confirm that retrieved chunks show the selected chapter and relevant textbook pages. Repeat for both classes.

## Common errors

- **FAISS index missing:** run `python ingest.py`, then commit and push `vector_store/`.
- **API key missing:** add `GROQ_API_KEY` to Streamlit app Secrets.
- **No chapters appear:** chapter headings were not detected in the PDF. Inspect metadata and adjust `CHAPTER_RE` in `ingest.py` to the book's exact heading format.
- **Wrong chapter retrieval:** confirm metadata chapter labels, then rebuild the indexes.
- **Model unavailable:** change `LLM_MODEL` in Streamlit Secrets to a currently available Groq chat model that supports JSON output.
- **Slow first load:** SentenceTransformer downloads on the first Streamlit boot, then is cached.
- **PDF symbols:** DOCX generally preserves Unicode physics symbols better. ReportLab export uses available Unicode-compatible fonts where possible.

## Security

Never place an API key in `app.py`, GitHub commits, README, or `.env` committed to GitHub. Use Streamlit Secrets in production.
