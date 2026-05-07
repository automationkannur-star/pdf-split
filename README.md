## PDF OCR Playground (Python)

Simple web UI to upload a PDF, switch between OCR engines, and compare extracted text.

### What you get

- Upload a PDF
- Pick an OCR engine (Tesseract, docTR, Surya, AWS Textract if installed/configured)
- View per-page + combined extracted text
- Download extracted text as `.txt`

### Quickstart

1) Create and activate a venv

```bash
python -m venv .venv
.\.venv\Scripts\activate
```

2) Install base dependencies

```bash
pip install -r requirements.txt
```

3) Run the app

```bash
streamlit run app.py
```

### OCR engines

#### Tesseract (recommended to start)

- Install the **Tesseract** binary on Windows (e.g. from UB Mannheim build).
- Then install Python bindings:

```bash
pip install -r requirements-tesseract.txt
```

If Tesseract is not on PATH, set `TESSERACT_CMD` in the UI (or as an environment variable).

#### docTR

```bash
pip install -r requirements-doctr.txt
```

Notes:
- docTR can use PyTorch; GPU is optional but helps.
- First run can download weights.

#### Surya OCR

```bash
pip install -r requirements-surya.txt
```

Notes:
- Surya packaging changes over time; if import fails the app will show the exact error and you can adjust.

#### AWS Textract

```bash
pip install -r requirements-textract.txt
```

Configure credentials in either way:

- `AWS_PROFILE` + configured AWS CLI profile, or
- `AWS_ACCESS_KEY_ID` + `AWS_SECRET_ACCESS_KEY` (plus optional `AWS_SESSION_TOKEN`)

Also set region:

- `AWS_REGION` (for example `ap-south-1`)

In the app, these can be entered in the sidebar fields.

### Tips for evaluation

- Use the same PDF and language across engines.
- Compare speed vs accuracy:
  - total runtime
  - per-page runtime
  - text quality (missing lines, ordering, tables)

