# StemgenRT-5.8 draft paper

[Read the PDF](stemgenrt-5.8-draft-v0.1.pdf) · [Markdown source](draft.md) · [Bibliography](references.bib)

This working manuscript retains its 11 September 2026 v0.1 results, with a
naming revision dated 20 September. It describes the **historical four-state
C204 predecessor**. Its architecture and measurements do not evaluate the
current eight-state `StemgenRT58` implementation in `stemgenrt`.

The 5.8 suffix means 256 samples of graph-plus-host algorithmic delay at
44.1 kHz, rounded to 5.8 ms; model revisions are separate. The underlying
evidence ledger and evaluation artifacts remain unpublished. Author details,
untouched-song evaluation, and other submission work remain open in the draft.

## Rebuild

Use Python 3.12, DejaVu fonts, and the system libraries required by WeasyPrint.
From the repository root, install the document dependencies in a separate
Python environment and run:

```bash
python -m pip install matplotlib==3.10.5 Markdown==3.10.3 weasyprint==70.0 PyMuPDF==1.28.2
python paper/render_figure.py
python paper/build_pdf.py
```

The scripts regenerate `architecture.svg` and `stemgenrt-5.8-draft-v0.1.pdf`.
The PDF builder checks manuscript content and links, then prints its validation
summary. Document generation does not require model weights or training data.
