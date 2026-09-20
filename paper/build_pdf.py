"""Build a readable PDF from the manuscript without changing its contents.

Install the dependencies listed in paper/README.md, then run from the
repository root: python paper/build_pdf.py
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
import re

import markdown
import pymupdf
from weasyprint import HTML

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "draft.md"
OUTPUT = HERE / "stemgenrt-5.8-draft-v0.1.pdf"

source = SOURCE.read_text()
body = markdown.markdown(source, extensions=["tables", "fenced_code", "toc"])
# Keep the vector figure and its caption together on a landscape page.
body, figure_count = re.subn(
    r'<p>(<img[^>]+src="architecture\.svg"[^>]*>)</p>\s*<p><em>(Figure 1\..*?)</em></p>',
    r'<figure>\1<figcaption>\2</figcaption></figure>',
    body,
    flags=re.S,
)
if figure_count != 1:
    raise ValueError("Expected exactly one architecture figure and caption")
# Place the full-size figure after the manuscript, as a standalone plate. This
# keeps the portrait text flowing without sparse pages around a landscape insert.
figure = re.search(r"<figure>.*?</figure>", body, flags=re.S).group(0)
body = body.replace(figure, "", 1)
body += figure
# Local companion documents remain named in the text; a shared PDF should not
# contain machine-specific file:// links. Published references stay clickable.
body = re.sub(
    r'<a href="([^"#]+)"[^>]*>(.*?)</a>',
    lambda m: m.group(0) if m[1].startswith(("https://", "http://")) else m[2],
    body,
    flags=re.S,
)

css = """
@page {
  size: A4;
  margin: 20mm 19mm 20mm 21mm;
  @top-left {
    content: "STEMGENRT-5.8 · WORKING MANUSCRIPT";
    font: 8pt "DejaVu Sans"; color: #61717d;
  }
  @bottom-left {
    content: "Draft v0.1 · 11 Sep 2026 · Naming 20 Sep 2026";
    font: 8pt "DejaVu Sans"; color: #61717d;
  }
  @bottom-right {
    content: counter(page);
    font: 8pt "DejaVu Sans"; color: #61717d;
  }
}
@page :first { @top-left { content: none; } }
@page architecture {
  size: A4 landscape;
  margin: 18mm;
}
html { font: 10.5pt/1.35 "DejaVu Serif", serif; color: #182b3a; }
body { margin: 0; }
h1, h2, h3 { font-family: "DejaVu Sans", sans-serif; line-height: 1.22; }
h1 { font-size: 20pt; font-weight: 700; margin: 0 0 16pt; color: #163f54; }
h2 { font-size: 13pt; margin: 18pt 0 8pt; break-after: avoid; }
h3 { font-size: 11pt; margin: 13pt 0 6pt; break-after: avoid; }
p { margin: 0 0 9pt; orphans: 3; widows: 3; }
body > p:nth-of-type(1), body > p:nth-of-type(2) {
  font-family: "DejaVu Sans"; font-size: 9pt; color: #61717d;
}
a { color: #1d5674; text-decoration: none; overflow-wrap: anywhere; }
code { font: .88em "DejaVu Sans Mono", monospace; overflow-wrap: anywhere; }
pre {
  font: 8.3pt/1.45 "DejaVu Sans Mono", monospace;
  white-space: pre-wrap; overflow-wrap: anywhere;
  padding: 9pt 11pt; margin: 10pt 0;
  background: #f2f5f7; border-left: 2pt solid #7292a4;
  break-inside: avoid;
}
pre code { font: inherit; }
table { width: 100%; border-collapse: collapse; font-size: 8.6pt; line-height: 1.35; margin: 11pt 0; }
thead { display: table-header-group; }
th { text-align: left; font-family: "DejaVu Sans"; background: #eaf1f5; font-weight: 600; }
td, th { padding: 6pt 5pt; border-bottom: .5pt solid #c8d3dc; vertical-align: top; overflow-wrap: anywhere; }
tr { break-inside: avoid; }
table + p:has(> em:only-child) { font-size: 9pt; color: #51616b; }
figure { page: architecture; break-before: page; break-after: page; margin: 0; }
figure img { display: block; width: 100%; height: auto; margin: 5mm 0 5mm; }
figcaption { font: 9.5pt/1.4 "DejaVu Serif"; color: #43535f; }
ol, ul { padding-left: 19pt; margin: 6pt 0 10pt; }
li { margin-bottom: 6pt; orphans: 2; widows: 2; }
#references + ol { font-size: 9pt; line-height: 1.35; }
#references + ol li { margin-bottom: 4pt; }
#author-decisions-for-the-next-revision { font-size: 11pt; margin-top: 12pt; }
#author-decisions-for-the-next-revision + p { font-size: 9pt; line-height: 1.35; }
"""
html = (
    '<!doctype html><html lang="en"><head><meta charset="utf-8">'
    '<title>StemgenRT-5.8 — Real-Time Low-Latency Music Source Separation — Draft v0.1</title>'
    '<meta name="description" content="Working technical manuscript, version 0.1.">'
    f'<style>{css}</style></head><body>{body}</body></html>'
)
HTML(string=html, base_url=str(HERE)).write_pdf(OUTPUT)

pdf = pymupdf.open(OUTPUT)
page_texts = [page.get_text() for page in pdf]
text = "\n".join(page_texts)
for required in ["Abstract", "3. Architecture", "6. Results", "9. Conclusion",
                 "References", "Author decisions", "4.069079", "4.238471", "27,823,208",
                 "StemgenRT-5.8", "Historical scope", "StemgenRT58"]:
    if required not in text:
        raise ValueError(f"PDF is missing expected manuscript content: {required}")
if any(len(t.strip()) < 50 for t in page_texts):
    raise ValueError("Unexpected empty or nearly empty PDF page")
file_links = [link for page in pdf for link in page.get_links()
              if str(link.get("uri", "")).startswith("file:") or link.get("kind") == pymupdf.LINK_LAUNCH]
if file_links:
    raise ValueError("PDF contains local file links")
validation = {
    "source": SOURCE.name,
    "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    "figure_sha256": hashlib.sha256((HERE / "architecture.svg").read_bytes()).hexdigest(),
    "pdf": OUTPUT.name,
    "pdf_sha256": hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),
    "pages": len(pdf),
    "page_sizes_points": [[page.rect.width, page.rect.height] for page in pdf],
    "pdf_bytes": OUTPUT.stat().st_size,
    "external_links": sum(link.get("kind") == pymupdf.LINK_URI for page in pdf for link in page.get_links()),
    "package_versions": {name: importlib.metadata.version(name) for name in ("Markdown", "weasyprint", "PyMuPDF")},
}
print(json.dumps(validation, indent=2))
