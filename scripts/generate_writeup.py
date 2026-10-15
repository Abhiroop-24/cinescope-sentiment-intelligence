"""Generate a minimal two-page PDF from docs/project_writeup.md.

This avoids making report generation a prerequisite for running the model.
For polished typesetting, the Markdown source can be rendered with Pandoc.
"""
from pathlib import Path


def pdf_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def build() -> None:
    source = Path("docs/project_writeup.md").read_text(encoding="utf-8")
    lines = [line.lstrip("# ").strip() for line in source.splitlines() if line.strip() and not line.startswith("## Reproducibility")]
    pages = [lines[:34], lines[34:68]]
    objects: list[bytes] = []
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objects.append(b"<< /Type /Pages /Kids [3 0 R 4 0 R] /Count 2 >>")
    for page_index in range(2):
        content = "BT /F1 11 Tf 54 760 Td 15 TL " + " ".join(
            f"({pdf_escape(line[:105])}) Tj T*" for line in pages[page_index]
        ) + " ET"
        stream = content.encode()
        objects.append(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 5 0 R >> >> /Contents {6 + page_index} 0 R >>".encode())
        objects.append(f"<< /Length {len(stream)} >>\nstream\n".encode() + stream + b"\nendstream")
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    output = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for number, obj in enumerate(objects, 1):
        offsets.append(len(output)); output.extend(f"{number} 0 obj\n".encode()); output.extend(obj); output.extend(b"\nendobj\n")
    xref = len(output); output.extend(f"xref\n0 {len(objects)+1}\n0000000000 65535 f \n".encode())
    for offset in offsets[1:]: output.extend(f"{offset:010d} 00000 n \n".encode())
    output.extend(f"trailer << /Size {len(objects)+1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF".encode())
    Path("docs/project_writeup.pdf").write_bytes(output)


if __name__ == "__main__":
    build()
