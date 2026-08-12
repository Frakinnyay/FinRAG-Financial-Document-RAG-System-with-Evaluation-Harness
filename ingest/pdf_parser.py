"""
Table-aware PDF parsing for financial documents.

Why this is the hard part:
Naive RAG pipelines run the whole PDF through a generic text splitter.
That shreds tables mid-row, loses column headers, and turns
"Revenue | FY23 | FY22 \n 59,381 | 46,867" into two meaningless chunks.
For financial documents, numbers WITHOUT their row/column context are
worse than no answer at all.

Strategy used here:
1. Parse each page with pdfplumber, which detects tables separately
   from free text (uses PDF layout, not regex).
2. Tables are extracted as structured rows and re-serialized as
   markdown tables, with the header row PRE-PENDED to every chunk
   if a table has to be split (so a chunk of rows 26-40 still knows
   what column 3 means).
3. Free-flowing text (the area of the page NOT covered by a table
   bounding box) is chunked separately with a simple sliding window.
4. Every chunk carries metadata: company, fiscal_year, source_file,
   page_number, chunk_type ("text" | "table"), and a chunk_id.

This keeps tables intact and queryable, and keeps prose chunks from
being polluted by half a table row.
"""
import re
import json
import hashlib
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List

import pdfplumber

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import TEXT_CHUNK_SIZE, TEXT_CHUNK_OVERLAP, MAX_TABLE_ROWS_PER_CHUNK


@dataclass
class Chunk:
    chunk_id: str
    text: str
    company: str
    fiscal_year: str
    source_file: str
    page_number: int
    chunk_type: str  # "text" or "table"


def _make_chunk_id(source_file: str, page: int, idx: int) -> str:
    raw = f"{source_file}-{page}-{idx}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def _table_to_markdown(table: List[List[str]]) -> str:
    """Convert a pdfplumber-extracted table (list of rows) to markdown."""
    if not table or not table[0]:
        return ""
    # Clean cells: pdfplumber gives None for empty cells
    clean = [[(c or "").strip().replace("\n", " ") for c in row] for row in table]
    header, *rows = clean
    md = "| " + " | ".join(header) + " |\n"
    md += "| " + " | ".join(["---"] * len(header)) + " |\n"
    for row in rows:
        # pad short rows
        row = row + [""] * (len(header) - len(row))
        md += "| " + " | ".join(row[:len(header)]) + " |\n"
    return md


def _chunk_table(table: List[List[str]], max_rows: int) -> List[str]:
    """Split a long table into row-chunks, repeating the header each time."""
    if not table:
        return []
    header, *rows = table
    if len(rows) <= max_rows:
        return [_table_to_markdown(table)]
    chunks = []
    for i in range(0, len(rows), max_rows):
        sub_table = [header] + rows[i:i + max_rows]
        chunks.append(_table_to_markdown(sub_table))
    return chunks


def _chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Simple sliding-window splitter on paragraph boundaries where possible."""
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    paragraphs = text.split("\n\n")
    chunks, current = [], ""
    for para in paragraphs:
        if len(current) + len(para) + 2 <= chunk_size:
            current = f"{current}\n\n{para}" if current else para
        else:
            if current:
                chunks.append(current.strip())
            # if a single paragraph is itself too long, hard-split it
            if len(para) > chunk_size:
                start = 0
                while start < len(para):
                    end = start + chunk_size
                    chunks.append(para[start:end].strip())
                    start = end - overlap
                current = ""
            else:
                current = para
    if current:
        chunks.append(current.strip())

    # add overlap between consecutive chunks for continuity
    overlapped = []
    for i, c in enumerate(chunks):
        if i == 0:
            overlapped.append(c)
        else:
            prev_tail = chunks[i - 1][-overlap:]
            overlapped.append(f"{prev_tail}\n{c}")
    return overlapped


def parse_pdf(filepath: Path, company: str, fiscal_year: str) -> List[Chunk]:
    """Parse a single annual report PDF into table-aware chunks."""
    chunks: List[Chunk] = []
    source_file = filepath.name

    with pdfplumber.open(str(filepath)) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            idx_on_page = 0

            try:
                # --- 1. Extract tables and their bounding boxes ---
                found_tables = page.find_tables(
                    table_settings={
                        "vertical_strategy": "lines_strict",
                        "horizontal_strategy": "lines_strict",
                    }
                )
                if not found_tables:
                    found_tables = page.find_tables(
                        table_settings={
                            "vertical_strategy": "text",
                            "horizontal_strategy": "text",
                        }
                    )

                table_bboxes = []
                for t in found_tables:
                    table_data = t.extract()
                    if not table_data or len(table_data) < 2:
                        continue

                    # Clamp bbox to the page's actual boundaries. Some PDFs
                    # (rotated pages, odd page sizes) report table coords
                    # slightly outside the page, which crashes outside_bbox.
                    x0, top, x1, bottom = t.bbox
                    x0 = max(0, min(x0, page.width))
                    x1 = max(0, min(x1, page.width))
                    top = max(0, min(top, page.height))
                    bottom = max(0, min(bottom, page.height))
                    if x1 <= x0 or bottom <= top:
                        continue  # invalid box, skip subtracting it from text
                    safe_bbox = (x0, top, x1, bottom)
                    table_bboxes.append(safe_bbox)

                    for md_chunk in _chunk_table(table_data, MAX_TABLE_ROWS_PER_CHUNK):
                        if md_chunk.strip():
                            idx_on_page += 1
                            chunks.append(Chunk(
                                chunk_id=_make_chunk_id(source_file, page_num, idx_on_page),
                                text=md_chunk,
                                company=company,
                                fiscal_year=fiscal_year,
                                source_file=source_file,
                                page_number=page_num,
                                chunk_type="table",
                            ))

                # --- 2. Extract text OUTSIDE the table bounding boxes ---
                page_for_text = page
                for bbox in table_bboxes:
                    try:
                        page_for_text = page_for_text.outside_bbox(bbox)
                    except Exception:
                        # If subtracting this box fails for any reason,
                        # just skip subtracting it and keep going - worst
                        # case some table text also appears in a prose
                        # chunk, which is harmless.
                        continue

                raw_text = page_for_text.extract_text() or ""

                for t_chunk in _chunk_text(raw_text, TEXT_CHUNK_SIZE, TEXT_CHUNK_OVERLAP):
                    if len(t_chunk.strip()) < 30:
                        continue
                    idx_on_page += 1
                    chunks.append(Chunk(
                        chunk_id=_make_chunk_id(source_file, page_num, idx_on_page),
                        text=t_chunk,
                        company=company,
                        fiscal_year=fiscal_year,
                        source_file=source_file,
                        page_number=page_num,
                        chunk_type="text",
                    ))

            except Exception as e:
                # Don't let one bad page kill the whole file - skip it
                # and keep processing the rest of the document.
                print(f"    [page {page_num}] skipped due to error: {e}")
                continue

    return chunks


def parse_pdf_dir(raw_dir: Path) -> List[Chunk]:
    """
    Expects files named like: COMPANY_FY.pdf, e.g. TCS_2023.pdf, RELIANCE_2022.pdf
    Company and fiscal year are inferred from the filename.
    """
    all_chunks: List[Chunk] = []
    pdf_files = sorted(Path(raw_dir).glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(
            f"No PDFs found in {raw_dir}. Put your annual report PDFs there, "
            f"named like TCS_2023.pdf"
        )

    for filepath in pdf_files:
        stem = filepath.stem  # e.g. TCS_2023
        parts = stem.split("_")
        if len(parts) >= 2 and parts[-1].isdigit():
            company = "_".join(parts[:-1]).upper()
            fiscal_year = parts[-1]
        else:
            company = stem.upper()
            fiscal_year = "UNKNOWN"

        print(f"Parsing {filepath.name}  (company={company}, year={fiscal_year}) ...")
        try:
            file_chunks = parse_pdf(filepath, company, fiscal_year)
            print(f"  -> {len(file_chunks)} chunks "
                  f"({sum(1 for c in file_chunks if c.chunk_type=='table')} table, "
                  f"{sum(1 for c in file_chunks if c.chunk_type=='text')} text)")
            all_chunks.extend(file_chunks)
        except Exception as e:
            print(f"  !! FAILED to parse {filepath.name}: {e}")

    return all_chunks


def save_chunks_jsonl(chunks: List[Chunk], out_path: Path):
    with open(out_path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")


if __name__ == "__main__":
    from config import RAW_DIR, PROCESSED_DIR
    chunks = parse_pdf_dir(RAW_DIR)
    out_path = PROCESSED_DIR / "raw_chunks.jsonl"
    save_chunks_jsonl(chunks, out_path)
    print(f"\nSaved {len(chunks)} total chunks -> {out_path}")
