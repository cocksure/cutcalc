from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pdfplumber

PDF_DIR = Path("/Users/sanjarmaxmudov/Downloads/pastal")
OUT_XLSX = Path(__file__).resolve().parent / "markers_data.xlsx"


def _to_float(num: str) -> float | None:
    """Convert '85,37' or '10.656' to float."""
    if not num:
        return None
    num = num.strip().replace(" ", "").replace(",", ".")
    try:
        return float(num)
    except ValueError:
        return None


def _parse_length_m(text: str) -> float | None:
    t = re.sub(r"\s+", " ", text)

    # M can be латинская M or кириллическая М
    # CM can be 'CM' or 'См' (кириллица)
    m = re.search(r"Длина:\s*(\d+)\s*[MМ]\s*([\d,\.]+)\s*(?:CM|См)\b", t, flags=re.IGNORECASE)
    if m:
        meters = int(m.group(1))
        cm = _to_float(m.group(2))
        return meters + (cm / 100.0) if cm is not None else None

    # Sometimes length only in cm: 'Длина: 748,95См'
    m = re.search(r"Длина:\s*([\d,\.]+)\s*(?:CM|См)\b", t, flags=re.IGNORECASE)
    if m:
        cm_total = _to_float(m.group(1))
        return (cm_total / 100.0) if cm_total is not None else None

    # Fallback meters number
    m = re.search(r"Длина:\s*([\d,\.]+)\s*(?:m|метр|метров)?\b", t, flags=re.IGNORECASE)
    val = _to_float(m.group(1)) if m else None
    if val is not None and val >= 0.5:
        return val

    return None


def sort_size_pairs(pairs: list[tuple[str, int]]) -> list[tuple[str, int]]:
    """
    pairs: [("XS", 1), ("S", 2), ("M", 3)] or [("44", 2), ("46", 5)]
    """
    SORT_ORDER = {
        "3XS": 0, "2XS": 1, "XS": 2, "XS/S": 3, "S": 4, "S/M": 5,
        "M": 6, "M/L": 7, "L": 8, "L/50": 9, "L/XL": 10, "XL/L": 11,
        "XL": 12, "XL/2XL": 13, "2XL": 14, "2XL/3XL": 15, "XXL": 16,
        "3XL": 17, "3XL/4XL": 18, "XXXL": 19, "4XL": 20, "4XL/5XL": 21,
        "5XL": 22, "6XL": 23,
        # доп. варианты, чтобы не падало
        "OS": 100, "ONE": 101
    }

    def is_letter_size(s: str) -> bool:
        s = s.upper()
        return any(ch in s for ch in ["X", "S", "M", "L"]) or s in ("OS", "ONE")

    # normalize
    norm = [(s.strip().upper(), int(q)) for s, q in pairs]

    if norm and is_letter_size(norm[0][0]):
        # если размер не найден в SORT_ORDER — кидаем в конец
        return sorted(norm, key=lambda x: SORT_ORDER.get(x[0], 999))
    else:
        # numeric sizes
        def to_num(v: str) -> int:
            v = v.strip().upper()
            # убираем ведущие буквы типа I44 -> 44
            v = re.sub(r"^[A-ZА-Я]+", "", v)
            try:
                return int(v)
            except ValueError:
                return 999999

        return sorted(norm, key=lambda x: to_num(x[0]))


def _parse_sizes_and_total(text: str) -> tuple[str | None, int | None]:
    # normalize spaces/newlines
    t = re.sub(r"\s+", " ", text)

    # 0) combined sizes: S_44/1 M_46/4 3XL_54/2 — sort by numeric part
    combo_pairs = re.findall(r"\b([0-9A-Za-zА-Я]+_\d{2,3})\s*/\s*(\d+)\b", t)
    if combo_pairs:
        pairs2 = sorted(
            [(s, int(q)) for s, q in combo_pairs],
            key=lambda x: int(x[0].split("_")[-1]),
        )
        sizes_str = " ".join(f"{s}/{q}" for s, q in pairs2)
        return sizes_str, sum(q for _, q in pairs2)

    # 1) letter sizes (incl 2XS, 3XL, XS/S, XL/2XL, OS, ONE)
    letter_pairs = re.findall(
        r"\b((?:\d)?(?:XS|S|M|L|XL|XXL|XXXL|XXXXL|OS|ONE|XS/S|S/M|M/L|L/XL|XL/L|XL/2XL|2XL/3XL|3XL/4XL|4XL/5XL))\s*/\s*(\d+)\b",
        t,
        flags=re.IGNORECASE,
    )

    # 2) numeric sizes: 44/3 46/5 48/4 ...
    numeric_pairs = re.findall(r"\b([A-ZА-Я]?\d{2,3})\s*/\s*(\d+)\b", t, flags=re.IGNORECASE)

    pairs = letter_pairs if letter_pairs else numeric_pairs
    if not pairs:
        return None, None

    pairs2 = sort_size_pairs([(s, int(q)) for s, q in pairs])
    sizes_str = " ".join([f"{s}/{q}" for s, q in pairs2])
    total = sum(q for _, q in pairs2)
    return sizes_str, total


def parse_one_pdf(pdf_path: Path) -> dict:
    with pdfplumber.open(str(pdf_path)) as pdf:
        # usually 1 page, but just in case join all pages
        text = "\n".join((p.extract_text() or "") for p in pdf.pages)

    name = None
    m = re.search(r"Имя\s+Раскладки:\s*(.+?)(?=\s+Длина:|$)", text, flags=re.DOTALL)
    name = m.group(1).strip() if m else None
    if m:
        name = m.group(1).strip()

    length_m = _parse_length_m(text)

    width_cm = None
    m = re.search(r"Ширина:\s*([\d,\.]+)\s*(?:CM|См)\b", re.sub(r"\s+", " ", text), flags=re.IGNORECASE)
    if m:
        width_cm = _to_float(m.group(1))

    efficiency_pct = None
    m = re.search(r"Использование:\s*([\d,\.]+)\s*%", re.sub(r"\s+", " ", text), flags=re.IGNORECASE)
    if m:
        efficiency_pct = _to_float(m.group(1))

    unplaced = placed = None
    m = re.search(r"Не\s*размещенные\s*/\s*Размещенные:\s*(\d+)\s*/\s*(\d+)", text, flags=re.IGNORECASE)
    if not m:
        m = re.search(r"Неразмещенные/Размещенные:\s*(\d+)\s*/\s*(\d+)", text, flags=re.IGNORECASE)
    if m:
        unplaced, placed = int(m.group(1)), int(m.group(2))

    shrink_pct = stretch_pct = None
    m = re.search(r"Усадка/Растяжение:\s*([-\d,\.]+)\s*/\s*([-\d,\.]+)", text)
    if m:
        shrink_pct = _to_float(m.group(1))
        stretch_pct = _to_float(m.group(2))

    block_table = None

    m = re.search(r"Таблица\s+Блок\s*/\s*Буфер:\s*(.+?)(?=\s+Дата:|$)", text, flags=re.DOTALL)
    if m:
        block_table = m.group(1).strip()

    if not block_table:
        m = re.search(r"Таблица\s+Ограничений:\s*(.+)$", text, flags=re.IGNORECASE | re.MULTILINE)
        if m:
            block_table = m.group(1).strip()

    if block_table:
        block_table = re.sub(r"\s+", " ", block_table).replace(" -", "-").replace("- ", "-")

    date = time = None
    m = re.search(r"Дата:\s*(\d{2}\.\d{2}\.\d{4})", text)
    if m:
        date = m.group(1)
    m = re.search(r"Время:\s*(\d{2}:\d{2}:\d{2})", text)
    if m:
        time = m.group(1)

    sizes_str, total_garments = _parse_sizes_and_total(text)

    consumption_m_per_piece = None
    if length_m is not None and total_garments:
        consumption_m_per_piece = length_m / total_garments

    return {
        "file": pdf_path.name,
        "marker_name": name,
        "date": date,
        "time": time,
        "length_m": length_m,
        "width_cm": width_cm,
        "efficiency_pct": efficiency_pct,
        "sizes": sizes_str,
        "total_garments": total_garments,
        "unplaced": unplaced,
        "placed": placed,
        "shrink_pct": shrink_pct,
        "stretch_pct": stretch_pct,
        "block_table": block_table,
        "consumption_m_per_piece": consumption_m_per_piece,
    }


def main() -> None:
    if not PDF_DIR.exists():
        raise SystemExit(f"PDF folder not found: {PDF_DIR}")

    pdfs = sorted(PDF_DIR.glob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"No PDFs found in: {PDF_DIR}")

    rows = []
    for i, p in enumerate(pdfs, start=1):
        try:
            rows.append(parse_one_pdf(p))
        except Exception as e:
            rows.append({"file": p.name, "error": str(e)})
        if i % 25 == 0:
            print(f"Parsed {i}/{len(pdfs)}...")

    df = pd.DataFrame(rows)

    # nice column order
    cols = [
        "file", "marker_name", "date", "time",
        "length_m", "width_cm", "efficiency_pct",
        "sizes", "total_garments",
        "consumption_m_per_piece",
        "unplaced", "placed",
        "shrink_pct", "stretch_pct",
        "block_table", "error",
    ]
    for c in cols:
        if c not in df.columns:
            df[c] = None
    df = df[cols]

    df.to_excel(OUT_XLSX, index=False)
    print(f"✅ Saved: {OUT_XLSX}")


if __name__ == "__main__":
    main()
