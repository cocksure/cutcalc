import io
import os
import json
import colorsys
import tempfile
from collections import Counter

import ezdxf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

DATA_PATH = os.path.join(os.path.dirname(__file__), "markers_labeled.xlsx")
WIDTH_TOLERANCE = 30  # cm
TOP_N = 20

DISPLAY_COLUMNS = [
    "marker_name",
    "product_type",
    "width_cm",
    "block_table",
    "sizes",
    "total_garments",
    "length_m",
    "efficiency_pct",
    "consumption_m_per_piece",
    "shrink_pct",
    "stretch_pct",
]


# ── Helpers ───────────────────────────────────────────────────────────────────


def parse_sizes(sizes_str: str) -> Counter:
    """Parse '116/1 128/2 140/3' or 'S/2 M/3 L/2' into Counter."""
    result = Counter()
    for token in str(sizes_str).split():
        parts = token.rsplit("/", 1)
        if len(parts) == 2:
            try:
                result[parts[0].strip()] = int(parts[1])
            except ValueError:
                pass
    return result


def sizes_total(sizes_str: str) -> int:
    return sum(parse_sizes(sizes_str).values())


def sizes_count(sizes_str: str) -> int:
    return len(parse_sizes(sizes_str))


def size_similarity(a: str, b: str) -> float:
    """Score 0..1 — how similar two size assortments are by proportion overlap."""
    ca, cb = parse_sizes(a), parse_sizes(b)
    if not ca or not cb:
        return 0.0
    all_keys = set(ca) | set(cb)
    total_a = sum(ca.values()) or 1
    total_b = sum(cb.values()) or 1
    overlap = 0.0
    for k in all_keys:
        ra = ca.get(k, 0) / total_a
        rb = cb.get(k, 0) / total_b
        overlap += min(ra, rb)
    return round(overlap, 3)


def compute_stats(similar: pd.DataFrame) -> dict:
    c = similar["consumption_m_per_piece"]
    e = similar["efficiency_pct"]
    g = similar["total_garments"]
    ln = similar["length_m"]
    return {
        "count": len(similar),
        "consumption_mean": round(c.mean(), 4),
        "consumption_median": round(c.median(), 4),
        "consumption_min": round(c.min(), 4),
        "consumption_max": round(c.max(), 4),
        "consumption_std": round(c.std(), 4) if len(c) > 1 else 0.0,
        "efficiency_mean": round(e.mean(), 2),
        "efficiency_min": round(e.min(), 2),
        "efficiency_max": round(e.max(), 2),
        "length_mean": round(ln.mean(), 3),
        "length_min": round(ln.min(), 3),
        "length_max": round(ln.max(), 3),
        "garments_mean": round(g.mean(), 1),
        "garments_min": int(g.min()),
        "garments_max": int(g.max()),
        "widths_used": sorted(similar["width_cm"].unique().tolist()),
        "fabrics_used": sorted(similar["block_table"].unique().tolist()),
        "shrink_mean": round(similar["shrink_pct"].mean(), 2),
        "stretch_mean": round(similar["stretch_pct"].mean(), 2),
    }


# ── Mathematical model ────────────────────────────────────────────────────────


@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_excel(DATA_PATH)


@st.cache_data
def compute_area_coefficients(df_json: str) -> dict:
    """Calculate empirical area per garment (m²) for each product_type."""
    df = pd.read_json(df_json)
    df["_area"] = (
        df["consumption_m_per_piece"]
        * (df["width_cm"] / 100)
        * (df["efficiency_pct"] / 100)
    )
    coeffs = {}
    for pt, group in df.groupby("product_type"):
        areas = group["_area"]
        effs = group["efficiency_pct"]
        coeffs[pt] = {
            "area_mean": round(areas.mean(), 4),
            "area_std": round(areas.std(), 4) if len(areas) > 1 else 0.0,
            "area_min": round(areas.min(), 4),
            "area_max": round(areas.max(), 4),
            "efficiency_mean": round(effs.mean(), 2),
            "efficiency_std": round(effs.std(), 2) if len(effs) > 1 else 0.0,
            "count": len(group),
        }
    return coeffs


@st.cache_data
def build_efficiency_model(df_json: str) -> pd.DataFrame:
    """Build width→efficiency lookup from all historical data."""
    df = pd.read_json(df_json)
    # Bin widths into ranges and calculate average efficiency per bin
    df["_width_bin"] = pd.cut(df["width_cm"], bins=[0, 60, 80, 100, 120, 160, 200, 300])
    return df.groupby("_width_bin", observed=True)["efficiency_pct"].agg(["mean", "count"]).reset_index()


def estimate_efficiency(df: pd.DataFrame, width_cm: float, product_type: str | None) -> float:
    """Estimate realistic efficiency for given fabric width.

    Narrow fabric → lower efficiency. Uses actual data.
    """
    # First try: records with same product_type and close width
    tolerance = 30
    mask = (df["width_cm"] >= width_cm - tolerance) & (df["width_cm"] <= width_cm + tolerance)
    if product_type:
        subset = df.loc[mask & (df["product_type"] == product_type), "efficiency_pct"]
        if len(subset) >= 3:
            return round(subset.mean(), 2)

    # Fallback: all records with close width (any product type)
    subset = df.loc[mask, "efficiency_pct"]
    if len(subset) >= 3:
        return round(subset.mean(), 2)

    # Last resort: global regression — narrow fabric ≈ lower efficiency
    # Empirical: efficiency ≈ 60% at 50cm, 75% at 100cm, 83% at 175cm, 85% at 200cm
    eff = 55 + (width_cm - 50) * 0.2
    return round(max(55, min(90, eff)), 2)


SEAM_ALLOWANCE = 1.0  # cm, default side seam allowance
HEM_ALLOWANCE = 2.0   # cm, default bottom hem fold


def calc_area_from_measurements(m: dict, stitch: dict | None = None) -> tuple[float, str, list]:
    """Calculate pattern area (m²) from detailed garment measurements.

    Returns (area_m2, breakdown_text, pieces_info).
    Measurements (all in cm):
        A  - front length (HPS to bottom)
        B  - back length (by center)
        C  - 1/2 chest width
        K1 - armhole depth (armpit level from HPS)
        G1 - sleeve inseam length
        Z3 - 1/2 bottom sleeve width
        E2 - 1/2 placket length (optional)
        T4 - placket width (optional)
    Stitch widths (all in cm, optional):
        needle  - needle stitching width (for side seams)
        cover   - coverstitch width (for hems)
        placket - placket stitch width
    """
    sa = SEAM_ALLOWANCE
    hem = HEM_ALLOWANCE
    sleeve_hem = 1.5  # cm, sleeve cuff fold

    if stitch:
        # Needle stitching → side seam allowance (stitch overlaps two pieces)
        needle_w = stitch.get("needle", 0)
        if needle_w > 0:
            sa = max(0.8, needle_w * 0.6)

        # Coverstitch → hem fold allowance (fabric folds up under the stitch)
        cover_w = stitch.get("cover", 0)
        if cover_w > 0:
            hem = cover_w + 1.5
            sleeve_hem = cover_w + 1.0

    parts = []

    # Front panel: height = A + sa(shoulder) + hem(bottom), width = C + sa(side)
    A = m.get("A", 0)
    C = m.get("C", 0)
    front = (A + sa + hem) * (C + sa) if A and C else 0
    if front:
        parts.append(f"Перед: {front:.0f} см²")

    # Back panel
    B = m.get("B", 0)
    back = (B + sa + hem) * (C + sa) if B and C else 0
    if back:
        parts.append(f"Спинка: {back:.0f} см²")

    # Sleeves (×2)
    K1 = m.get("K1", 0)
    G1 = m.get("G1", 0)
    Z3 = m.get("Z3", 0)
    sleeves = 0
    if Z3 > 0 and (G1 > 0 or K1 > 0):
        cap_height = K1 * 0.45 if K1 > 0 else 10
        sleeve_height = cap_height + G1 + sa + sleeve_hem
        cap_width = Z3 * 2 * 1.3 + sa * 2
        bottom_width = Z3 * 2 + sa * 2
        one_sleeve = (cap_width + bottom_width) / 2 * sleeve_height
        sleeves = one_sleeve * 2
        parts.append(f"Рукава ×2: {sleeves:.0f} см²")

    # Placket / collar
    E2 = m.get("E2", 0)
    T4 = m.get("T4", 0)
    collar_placket = 0
    if E2 > 0 and T4 > 0:
        placket_sa = sa
        if stitch and stitch.get("placket", 0) > 0:
            placket_sa = stitch["placket"] * 0.5
        collar_placket = E2 * 2 * (T4 + placket_sa) * 2
        parts.append(f"Планка/воротник: {collar_placket:.0f} см²")

    body = front + back
    total = body + sleeves + collar_placket

    # Small parts allowance (pocket, labels, reinforcements) = 3%
    small = total * 0.03
    total += small

    total_m2 = round(total / 10000, 4)

    # Collect piece dimensions for layout analysis
    pieces_info = []
    if front:
        pieces_info.append((C + sa, A + sa + hem, "Перед", 1))
    if back:
        pieces_info.append((C + sa, B + sa + hem, "Спинка", 1))
    if sleeves:
        pieces_info.append((max(cap_width, bottom_width), cap_height + G1 + sa + sleeve_hem, "Рукав", 2))
    if collar_placket:
        p_sa = stitch["placket"] * 0.5 if (stitch and stitch.get("placket", 0) > 0) else sa
        pieces_info.append((T4 + p_sa, E2 * 2, "Планка", 2))

    stitch_info = ""
    if stitch and any(v > 0 for v in stitch.values()):
        stitch_info = f" (припуск шва: {sa:.1f} см, подгиб низа: {hem:.1f} см)"
    breakdown = " + ".join(parts) + f" + мелкие детали ≈ {total:.0f} см²{stitch_info} = {total_m2} м²"
    return total_m2, breakdown, pieces_info


MEAS_COLS = ["A", "B", "C", "K1", "G1", "Z3", "E2", "T4"]


def calc_sized_pieces(
    sizes_dict: dict,
    measurements_table: dict[str, dict],
    stitch: dict | None = None,
) -> tuple[float, str, list]:
    """Calculate pieces for all sizes in the assortment.

    Args:
        sizes_dict: {"S": 1, "M": 2, "L": 3} — size → garment count
        measurements_table: {"S": {"A": 54, ...}, ...} — size → measurements
        stitch: stitch parameters (optional)

    Returns:
        (avg_area_m2, breakdown_text, sized_pieces)
        sized_pieces: [(w, h, name, total_count, size_label), ...]  ← 5-tuple
    """
    all_pieces: list[tuple] = []
    total_area = 0.0
    total_garments = sum(sizes_dict.values())
    breakdowns = []

    for size_name, count in sizes_dict.items():
        meas = measurements_table.get(size_name, {})
        if not (meas.get("A", 0) > 0 and meas.get("C", 0) > 0):
            continue
        area, _bkdn, pieces = calc_area_from_measurements(meas, stitch)
        total_area += area * count
        breakdowns.append(f"{size_name} (×{count}): {area} м²")
        for pw, ph, name, piece_count in pieces:
            all_pieces.append((pw, ph, name, piece_count * count, size_name))

    avg_area = round(total_area / total_garments, 4) if total_garments > 0 else 0.0
    bkdn_text = f"Средняя площадь: {avg_area} м²/изд. | " + ", ".join(breakdowns)

    return avg_area, bkdn_text, all_pieces


def _shoelace_area(points: list[tuple[float, float]]) -> float:
    """Polygon area via Shoelace formula."""
    n = len(points)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += points[i][0] * points[j][1]
        area -= points[j][0] * points[i][1]
    return abs(area) / 2.0


def _bbox(points: list[tuple[float, float]]) -> tuple[float, float]:
    """Bounding box width × height."""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return max(xs) - min(xs), max(ys) - min(ys)


def _is_nearly_closed(points: list[tuple[float, float]], tol: float = 0.5) -> bool:
    """Check if first ≈ last point (within tolerance)."""
    if len(points) < 3:
        return False
    dx = abs(points[0][0] - points[-1][0])
    dy = abs(points[0][1] - points[-1][1])
    return dx < tol and dy < tol


def _extract_points_from_entity(entity) -> list[tuple[float, float]] | None:
    """Extract polygon points from a DXF entity. Returns None if not a polygon."""
    etype = entity.dxftype()

    if etype == "LWPOLYLINE":
        pts = [(p[0], p[1]) for p in entity.get_points(format="xy")]
        if entity.closed or _is_nearly_closed(pts):
            return pts

    elif etype == "POLYLINE":
        pts = [(v.dxf.location.x, v.dxf.location.y) for v in entity.vertices]
        if entity.is_closed or _is_nearly_closed(pts):
            return pts

    elif etype == "SPLINE":
        # Approximate spline with control points for bounding box
        try:
            pts = [(p[0], p[1]) for p in entity.control_points]
            if len(pts) >= 3 and (entity.closed or _is_nearly_closed(pts)):
                return pts
            # Try flattening for better approximation
            pts = [(p[0], p[1]) for p in entity.flattening(0.5)]
            if len(pts) >= 3 and _is_nearly_closed(pts):
                return pts
        except Exception:
            pass

    elif etype == "HATCH":
        # HATCH entities have boundary paths with exact area info
        try:
            for path in entity.paths:
                pts = []
                if hasattr(path, "vertices"):
                    pts = [(v[0], v[1]) for v in path.vertices]
                elif hasattr(path, "edges"):
                    for edge in path.edges:
                        if hasattr(edge, "start"):
                            pts.append((edge.start[0], edge.start[1]))
                if len(pts) >= 3:
                    return pts
        except Exception:
            pass

    return None


def _collect_entities(doc) -> list:
    """Collect entities from modelspace + inside block definitions (INSERT)."""
    msp = doc.modelspace()
    entities = []

    for entity in msp:
        if entity.dxftype() == "INSERT":
            # Expand block reference into its entities
            try:
                block = doc.blocks.get(entity.dxf.name)
                if block:
                    # Get insert transform (position + scale)
                    ix = entity.dxf.get("insert", (0, 0, 0))
                    sx = entity.dxf.get("xscale", 1.0)
                    sy = entity.dxf.get("yscale", 1.0)
                    for bent in block:
                        # Store transform info for later use
                        bent._dxf_insert_offset = (float(ix[0]), float(ix[1]))
                        bent._dxf_insert_scale = (float(sx), float(sy))
                        entities.append(bent)
            except Exception:
                pass
        else:
            entity._dxf_insert_offset = (0.0, 0.0)
            entity._dxf_insert_scale = (1.0, 1.0)
            entities.append(entity)

    return entities


def _chain_lines_to_polygons(line_entities: list) -> list[list[tuple[float, float]]]:
    """Chain LINE entities into closed polygons by connecting endpoints."""
    if not line_entities:
        return []

    segments = []
    for ent in line_entities:
        try:
            s = ent.dxf.start
            e = ent.dxf.end
            segments.append(((float(s.x), float(s.y)), (float(e.x), float(e.y))))
        except Exception:
            continue

    if not segments:
        return []

    polygons = []
    used = [False] * len(segments)
    tol = 0.5

    for start_idx in range(len(segments)):
        if used[start_idx]:
            continue

        chain = [segments[start_idx][0], segments[start_idx][1]]
        used[start_idx] = True

        changed = True
        while changed:
            changed = False
            for i in range(len(segments)):
                if used[i]:
                    continue
                s, e = segments[i]
                tail = chain[-1]
                # Connect to tail
                if abs(s[0] - tail[0]) < tol and abs(s[1] - tail[1]) < tol:
                    chain.append(e)
                    used[i] = True
                    changed = True
                elif abs(e[0] - tail[0]) < tol and abs(e[1] - tail[1]) < tol:
                    chain.append(s)
                    used[i] = True
                    changed = True

        if len(chain) >= 4 and _is_nearly_closed(chain, tol=1.0):
            polygons.append(chain)

    return polygons


def parse_dxf_pieces(file_bytes: bytes) -> tuple[list[dict], dict]:
    """Parse DXF file and extract pattern pieces.

    Handles: LWPOLYLINE, POLYLINE, SPLINE, HATCH, LINE chains, INSERT/BLOCK.
    Returns (pieces_list, diagnostics_dict).
    """
    # Save to temp file — ezdxf.readfile() handles binary DXF, encoding, etc.
    with tempfile.NamedTemporaryFile(suffix=".dxf", delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        doc = ezdxf.readfile(tmp_path)
    finally:
        os.unlink(tmp_path)

    all_entities = _collect_entities(doc)

    # Count entity types for diagnostics
    type_counts: dict[str, int] = {}
    for ent in all_entities:
        t = ent.dxftype()
        type_counts[t] = type_counts.get(t, 0) + 1

    diag = {
        "entity_types": type_counts,
        "total_entities": len(all_entities),
        "blocks": len(doc.blocks) - 2,  # minus *Model_Space and *Paper_Space
    }

    raw_pieces = []
    line_entities = []

    for entity in all_entities:
        etype = entity.dxftype()

        if etype == "LINE":
            line_entities.append(entity)
            continue

        points = _extract_points_from_entity(entity)
        if not points or len(points) < 3:
            continue

        # Apply INSERT transform if present
        offset = getattr(entity, "_dxf_insert_offset", (0.0, 0.0))
        scale = getattr(entity, "_dxf_insert_scale", (1.0, 1.0))
        if offset != (0.0, 0.0) or scale != (1.0, 1.0):
            points = [
                (p[0] * scale[0] + offset[0], p[1] * scale[1] + offset[1])
                for p in points
            ]

        area = _shoelace_area(points)
        w, h = _bbox(points)
        raw_pieces.append({"area": area, "width": w, "height": h})

    # Try chaining LINE entities into polygons
    if line_entities:
        polygons = _chain_lines_to_polygons(line_entities)
        diag["line_chains_found"] = len(polygons)
        for poly_pts in polygons:
            area = _shoelace_area(poly_pts)
            w, h = _bbox(poly_pts)
            raw_pieces.append({"area": area, "width": w, "height": h})

    diag["raw_shapes_found"] = len(raw_pieces)

    if not raw_pieces:
        return [], diag

    # Auto-detect units: if average area > 5000 → likely mm², convert to cm²
    avg_area = sum(p["area"] for p in raw_pieces) / len(raw_pieces)
    if avg_area > 5000:
        scale = 0.01  # mm² → cm²
        dim_scale = 0.1  # mm → cm
        diag["units_detected"] = "мм (конвертировано в см)"
    else:
        scale = 1.0
        dim_scale = 1.0
        diag["units_detected"] = "см"

    pieces = []
    for p in raw_pieces:
        area_cm2 = p["area"] * scale
        w_cm = p["width"] * dim_scale
        h_cm = p["height"] * dim_scale

        # Filter tiny pieces (notches, drill holes, marks)
        if area_cm2 < 20:
            continue

        pieces.append({
            "area_cm2": round(area_cm2, 1),
            "width_cm": round(w_cm, 1),
            "height_cm": round(h_cm, 1),
        })

    # Sort by area descending
    pieces.sort(key=lambda p: -p["area_cm2"])

    # Auto-label pieces
    for i, p in enumerate(pieces):
        if i == 0:
            p["name"] = "Перед"
        elif i == 1:
            p["name"] = "Спинка"
        elif i <= 3:
            p["name"] = "Рукав"
        elif i <= 5:
            p["name"] = "Планка"
        else:
            p["name"] = f"Деталь {i - 5}"

    diag["pieces_after_filter"] = len(pieces)
    return pieces, diag


CUTTING_GAP = 0.3   # cm between pieces
END_LOSS = 2.0      # cm at each end of marker layout


def layout_analysis(pieces: list, fabric_width: float, total_garments: int) -> dict:
    """Analyze how pattern pieces fit on fabric and estimate cutting losses.

    pieces: 4-tuple (w, h, name, count_per_garment)
         or 5-tuple (w, h, name, total_count, size_label)
    Returns layout info with piece placement and cutting loss estimate.
    """
    is_sized = len(pieces[0]) == 5 if pieces else False
    details = []

    for item in pieces:
        if is_sized:
            pw, ph, name, total_count, size_label = item
            display_name = f"{name} {size_label}"
        else:
            pw, ph, name, count = item
            total_count = count * total_garments
            display_name = name

        # Check orientation: piece width vs fabric width
        fits_normal = pw <= fabric_width
        fits_rotated = ph <= fabric_width

        if fits_normal:
            per_row = max(1, int(fabric_width // (pw + CUTTING_GAP)))
            used_height = ph
            orientation = "вертикально"
            used_width = pw
        elif fits_rotated:
            per_row = max(1, int(fabric_width // (ph + CUTTING_GAP)))
            used_height = pw
            orientation = "горизонтально (повёрнут)"
            used_width = ph
        else:
            per_row = 1
            used_height = min(pw, ph)
            used_width = max(pw, ph)
            orientation = "не помещается!"

        waste_width = fabric_width - per_row * used_width - (per_row - 1) * CUTTING_GAP
        rows_needed = (total_count + per_row - 1) // per_row
        length_cm = rows_needed * used_height

        details.append({
            "name": display_name,
            "size": f"{pw:.1f}×{ph:.1f}",
            "orientation": orientation,
            "per_row": per_row,
            "count": total_count,
            "total_count": total_count,
            "rows_needed": rows_needed,
            "length_cm": round(length_cm, 1),
            "waste_width_cm": round(waste_width, 1),
        })

    # Total rows across all piece types
    total_rows = sum(d["rows_needed"] for d in details)

    # Cutting losses
    gap_loss_cm = max(0, total_rows - 1) * CUTTING_GAP
    end_loss_cm = END_LOSS * 2
    total_loss_cm = gap_loss_cm + end_loss_cm
    loss_per_piece_m2 = (total_loss_cm * fabric_width / 10000) / total_garments

    return {
        "details": details,
        "total_rows": total_rows,
        "gap_loss_cm": round(gap_loss_cm, 1),
        "end_loss_cm": round(end_loss_cm, 1),
        "total_loss_cm": round(total_loss_cm, 1),
        "loss_per_piece_m2": round(loss_per_piece_m2, 4),
    }


PIECE_COLORS = {
    "Перед": "#5B9BD5",
    "Спинка": "#70AD47",
    "Рукав": "#FFC000",
    "Планка": "#FF6B6B",
}


def draw_marker_layout(
    pieces_info: list,
    fabric_width_cm: float,
    total_garments: int = 1,
    gap_cm: float = 0.3,
    sizes_breakdown: dict | None = None,
) -> tuple:
    """Draw a professional marker layout visualization (like Gerber/Lectra).

    pieces_info: 4-tuple (w, h, name, count_per_garment) — old format
              or 5-tuple (w, h, name, total_count, size_label) — sized format
    sizes_breakdown: {"S": 1, "M": 2, "L": 3} — optional, for accurate title

    Returns (fig, total_length_m, packing_efficiency_pct).
    """
    is_sized = len(pieces_info[0]) == 5 if pieces_info else False

    # ── 1. Build piece list ───────────────────────────────────────────────
    all_pieces = []

    def _orient_and_add(pw, ph, name, label, rotated=False):
        if pw <= fabric_width_cm:
            all_pieces.append({"w": pw, "h": ph, "name": name, "g": label, "rot": False})
        elif ph <= fabric_width_cm:
            all_pieces.append({"w": ph, "h": pw, "name": name, "g": label, "rot": True})
        else:
            all_pieces.append(
                {"w": min(pw, ph), "h": max(pw, ph), "name": name, "g": label, "rot": True}
            )

    if is_sized:
        # 5-tuple: (w, h, name, total_count, size_label)
        for pw, ph, name, total_count, size_label in pieces_info:
            for _ in range(total_count):
                _orient_and_add(pw, ph, name, size_label)
        if sizes_breakdown:
            total_garments_display = sum(sizes_breakdown.values())
            size_str = ", ".join(f"{s}×{c}" for s, c in sizes_breakdown.items())
        else:
            total_garments_display = total_garments
            size_str = ""
    else:
        # 4-tuple: (w, h, name, count_per_garment)
        for pw, ph, name, count in pieces_info:
            for gnum in range(1, total_garments + 1):
                for _ in range(count):
                    _orient_and_add(pw, ph, name, str(gnum))
        total_garments_display = total_garments
        size_str = ""

    # Sort: tallest (longest along fabric) first → better shelf packing
    all_pieces.sort(key=lambda p: (-p["h"], -p["w"]))

    # ── 2. FFDH strip packing ─────────────────────────────────────────────
    # Shelves run across fabric width; stacked along fabric length.
    shelves: list[dict] = []

    for pc in all_pieces:
        placed = False
        for shelf in shelves:
            if pc["w"] + gap_cm <= shelf["rem"] + 0.1:
                pc["x"] = fabric_width_cm - shelf["rem"]
                pc["y"] = shelf["y"]
                shelf["rem"] -= pc["w"] + gap_cm
                placed = True
                break
        if not placed:
            y0 = sum(s["h"] + gap_cm for s in shelves)
            pc["x"] = 0.0
            pc["y"] = y0
            shelves.append(
                {"y": y0, "h": pc["h"], "rem": fabric_width_cm - pc["w"] - gap_cm}
            )

    total_len = sum(s["h"] + gap_cm for s in shelves) if shelves else 0.0
    piece_area = sum(p["w"] * p["h"] for p in all_pieces)
    fabric_area = fabric_width_cm * total_len
    eff = piece_area / fabric_area * 100 if fabric_area > 0 else 0.0

    # ── 3. Matplotlib figure ──────────────────────────────────────────────
    ratio = fabric_width_cm / total_len if total_len else 0.1
    fig_w = min(20, max(12, total_len / 45))
    fig_h = max(3.5, min(8, fig_w * ratio * 2.2))

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("#FAFAFA")
    ax.set_facecolor("#F0EDE4")

    # Fabric strip
    ax.add_patch(
        mpatches.FancyBboxPatch(
            (0, 0),
            total_len,
            fabric_width_cm,
            boxstyle="round,pad=0",
            lw=1.5,
            ec="#555",
            fc="#FAF8F0",
            zorder=0,
        )
    )

    # Subtle grid every 10 cm
    for gx in range(0, int(total_len) + 1, 10):
        ax.axvline(gx, color="#e0ddd5", lw=0.3, zorder=1)
    for gy in range(0, int(fabric_width_cm) + 1, 10):
        ax.axhline(gy, color="#e0ddd5", lw=0.3, zorder=1)

    # Shelf boundary lines
    for shelf in shelves:
        yend = shelf["y"] + shelf["h"]
        ax.axvline(yend, color="#bbb", lw=0.5, ls="--", alpha=0.4, zorder=1)

    # ── 4. Draw each piece ────────────────────────────────────────────────
    # Map unique "g" labels to indices for color variation
    g_labels = list(dict.fromkeys(pc["g"] for pc in all_pieces))

    for pc in all_pieces:
        base = PIECE_COLORS.get(pc["name"], "#9B59B6")
        g_idx = g_labels.index(pc["g"])
        r, g, b = (int(base[i : i + 2], 16) / 255 for i in (1, 3, 5))
        hue, sat, val = colorsys.rgb_to_hsv(r, g, b)
        val2 = max(0.45, min(1.0, val * (0.82 + 0.18 * (g_idx % 4) / 3)))
        sat2 = max(0.3, min(1.0, sat * (1.0 - 0.12 * (g_idx % 4) / 3)))
        rc, gc, bc = colorsys.hsv_to_rgb(hue, sat2, val2)

        inset = 0.25
        rect = mpatches.FancyBboxPatch(
            (pc["y"] + inset, pc["x"] + inset),
            pc["h"] - inset * 2,
            pc["w"] - inset * 2,
            boxstyle="round,pad=0.15",
            lw=0.6,
            ec="#444",
            fc=(rc, gc, bc),
            alpha=0.88,
            zorder=2,
        )
        ax.add_patch(rect)

        # Label inside the piece
        cx = pc["y"] + pc["h"] / 2
        cy = pc["x"] + pc["w"] / 2
        min_dim = min(pc["h"], pc["w"])

        # For sized format: "Перед M", for old format: "Перед #3"
        g_display = pc["g"] if is_sized else f"#{pc['g']}"

        if min_dim > 10:
            fs = min(7.5, min_dim / 5)
            rot_mark = " ↺" if pc["rot"] else ""
            ax.text(
                cx, cy, f"{pc['name']}{rot_mark}\n{g_display}",
                ha="center", va="center", fontsize=fs,
                color="white", fontweight="bold", zorder=3,
            )
        elif min_dim > 4:
            ax.text(
                cx, cy, g_display,
                ha="center", va="center", fontsize=5,
                color="white", fontweight="bold", zorder=3,
            )

    # ── 5. Dimension annotations ──────────────────────────────────────────
    ax.annotate(
        "", xy=(total_len, -2.5), xytext=(0, -2.5),
        arrowprops=dict(arrowstyle="<->", color="#333", lw=1.2),
    )
    ax.text(
        total_len / 2, -4.2,
        f"{total_len / 100:.2f} м",
        ha="center", fontsize=10, fontweight="bold", color="#222",
    )

    ax.annotate(
        "", xy=(-2.5, fabric_width_cm), xytext=(-2.5, 0),
        arrowprops=dict(arrowstyle="<->", color="#333", lw=1.2),
    )
    ax.text(
        -4, fabric_width_cm / 2,
        f"{fabric_width_cm:.0f}\nсм",
        ha="center", va="center", fontsize=8, color="#333", rotation=90,
    )

    # ── 6. Title, legend, cleanup ─────────────────────────────────────────
    size_info = f"  ({size_str})" if size_str else ""
    ax.set_title(
        f"Маркер-раскладка   •   {total_len / 100:.2f} м × {fabric_width_cm:.0f} см   •   "
        f"{len(all_pieces)} дет. / {total_garments_display} изд.{size_info}   •   "
        f"КПД раскладки: {eff:.1f}%",
        fontsize=10, fontweight="bold", pad=14, color="#222",
    )

    used = {p["name"] for p in all_pieces}
    handles = [
        mpatches.Patch(fc=PIECE_COLORS[n], ec="#444", label=n, alpha=0.85)
        for n in PIECE_COLORS
        if n in used
    ]
    ax.legend(
        handles=handles, loc="upper right", fontsize=7.5,
        framealpha=0.92, edgecolor="#ccc", fancybox=True,
    )

    ax.set_xlim(-6, total_len + 4)
    ax.set_ylim(-6, fabric_width_cm + 4)
    ax.set_aspect("equal")
    ax.axis("off")
    plt.tight_layout()

    return fig, round(total_len / 100, 2), round(eff, 1)


def math_predict(
    area_coeff: dict,
    width_cm: float,
    total_garments: int,
    shrink_pct: float = 0.0,
    stretch_pct: float = 0.0,
    manual_area: float | None = None,
    adjusted_efficiency: float | None = None,
    cutting_loss_m2: float = 0.0,
) -> dict:
    """Predict consumption using area coefficient and physics."""
    if manual_area and manual_area > 0:
        area_per_piece = manual_area + cutting_loss_m2
        area_source = "размеры изделия"
        if cutting_loss_m2 > 0:
            area_source += f" + потери раскроя ({cutting_loss_m2:.4f} м²)"
    else:
        area_per_piece = area_coeff["area_mean"]
        area_source = f"база ({area_coeff['count']} записей)"

    # Use width-adjusted efficiency if available, otherwise fallback to type average
    if adjusted_efficiency and adjusted_efficiency > 0:
        eff = adjusted_efficiency / 100
        eff_display = adjusted_efficiency
    else:
        eff = area_coeff["efficiency_mean"] / 100
        eff_display = area_coeff["efficiency_mean"]

    width_m = width_cm / 100

    shrink_factor = 1 + abs(shrink_pct) / 100
    stretch_factor = 1 + abs(stretch_pct) / 100
    correction = (shrink_factor + stretch_factor) / 2

    consumption = (area_per_piece / (width_m * eff)) * correction
    length = consumption * total_garments

    cons_min = (area_coeff["area_min"] / (width_m * eff)) * correction
    cons_max = (area_coeff["area_max"] / (width_m * eff)) * correction

    return {
        "consumption_m_per_piece": round(consumption, 3),
        "estimated_length_m": round(length, 2),
        "estimated_efficiency_pct": round(eff_display, 1),
        "range_min": round(cons_min, 3),
        "range_max": round(cons_max, 3),
        "area_per_piece_m2": round(area_per_piece, 4),
        "area_source": area_source,
        "records_used": area_coeff["count"],
    }


# ── Search ────────────────────────────────────────────────────────────────────


def find_similar(
    df: pd.DataFrame,
    product_type: str | None,
    width_cm: float,
    block_table: str | None,
    sizes: str = "",
    shrink_pct: float = 0.0,
    stretch_pct: float = 0.0,
) -> pd.DataFrame:
    mask = (df["width_cm"] >= width_cm - WIDTH_TOLERANCE) & (
        df["width_cm"] <= width_cm + WIDTH_TOLERANCE
    )
    if product_type:
        mask &= df["product_type"] == product_type
    if block_table:
        mask &= df["block_table"] == block_table
    similar = df.loc[mask].copy()
    if similar.empty:
        return similar

    similar["_w"] = 1 - (similar["width_cm"] - width_cm).abs() / (WIDTH_TOLERANCE + 1)

    if sizes:
        similar["_s"] = similar["sizes"].apply(lambda s: size_similarity(sizes, s))
    else:
        similar["_s"] = 0.5

    if shrink_pct != 0 or stretch_pct != 0:
        max_diff = 5.0
        similar["_sh"] = 1 - (similar["shrink_pct"] - shrink_pct).abs().clip(upper=max_diff) / max_diff
        similar["_st"] = 1 - (similar["stretch_pct"] - stretch_pct).abs().clip(upper=max_diff) / max_diff
        similar["_score"] = similar["_w"] * 0.3 + similar["_s"] * 0.4 + similar["_sh"] * 0.15 + similar["_st"] * 0.15
    else:
        similar["_score"] = similar["_w"] * 0.4 + similar["_s"] * 0.6

    similar = similar.sort_values("_score", ascending=False).head(TOP_N)
    return similar.drop(columns=[c for c in similar.columns if c.startswith("_")])


# ── Prompt ────────────────────────────────────────────────────────────────────


def build_prompt(
    model_name: str,
    product_type: str,
    width_cm: float,
    sizes: str,
    block_table: str,
    shrink_pct: float,
    stretch_pct: float,
    similar: pd.DataFrame,
    stats: dict,
    input_total: int,
    input_num_sizes: int,
    math_result: dict | None,
) -> list[dict]:
    records_text = similar[DISPLAY_COLUMNS].to_csv(index=False)

    math_block = ""
    if math_result:
        math_block = (
            f"\n=== МАТЕМАТИЧЕСКИЙ ПРОГНОЗ (на основе площади лекал) ===\n"
            f"Площадь лекал на 1 изделие: {math_result['area_per_piece_m2']} м² "
            f"(среднее по {math_result['records_used']} записям типа \"{product_type}\")\n"
            f"Расход: {math_result['consumption_m_per_piece']} м/шт\n"
            f"Длина раскладки: {math_result['estimated_length_m']} м\n"
            f"Ожидаемый КПД: {math_result['estimated_efficiency_pct']}%\n"
            f"Диапазон расхода: {math_result['range_min']} – {math_result['range_max']} м/шт\n"
            f"Формула: расход = площадь_лекал / (ширина_ткани × КПД) × коррекция_усадки\n"
        )

    system = (
        "Ты — эксперт по раскладке лекал и нормированию расхода ткани на швейном производстве.\n\n"
        "Тебе дают:\n"
        "1. Параметры нового заказа\n"
        "2. МАТЕМАТИЧЕСКИЙ ПРОГНОЗ — расчёт по формуле площади лекал (это физически обоснованная оценка, "
        "КПД уже скорректирован под ширину ткани)\n"
        "3. Историческую базу похожих раскладок со статистикой\n\n"
        "Твоя задача — дать ФИНАЛЬНЫЙ прогноз, объединив математику и исторические данные.\n\n"
        "КРИТИЧЕСКИ ВАЖНЫЕ ПРАВИЛА:\n"
        "1. Математический прогноз — это твоя БАЗОВАЯ ЛИНИЯ. Особенно если он рассчитан по размерам "
        "изделия — это ПРЯМОЙ расчёт площади лекал. Твой финальный прогноз ДОЛЖЕН быть близок "
        "к математическому (отклонение не более ±15%).\n"
        "2. Если похожих записей МАЛО (< 5) — доверяй математике БОЛЬШЕ чем истории. "
        "Не тяни прогноз к среднему по нерелевантным записям!\n"
        "3. Корректируй математику ТОЛЬКО при наличии веских оснований:\n"
        "   - Много (5+) записей с очень похожими параметрами показывают другой расход\n"
        "   - Размерный ассортимент сильно отличается от среднего (много крупных/мелких)\n"
        "   - Усадка/растяжение существенно отличается\n"
        "4. Кол-во изделий: мало изделий (< 8) = хуже оптимизация = ниже КПД = выше расход\n"
        "5. Узкая ткань (< 100 см) даёт ЗНАЧИТЕЛЬНО ниже КПД (60-70%) чем широкая (80-85%)\n\n"
        "Ответ строго в JSON:\n"
        "{\n"
        '  "consumption_m_per_piece": число (метры на 1 изделие, точность 0.001),\n'
        '  "estimated_length_m": число (длина раскладки в метрах, точность 0.01),\n'
        '  "estimated_efficiency_pct": число (ожидаемый КПД в %, точность 0.1),\n'
        '  "confidence": "low" | "medium" | "high",\n'
        '  "explanation": "подробный анализ на русском (4-6 предложений): '
        "как математический прогноз был скорректирован и почему, "
        "какие факторы увеличили или уменьшили расход относительно базовой линии\",\n"
        '  "range_min": число (нижняя граница расхода м/шт),\n'
        '  "range_max": число (верхняя граница расхода м/шт)\n'
        "}"
    )

    user = (
        f"=== НОВЫЙ ЗАКАЗ ===\n"
        f"Модель: {model_name or 'не указана'}\n"
        f"Тип изделия: {product_type or 'не указан'}\n"
        f"Ширина ткани: {width_cm} см\n"
        f"Размерный ассортимент: {sizes}\n"
        f"  → Всего изделий: {input_total}, уникальных размеров: {input_num_sizes}\n"
        f"Тип ткани: {block_table}\n"
        f"Усадка: {shrink_pct}%, Растяжение: {stretch_pct}%\n"
        f"{math_block}\n"
        f"=== СТАТИСТИКА ПО {stats['count']} ПОХОЖИМ РАСКЛАДКАМ ===\n"
        f"Расход (м/шт): среднее={stats['consumption_mean']}, "
        f"медиана={stats['consumption_median']}, "
        f"мин={stats['consumption_min']}, макс={stats['consumption_max']}, "
        f"стд.откл.={stats['consumption_std']}\n"
        f"Длина (м): среднее={stats['length_mean']}, "
        f"мин={stats['length_min']}, макс={stats['length_max']}\n"
        f"КПД (%): среднее={stats['efficiency_mean']}, "
        f"мин={stats['efficiency_min']}, макс={stats['efficiency_max']}\n"
        f"Кол-во изделий: среднее={stats['garments_mean']}, "
        f"мин={stats['garments_min']}, макс={stats['garments_max']}\n"
        f"Ширины: {stats['widths_used']}\n"
        f"Ткани: {stats['fabrics_used']}\n"
        f"Усадка среднее: {stats['shrink_mean']}%, Растяжение среднее: {stats['stretch_mean']}%\n\n"
        f"=== ИСТОРИЧЕСКИЕ ЗАПИСИ ===\n"
        f"{records_text}\n"
        f"Дай финальный прогноз, объединив математический расчёт и исторические данные."
    )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def call_openai(messages: list[dict]) -> dict:
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.3,
        response_format={"type": "json_object"},
    )
    text = response.choices[0].message.content
    return json.loads(text)


# ── UI ────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Прогноз расхода ткани", layout="wide")
st.title("Прогноз нормы расхода ткани")
st.caption(
    "Гибридный прогноз: математический расчёт по площади лекал + "
    "ИИ-анализ исторических раскладок."
)

df = load_data()
area_coeffs = compute_area_coefficients(df.to_json())

with st.sidebar:
    st.header("Параметры заказа")

    model_name = st.text_input("Модель", placeholder="AW26-105766")

    HIDE_TYPES = {"other", "tshirt_women", "Женская ", "Женская"}
    product_types = sorted(t for t in df["product_type"].dropna().unique() if t not in HIDE_TYPES)
    product_type = st.selectbox(
        "Тип изделия",
        product_types,
    )

    width_cm = st.number_input(
        "Ширина ткани (см)",
        min_value=50.0,
        max_value=300.0,
        value=175.0,
        step=5.0,
    )

    sizes = st.text_input(
        "Размерный ассортимент",
        placeholder="116/1 128/1 140/2 152/3 164/5",
        help="Формат: Размер/Количество через пробел",
    )

    block_table = st.selectbox(
        "Тип ткани (block_table)",
        [""] + sorted(df["block_table"].dropna().unique()),
        format_func=lambda x: "— не указан —" if x == "" else x,
    )

    # ── DXF file upload ─────────────────────────────────────────────────
    dxf_pieces = []
    with st.expander("Загрузка лекал (DXF)"):
        st.caption("Загрузите DXF файл лекал из CAD-системы (Gerber, Lectra и др.)")
        uploaded_dxf = st.file_uploader(
            "DXF файл лекал", type=["dxf"],
            help="Файл с контурами деталей из CAD-системы",
        )
        if uploaded_dxf is not None:
            try:
                dxf_pieces, dxf_diag = parse_dxf_pieces(uploaded_dxf.read())
                if dxf_pieces:
                    df_dxf = pd.DataFrame(dxf_pieces)
                    df_dxf = df_dxf[["name", "area_cm2", "width_cm", "height_cm"]]
                    df_dxf.columns = ["Деталь", "Площадь (см²)", "Ширина (см)", "Высота (см)"]
                    st.dataframe(df_dxf, use_container_width=True, hide_index=True)
                    total_dxf_area = sum(p["area_cm2"] for p in dxf_pieces)
                    units = dxf_diag.get("units_detected", "")
                    st.success(
                        f"Найдено {len(dxf_pieces)} деталей, "
                        f"общая площадь: {total_dxf_area:.0f} см² "
                        f"({total_dxf_area / 10000:.4f} м²)"
                        f"{f' | Единицы: {units}' if units else ''}"
                    )
                else:
                    st.warning("Не найдено деталей в DXF файле.")
                    # Show diagnostics to help debug
                    types_str = ", ".join(
                        f"{t}: {c}" for t, c in dxf_diag.get("entity_types", {}).items()
                    )
                    st.caption(
                        f"Диагностика: {dxf_diag.get('total_entities', 0)} объектов "
                        f"({types_str}), "
                        f"блоков: {dxf_diag.get('blocks', 0)}, "
                        f"контуров найдено: {dxf_diag.get('raw_shapes_found', 0)}"
                    )
            except Exception as e:
                st.error(f"Ошибка чтения DXF: {e}")

    # ── Single-size measurements (quick math) ───────────────────────────
    single_measurement = {}
    with st.expander("Измерения — один размер (быстрый расчёт)"):
        st.caption("Введите измерения среднего размера (напр. M). Используется только для математического расчёта площади.")
        _mc1, _mc2, _mc3, _mc4 = st.columns(4)
        with _mc1:
            _A = st.number_input("A (дл. переда)", min_value=0.0, max_value=200.0, value=0.0, step=0.1, format="%.1f", key="s_A")
            _B = st.number_input("B (дл. спинки)", min_value=0.0, max_value=200.0, value=0.0, step=0.1, format="%.1f", key="s_B")
        with _mc2:
            _C = st.number_input("C (1/2 груди)", min_value=0.0, max_value=100.0, value=0.0, step=0.1, format="%.1f", key="s_C")
            _K1 = st.number_input("K1 (гл. проймы)", min_value=0.0, max_value=50.0, value=0.0, step=0.1, format="%.1f", key="s_K1")
        with _mc3:
            _G1 = st.number_input("G1 (дл. рукава)", min_value=0.0, max_value=80.0, value=0.0, step=0.1, format="%.1f", key="s_G1")
            _Z3 = st.number_input("Z3 (1/2 шир. рук.)", min_value=0.0, max_value=40.0, value=0.0, step=0.1, format="%.1f", key="s_Z3")
        with _mc4:
            _E2 = st.number_input("E2 (1/2 планки)", min_value=0.0, max_value=50.0, value=0.0, step=0.1, format="%.1f", key="s_E2")
            _T4 = st.number_input("T4 (шир. планки)", min_value=0.0, max_value=10.0, value=0.0, step=0.1, format="%.1f", key="s_T4")
        single_measurement = {"A": _A, "B": _B, "C": _C, "K1": _K1, "G1": _G1, "Z3": _Z3, "E2": _E2, "T4": _T4}

    # ── Per-size measurement table ───────────────────────────────────────
    size_measurements = {}
    with st.expander("Измерения — все размеры (точная раскладка)"):
        parsed_sizes = parse_sizes(sizes) if sizes.strip() else Counter()
        unique_sizes = list(parsed_sizes.keys())

        if unique_sizes:
            st.caption("Заполните измерения для каждого размера. Чем точнее — тем лучше прогноз.")

            # Initial DataFrame — data_editor preserves edits via key
            data = {"Кол": [parsed_sizes[s] for s in unique_sizes]}
            for col in MEAS_COLS:
                data[col] = [0.0] * len(unique_sizes)
            df_meas_input = pd.DataFrame(data, index=unique_sizes)
            df_meas_input.index.name = "Размер"

            # Key changes when sizes change → resets table for new sizes
            editor_key = f"meas_{'_'.join(unique_sizes)}"

            edited_meas = st.data_editor(
                df_meas_input,
                use_container_width=True,
                column_config={
                    "Кол": st.column_config.NumberColumn("Кол", disabled=True),
                    "A": st.column_config.NumberColumn("A", help="Длина переда", min_value=0.0, max_value=200.0, format="%.1f"),
                    "B": st.column_config.NumberColumn("B", help="Длина спинки", min_value=0.0, max_value=200.0, format="%.1f"),
                    "C": st.column_config.NumberColumn("C", help="1/2 обхвата груди", min_value=0.0, max_value=100.0, format="%.1f"),
                    "K1": st.column_config.NumberColumn("K1", help="Глубина проймы", min_value=0.0, max_value=50.0, format="%.1f"),
                    "G1": st.column_config.NumberColumn("G1", help="Длина рукава", min_value=0.0, max_value=80.0, format="%.1f"),
                    "Z3": st.column_config.NumberColumn("Z3", help="1/2 ширины рукава", min_value=0.0, max_value=40.0, format="%.1f"),
                    "E2": st.column_config.NumberColumn("E2", help="1/2 длины планки", min_value=0.0, max_value=50.0, format="%.1f"),
                    "T4": st.column_config.NumberColumn("T4", help="Ширина планки", min_value=0.0, max_value=10.0, format="%.1f"),
                },
                key=editor_key,
            )

            # Build measurements dict from edited table
            for s in unique_sizes:
                size_measurements[s] = {
                    col: float(edited_meas.loc[s, col]) for col in MEAS_COLS
                }
        else:
            st.caption("Сначала укажите размерный ассортимент выше.")

    with st.expander("Параметры строчки (из спецификации клиента)"):
        st.caption("Ширины строчек влияют на припуски швов и подгибы — точнее расчёт площади.")
        col_nst, col_cst = st.columns(2)
        with col_nst:
            stitch_needle = st.number_input(
                "Игольная строчка (см)", min_value=0.0, max_value=5.0,
                value=0.0, step=0.1, format="%.1f",
                help="|| — needle stitching, для боковых швов",
            )
        with col_cst:
            stitch_cover = st.number_input(
                "Покровная строчка (см)", min_value=0.0, max_value=5.0,
                value=0.0, step=0.1, format="%.1f",
                help="Coverstitch, для подгибов низа и рукавов",
            )
        stitch_placket = st.number_input(
            "Планочная строчка (см)", min_value=0.0, max_value=5.0,
            value=0.0, step=0.1, format="%.1f",
            help="Cover stitch планки (placket from shell)",
        )

    stitch_params = {
        "needle": stitch_needle,
        "cover": stitch_cover,
        "placket": stitch_placket,
    }

    st.markdown("**Усадка / Растяжение (%)**")
    col_sh, col_st = st.columns(2)
    with col_sh:
        shrink_pct_input = st.number_input(
            "Усадка", min_value=0.0, max_value=10.0,
            value=0.0, step=0.01, format="%.2f", help="Например: 2.87",
        )
    with col_st:
        stretch_pct_input = st.number_input(
            "Растяжение", min_value=0.0, max_value=10.0,
            value=0.0, step=0.01, format="%.2f", help="Например: 2.70",
        )
    shrink_pct = -shrink_pct_input
    stretch_pct = -stretch_pct_input

    predict_btn = st.button("Рассчитать", type="primary", use_container_width=True)

# ── Parsed sizes info ─────────────────────────────────────────────────────────

if sizes.strip():
    parsed = parse_sizes(sizes)
    if parsed:
        with st.sidebar:
            st.caption(
                f"Распознано: {sizes_count(sizes)} размеров, "
                f"{sizes_total(sizes)} изделий"
            )

# ── Prediction ────────────────────────────────────────────────────────────────

if predict_btn:
    if not sizes.strip():
        st.warning("Укажите размерный ассортимент.")
        st.stop()

    if not os.environ.get("OPENAI_API_KEY"):
        st.error("OPENAI_API_KEY не найден. Добавьте его в файл `.env`.")
        st.stop()

    input_total = sizes_total(sizes)
    input_num_sizes = sizes_count(sizes)

    if input_total == 0:
        st.warning("Не удалось распознать размеры. Формат: Размер/Кол через пробел.")
        st.stop()

    # ── Mathematical prediction ───────────────────────────────────────────
    manual_area = None
    area_breakdown = None
    pieces_info = None
    layout_info = None
    cutting_loss = 0.0
    has_stitch = any(v > 0 for v in stitch_params.values())
    stitch_arg = stitch_params if has_stitch else None

    # Check if per-size measurements are filled
    parsed_sizes_dict = parse_sizes(sizes)
    has_sized_measurements = any(
        size_measurements.get(s, {}).get("A", 0) > 0
        and size_measurements.get(s, {}).get("C", 0) > 0
        for s in parsed_sizes_dict
    )

    # Priority: DXF > manual measurements > historical data only
    has_dxf = len(dxf_pieces) > 0

    if has_dxf:
        # Use DXF-parsed exact areas and bounding boxes
        total_dxf_area_cm2 = sum(p["area_cm2"] for p in dxf_pieces)
        manual_area = round(total_dxf_area_cm2 / 10000, 4)  # cm² → m²
        area_breakdown = (
            f"Площадь из DXF: {total_dxf_area_cm2:.0f} см² = {manual_area} м²/изделие "
            f"({len(dxf_pieces)} деталей)"
        )
        # Convert DXF pieces to 5-tuple format for marker visualization
        # Each DXF piece × total_garments (one DXF = one garment set)
        pieces_info = []
        for p in dxf_pieces:
            pieces_info.append((
                p["width_cm"], p["height_cm"], p["name"],
                input_total,  # total count across all garments
                "DXF",
            ))
        if pieces_info:
            layout_info = layout_analysis(pieces_info, width_cm, input_total)
            cutting_loss = layout_info["loss_per_piece_m2"]
    elif has_sized_measurements:
        manual_area, area_breakdown, pieces_info = calc_sized_pieces(
            parsed_sizes_dict, size_measurements, stitch_arg,
        )
        if pieces_info:
            layout_info = layout_analysis(pieces_info, width_cm, input_total)
            cutting_loss = layout_info["loss_per_piece_m2"]
    elif single_measurement.get("A", 0) > 0 and single_measurement.get("C", 0) > 0:
        # Single-size = reference size → grade area across assortment
        ref_area, area_breakdown, single_pieces = calc_area_from_measurements(
            single_measurement, stitch_arg,
        )
        n_sizes = len(parsed_sizes_dict)
        if n_sizes > 1:
            # ~5% area change per size step (standard grading)
            AREA_GRADE_PCT = 0.05
            mid_idx = n_sizes // 2
            total_graded = 0.0
            for i, (sz, cnt) in enumerate(parsed_sizes_dict.items()):
                step = i - mid_idx
                total_graded += ref_area * (1 + step * AREA_GRADE_PCT) * cnt
            manual_area = round(total_graded / input_total, 4)
            area_breakdown += (
                f"\n→ Грейдинг по ассортименту ({n_sizes} размеров): "
                f"средневзвешенная {manual_area} м²/изд."
            )
        else:
            manual_area = ref_area
        if single_pieces:
            layout_info = layout_analysis(single_pieces, width_cm, input_total)
            cutting_loss = layout_info["loss_per_piece_m2"]

    math_result = None
    adj_eff = None
    if product_type and product_type in area_coeffs:
        adj_eff = estimate_efficiency(df, width_cm, product_type)
        math_result = math_predict(
            area_coeffs[product_type], width_cm, input_total,
            shrink_pct, stretch_pct, manual_area, adj_eff, cutting_loss,
        )

    # ── Similar records search ────────────────────────────────────────────
    similar = find_similar(
        df, product_type or None, width_cm, block_table or None,
        sizes, shrink_pct, stretch_pct,
    )

    if similar.empty and block_table:
        similar = find_similar(
            df, product_type or None, width_cm, None,
            sizes, shrink_pct, stretch_pct,
        )
        if not similar.empty:
            st.info("По ткани совпадений нет — показаны результаты без фильтра по ткани.")

    if similar.empty and product_type:
        similar = find_similar(
            df, None, width_cm, block_table or None,
            sizes, shrink_pct, stretch_pct,
        )
        if not similar.empty:
            st.info("По типу изделия совпадений нет — показаны результаты по всем типам.")

    if similar.empty and math_result is None:
        st.warning("Похожих раскладок не найдено и нет данных для математического расчёта.")
        st.stop()

    # ── Show math prediction ──────────────────────────────────────────────
    if math_result:
        st.subheader("Математический прогноз (по площади лекал)")
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Расход", f"{math_result['consumption_m_per_piece']} м/шт")
        mc2.metric("Диапазон", f"{math_result['range_min']} – {math_result['range_max']}")
        mc3.metric("Длина раскладки", f"{math_result['estimated_length_m']} м")
        mc4.metric("КПД (средний)", f"{math_result['estimated_efficiency_pct']}%")
        source_text = f"Площадь лекал: {math_result['area_per_piece_m2']} м²/изделие (источник: {math_result['area_source']})"
        if area_breakdown:
            source_text += f"\n{area_breakdown}"
        st.caption(source_text)

        # Show layout analysis if available
        if layout_info:
            with st.expander("Анализ раскладки деталей на ткани"):
                st.markdown(f"**Ширина ткани: {width_cm} см**")
                for d in layout_info["details"]:
                    orient_icon = "↕" if "верт" in d["orientation"] else "↔"
                    st.markdown(
                        f"- **{d['name']}** ({d['size']} см) {orient_icon} {d['orientation']} — "
                        f"{d['per_row']} шт/ряд, "
                        f"{d['rows_needed']} рядов × {d['total_count']} шт = "
                        f"**{d['length_cm'] / 100:.2f} м** по длине, "
                        f"остаток по ширине: {d['waste_width_cm']:.1f} см"
                    )
                st.markdown(
                    f"\n**Потери раскроя:** зазоры между деталями ({CUTTING_GAP} см × "
                    f"{layout_info['total_rows']} рядов) = {layout_info['gap_loss_cm']} см + "
                    f"концы раскладки = {layout_info['end_loss_cm']} см → "
                    f"**итого {layout_info['total_loss_cm']} см** "
                    f"(+{layout_info['loss_per_piece_m2']} м²/изделие)"
                )

    # ── Marker layout visualization (only for per-size or DXF) ──────────
    if pieces_info and (has_sized_measurements or has_dxf):
        st.subheader("Визуализация маркер-раскладки")
        try:
            sb = None
            if has_dxf:
                sb = {"DXF": input_total}
            elif has_sized_measurements:
                sb = dict(parsed_sizes_dict)
            marker_fig, marker_len, marker_eff = draw_marker_layout(
                pieces_info, width_cm, input_total,
                sizes_breakdown=sb,
            )
            st.pyplot(marker_fig)
            plt.close(marker_fig)
            st.caption(
                f"Симуляция раскладки (FFDH-алгоритм): длина {marker_len} м, "
                f"КПД упаковки {marker_eff}%. "
                f"Реальная раскладка в CAD-системе будет эффективнее за счёт нестинга."
            )
        except Exception as e:
            st.warning(f"Не удалось построить визуализацию: {e}")

    # ── AI prediction ─────────────────────────────────────────────────────
    if not similar.empty:
        stats = compute_stats(similar)

        messages = build_prompt(
            model_name, product_type or "не указан", width_cm, sizes,
            block_table or "не указан", shrink_pct, stretch_pct,
            similar, stats, input_total, input_num_sizes, math_result,
        )

        with st.spinner("ИИ анализирует данные..."):
            try:
                result = call_openai(messages)
            except Exception as e:
                st.error(f"Ошибка при обращении к OpenAI: {e}")
                st.stop()

        st.subheader("Финальный прогноз ИИ")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Расход", f"{result.get('consumption_m_per_piece', '—')} м/шт")
        with col2:
            r_min = result.get("range_min", "—")
            r_max = result.get("range_max", "—")
            st.metric("Диапазон", f"{r_min} – {r_max}")
        with col3:
            st.metric("Длина раскладки", f"{result.get('estimated_length_m', '—')} м")
        with col4:
            st.metric("КПД раскладки", f"{result.get('estimated_efficiency_pct', '—')}%")

        confidence = result.get("confidence", "—")
        confidence_map = {"high": "Высокая", "medium": "Средняя", "low": "Низкая"}
        confidence_colors = {"high": "green", "medium": "orange", "low": "red"}
        color = confidence_colors.get(confidence, "gray")
        label = confidence_map.get(confidence, confidence)
        st.markdown(f"**Уверенность:** :{color}[{label}]")

        st.subheader("Анализ ИИ")
        st.info(result.get("explanation", "—"))

        # ── Stats & table ─────────────────────────────────────────────────
        st.subheader("Статистика по похожим раскладкам")
        sc1, sc2, sc3, sc4 = st.columns(4)
        sc1.metric("Расход среднее", f"{stats['consumption_mean']} м/шт")
        sc2.metric("Расход медиана", f"{stats['consumption_median']} м/шт")
        sc3.metric("КПД среднее", f"{stats['efficiency_mean']}%")
        sc4.metric("Длина среднее", f"{stats['length_mean']} м")

        st.subheader(f"Похожие раскладки ({stats['count']} записей)")
        st.dataframe(similar[DISPLAY_COLUMNS], use_container_width=True, hide_index=True)
    elif math_result:
        st.info("Похожих раскладок не найдено — прогноз только по математической модели.")
