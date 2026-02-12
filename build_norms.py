import pandas as pd

IN_XLSX = "markers_labeled.xlsx"
OUT_XLSX = "norms_by_type.xlsx"


def main():
    df = pd.read_excel(IN_XLSX)

    # Берем только строки с расходом и заполненным типом
    df = df[df["consumption_m_per_piece"].notna()].copy()
    df = df[df["product_type"].astype(str).str.strip() != ""].copy()

    # Убираем rib/test из норм одежды (если хотите отдельно — просто не фильтруйте)
    df = df[df["consumption_m_per_piece"] >= 0.10].copy()

    # Нормализуем ширину и блок
    df["width_cm_round"] = df["width_cm"].round(0)
    df["block_table_clean"] = df["block_table"].fillna("UNKNOWN").astype(str).str.strip()
    df["product_type_clean"] = df["product_type"].astype(str).str.strip().str.lower()

    g = df.groupby(["product_type_clean", "width_cm_round", "block_table_clean"])["consumption_m_per_piece"].agg(
        count="count",
        mean="mean",
        std="std",
        min="min",
        max="max",
    ).reset_index()

    g["normal_m_per_piece"] = g["mean"]
    g["safe_m_per_piece"] = g["mean"] + g["std"].fillna(0)  # безопасная норма

    g = g.sort_values(["count", "product_type_clean"], ascending=[False, True])

    with pd.ExcelWriter(OUT_XLSX) as w:
        df.to_excel(w, index=False, sheet_name="data_clean")
        g.to_excel(w, index=False, sheet_name="norms")

    print(f"✅ Saved: {OUT_XLSX}")


if __name__ == "__main__":
    main()
