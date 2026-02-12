import pandas as pd

df = pd.read_excel("markers_data.xlsx")

need = ["length_m", "width_cm", "total_garments", "consumption_m_per_piece"]
print("Rows:", len(df))

for c in need:
    miss = df[c].isna().sum()
    print(f"{c} missing: {miss}")

# 1) Статистика расхода
print("\nConsumption stats (m/pcs):")
print(df["consumption_m_per_piece"].describe())

# 2) Топ-10 самых больших расходов
print("\nTop 10 biggest consumption:")
print(
    df.sort_values("consumption_m_per_piece", ascending=False)[
        ["file", "marker_name", "width_cm", "total_garments", "length_m", "consumption_m_per_piece"]
    ].head(10)
)

# 3) Найти подозрительно маленькие/большие расходы (аномалии)
tiny = df[df["consumption_m_per_piece"].notna() & (df["consumption_m_per_piece"] < 0.10)]
huge = df[df["consumption_m_per_piece"].notna() & (df["consumption_m_per_piece"] > 3.00)]

print("\nSuspicious consumption (<0.10 m/pcs):", len(tiny))
if len(tiny):
    print(tiny[["file","marker_name","length_m","total_garments","consumption_m_per_piece"]].head(30))

print("\nSuspicious consumption (>3.00 m/pcs):", len(huge))
if len(huge):
    print(huge[["file","marker_name","length_m","total_garments","consumption_m_per_piece"]].head(30))

# 4) Сохранить все проблемные строки в отдельный Excel
bad = df[
    df["length_m"].isna()
    | df["width_cm"].isna()
    | df["total_garments"].isna()
    | (df["consumption_m_per_piece"].notna() & (df["consumption_m_per_piece"] < 0.10))
]

bad_cols = [
    "file", "marker_name", "date", "time",
    "length_m", "width_cm", "efficiency_pct",
    "sizes", "total_garments",
    "consumption_m_per_piece",
    "shrink_pct", "stretch_pct",
    "block_table", "error"
]
bad_cols = [c for c in bad_cols if c in df.columns]

bad.to_excel("markers_bad_rows.xlsx", index=False)
print("\n✅ Saved markers_bad_rows.xlsx (bad rows):", len(bad))