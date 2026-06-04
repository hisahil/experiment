from __future__ import annotations

from pathlib import Path
import re

import pandas as pd
from datasets import load_dataset


OUTPUT_DIR = Path("data/math500")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def extract_boxed(solution: str) -> str | None:
    match = re.findall(r"\\boxed\{([^{}]+)\}", solution)
    return match[-1].strip() if match else None


ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
df = pd.DataFrame(ds)

df["level_num"] = df["level"].astype(str).str.extract(r"(\d+)").astype(int)

df["gold_answer"] = df["answer"]
df["boxed_answer_fallback"] = df["solution"].apply(extract_boxed)

df_level_45 = df[df["level_num"].isin([4, 5])].copy()

df_level_45 = df_level_45[
    [
        "problem",
        "solution",
        "gold_answer",
        "boxed_answer_fallback",
        "subject",
        "level",
        "unique_id",
    ]
]

df_80 = df_level_45.sample(n=80, random_state=42)

df_level_45.to_csv(OUTPUT_DIR / "math500_level_4_5_all.csv", index=False)
df_80.to_csv(OUTPUT_DIR / "math500_level_4_5_sample_80.csv", index=False)

with pd.ExcelWriter(OUTPUT_DIR / "math500_overview.xlsx", engine="openpyxl") as writer:
    df.to_excel(writer, sheet_name="all_math500", index=False)
    df_level_45.to_excel(writer, sheet_name="level_4_5", index=False)
    df_80.to_excel(writer, sheet_name="sample_80", index=False)

print("Saved files in:", OUTPUT_DIR)
print("Total MATH-500:", len(df))
print("Level 4 and 5:", len(df_level_45))
print("Sample:", len(df_80))
