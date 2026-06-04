from pathlib import Path
from datasets import load_dataset
import pandas as pd

output_dir = Path("data/gsm8k")
output_dir.mkdir(parents=True, exist_ok=True)

dataset = load_dataset("openai/gsm8k", "main")

excel_path = output_dir / "gsm8k_overview.xlsx"

with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
    for split_name in dataset.keys():
        df = pd.DataFrame(dataset[split_name])

        df.insert(0, "id", range(1, len(df) + 1))
        df["final_answer"] = df["answer"].str.extract(r"####\s*(.*)")
        df["question_preview"] = df["question"].str.slice(0, 180)

        df = df[["id", "question", "question_preview", "answer", "final_answer"]]

        csv_path = output_dir / f"gsm8k_{split_name}.csv"
        df.to_csv(csv_path, index=False)

        df.to_excel(writer, sheet_name=split_name[:31], index=False)

print(f"Saved Excel file to: {excel_path}")
print(f"Saved CSV files to: {output_dir}")
