import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))

import config as C
from dataset_split import canonical_split

OUT = C.DATA_PROCESSED.parent / "splits"
OUT.mkdir(parents=True, exist_ok=True)

df = canonical_split()
for name in ("train", "valid", "test"):
    part = df[df["split"] == name].sort_values("date")
    part.to_csv(OUT / f"{name}.csv", index=False)
    print(f"{name:5s}: {len(part):4d} hari | {part['date'].min().date()} .. {part['date'].max().date()}")

assert df["date"].is_unique
print(f"Tersimpan di {OUT}")