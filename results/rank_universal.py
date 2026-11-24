
from pathlib import Path
import csv
from wordfreq import zipf_frequency

HERE = Path(__file__).resolve().parent
THRESHOLD = 5.0

words = sorted(set(
    w.strip()
    for w in (HERE / "universal_words.txt").read_text(
        encoding="utf-8"
    ).splitlines()
    if w.strip()
))

results = []

for word in words:
    zipf = zipf_frequency(word, "en", wordlist="best")

    if zipf == 0:
        category = "unlisted"
    elif zipf >= THRESHOLD:
        category = "common"
    else:
        category = "less_common"

    per_million = 10 ** (zipf - 3) if zipf > 0 else None
    results.append((word, zipf, per_million, category))

# Sort by frequency, with unlisted words last.
results.sort(key=lambda r: (r[3] == "unlisted", r[1], r[0]))

# Complete frequency table
with (HERE / "universal_word_frequencies.csv").open(
    "w", newline="", encoding="utf-8-sig"
) as f:
    writer = csv.writer(f)
    writer.writerow([
        "word", "zipf", "estimated_per_million", "category"
    ])
    for word, zipf, pm, category in results:
        writer.writerow([
            word, zipf, round(pm, 5) if pm is not None else "",
            category
        ])

# Exclude common words
remaining = [
    r for r in results if r[3] != "common"
]

(HERE / "uncommon_universal_words.txt").write_text(
    "\n".join(r[0] for r in remaining) + "\n",
    encoding="utf-8"
)

# Keep unlisted words separately:
# unlisted does not necessarily mean genuinely rare.
unlisted = [r[0] for r in results if r[3] == "unlisted"]

(HERE / "unlisted_universal_words.txt").write_text(
    "\n".join(unlisted) + "\n",
    encoding="utf-8"
)

print(f"Total universal words: {len(words)}")
print(f"Common words removed: {len(words) - len(remaining)}")
print(f"Remaining: {len(remaining)}")
print(f"Unlisted in wordfreq: {len(unlisted)}")

print("\nLess-common universal words:")
for word, zipf, pm, category in remaining:
    if category == "less_common":
        print(f"{word:25} Zipf={zipf:.2f}")
