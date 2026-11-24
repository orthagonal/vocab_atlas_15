# Vocabulary Atlas: Overlap Across 15 Hugging Face Tokenizers

**An exploratory, reproducible study of which token strings are shared across language-model vocabularies — and why.**

**Status:** Paused after the initial 15-tokenizer census and English-frequency analysis. The results below are measurements from this sample, **not** estimates for every tokenizer on Hugging Face.

## Overview

Tokenizers break text into reusable units: complete words, word fragments, symbols, and sometimes byte sequences. Different model families often learn different vocabularies, but how much do those vocabularies have in common?

This project downloads only public `tokenizer.json` files (no model weights), extracts lexical token inventories, and compares their exact token strings and approximate normalized byte representations. A follow-up experiment asks why certain strings occur in **all 15** vocabularies, including relatively uncommon English words, place-name fragments, and programming terms.

## Results at a glance

| Metric | Observed |
|---|---:|
| Tokenizer vocabularies analyzed | **15** |
| Total lexical token entries across the 15 vocabularies | **1,780,381** |
| Distinct raw lexical token strings (union) | **867,110** |
| Distinct normalized byte surfaces (union) | **796,695** |
| Raw strings appearing in exactly one vocabulary | **519,745** |
| Raw strings appearing in two or more vocabularies | **347,365** |
| Raw strings appearing in all 15 vocabularies | **998** |
| Distinct exact raw token sets in this sample | **15** |
| Distinct token-to-ID mappings in this sample | **15** |

About **60%** of the distinct raw strings appear in only one of these 15 vocabularies. Only **998** appear in every one. This doesn't mean the rest are never useful or that different tokenizers encode the same text differently in every case; it is an exact *inventory membership* comparison.

### The 15 sampled vocabularies

| Tokenizer | Lexical entries | Newly added raw strings, in analysis order |
|---|---:|---:|
| GPT-2 | 50,256 | 50,256 |
| BERT uncased | 30,517 | 26,197 |
| RoBERTa | 50,260 | 4 |
| T5 | 31,997 | 26,740 |
| mT5 | 250,097 | 222,697 |
| XLM-R | 249,997 | 134,290 |
| BLOOM | 250,676 | 208,663 |
| GPT-NeoX | 50,275 | 5,548 |
| Llama 2 family | 31,997 | 3,760 |
| Mistral 7B | 31,997 | 1,356 |
| Llama 3 | 128,000 | 54,372 |
| Qwen 2.5 | 151,651 | 27,783 |
| Qwen 3 | 151,655 | 4 |
| Gemma 2 | 255,994 | 100,759 |
| Falcon 7B | 65,012 | 4,681 |

**Important:** “Newly added” means new relative to preceding rows in **this chosen order**, not historical novelty or proof that one model derived its vocabulary from another. The sample is intentionally varied but not random or representative of the entire Hub.

### Strongest overlaps

| Pair | Shared normalized byte surfaces | Jaccard similarity |
|---|---:|---:|
| Qwen 2.5 / Qwen 3 | 151,651 | ~100.00% |
| GPT-2 / RoBERTa | 50,256 | ~99.99% |
| Llama 3 / Qwen 2.5 | 109,566 | 64.42% |
| GPT-2 / Falcon 7B | 42,465 | 58.33% |
| GPT-2 / GPT-NeoX | 36,937 | 58.10% |

Very similar vocabularies are **not independent observations**. The Qwen 2.5 and Qwen 3 raw inventories differ by only four entries in this sample, and GPT-2 and RoBERTa also have near-identical raw token inventories. Jaccard overlap alone doesn't establish shared lineage, and byte-surface overlap doesn't establish identical tokenization behavior.

## What are the 998 universally shared strings?

Many are unsurprising: ASCII characters, short strings, punctuation, and very common words, including `the` and `and`.

The more interesting cases are strings whose **standalone English word frequency** is not especially high:

| Pattern | Observed examples | Possible explanation |
|---|---|---|
| Place-name and surname components | `ville`, `shire`, `berg`, `berger`, `borough`, `stein` | Common *inside* names even if rare alone |
| Web and development terminology | `html`, `php`, `java`, `http`, `www`, `api`, `default`, `index` | Recurrence in specialized technical/web text |
| Multilingual words or fragments | `und`, `oder`, `och`, `que`, `aux`, `ich` | English-only word frequency misses other languages |
| Reusable morphology | `ing`, `ness`, `inter`, `graph`, `maker` | Subword recurrence across many host words |

These categories are **descriptive hypotheses**, not proven causal explanations. A string present in a vocabulary may be encoded differently depending on spaces, context, normalization, and pre-tokenization.

### Follow-up: compare word frequency with substring frequency

We used [`wordfreq`](https://pypi.org/project/wordfreq/) to score standalone English words and to estimate how often each string occurs *inside other English words*. The latter is a frequency-weighted scan of a word list, **not** a raw-corpus count of tokenizer output.

A few observed examples:

| Token | English Zipf score | Estimated embedded occurrences / million alphabetic words | Example host words |
|---|---:|---:|---|
| `html` | 3.65 | 0.253 | `xhtml`, `shtml`, `dhtml` |
| `php` | 3.42 | 0.131 | `phpbb`, `phpunit`, `phpstorm` |
| `java` | 4.00 | 4.873 | `javascript`, `javanese` |
| `ville` | 3.14 | 93.776 | `nashville`, `louisville` |
| `berg` | 3.32 | 66.792 | `bloomberg`, `goldberg`, `iceberg` |
| `api` | 3.68 | 298.137 | `capital`, `rapid`, `therapist` |
| `und` | 3.94 | 4,352.732 | `around`, `under`, `found` |

**Interpretation:** A token like `ville` is far more prevalent as a substring than as a standalone word. `html` and `php` remain comparatively uncommon in this *general English* reference, making them interesting candidates for domain-specific investigation. For `api` and `und`, many matches are merely incidental substrings of unrelated English words: frequency of a character sequence is not evidence that it forms a meaningful linguistic unit or that a tokenizer emits it in context.

An earlier shortlist ranked by *embedded* occurrence rate alone. That ranking can make common words such as `learning` look artificially rare, because standalone occurrences are excluded. The follow-up CSV also includes `all_substring_occurrences_per_million_words`, which includes standalone and embedded matches on a consistent reference basis.

## Method

1. Retrieve `tokenizer.json` from 15 specified Hugging Face model repositories.
2. Extract token strings and IDs; exclude tokens explicitly marked as special in `added_tokens` metadata.
3. Compute exact raw-string unions, prevalence counts, unique-to-sample strings, and all 105 pairwise overlaps.
4. Construct a *diagnostic* normalized byte surface (e.g., decode GPT-style byte representations, map SentencePiece's `▁`, and strip WordPiece's `##` continuation marker). Compare normalized surfaces with Jaccard similarity.
5. Isolate the 998 strings shared by all 15 and rank lowercase alphabetic strings with `wordfreq`.
6. Use a weighted English word inventory to estimate standalone and embedded substring occurrence rates and, optionally, compare with strings appearing in 10–14 of the vocabularies.

**Scope limits:** The analysis counts **lexical token strings**, not unique English words, semantic units, actual model token usage, or every possible byte sequence. Normalized surfaces are a convenience for comparison, not proof of equivalent encoding. Dataset composition, tokenizer algorithms, merge rules, and shared ancestry are not controlled in this initial experiment.

## Reproduce

Python 3 is required. The initial census can run on Windows with:

```powershell
py -m pip install -r requirements.txt
py vocab_atlas_15.py --out results
```

Or use `run_windows.bat` / `run_windows.ps1`. Public tokenizer files are downloaded; model weights are **not** downloaded. `results/manifest.json` records the resolved repository revisions so a run can be traced back to its inputs. For an offline run using downloaded tokenizer JSON files, see `vocab_atlas_15.py --help`.

For the English substring investigation, install the additional dependency and place `substring_frequency_atlas.py` beside `universal_words.txt` (and ideally `all_token_prevalence.csv.gz`):

```powershell
py -m pip install wordfreq
py substring_frequency_atlas.py --words results/universal_words.txt --out results
```

`universal_words.txt` is an intermediate generated from universal raw tokens, filtered to lowercase ASCII alphabetic strings of at least two characters. It is **not** included in the original base census outputs unless you have added that export to the script. If your file is elsewhere, pass its actual path to `--words`.

Key output files:

- `results/REPORT.md`, `summary.json`, `manifest.json`: census findings and provenance.
- `results/vocabulary_stats.csv`, `pairwise_overlap.csv`, `all_token_prevalence.csv.gz`: sample inventories and overlap data.
- `results/accumulation.png`, `overlap_heatmap.png`, `prevalence.png`: visualizations, when plotting dependencies are installed.
- `results/substring_frequency_universal.csv`, `substring_frequency_all.csv`, `unexpected_universal_tokens.csv`, `prevalence_comparison.csv`: frequency-based follow-up analyses.

Run the offline unit tests with `py -m unittest -v test_atlas.py`. They test the mechanics using synthetic fixtures; the reported headline numbers come from the user's actual tokenizer run.

## Open questions / future work

- **Frequency vs. universality:** After controlling for token length and estimated substring frequency, which strings are *more prevalent across vocabularies than expected*?
- **Domain-specificity:** Are technical strings such as `html` and `php` more frequent in code/web corpora than in general English? Test on appropriate corpora rather than inferring from labels alone.
- **Languages:** Repeat the frequency comparison using multilingual reference distributions.
- **Family independence:** Collapse near-identical vocabularies before treating common tokens as independent convergence.
- **Actual encoding:** Measure how frequently each tokenizer *emits* a candidate token on the same evaluation corpus, not just whether the string is in the inventory.
- **Scale:** Expand beyond 15 tokenizers using exact vocabulary deduplication and reproducible revision pinning.

## Project status

This is an exploratory research repository, **paused intentionally** after the first real-data census and initial substring-frequency study. Findings are descriptive; stronger claims about convergence, corpus influence, and tokenizer optimization require controlled experiments.
