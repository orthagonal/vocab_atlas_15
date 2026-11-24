#!/usr/bin/env python3
"""Find universal tokenizer strings that are rare even as English substrings.

Inputs: universal_words.txt (one lowercase token per line), and optionally the
existing all_token_prevalence.csv.gz from Vocabulary Atlas (15 tokenizers).

Run: pip install wordfreq
     python substring_frequency_atlas.py

No model downloads or changes to vocab_atlas_15.py are required.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import heapq
from collections import defaultdict, deque
from pathlib import Path
from statistics import median


def load_tokens(words_path: Path, index_path: Path | None, min_prevalence: int):
    if not words_path.is_file():
        raise FileNotFoundError(f"Cannot find {words_path}. Put this script beside universal_words.txt or pass --words PATH.")
    universal = {
        x.strip() for x in words_path.read_text(encoding="utf-8-sig").splitlines()
        if x.strip().isascii() and x.strip().isalpha() and x.strip().islower() and len(x.strip()) >= 2
    }
    prevalence = {token: 15 for token in universal}
    if index_path and index_path.is_file():
        opener = gzip.open if index_path.suffix == '.gz' else open
        with opener(index_path, 'rt', encoding='utf-8-sig', newline='') as f:
            for row in csv.DictReader(f):
                token = row['raw_token']
                if not (token.isascii() and token.isalpha() and token.islower() and len(token) >= 2):
                    continue
                count = int(row['vocabulary_count'])
                if count >= min_prevalence:
                    prevalence[token] = count
        print(f"Using control group from {index_path}: {len(prevalence):,} tokens")
    else:
        print("No prevalence index found; analyzing universal_words.txt only.")
    return prevalence, universal


def make_automaton(tokens: list[str]):
    """Pure Python Aho-Corasick: enumerate overlaps in one scan per word."""
    edges: list[dict[str, int]] = [{}]
    fail = [0]
    output: list[list[int]] = [[]]
    for ix, token in enumerate(tokens):
        state = 0
        for char in token:
            nxt = edges[state].get(char)
            if nxt is None:
                nxt = len(edges)
                edges[state][char] = nxt
                edges.append({})
                fail.append(0)
                output.append([])
            state = nxt
        output[state].append(ix)
    q = deque(edges[0].values())
    while q:
        state = q.popleft()
        for char, child in edges[state].items():
            q.append(child)
            fallback = fail[state]
            while fallback and char not in edges[fallback]:
                fallback = fail[fallback]
            fail[child] = edges[fallback].get(char, 0)
            output[child].extend(output[fail[child]])
    return edges, fail, output


def count_substrings(tokens: list[str], word_frequencies: dict[str, float]):
    """Weighted substring occurrences, including overlapping matches.

    Rates are normalized per million lowercase ASCII alphabetic word tokens in
    wordfreq's estimated English word distribution, not per million characters.
    """
    edges, fail, output = make_automaton(tokens)
    all_mass = [0.0] * len(tokens)
    embedded_mass = [0.0] * len(tokens)
    host_count = [0] * len(tokens)
    top_hosts: list[list[tuple[float, str]]] = [[] for _ in tokens]
    total = 0.0
    num_words = 0
    for word, freq in word_frequencies.items():
        if not (word.isascii() and word.isalpha() and word.islower()) or freq <= 0:
            continue
        num_words += 1
        total += freq
        state = 0
        matched_embedded = set()
        for char in word:
            while state and char not in edges[state]:
                state = fail[state]
            state = edges[state].get(char, 0)
            for ix in output[state]:
                all_mass[ix] += freq
                if len(word) > len(tokens[ix]):
                    embedded_mass[ix] += freq
                    matched_embedded.add(ix)
        for ix in matched_embedded:
            host_count[ix] += 1
            heap = top_hosts[ix]
            item = (freq, word)
            if len(heap) < 5:
                heapq.heappush(heap, item)
            elif item > heap[0]:
                heapq.heapreplace(heap, item)
    if total <= 0:
        raise ValueError("No alphabetic words found in reference frequency data")
    scale = 1_000_000.0 / total
    return [
        (all_mass[i] * scale, embedded_mass[i] * scale,
         host_count[i], ', '.join(w for _, w in sorted(top_hosts[i], reverse=True)))
        for i in range(len(tokens))
    ], num_words


def export_csv(path: Path, rows: list[dict], headers: list[str] | None = None):
    if headers is None:
        headers = list(rows[0]) if rows else []
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--words', type=Path, default=Path('universal_words.txt'))
    parser.add_argument('--index', type=Path, default=None,
                        help='Optional all_token_prevalence.csv.gz; autodetected beside --words')
    parser.add_argument('--out', type=Path, default=None)
    parser.add_argument('--min-prevalence', type=int, default=10,
                        help='Minimum tokenizer prevalence to use as controls (default 10)')
    parser.add_argument('--zipf-cutoff', type=float, default=5.0,
                        help='Exclude common whole words at or above this English Zipf (default 5)')
    parser.add_argument('--embedded-cutoff', type=float, default=10.0,
                        help='Only keep rare embedded substrings below occurrences per million (default 10)')
    parser.add_argument('--min-length', type=int, default=4,
                        help='Minimum token length in rare shortlist (default 4)')
    args = parser.parse_args()

    from wordfreq import get_frequency_dict, zipf_frequency

    if args.index is None:
        for candidate in (args.words.parent / 'all_token_prevalence.csv.gz',
                          args.words.parent / 'results' / 'all_token_prevalence.csv.gz'):
            if candidate.is_file():
                args.index = candidate
                break
    out = args.out or args.words.parent
    out.mkdir(parents=True, exist_ok=True)
    prevalence, universal = load_tokens(args.words, args.index, args.min_prevalence)
    tokens = sorted(prevalence)
    if not tokens:
        raise ValueError('No valid lowercase alphabetic tokens found')

    print(f"Loading English frequency inventory and matching {len(tokens):,} tokens ...", flush=True)
    english = get_frequency_dict('en', wordlist='best')
    matches, n_words = count_substrings(tokens, english)
    print(f"Analyzed {n_words:,} lowercase alphabetic English word types")

    columns = [
        'token', 'vocabulary_count', 'english_zipf',
        'all_substring_occurrences_per_million_words',
        'embedded_occurrences_per_million_words',
        'number_of_embedded_host_words', 'frequent_host_examples',
    ]
    rows = []
    for token, (all_per_m, embed_per_m, hosts, examples) in zip(tokens, matches):
        z = zipf_frequency(token, 'en', wordlist='best')
        rows.append({
            'token': token,
            'vocabulary_count': prevalence[token],
            'english_zipf': round(z, 2),
            'all_substring_occurrences_per_million_words': round(all_per_m, 5),
            'embedded_occurrences_per_million_words': round(embed_per_m, 5),
            'number_of_embedded_host_words': hosts,
            'frequent_host_examples': examples,
        })

    rows.sort(key=lambda x: (x['vocabulary_count'], x['embedded_occurrences_per_million_words'], x['token']), reverse=True)
    export_csv(out / 'substring_frequency_all.csv', rows, columns)
    universal_rows = [r for r in rows if r['token'] in universal]
    universal_rows.sort(key=lambda r: (r['embedded_occurrences_per_million_words'], -len(r['token']), r['token']))
    export_csv(out / 'substring_frequency_universal.csv', universal_rows, columns)

    eligible = [r for r in universal_rows if
                len(r['token']) >= args.min_length and
                r['english_zipf'] < args.zipf_cutoff]
    rare = [r for r in eligible if
            0 < r['embedded_occurrences_per_million_words'] < args.embedded_cutoff]
    unobserved = [r for r in eligible if r['embedded_occurrences_per_million_words'] == 0]
    export_csv(out / 'unexpected_universal_tokens.csv', rare, columns)
    (out / 'unexpected_universal_tokens.txt').write_text(
        '\n'.join(r['token'] for r in rare) + ('\n' if rare else ''), encoding='utf-8'
    )
    export_csv(out / 'unobserved_embedded_tokens.csv', unobserved, columns)

    groups = defaultdict(list)
    for row in rows:
        length = len(row['token'])
        if length >= args.min_length:
            length_band = ('2-3' if length <= 3 else
                           '4-5' if length <= 5 else
                           '6-8' if length <= 8 else '9+')
            key = (row['vocabulary_count'], length_band)
            groups[key].append(row['embedded_occurrences_per_million_words'])
    group_rows = [
        {'vocabularies': count, 'length_band': length_band,
         'token_count': len(v), 'median_embedded_per_million': round(median(v), 4)}
        for (count, length_band), v in sorted(groups.items())
    ]
    export_csv(out / 'prevalence_comparison.csv', group_rows,
               ['vocabularies', 'length_band', 'token_count', 'median_embedded_per_million'])

    print(f"\nUniversal tokens measured: {len(universal_rows):,}")
    print(f"Rare embedded universal tokens:  {len(rare):,}")
    print(f"No embedded match in reference: {len(unobserved):,}")
    print(f"Outputs in: {out.resolve()}")
    print('Top 25 candidates (alphabetical ties and longer tokens first):')
    for row in rare[:25]:
        print(f"  {row['token']:18} Zipf={row['english_zipf']:4.2f} "
              f"embedded/million={row['embedded_occurrences_per_million_words']:8.3f} "
              f"hosts={row['frequent_host_examples']}")
    interesting = {'ville', 'berg', 'shire', 'html', 'php', 'java', 'api', 'und', 'que'}
    print('\nSelected tokens:')
    for row in sorted(universal_rows, key=lambda r: r['token']):
        if row['token'] in interesting:
            print(f"  {row['token']:8} English Zipf={row['english_zipf']:.2f}, "
                  f"embedded/million={row['embedded_occurrences_per_million_words']:.3f}, "
                  f"hosts={row['frequent_host_examples']}")
    print('\nUnobserved in reference (first 20):', ', '.join(r['token'] for r in unobserved[:20]))
    print('NOTE: Estimated rates come from a weighted word-frequency list, not a raw corpus.')
    print('Zero matches means absent from this reference, NOT proven rare in actual text.')


if __name__ == '__main__':
    main()
