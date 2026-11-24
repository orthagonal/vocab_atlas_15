#!/usr/bin/env python3
"""Count and compare the actual token inventories in 15 Hugging Face tokenizers.

Loads only tokenizer.json files; does not execute repository code or load weights.
"""
from __future__ import annotations
from collections import Counter

import argparse
import csv
import gzip
import hashlib
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

# The order is intentional: diverse historic tokenizers first, then newer LLMs.
# Distinct model repositories do not guarantee distinct token *sets*.
MODELS = [
    ("GPT-2", "openai-community/gpt2"),
    ("BERT uncased", "google-bert/bert-base-uncased"),
    ("RoBERTa", "FacebookAI/roberta-base"),
    ("T5", "google-t5/t5-small"),
    ("mT5", "Xenova/mt5-small"),
    ("XLM-R", "FacebookAI/xlm-roberta-base"),
    ("BLOOM", "bigscience/bloom-560m"),
    ("GPT-NeoX", "EleutherAI/gpt-neox-20b"),
    ("Llama 2 family", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"),
    ("Mistral 7B", "mistralai/Mistral-7B-v0.1"),
    ("Llama 3", "NousResearch/Meta-Llama-3-8B"),
    ("Qwen 2.5", "Qwen/Qwen2.5-0.5B"),
    ("Qwen 3", "Qwen/Qwen3-0.6B"),
    ("Gemma 2", "unsloth/gemma-2-2b"),
    ("Falcon 7B", "tiiuae/falcon-7b"),
]

categories = Counter()


def _bytes_decoder() -> dict[str, int]:
    # The byte-to-visible-Unicode alphabet used by GPT-2/ByteLevel BPE.
    bs = list(range(ord('!'), ord('~') + 1))
    bs += list(range(ord('¡'), ord('¬') + 1))
    bs += list(range(ord('®'), ord('ÿ') + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(cp): b for cp, b in zip(cs, bs)}


BYTE_MAP = _bytes_decoder()
BYTE_FALLBACK = re.compile(r'^<0x([0-9A-Fa-f]{2})>$')


def _has_component(node: object, target: str) -> bool:
    if isinstance(node, dict):
        if node.get('type') == target:
            return True
        return any(_has_component(value, target) for value in node.values())
    if isinstance(node, list):
        return any(_has_component(value, target) for value in node)
    return False


def detect_encoding(data: dict) -> str:
    """Identify the display alphabet used in the serialized tokenizer vocabulary."""
    decoder = data.get('decoder')
    pretokenizer = data.get('pre_tokenizer')
    if _has_component(decoder, 'ByteLevel') or _has_component(pretokenizer, 'ByteLevel'):
        return 'bytelevel'
    if _has_component(decoder, 'Metaspace') or _has_component(pretokenizer, 'Metaspace'):
        return 'metaspace'
    if data.get('model', {}).get('type') == 'WordPiece':
        return 'wordpiece'
    if data.get('model', {}).get('type') == 'Unigram':
        # SentencePiece vocabularies commonly represent whitespace as ▁.
        return 'sentencepiece'
    return 'literal'


def surface_bytes(token: str, mode: str, continuation_prefix: str = '##') -> bytes:
    """A *diagnostic* normalized byte representation, not contextual token semantics.

    Whole byte-level tokens can be incomplete UTF-8 sequences; bytes preserve
    that information, rather than lossy Unicode replacement-decoding.
    """
    m = BYTE_FALLBACK.fullmatch(token)
    if m and mode in ('sentencepiece', 'metaspace'):
        return bytes([int(m.group(1), 16)])
    if mode == 'bytelevel':
        out = bytearray()
        for char in token:
            if char in BYTE_MAP:
                out.append(BYTE_MAP[char])
            else:
                # Covers genuinely literal/custom-added Unicode characters.
                out.extend(char.encode('utf-8', 'surrogatepass'))
        return bytes(out)
    if mode in ('metaspace', 'sentencepiece'):
        return token.replace('▁', ' ').encode('utf-8', 'surrogatepass')
    if mode == 'wordpiece' and token.startswith(continuation_prefix):
        token = token[len(continuation_prefix):]
    return token.encode('utf-8', 'surrogatepass')


def infer_model_type(data: dict) -> str:
    """Read model.type, or infer a conservative label from tokenizer.json.

    A number of tokenizer exports omit the optional model.type field even
    though model.vocab (and, for BPE, model.merges) is present. Missing type
    metadata must never abort a vocabulary-only census.
    """
    model = data.get('model') or {}
    declared = model.get('type')
    if isinstance(declared, str) and declared:
        return declared
    vocab = model.get('vocab')
    if isinstance(vocab, list):
        return 'Unigram (inferred)'
    if isinstance(vocab, dict):
        if 'merges' in model:
            return 'BPE (inferred)'
        if 'continuing_subword_prefix' in model or 'max_input_chars_per_word' in model:
            return 'WordPiece (inferred)'
        return 'Unspecified mapping vocabulary'
    return 'Unknown'


def read_vocabulary(data: dict) -> tuple[dict[str, int], set[str]]:
    model = data.get('model') or {}
    entries = model.get('vocab')
    if isinstance(entries, dict):
        mapping = {str(token): int(token_id) for token, token_id in entries.items()}
    elif isinstance(entries, list):
        # Unigram entries are [piece, log_probability]; index is token ID.
        mapping = {str(entry[0]): i for i, entry in enumerate(entries)}
        if len(mapping) != len(entries):
            raise ValueError('Duplicate token string inside Unigram vocabulary')
    else:
        raise ValueError(f'Unknown model.vocab format: {type(entries).__name__}')

    special = set()
    for added in data.get('added_tokens') or []:
        token = added['content']
        token_id = int(added['id'])
        if added.get('special', False):
            special.add(token)
        if token not in mapping:
            mapping[token] = token_id
        # Some exports remap IDs in added-token metadata; retain the
        # existing base ID if the string is already in model.vocab.
    return mapping, special


def fingerprint_strings(tokens: set[str]) -> str:
    h = hashlib.sha256()
    for s in sorted(tokens):
        b = s.encode('utf-8', 'surrogatepass')
        h.update(len(b).to_bytes(4, 'big'))
        h.update(b)
    return h.hexdigest()


def fingerprint_bytes(tokens: set[bytes]) -> str:
    h = hashlib.sha256()
    for b in sorted(tokens):
        h.update(len(b).to_bytes(4, 'big'))
        h.update(b)
    return h.hexdigest()


def fingerprint_mapping(mapping: dict[str, int]) -> str:
    h = hashlib.sha256()
    for token, token_id in sorted(mapping.items()):
        b = token.encode('utf-8', 'surrogatepass')
        h.update(len(b).to_bytes(4, 'big'))
        h.update(b)
        h.update(int(token_id).to_bytes(8, 'big', signed=True))
    return h.hexdigest()


@dataclass
class Vocab:
    name: str
    repo: str
    sha: str
    encoding: str
    model_type: str
    base_size: int
    full_size: int
    special_size: int
    raw: set[str]
    surface: set[bytes]
    raw_hash: str
    mapped_hash: str
    surface_hash: str
    surface_collisions: int


def load_vocab(name: str, repo: str, local_dir: Path | None) -> Vocab:
    if local_dir is not None:
        path = local_dir / (repo.replace('/', '__') + '.json')
        if not path.exists():
            raise FileNotFoundError(f'Local tokenizer not found: {path}')
        revision = 'local-file'
    else:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise RuntimeError('Install dependencies: pip install -r requirements.txt') from exc
        path = Path(hf_hub_download(repo_id=repo, filename='tokenizer.json', repo_type='model'))
        revision = path.parent.name  # Hugging Face snapshot cache directory SHA
    data = json.loads(path.read_text(encoding='utf-8'))
    mapping, special = read_vocabulary(data)
    mode = detect_encoding(data)
    continuation = data.get('model', {}).get('continuing_subword_prefix') or '##'
    lexical = set(mapping) - special
    decoded = {surface_bytes(token, mode, continuation) for token in lexical}
    return Vocab(
        name=name, repo=repo, sha=revision, encoding=mode,
        model_type=infer_model_type(data),
        base_size=len(data['model']['vocab']), full_size=len(mapping),
        special_size=len(special), raw=lexical, surface=decoded,
        raw_hash=fingerprint_strings(lexical),
        mapped_hash=fingerprint_mapping(mapping),
        surface_hash=fingerprint_bytes(decoded),
        surface_collisions=len(lexical) - len(decoded),
    )


def write_csv(path: Path, rows: list[dict], headers: list[str] | None = None) -> None:
    if headers is None:
        headers = list(rows[0]) if rows else []
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def pct(x: float) -> str:
    return f'{100*x:.2f}%'


def graph(vocabs: list[Vocab], stats: list[dict], pairs: list[dict], raw_counter: Counter, out: Path) -> bool:
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print('Plot dependencies unavailable; CSV and Markdown outputs still generated.', file=sys.stderr)
        return False

    x = np.arange(1, len(vocabs) + 1)
    fig, ax = plt.subplots(figsize=(10.6, 5.2))
    ax.plot(x, [r['cumulative_raw'] for r in stats], marker='o', label='Exact raw token strings')
    ax.plot(x, [r['cumulative_surface'] for r in stats], marker='s', label='Normalized byte surfaces')
    ax.set_xticks(x, [v.name for v in vocabs], rotation=60, ha='right', fontsize=8)
    ax.set_ylabel('Distinct tokens in union')
    ax.set_title('Vocabulary accumulation (selected order, not a statistical estimate)')
    ax.ticklabel_format(axis='y', style='plain')
    ax.grid(alpha=.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out/'accumulation.png', dpi=170)
    plt.close(fig)

    n = len(vocabs)
    mat = np.eye(n)
    idx = {v.repo: i for i,v in enumerate(vocabs)}
    for row in pairs:
        i, j = idx[row['repo_a']], idx[row['repo_b']]
        mat[i,j] = mat[j,i] = row['surface_jaccard']
    fig, ax = plt.subplots(figsize=(10, 8.6))
    im = ax.imshow(mat, vmin=0, vmax=1, cmap='viridis')
    ax.set_xticks(range(n), [v.name for v in vocabs], rotation=70, ha='right', fontsize=8)
    ax.set_yticks(range(n), [v.name for v in vocabs], fontsize=8)
    fig.colorbar(im, label='Normalized-byte Jaccard', ax=ax)
    ax.set_title('Overlap of token byte surfaces (not contextual meaning)')
    fig.tight_layout()
    fig.savefig(out/'overlap_heatmap.png', dpi=170)
    plt.close(fig)

    freq_counts = Counter(raw_counter.values())
    fig, ax = plt.subplots(figsize=(10.5, 4.9))
    ks = range(1, n+1)
    ax.bar(list(ks), [freq_counts.get(k, 0) for k in ks])
    ax.set_yscale('log')
    ax.set_xlabel('Number of vocabularies containing a raw token string')
    ax.set_ylabel('Number of distinct strings (log scale)')
    ax.set_xticks(list(ks))
    ax.set_title('Distribution of vocabulary prevalence')
    fig.tight_layout()
    fig.savefig(out/'prevalence.png', dpi=170)
    plt.close(fig)
    return True


def analyze(vocabs: list[Vocab], out: Path, make_plots=True, write_full_index=True) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    n = len(vocabs)
    raw_counter = Counter(token for v in vocabs for token in v.raw)
    raw_union, surface_union = set(), set()
    stats = []
    for v in vocabs:
        added_raw = len(v.raw - raw_union)
        added_surface = len(v.surface - surface_union)
        raw_union.update(v.raw)
        surface_union.update(v.surface)
        stats.append({
            'name':v.name,'repo':v.repo,'revision':v.sha,'model_type':v.model_type,
            'encoding':v.encoding,'base_vocab_size':v.base_size,'all_vocab_size':v.full_size,
            'explicit_special_token_count':v.special_size,
            'lexical_raw_token_count':len(v.raw),
            'normalized_byte_surface_count':len(v.surface),
            'normalization_collisions':v.surface_collisions,
            'globally_private_raw_tokens':sum(raw_counter[t] == 1 for t in v.raw),
            'new_raw_tokens_in_order':added_raw,'new_byte_surfaces_in_order':added_surface,
            'cumulative_raw':len(raw_union),'cumulative_surface':len(surface_union),
            'raw_set_sha256':v.raw_hash,'mapping_sha256':v.mapped_hash,
            'surface_set_sha256':v.surface_hash,
        })

    pairs = []
    for a, b in combinations(vocabs, 2):
        raw_shared = len(a.raw & b.raw)
        surf_shared = len(a.surface & b.surface)
        pairs.append({
            'a':a.name,'b':b.name,'repo_a':a.repo,'repo_b':b.repo,
            'raw_shared':raw_shared,
            'raw_jaccard':raw_shared/(len(a.raw)+len(b.raw)-raw_shared),
            'raw_overlap_smaller':raw_shared/min(len(a.raw),len(b.raw)),
            'surface_shared':surf_shared,
            'surface_jaccard':surf_shared/(len(a.surface)+len(b.surface)-surf_shared),
            'surface_overlap_smaller':surf_shared/min(len(a.surface),len(b.surface)),
            'same_raw_set':a.raw_hash == b.raw_hash,
            'same_token_to_id_mapping':a.mapped_hash == b.mapped_hash,
        })

    write_csv(out/'vocabulary_stats.csv',stats)
    write_csv(out/'pairwise_overlap.csv',pairs)
    top = sorted(raw_counter.items(), key=lambda kv:(-kv[1],kv[0]))[:500]
    top_rows = [{'raw_token':t, 'vocabulary_count':f,
                 'vocabularies':'; '.join(v.name for v in vocabs if t in v.raw)} for t,f in top]
    write_csv(out/'top_shared_tokens.csv',top_rows)
    # Export every token reused across 2+ vocabularies
    reused = sorted(
        ((token, count) for token, count in raw_counter.items() if count >= 2),
        key=lambda x: (-x[1], x[0])
    )

    with (out / 'reused_tokens.csv').open(
        'w', encoding='utf-8-sig', newline=''
    ) as f:
        writer = csv.writer(f)
        writer.writerow(['token', 'vocabulary_count', 'vocabularies'])

        for token, count in reused:
            members = [v.name for v in vocabs if token in v.raw]
            writer.writerow([token, count, '; '.join(members)])

    # Export tokens shared by ALL vocabularies
    shared_all = sorted(
        token for token, count in raw_counter.items()
        if count == n
    )

    (out / 'shared_by_all.txt').write_text(
        '\n'.join(repr(t) for t in shared_all) + '\n',
        encoding='utf-8'
    )

    # Show a sample in the console
    print(f'\n{len(reused):,} tokens reused by 2+ vocabularies')
    print(f'{len(shared_all):,} tokens shared by all {n} vocabularies')

    print('\nuniversally shared tokens:')
    for token in shared_all:
        print(repr(token))


        for token in shared_all:
            if len(token) == 1:
                category = "Single character"
            elif token.isascii() and token.isdigit():
                category = "Numbers"
            elif token.isascii() and token.isalpha():
                if len(token) <= 3:
                    category = "Short alphabetic (2-3)"
                else:
                    category = "Alphabetic (4+)"
            elif token.isascii() and not any(c.isalnum() for c in token):
                category = "Punctuation / symbols"
            elif token.isalpha():
                category = "Non-ASCII alphabetic"
            else:
                category = "Mixed / other"

            categories[category] += 1

        print("\nUNIVERSAL TOKEN CATEGORIES")
        for category, count in categories.most_common():
            print(f"{category:30} {count:5}")

        # Isolate longer alphabetic tokens for investigation
        words = sorted(
            t for t in shared_all
            if t.isascii()
            and t.isalpha()
            and t.islower()
            and len(t) >= 2
        )

        print(f"\n{len(words)} lowercase alphabetic tokens (2+ chars)")

        (out / "universal_words.txt").write_text(
            "\n".join(words) + "\n",
            encoding="utf-8"
        )
    if write_full_index:
        with gzip.open(out/'all_token_prevalence.csv.gz', mode='wt', encoding='utf-8',newline='') as f:
            w = csv.writer(f)
            w.writerow(['raw_token','vocabulary_count','vocabulary_indices_1_based'])
            for token in sorted(raw_counter):
                w.writerow([token,raw_counter[token],';'.join(str(i+1) for i,v in enumerate(vocabs) if token in v.raw)])

    hist = Counter(raw_counter.values())
    summary = {
        'generated_utc':datetime.now(timezone.utc).isoformat(),
        'model_count':n,
        'raw_token_union':len(raw_union),
        'normalized_byte_surface_union':len(surface_union),
        'total_raw_token_occurrences':sum(len(v.raw) for v in vocabs),
        'raw_tokens_found_in_exactly_one_vocabulary':hist[1],
        'raw_tokens_found_in_every_vocabulary':hist[n],
        'raw_token_prevalence_histogram':{str(k):hist[k] for k in range(1,n+1)},
        'unique_raw_vocabulary_sets':len({v.raw_hash for v in vocabs}),
        'unique_token_to_id_mappings':len({v.mapped_hash for v in vocabs}),
        'unique_normalized_byte_surface_sets':len({v.surface_hash for v in vocabs}),
        'model_repos_and_revisions':{v.repo:v.sha for v in vocabs},
        'notes':[
            'Raw token equality compares exact strings in tokenizer.json, excluding entries explicitly flagged special.',
            'Byte-surface equality strips ByteLevel/Metaspace/WordPiece display conventions; it is heuristic across tokenizer families.',
            'A byte surface can occur in multiple tokens within one tokenizer; these are counted once per vocabulary.',
            'Tokenizer normalization and contextual encoding behavior are not comparable by vocabulary alone.',
            'Novelty counts depend on the chosen order and cannot be interpreted as historical ancestry.',
        ],
    }
    (out/'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    (out/'manifest.json').write_text(json.dumps(
        [{'name':v.name,'repo':v.repo,'revision':v.sha,'model_type':v.model_type,
          'encoding':v.encoding, 'tokenizer_json_url':f'https://huggingface.co/{v.repo}/blob/{v.sha}/tokenizer.json'}
          for v in vocabs],indent=2,ensure_ascii=False),encoding='utf-8')

    plotted = graph(vocabs,stats,pairs,raw_counter,out) if make_plots else False
    best = sorted(pairs,key=lambda p:-p['surface_jaccard'])[:8]
    md = [
        '# Vocabulary Atlas: 15 Hugging Face tokenizers',
        '',
        f'Generated: {summary["generated_utc"]}',
        '',
        '## Main results',
        '',
        f'- **{n}** model tokenizer inventories',
        f'- **{summary["raw_token_union"]:,}** distinct raw lexical token strings',
        f'- **{summary["normalized_byte_surface_union"]:,}** distinct normalized byte surfaces',
        f'- **{summary["total_raw_token_occurrences"]:,}** raw token occurrences across vocabularies',
        f'- **{summary["raw_tokens_found_in_exactly_one_vocabulary"]:,}** raw strings appear in exactly one vocabulary',
        f'- **{summary["raw_tokens_found_in_every_vocabulary"]:,}** raw strings occur in all {n} vocabularies',
        f'- **{summary["unique_raw_vocabulary_sets"]}** distinct exact token-string sets',
        f'- **{summary["unique_token_to_id_mappings"]}** distinct string-to-ID mappings',
        '', '## Vocabulary inventory', '',
        '| Model | Lexical tokens | New raw strings | Cumulative raw union | Unique to this vocabulary |',
        '|---|---:|---:|---:|---:|',
    ]
    for r in stats:
        md.append('| {} | {:,} | {:,} | {:,} | {:,} |'.format(
            r['name'],r['lexical_raw_token_count'],r['new_raw_tokens_in_order'],
            r['cumulative_raw'],r['globally_private_raw_tokens']))
    md.extend(['', '## Closest pairs (normalized byte-surface Jaccard)', '',
               '| A | B | Shared byte surfaces | Jaccard |', '|---|---|---:|---:|'])
    for p in best:
        md.append(f"| {p['a']} | {p['b']} | {p['surface_shared']:,} | {pct(p['surface_jaccard'])} |")
    if plotted:
        md.extend(['', '## Plots', '',
                   '![Vocabulary accumulation](accumulation.png)', '',
                   '![Overlap matrix](overlap_heatmap.png)', '',
                   '![Token prevalence](prevalence.png)'])
    md.extend(['', '## Interpretation', '',
        'The raw-string union is a literal count, not a count of unique words. '
        'The normalized-byte union is a comparison aid, not a proof of equivalent tokenization. '
        'For example, WordPiece `##word` and `word` have different contextual roles, '
        'and byte-level tokens can contain incomplete UTF-8 sequences.', '',
        'Files: `manifest.json`, `vocabulary_stats.csv`, `pairwise_overlap.csv`, '
        '`top_shared_tokens.csv`, `all_token_prevalence.csv.gz`, `summary.json`.', ''])
    (out/'REPORT.md').write_text('\n'.join(md), encoding='utf-8')
    return summary


def main() -> int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, default=Path('results'), help='Output directory')
    p.add_argument('--local-dir', type=Path, help='Offline input: <org>__<model>.json files')
    p.add_argument('--skip-plots', action='store_true',help='Do not produce PNG charts')
    p.add_argument('--skip-full-index',action='store_true',help='Skip large compressed prevalence CSV')
    p.add_argument('--model', action='append', default=[],
        help='Override default 15; format Display Name=org/repo (repeat for each model)')
    args=p.parse_args()
    selected=[]
    if args.model:
        for value in args.model:
            if '=' not in value:
                p.error('--model must be Display Name=org/repo')
            name,repo=value.split('=',1)
            selected.append((name,repo))
    else:
        selected=MODELS
    found=[]
    for i,(name,repo) in enumerate(selected,1):
        print(f'[{i}/{len(selected)}] Reading {name} ({repo})...',flush=True)
        try:
            voc=load_vocab(name,repo,args.local_dir)
        except Exception as exc:
            print(f'FAILED fetching/loading {repo}: {type(exc).__name__}: {exc}',file=sys.stderr)
            print('Stopping rather than fabricating a partial 15-tokenizer result.',file=sys.stderr)
            return 1
        found.append(voc)
        print(f'    {len(voc.raw):,} lexical tokens; {len(voc.surface):,} normalized surfaces; {voc.encoding}',flush=True)
    result=analyze(found,args.out,make_plots=not args.skip_plots,
                   write_full_index=not args.skip_full_index)
    print('\nRESULTS')
    print(f'  Raw lexical-token union:        {result["raw_token_union"]:,}')
    print(f'  Normalized byte-surface union:  {result["normalized_byte_surface_union"]:,}')
    print(f'  In exactly one vocabulary:      {result["raw_tokens_found_in_exactly_one_vocabulary"]:,}')
    print(f'  In every vocabulary:           {result["raw_tokens_found_in_every_vocabulary"]:,}')
    print(f'  Full report: {args.out.resolve() / "REPORT.md"}')
    return 0


if __name__=='__main__':
    sys.exit(main())
