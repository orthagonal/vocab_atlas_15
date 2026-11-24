"""Inspect next-token uncertainty during greedy Qwen generation.

Usage:
    pip install torch transformers matplotlib
    python qwen_token_entropy.py --prompt "Invent the name of a strange alien artifact."

Writes <output-prefix>.csv and <output-prefix>.png.
"""

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def printable_token(tokenizer, token_id: int) -> str:
    """Make whitespace and unusual token pieces recognizable in logs/plots."""
    decoded = tokenizer.decode(
        [token_id],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    if not decoded or "\ufffd" in decoded:
        return repr(tokenizer.convert_ids_to_tokens(token_id))
    return repr(decoded)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument(
        "--prompt",
        default="Invent a strange object discovered in an abandoned spaceship. Describe it in four sentences.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--output-prefix", default="token_entropy")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"Loading {args.model} on {device} ({dtype})")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(device)
    model.eval()

    messages = [{"role": "user", "content": args.prompt}]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(device)
    prompt_length = inputs["input_ids"].shape[-1]

    # Hugging Face returns a tuple of raw, *unprocessed* logits, one per
    # generation step. Greedy decoding selects the next token at each step.
    with torch.inference_mode():
        result = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            num_beams=1,
            return_dict_in_generate=True,
            output_logits=True,
        )

    generated_ids = result.sequences[0, prompt_length:].tolist()
    answer = tokenizer.decode(generated_ids, skip_special_tokens=True)
    print("\nGENERATED RESPONSE:\n" + answer + "\n")

    rows = []
    print(f"{'Step':>4}  {'Selected token':<22} {'P(chosen)':>9} {'Entropy':>9} {'Eff. choices':>12}")
    print("-" * 68)

    for step, (chosen_id, step_logits) in enumerate(zip(generated_ids, result.logits), start=1):
        # Scores shape: (batch=1, vocabulary_size). Calculate in float32.
        scores = step_logits[0].float()
        log_probs = torch.log_softmax(scores, dim=-1)
        probs = log_probs.exp()
        entropy = -(probs * log_probs).sum().item()  # Natural-log units: nats
        effective_choices = math.exp(entropy)
        top_probs, top_ids = torch.topk(probs, k=5)
        chosen_prob = probs[chosen_id].item()
        most_likely_prob = top_probs[0].item()
        token_label = printable_token(tokenizer, chosen_id)
        top_five = "; ".join(
            f"{printable_token(tokenizer, tid)}={pr:.3f}"
            for tid, pr in zip(top_ids.tolist(), top_probs.tolist())
        )
        rows.append({
            "step": step,
            "token_id": chosen_id,
            "token": token_label,
            "chosen_probability": chosen_prob,
            "top_probability": most_likely_prob,
            "entropy_nats": entropy,
            "effective_choices": effective_choices,
            "top_five": top_five,
        })
        print(f"{step:4d}  {token_label:<22.22} {chosen_prob:9.3f} {entropy:9.3f} {effective_choices:12.1f}")

    if not rows:
        print("No tokens were generated.")
        return

    prefix = Path(args.output_prefix)
    csv_path = prefix.with_suffix(".csv")
    png_path = prefix.with_suffix(".png")
    with csv_path.open("w", newline="", encoding="utf-8-sig") as fp:
        writer = csv.DictWriter(fp, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    steps = [row["step"] for row in rows]
    entropies = [row["entropy_nats"] for row in rows]
    confidence = [row["top_probability"] for row in rows]
    token_labels = [row["token"] for row in rows]

    fig, ax = plt.subplots(figsize=(max(12, len(rows) * 0.31), 6))
    ax.plot(steps, entropies, linewidth=1.5, alpha=0.65)
    markers = ax.scatter(steps, entropies, c=confidence, vmin=0, vmax=1, s=40, zorder=3)
    colorbar = fig.colorbar(markers, ax=ax, pad=0.012)
    colorbar.set_label("Highest next-token probability (0–1)")
    ax.set_xticks(steps)
    ax.set_xticklabels(token_labels, rotation=80, fontsize=8)
    ax.set_xlabel("Generated token (as chosen by greedy decoding)")
    ax.set_ylabel("Next-token entropy (nats)")
    ax.set_title("Qwen next-token uncertainty by generation step")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(png_path, dpi=170, bbox_inches="tight")
    print(f"\nSaved {csv_path} and {png_path}")
    print("\nMost uncertain steps:")
    for row in sorted(rows, key=lambda row: row["entropy_nats"], reverse=True)[:5]:
        print(
            f"  Step {row['step']:>3} {row['token']}: "
            f"H={row['entropy_nats']:.3f}, "
            f"effective choices={row['effective_choices']:.1f}; "
            f"top 5: {row['top_five']}"
        )
    if not args.no_show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
