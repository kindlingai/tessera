# /// script
# requires-python = ">=3.11"
# dependencies = ["torch", "transformers", "huggingface_hub", "numpy"]
# ///
"""Writes the logits reference that a model's golden test reads.

usage: uv run tools/parity/reference.py <model> <name> [--case <case>=<text>]... [--out <dir>]

<model> is a Hugging Face repository, `owner/name` or `owner/name@revision`,
or a local directory. It must hold one `model.safetensors`.
<name> is the `name` of the model's `Spec` in tools/parity/tests.

The files go to ~/.cache/tessera/references/<name>, or to <dir>/<name>:

    model.safetensors          a link to the checkpoint
    cases/<case>/text.txt      the text
    cases/<case>/tokens.txt    the token ids of one sequence, on one line
    cases/<case>/logits.f32    little-endian f32, [tokens, vocab]

Transformers computes the logits in float32. The tokenizer adds its default
special tokens and no chat template.
"""

import argparse
import os
import sys

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = "~/.cache/tessera/references"
TEXT = "The quick brown fox jumps over the lazy dog. The capital of France is"
FILES = ["*.json", "*.safetensors", "*.txt", "*.model"]


def arguments():
    p = argparse.ArgumentParser(usage=__doc__)
    p.add_argument("model")
    p.add_argument("name")
    p.add_argument("--case", action="append", default=[])
    p.add_argument("--out", default=ROOT)
    return p.parse_args()


def cases(given):
    """Each `<case>=<text>` as a pair, or the default case."""
    out = []
    for entry in given:
        case, sep, text = entry.partition("=")
        if not (case and sep and text):
            sys.exit(f"--case needs <case>=<text>, not {entry!r}")
        out.append((case, text))
    return out or [("prompt", TEXT)]


def checkpoint(model):
    """The directory that holds `model`'s files."""
    if os.path.isdir(model):
        return model
    repo, _, revision = model.partition("@")
    return snapshot_download(repo, revision=revision or None, allow_patterns=FILES)


def link(source, target):
    """Points `target` at the file behind `source`."""
    if os.path.lexists(target):
        os.remove(target)
    os.symlink(os.path.realpath(source), target)


def main():
    args = arguments()
    wanted = cases(args.case)
    source = checkpoint(args.model)
    weights = os.path.join(source, "model.safetensors")
    if not os.path.isfile(weights):
        sys.exit(f"{source} has no model.safetensors. The golden test reads one file.")

    tokenizer = AutoTokenizer.from_pretrained(source)
    model = AutoModelForCausalLM.from_pretrained(source, dtype=torch.float32).eval()

    out = os.path.join(os.path.expanduser(args.out), args.name)
    os.makedirs(out, exist_ok=True)
    link(weights, os.path.join(out, "model.safetensors"))
    for case, text in wanted:
        ids = tokenizer(text, return_tensors="pt")["input_ids"]
        with torch.no_grad():
            logits = model(input_ids=ids, use_cache=False).logits[0]
        folder = os.path.join(out, "cases", case)
        os.makedirs(folder, exist_ok=True)
        with open(os.path.join(folder, "text.txt"), "w") as f:
            f.write(text + "\n")
        with open(os.path.join(folder, "tokens.txt"), "w") as f:
            f.write(" ".join(str(i) for i in ids[0].tolist()) + "\n")
        logits.float().contiguous().numpy().astype("<f4").tofile(os.path.join(folder, "logits.f32"))
        print(f"{case}: {ids.shape[1]} tokens, logits {tuple(logits.shape)}")
    print(out)


if __name__ == "__main__":
    main()
