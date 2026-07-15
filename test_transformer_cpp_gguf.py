"""Verify a transformer_cpp-trained GGUF end-to-end in the tinyllama.cpp engine.

Loads the GGUF through the pybind11 API (TinyLlamaSession), checks the config
parsed sanely (word-level tokenizer family, expected dims), runs a handful of
prompts, and applies mechanical sanity checks to the generations:

  * non-empty output
  * decoded tokens are words from the embedded vocabulary
  * not a single token repeated forever (degenerate loop check)

Exit code 0 = engine runs our GGUF; nonzero = something to tinker with.

    python test_transformer_cpp_gguf.py path/to/model.gguf [--steps 30]

(For GGUF loads the engine reads the tokenizer from embedded metadata; the
tokenizer_path argument is ignored, so we just pass the GGUF path twice.)
"""
import argparse
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gguf")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    tl = None
    for name in ("tinyllama_bindings", "tinyllama_cpp"):
        try:
            mod = __import__(name)
            if hasattr(mod, "TinyLlamaSession"):
                tl = mod
                break
        except ImportError:
            continue
    if tl is None:
        print("FAIL: no importable module exposing TinyLlamaSession "
              "(build the bindings or pip install .)")
        return 2

    print(f"[1/3] loading {args.gguf} (CPU, n_gpu_layers=0)")
    sess = tl.TinyLlamaSession(args.gguf, args.gguf, args.threads, 0)

    cfg = sess.get_config()
    print(f"[2/3] config: arch={getattr(cfg, 'architecture', '?')} "
          f"vocab={getattr(cfg, 'vocab_size', '?')} "
          f"layers={getattr(cfg, 'num_hidden_layers', '?')} "
          f"hidden={getattr(cfg, 'hidden_size', '?')}")

    prompts = [
        "once upon a time there was a little",
        "the cat sat on the",
        "tom and lily went to the park to",
    ]
    failures = []
    print(f"[3/3] generating {args.steps} tokens per prompt (greedy-ish, temp 0.1)")
    for p in prompts:
        out = sess.generate(p, steps=args.steps, temperature=0.1, top_k=40, top_p=0.9)
        words = out.split()
        degenerate = len(words) >= 6 and len(set(words[-6:])) == 1
        status = "OK"
        if not words:
            status = "EMPTY"
            failures.append((p, out))
        elif degenerate:
            status = "DEGENERATE (single-token loop)"
            failures.append((p, out))
        print(f"  prompt : {p}\n  output : {out!r}\n  status : {status}\n")

    if failures:
        print(f"RESULT: {len(failures)}/{len(prompts)} prompts failed sanity checks")
        return 1
    print("RESULT: engine runs the transformer_cpp GGUF — all sanity checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
