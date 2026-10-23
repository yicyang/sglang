"""GSM8K accuracy (greedy) for A/B-checking numerics of serving variants.

--mode chat (default): zero-shot via /v1/chat/completions; this model is a
  chat/thinking model and, on raw few-shot completion prompts, greedily emits
  <|im_end|> first for most questions. The answer is the number after the last
  '####', else the last number of the reply.
--mode fewshot: 5-shot raw completion via /generate (sglang few_shot_gsm8k format).
The first 5 test items are never scored. Every prediction is written out.
"""
import argparse
import concurrent.futures as cf
import json
import re

import requests

INVALID = -9999999


def answer_value(text):
    text = text.replace(",", "")
    nums = re.findall(r"-?\d+\.?\d*", text)
    if not nums:
        return INVALID
    try:
        return float(nums[-1])
    except ValueError:
        return INVALID


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--num-questions", type=int, default=1319)
    ap.add_argument("--shots", type=int, default=5)
    ap.add_argument("--parallel", type=int, default=64)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["chat", "fewshot"], default="chat")
    ap.add_argument("--max-tokens", type=int, default=2048)
    args = ap.parse_args()
    lines = [json.loads(l) for l in open(args.data)]
    shots = "".join(f"Question: {l['question']}\nAnswer: {l['answer']}\n\n" for l in lines[: args.shots])
    qs = lines[args.shots : args.shots + args.num_questions]  # shots are excluded from scoring

    def ask(item):
        if args.mode == "chat":
            r = requests.post(f"http://localhost:{args.port}/v1/chat/completions", json={
                "model": "default",
                "messages": [{"role": "user", "content": item["question"] +
                              "\nSolve it step by step and give the final numeric answer after '####'."}],
                "max_tokens": args.max_tokens, "temperature": 0,
            }, timeout=1200)
            r.raise_for_status()
            text = r.json()["choices"][0]["message"]["content"] or ""
            return text.split("####")[-1] if "####" in text else text
        prompt = shots + f"Question: {item['question']}\nAnswer:"
        r = requests.post(f"http://localhost:{args.port}/generate", json={
            "text": prompt,
            "sampling_params": {"max_new_tokens": 512, "temperature": 0, "stop": ["Question:", "</s>", "<|im_end|>"]},
        }, timeout=600)
        r.raise_for_status()
        return r.json()["text"]

    with cf.ThreadPoolExecutor(args.parallel) as ex:
        outs = list(ex.map(ask, qs))
    preds = [answer_value(o) for o in outs]
    labels = [answer_value(l["answer"].split("####")[-1]) for l in qs]
    acc = sum(p == l for p, l in zip(preds, labels)) / len(qs)
    inv = sum(p == INVALID for p in preds) / len(qs)
    print(f"Accuracy: {acc:.4f}\nInvalid: {inv:.4f}\nN: {len(qs)}")
    with open(args.out, "w") as f:
        for q, o, p, l in zip(qs, outs, preds, labels):
            f.write(json.dumps(dict(question=q["question"], output=o, pred=p, label=l)) + "\n")


if __name__ == "__main__":
    main()
