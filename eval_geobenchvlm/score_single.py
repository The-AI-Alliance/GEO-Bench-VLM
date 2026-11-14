#!/usr/bin/env python3
"""Score GEOBench-VLM Single predictions.

Usage:
    python score_single.py --data_path /datasets/GEOBench-VLM \
        --predictions /datasets/GEOBench-VLM/Results-qwen2-countcls/evaluation_results_GEOBench-VLM.json
"""

import argparse
import json
import os
import random
import re
from collections import Counter, defaultdict

OPTIONS = "ABCDE"
N_PROMPTS = 5
INVALID = "INVALID"


def extract_first_english_character(text):
    match = re.search(r"[a-zA-Z]", str(text))
    return match.group(0).upper() if match else ""


def parse_answer(answer):
    if isinstance(answer, list):
        answer = answer[0] if answer else ""
    char = extract_first_english_character(answer)
    return char if char in OPTIONS else INVALID


def majority_vote(answers, rng):
    counts = Counter(answers)
    top = max(counts.values())
    return rng.choice(sorted(a for a, c in counts.items() if c == top))


def score(questions, predictions, seed=0):
    rng = random.Random(seed)
    preds = {}
    for rec in predictions:
        qid = rec["question_id"]
        qid = int(qid[0] if isinstance(qid, list) else qid)
        preds[qid] = rec.get("predicted_answers", [])

    correct = defaultdict(int)
    total = defaultdict(int)
    for q in questions:
        answers = [parse_answer(a) for a in preds.get(int(q["question_id"]), [])[:N_PROMPTS]]
        answers += [INVALID] * (N_PROMPTS - len(answers))
        gt = extract_first_english_character(q["ground_truth_option"])
        correct[q["task"]] += int(majority_vote(answers, rng) == gt)
        total[q["task"]] += 1

    accuracy = {t: correct[t] / total[t] for t in sorted(total)}
    return accuracy, sum(accuracy.values()) / len(accuracy)


def main():
    parser = argparse.ArgumentParser(description="Score GEOBench-VLM Single predictions.")
    parser.add_argument("--data_path", default="/datasets/GEOBench-VLM", help="dataset root containing Single/qa.json")
    parser.add_argument("--predictions", required=True, help="evaluation_results_*.json written by a model script")
    parser.add_argument("--output", help="where to save the scores (default: scores.json next to --predictions)")
    parser.add_argument("--seed", type=int, default=0, help="seed for breaking ties in the majority vote")
    args = parser.parse_args()

    with open(os.path.join(args.data_path, "Single", "qa.json")) as f:
        questions = json.load(f)
    with open(args.predictions) as f:
        predictions = json.load(f)

    accuracy, average = score(questions, predictions, args.seed)

    print(f"{'Task':40s} {'Accuracy':>8s}")
    print("-" * 49)
    for task, acc in accuracy.items():
        print(f"{task:40s} {100 * acc:8.2f}")
    print("-" * 49)
    print(f"{'Average':40s} {100 * average:8.2f}")

    output = args.output or os.path.join(os.path.dirname(os.path.abspath(args.predictions)), "scores.json")
    results = {
        "predictions": args.predictions,
        "seed": args.seed,
        "average": round(100 * average, 2),
        "tasks": {task: round(100 * acc, 2) for task, acc in accuracy.items()},
    }
    with open(output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nScores saved to {output}")


if __name__ == "__main__":
    main()
