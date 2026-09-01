"""Score exported retrieval results with RAGAS, in isolation.

This script is run by a *separate* interpreter - the .venv-eval virtualenv -
and deliberately imports nothing from medrag_mini. RAGAS pins older versions of
openai and langchain-core than the pipeline environment carries, so the two
never share a process. The interface between them is a JSON file.

Usage:
    .venv-eval/bin/python scripts/ragas_score.py input.json output.json

Input JSON:  [{"retrieved_contexts": [...], "reference_contexts": [...]}, ...]
Output JSON: {"metrics": {...}, "notes": [...]}
"""

import asyncio
import json
import sys


def main(input_path: str, output_path: str) -> int:
    samples = json.loads(open(input_path).read())
    notes: list[str] = []

    try:
        from ragas import SingleTurnSample
        from ragas.metrics import (
            NonLLMContextPrecisionWithReference,
            NonLLMContextRecall,
        )
    except Exception as error:
        json.dump(
            {"metrics": {}, "notes": [f"RAGAS import failed: {type(error).__name__}: {error}"]},
            open(output_path, "w"),
        )
        return 1

    precision_metric = NonLLMContextPrecisionWithReference()
    recall_metric = NonLLMContextRecall()

    async def score_one(sample_dict):
        sample = SingleTurnSample(
            retrieved_contexts=sample_dict["retrieved_contexts"],
            reference_contexts=sample_dict["reference_contexts"],
        )
        precision = await precision_metric.single_turn_ascore(sample)
        recall = await recall_metric.single_turn_ascore(sample)
        return float(precision), float(recall)

    async def score_all():
        results = []
        for sample_dict in samples:
            if not sample_dict["retrieved_contexts"]:
                # The pipeline abstained. That is a legitimate outcome, and it
                # scores zero here rather than being dropped from the average.
                results.append((0.0, 0.0))
                continue
            try:
                results.append(await score_one(sample_dict))
            except Exception as error:
                notes.append(f"RAGAS failed on one sample: {type(error).__name__}")
        return results

    scored = asyncio.run(score_all())

    metrics = {}
    if scored:
        metrics["ragas_context_precision"] = sum(p for p, _ in scored) / len(scored)
        metrics["ragas_context_recall"] = sum(r for _, r in scored) / len(scored)
        metrics["ragas_n_samples"] = len(scored)

    json.dump({"metrics": metrics, "notes": notes}, open(output_path, "w"))
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1], sys.argv[2]))
