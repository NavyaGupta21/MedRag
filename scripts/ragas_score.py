"""
Score RAG pipeline results using RAGAS.

This script runs in a separate virtual environment (.venv-eval)
to avoid dependency conflicts with the main MedRAG pipeline.

Usage:
    .venv-eval/bin/python scripts/ragas_score.py input.json output.json

Input JSON format:

[
    {
        "user_input": "What are the symptoms of diabetes?",
        "response": "Common symptoms include increased thirst and frequent urination.",
        "retrieved_contexts": [
            "Diabetes can cause increased thirst and frequent urination."
        ],
        "reference_contexts": [
            "Common symptoms of diabetes include increased thirst,
             frequent urination and increased hunger."
        ],
        "reference": "Common symptoms include increased thirst,
                      frequent urination and increased hunger."
    }
]

Output JSON format:

{
    "metrics": {
        "ragas_context_precision": ...,
        "ragas_context_recall": ...,
        "ragas_faithfulness": ...,
        "ragas_answer_relevancy": ...,
        "ragas_answer_correctness": ...,
        "ragas_answer_similarity": ...,
        "ragas_n_samples": ...
    },
    "notes": [...]
}
"""

import asyncio
import json
import sys

from langchain_ollama import ChatOllama, OllamaEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper


def main(input_path: str, output_path: str) -> int:
    with open(input_path, "r", encoding="utf-8") as file:
        samples = json.load(file)

    notes = []

    try:
        from ragas import SingleTurnSample
        from ragas.metrics import (
            NonLLMContextPrecisionWithReference,
            NonLLMContextRecall,
            Faithfulness,
            AnswerRelevancy,
            AnswerCorrectness,
            AnswerSimilarity,
        )
    except Exception as error:
        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(
                {
                    "metrics": {},
                    "notes": [
                        f"RAGAS import failed: "
                        f"{type(error).__name__}: {error}"
                    ],
                },
                file,
                indent=4,
            )
        return 1

    ollama_llm = ChatOllama(
        model="llama3.1:8b",
        temperature=0,
    )
    llm = LangchainLLMWrapper(ollama_llm)

    ollama_embeddings = OllamaEmbeddings(
        model="nomic-embed-text",
    )
    embeddings = LangchainEmbeddingsWrapper(ollama_embeddings)

    context_precision_metric = NonLLMContextPrecisionWithReference()
    context_recall_metric = NonLLMContextRecall()
    faithfulness_metric = Faithfulness(llm=llm)
    answer_relevancy_metric = AnswerRelevancy(llm=llm, embeddings=embeddings)
    answer_similarity_metric = AnswerSimilarity(embeddings=embeddings)
    answer_correctness_metric = AnswerCorrectness(
        llm=llm,
        embeddings=embeddings,
        answer_similarity=answer_similarity_metric,
    )

    async def score_one(sample_dict):
        sample = SingleTurnSample(
            user_input=sample_dict["user_input"],
            response=sample_dict["response"],
            retrieved_contexts=sample_dict["retrieved_contexts"],
            reference_contexts=sample_dict["reference_contexts"],
            reference=sample_dict["reference"],
        )

        context_precision = await context_precision_metric.single_turn_ascore(sample)
        context_recall = await context_recall_metric.single_turn_ascore(sample)
        faithfulness = await faithfulness_metric.single_turn_ascore(sample)
        answer_relevancy = await answer_relevancy_metric.single_turn_ascore(sample)
        answer_correctness = await answer_correctness_metric.single_turn_ascore(sample)
        answer_similarity = await answer_similarity_metric.single_turn_ascore(sample)

        return {
            "context_precision": float(context_precision),
            "context_recall": float(context_recall),
            "faithfulness": float(faithfulness),
            "answer_relevancy": float(answer_relevancy),
            "answer_correctness": float(answer_correctness),
            "answer_similarity": float(answer_similarity),
        }

    async def score_all():
        results = []
        for index, sample_dict in enumerate(samples):
            try:
                if not sample_dict.get("retrieved_contexts"):
                    results.append({
                        "context_precision": 0.0,
                        "context_recall": 0.0,
                        "faithfulness": 0.0,
                        "answer_relevancy": 0.0,
                        "answer_correctness": 0.0,
                        "answer_similarity": 0.0,
                    })
                    notes.append(f"Sample {index}: No retrieved contexts.")
                    continue

                result = await score_one(sample_dict)
                results.append(result)
            except Exception as error:
                notes.append(f"Sample {index} failed: {type(error).__name__}: {error}")

        return results

    scored = asyncio.run(score_all())

    metrics = {}
    if scored:
        metrics["ragas_context_precision"] = sum(item["context_precision"] for item in scored) / len(scored)
        metrics["ragas_context_recall"] = sum(item["context_recall"] for item in scored) / len(scored)
        metrics["ragas_faithfulness"] = sum(item["faithfulness"] for item in scored) / len(scored)
        metrics["ragas_answer_relevancy"] = sum(item["answer_relevancy"] for item in scored) / len(scored)
        metrics["ragas_answer_correctness"] = sum(item["answer_correctness"] for item in scored) / len(scored)
        metrics["ragas_answer_similarity"] = sum(item["answer_similarity"] for item in scored) / len(scored)
        metrics["ragas_n_samples"] = len(scored)

    output = {
        "metrics": metrics,
        "notes": notes,
    }

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(output, file, indent=4)

    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)

    sys.exit(main(sys.argv[1], sys.argv[2]))