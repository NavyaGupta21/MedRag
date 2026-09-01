#!/usr/bin/env bash
# Create the isolated RAGAS evaluation environment.
#
# RAGAS is kept in its own virtualenv, with no access to system site-packages,
# because it requires older langchain and openai releases than the pipeline
# environment carries. The two never share a process: medrag_mini.evaluate
# exports retrieved contexts as JSON and invokes scripts/ragas_score.py with
# this interpreter.
#
# The pipeline works without this venv - evaluation simply reports its native
# metrics instead.
#
# The version pins below are not decoration. ragas 0.2.x imports
# `langchain_community.chat_models.vertexai`, which was removed in
# langchain-community 0.3, and its non-LLM string metrics need rapidfuzz.
set -euo pipefail

cd "$(dirname "$0")/.."

python3 -m venv .venv-eval
.venv-eval/bin/pip install --quiet --upgrade pip
.venv-eval/bin/pip install --quiet \
    "ragas==0.2.15" \
    "langchain-community==0.2.19" \
    "langchain-core<0.3" \
    "langchain-openai<0.2" \
    "langchain<0.3" \
    "langchain-text-splitters<0.3" \
    rapidfuzz

.venv-eval/bin/python -c "from ragas.metrics import NonLLMContextPrecisionWithReference; print('RAGAS evaluation environment ready')"
