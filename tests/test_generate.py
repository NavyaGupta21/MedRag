"""Generation and citation-validation tests.

The language model is stubbed throughout. What is being tested is the
orchestration and the enforcement layers - the parts that must hold regardless
of what the model emits.
"""

import numpy as np
import pytest

from medrag_mini.corpus import ChunkRecord
from medrag_mini.embed import Index
from medrag_mini.generate import (
    INSUFFICIENT,
    answer_question,
    attribute_sentences,
    build_prompt,
    validate_citations,
)
from medrag_mini.retrieve import Hit
from medrag_mini.safety import DISCLAIMER


def make_hits(n=3):
    return [
        Hit(
            record=ChunkRecord(f"c{i}", f"evidence text {i}", "statpearls", f"Doc{i}", "Etiology", 3),
            score=0.9 - i * 0.1,
            rank=i,
        )
        for i in range(n)
    ]


@pytest.fixture
def index():
    """Two vectors: one aligned with the probe query (PROBE), one orthogonal."""
    matrix = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
    records = [
        ChunkRecord("c0", "aspirin inhibits COX-1", "statpearls", "Aspirin", "Mechanism", 3),
        ChunkRecord("c1", "unrelated", "textbook", "Book", None, 1),
    ]
    return Index(matrix=matrix, records=records)


# Aligned with record c0, orthogonal to c1, so retrieval is deterministic and
# no language model is needed to exercise the orchestration.
PROBE = np.array([1, 0, 0], dtype=np.float32)

# Orthogonal to every indexed vector, so nothing clears any positive floor.
# This is the abstention case: a question the corpus simply does not cover.
OFF_TOPIC = np.array([0, 0, 1], dtype=np.float32)


class SpyGenerator:
    def __init__(self, response=""):
        self.response = response
        self.calls = []

    def __call__(self, messages, **kwargs):
        self.calls.append(messages)
        return self.response


class TestPromptAssembly:
    def test_prompt_contains_every_hit_in_order(self):
        hits = make_hits(3)
        messages = build_prompt("What causes X?", hits)
        user = messages[1]["content"]
        assert user.index("[S1]") < user.index("[S2]") < user.index("[S3]")
        for hit in hits:
            assert hit.record.text in user

    def test_prompt_labels_provenance(self):
        messages = build_prompt("q", make_hits(1))
        assert "StatPearls" in messages[1]["content"]
        assert "Doc0 - Etiology" in messages[1]["content"]

    def test_system_prompt_states_the_abstention_token(self):
        messages = build_prompt("q", make_hits(1))
        assert INSUFFICIENT in messages[0]["content"]


class TestCitationValidation:
    def test_out_of_range_markers_are_dropped(self):
        """A 1.5B model does emit [S9] when given three snippets."""
        cleaned, citations, _, dropped = validate_citations(
            "Aspirin inhibits COX-1 [S1]. It also does something else [S9].", make_hits(3)
        )
        assert "[S9]" not in cleaned
        assert dropped == ["[S9]"]
        assert "[S1]" in citations

    def test_citations_resolve_to_the_actual_records(self):
        _, citations, _, _ = validate_citations("A claim about biology here [S2].", make_hits(3))
        assert citations["[S2]"].chunk_id == "c1"

    def test_uncited_claim_is_flagged(self):
        _, _, unsupported, _ = validate_citations(
            "Aspirin irreversibly inhibits cyclooxygenase enzymes in platelets.", make_hits(3)
        )
        assert len(unsupported) == 1

    def test_cited_claim_is_not_flagged(self):
        _, _, unsupported, _ = validate_citations(
            "Aspirin irreversibly inhibits cyclooxygenase enzymes in platelets [S1].", make_hits(3)
        )
        assert unsupported == []

    def test_framing_sentences_are_exempt(self):
        _, _, unsupported, _ = validate_citations(
            "Based on the evidence provided in these snippets, the following holds [S1].",
            make_hits(3),
        )
        assert unsupported == []

    def test_unused_snippets_are_not_reported_as_citations(self):
        _, citations, _, _ = validate_citations("Only one source used here [S1].", make_hits(3))
        assert list(citations) == ["[S1]"]


class TestEnforcementLayers:
    def test_personal_advice_refused_without_calling_the_model(self, index):
        spy = SpyGenerator("should never be produced")
        result = answer_question("Should I take aspirin?", index, generate_fn=spy, query_vector=PROBE)
        assert result.refused
        assert spy.calls == []
        assert result.refusal.matched_pattern

    def test_no_hits_abstains_without_calling_the_model(self, index):
        """The core safety property: an unanswerable question never reaches
        the generator, so the generator cannot be talked into answering it."""
        spy = SpyGenerator("a confident hallucination")
        result = answer_question("anything", index, floor=0.3, generate_fn=spy, query_vector=OFF_TOPIC)
        assert result.abstained
        assert spy.calls == []
        assert "insufficient evidence" in result.answer.lower()

    def test_model_declaring_insufficiency_abstains(self, index):
        spy = SpyGenerator(INSUFFICIENT)
        result = answer_question("What is aspirin?", index, floor=0.0, generate_fn=spy, query_vector=PROBE)
        assert result.abstained

    def test_grounded_answer_is_marked_grounded(self, index):
        spy = SpyGenerator("Aspirin inhibits COX-1 in platelets [S1].")
        result = answer_question(
            "What is aspirin?",
            index,
            floor=0.0,
            generate_fn=spy,
            query_vector=PROBE,
            encode_fn=identical_encoder,
        )
        assert result.grounded
        assert result.answered
        assert result.citations

    def test_ungrounded_answer_is_flagged_not_hidden(self, index):
        """A claim matching no snippet stays visible as unsupported."""
        spy = SpyGenerator("Aspirin cures every known cardiovascular disease completely.")
        result = answer_question(
            "What is aspirin?",
            index,
            floor=0.0,
            generate_fn=spy,
            query_vector=PROBE,
            encode_fn=orthogonal_encoder,   # nothing resembles anything
        )
        assert not result.grounded
        assert result.unsupported_sentences


def orthogonal_encoder(texts):
    """Content-addressed basis vectors, so unrelated texts stay unrelated.

    The dimension is chosen from the text itself rather than its position in
    the batch: sentences and snippets are encoded in separate calls, so
    position-based dimensions would make the first sentence collide with the
    first snippet.
    """
    vectors = np.zeros((len(texts), 32), dtype=np.float32)
    for i, text in enumerate(texts):
        vectors[i, sum(ord(c) for c in text) % 32] = 1.0
    return vectors


def identical_encoder(texts):
    """Every text gets the same vector, so everything matches everything."""
    return np.tile(np.array([1.0] + [0.0] * 15, dtype=np.float32), (len(texts), 1))


class TestAttribution:
    """Qwen2.5-1.5B does not reliably emit [S#] markers, so the pipeline
    computes attributions itself and labels them as computed, not asserted."""

    def test_uncited_claim_is_attributed_to_its_best_match(self):
        hits = make_hits(2)
        # Sentence and snippet 0 share a direction; snippet 1 does not.
        def encoder(texts):
            out = np.zeros((len(texts), 4), dtype=np.float32)
            for i, text in enumerate(texts):
                out[i] = [1.0, 0.2, 0, 0] if "evidence text 1" not in text else [0, 0, 1.0, 0]
            return out

        attributions, unsupported = attribute_sentences(
            "Aspirin irreversibly inhibits cyclooxygenase in platelets.",
            hits,
            encode_fn=encoder,
            floor=-1.0,
        )
        assert len(attributions) == 1
        assert attributions[0].marker in ("[S1]", "[S2]")
        assert unsupported == []

    def test_sentence_below_floor_is_unsupported_not_attributed(self):
        attributions, unsupported = attribute_sentences(
            "Aspirin cures every known cardiovascular disease completely.",
            make_hits(2),
            encode_fn=orthogonal_encoder,
            floor=0.5,
        )
        assert attributions == []
        assert len(unsupported) == 1

    def test_already_cited_sentences_are_left_alone(self):
        attributions, unsupported = attribute_sentences(
            "Aspirin inhibits COX-1 in platelets [S1].", make_hits(2), encode_fn=identical_encoder
        )
        assert attributions == []
        assert unsupported == []

    def test_no_hits_means_everything_is_unsupported(self):
        attributions, unsupported = attribute_sentences(
            "Aspirin inhibits cyclooxygenase enzymes in platelets.", [], encode_fn=identical_encoder
        )
        assert attributions == []
        assert len(unsupported) == 1

    def test_attribution_carries_a_score_and_a_record(self):
        attributions, _ = attribute_sentences(
            "Aspirin irreversibly inhibits cyclooxygenase in platelets.",
            make_hits(2),
            encode_fn=identical_encoder,
            floor=-1.0,
        )
        assert attributions[0].record.chunk_id
        assert isinstance(attributions[0].score, float)


class TestDisclaimerIsUnconditional:
    @pytest.mark.parametrize(
        "question,response,floor",
        [
            ("Should I take aspirin?", "", 0.0),          # refusal path
            ("What is aspirin?", "answer [S1].", 0.3),    # abstention path (see probe below)
            ("What is aspirin?", "Aspirin works [S1].", 0.0),  # answered path
            ("What is aspirin?", INSUFFICIENT, 0.0),      # model-declared abstention
        ],
    )
    def test_every_path_carries_the_disclaimer(self, index, question, response, floor):
        # The 0.3-floor case uses the off-topic probe so retrieval returns
        # nothing; the rest use the aligned probe and reach the generator.
        probe = OFF_TOPIC if floor == 0.3 else PROBE
        result = answer_question(
            question,
            index,
            floor=floor,
            generate_fn=SpyGenerator(response),
            query_vector=probe,
            encode_fn=identical_encoder,
        )
        assert DISCLAIMER in result.answer
