import pytest

from medrag_mini.safety import (
    DISCLAIMER,
    PERSONAL_ADVICE_PATTERNS,
    screen,
    with_disclaimer,
)


class TestRefusals:
    @pytest.mark.parametrize(
        "question",
        [
            "Should I take aspirin for my chest pain?",
            "Should my father stop his warfarin?",
            "Can I take ibuprofen with my blood pressure medication?",
            "Is it safe for me to exercise with atrial fibrillation?",
            "How much metformin should I take?",
            "What dose should be given to a 40kg child of mine?",
            "My doctor said I have hypertension, is that right?",
            "I have chest pain and shortness of breath, what is wrong?",
            "My test results show high troponin, what does that mean for me?",
            "Do I have diabetes?",
            "Diagnose me based on these symptoms.",
        ],
    )
    def test_personal_clinical_decisions_are_refused(self, question):
        assert screen(question) is not None

    def test_refusal_names_the_rule_that_fired(self):
        refusal = screen("Should I take aspirin?")
        assert refusal.matched_pattern
        assert refusal.reason
        assert "clinician" in refusal.message

    def test_every_pattern_has_a_stated_reason(self):
        assert all(reason for _, reason in PERSONAL_ADVICE_PATTERNS)


class TestPassThrough:
    @pytest.mark.parametrize(
        "question",
        [
            "What causes atrial fibrillation?",
            "How is community-acquired pneumonia managed?",
            "What is the mechanism of action of metformin?",
            "What are the contraindications to thrombolysis?",
            "What is the typical dose range of amoxicillin in adults?",
            "What are the symptoms of aspirin overdose?",
            "How common is schizotypal personality disorder?",
        ],
    )
    def test_general_reference_questions_pass(self, question):
        """The gate must separate 'what should I do' from 'what is known'.

        Overdose, dosing ranges and contraindications are ordinary reference
        questions. A gate that blocked them on topic alone would be useless.
        """
        assert screen(question) is None


class TestKnownFalsePositives:
    def test_first_person_phrasing_of_a_general_question_is_refused(self):
        """A documented limitation, not a bug.

        "Should I be worried about..." is a general question in first-person
        clothing, and the rule cannot tell the difference. Erring toward refusal
        is the right direction for a medical tool, and the notebook demonstrates
        this misfire rather than hiding it - a guardrail whose failure mode has
        not been seen is not a guardrail that is understood.
        """
        assert screen("Should I be worried about antibiotic resistance generally?") is not None

    def test_academic_first_person_is_refused(self):
        assert screen("I have a question about the Krebs cycle.") is not None


class TestDisclaimer:
    def test_disclaimer_is_appended(self):
        out = with_disclaimer("Some answer.")
        assert out.startswith("Some answer.")
        assert DISCLAIMER in out

    def test_disclaimer_survives_trailing_whitespace(self):
        assert DISCLAIMER in with_disclaimer("Answer.   \n\n  ")
