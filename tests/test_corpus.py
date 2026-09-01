import json

import pytest

from medrag_mini.corpus import (
    ChunkRecord,
    load_records,
    parse_statpearls_title,
    normalize,
    save_records,
)


class TestParseStatpearlsTitle:
    def test_two_part_title(self):
        doc, sec = parse_statpearls_title(
            "Chronic Total Occlusion of the Coronary Artery -- Etiology"
        )
        assert doc == "Chronic Total Occlusion of the Coronary Artery"
        assert sec == "Etiology"

    def test_three_part_title_keeps_subsection(self):
        doc, sec = parse_statpearls_title("Asthma -- Treatment / Management -- Medications")
        assert doc == "Asthma"
        assert sec == "Treatment / Management -- Medications"

    def test_bare_hyphen_in_article_name_survives(self):
        # The exact failure this parser guards against: splitting on "-" would
        # mangle these into nonsense documents.
        doc, sec = parse_statpearls_title("Non-Hodgkin Lymphoma -- Epidemiology")
        assert doc == "Non-Hodgkin Lymphoma"
        assert sec == "Epidemiology"

    def test_hyphen_in_section_name_survives(self):
        doc, sec = parse_statpearls_title("Hypertension -- Continuing Education Activity")
        assert doc == "Hypertension"
        assert sec == "Continuing Education Activity"

    def test_no_separator_is_all_document(self):
        doc, sec = parse_statpearls_title("Diabetes Mellitus")
        assert doc == "Diabetes Mellitus"
        assert sec is None

    def test_trailing_separator_yields_no_section(self):
        doc, sec = parse_statpearls_title("Diabetes Mellitus -- ")
        assert doc == "Diabetes Mellitus"
        assert sec is None


class TestNormalize:
    def test_statpearls_record(self):
        raw = {
            "id": "article-100024_4",
            "title": "Chronic Total Occlusion of the Coronary Artery -- Etiology",
            "content": "CTO lesions arise from organized thrombus.",
            "contents": "Chronic Total Occlusion of the Coronary Artery -- Etiology. CTO lesions arise from organized thrombus.",
        }
        rec = normalize(raw, "statpearls")
        assert rec.chunk_id == "article-100024_4"
        assert rec.document == "Chronic Total Occlusion of the Coronary Artery"
        assert rec.section == "Etiology"
        assert rec.source_type == "statpearls"
        assert rec.token_estimate > 0

    def test_textbook_record_has_no_section(self):
        raw = {
            "id": "Anatomy_Gray_0",
            "title": "Anatomy_Gray",
            "content": "What is anatomy?",
            "contents": "Anatomy_Gray. What is anatomy?",
        }
        rec = normalize(raw, "textbook")
        assert rec.document == "Anatomy_Gray"
        assert rec.section is None

    def test_missing_contents_falls_back_to_title_and_content(self):
        raw = {"id": "x_0", "title": "T", "content": "C"}
        rec = normalize(raw, "textbook")
        assert rec.text == "T. C"

    def test_label_includes_section_when_present(self):
        rec = normalize(
            {"id": "a_1", "title": "Asthma -- Etiology", "content": "c", "contents": "x"},
            "statpearls",
        )
        assert rec.label == "Asthma - Etiology"


class TestRoundTrip:
    def test_save_and_load_preserves_records(self, tmp_path):
        records = [
            ChunkRecord("a_0", "text one", "textbook", "Book", None, 2),
            ChunkRecord("b_1", "text two", "statpearls", "Article", "Etiology", 2),
        ]
        path = tmp_path / "corpus.jsonl"
        save_records(records, path)
        assert load_records(path) == records

    def test_load_missing_file_raises_actionable_error(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="sample_corpus"):
            load_records(tmp_path / "nope.jsonl")


class TestSourceReferenceStripping:
    """StatPearls' own [7]-style markers collide with the pipeline's [S#]
    citations, so they are removed during normalization."""

    def test_single_marker_is_removed(self):
        from medrag_mini.corpus import strip_source_references

        assert strip_source_references("Mutations contribute to DTE. [7]") == (
            "Mutations contribute to DTE."
        )

    def test_multi_reference_marker_is_removed(self):
        from medrag_mini.corpus import strip_source_references

        assert strip_source_references("This is established [41, 42].") == "This is established."

    def test_pipeline_citation_markers_are_preserved(self):
        """[S1] must survive: only bare-numeric markers are source references."""
        from medrag_mini.corpus import strip_source_references

        assert strip_source_references("A claim [S1].") == "A claim [S1]."

    def test_normalize_strips_markers(self):
        rec = normalize(
            {
                "id": "a_1",
                "title": "Asthma -- Etiology",
                "content": "x",
                "contents": "Asthma -- Etiology. Airway inflammation is central [12].",
            },
            "statpearls",
        )
        assert "[12]" not in rec.text
        assert "inflammation is central." in rec.text
