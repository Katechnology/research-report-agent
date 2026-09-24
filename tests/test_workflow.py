"""Offline checks for orchestration and citation handling."""

import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from research_workflow import ReportConfig, render_report, run_workflow


SOURCES = [{
    "id": "S1", "title": "Example paper", "url": "https://example.org/paper",
    "year": 2025, "kind": "paper", "excerpt": "A documented finding.",
}]
PLAN = json.dumps({
    "queries": ["query one", "query two", "query three"],
    "outline": ["Introduction", "Method", "Evidence", "Limitations"],
})


class FakeCompletions:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content=self.outputs.pop(0)))])


class WorkflowTests(unittest.TestCase):
    def test_agent_outputs_and_source_links_flow_to_report(self):
        completions = FakeCompletions([
            PLAN, "Evidence: a documented finding [S1].",
            "# Draft\nA documented finding [S1].",
            "No factual corrections needed.",
            "# Final\nA documented finding is discussed with careful context [S1].",
        ])
        client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        config = ReportConfig(target_words=20, minimum_words=5)
        with patch("research_workflow.gather_sources", return_value=SOURCES):
            result = run_workflow("Example topic", client=client, config=config)
        self.assertIn("[[1]](https://example.org/paper)", result["report"])
        self.assertIn("## Sources", result["report"])
        self.assertEqual(result["quality"]["cited_source_count"], 1)
        self.assertEqual(len(completions.calls), 5)
        self.assertEqual(completions.calls[0]["model"], "openai:gpt-4.1-mini")
        self.assertIn("Evidence:", completions.calls[2]["messages"][1]["content"])

    def test_unknown_citation_requires_correction(self):
        with self.assertRaisesRegex(ValueError, "unknown source IDs"):
            render_report("An unsupported citation [S99].", SOURCES)

        completions = FakeCompletions([
            PLAN, "Evidence [S1].", "# Draft\nFinding [S1].", "Fix citations.",
            "# Final\nFinding [S99].", "# Final\nCorrected finding [S1].",
        ])
        client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        config = ReportConfig(target_words=20, minimum_words=1)
        with patch("research_workflow.gather_sources", return_value=SOURCES):
            result = run_workflow("Example topic", client=client, config=config)
        self.assertIn("[[1]](https://example.org/paper)", result["report"])
        self.assertEqual(len(completions.calls), 6)

    def test_grouped_citations_render_as_separate_links(self):
        sources = SOURCES + [{
            "id": "S2", "title": "Another paper", "url": "https://example.org/another",
            "year": 2024, "kind": "paper", "excerpt": "Another finding.",
        }]
        report = render_report("Two findings [S1, S2] and another [S1; S2].", sources)
        self.assertNotIn("[S1, S2]", report)
        self.assertEqual(report.count("[[1]](https://example.org/paper)"), 2)
        self.assertEqual(report.count("[[2]](https://example.org/another)"), 2)

    def test_short_report_gets_one_grounded_expansion_pass(self):
        completions = FakeCompletions([
            PLAN, "Evidence [S1].", "# Draft\nFinding [S1].", "Explain the evidence.",
            "# Final\nFinding [S1].",
            "# Final\nThe documented finding is explained in context with its limits "
            "and practical meaning for the reader [S1].",
        ])
        client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        config = ReportConfig(target_words=20, minimum_words=12)
        with patch("research_workflow.gather_sources", return_value=SOURCES):
            result = run_workflow("Example topic", client=client, config=config)
        self.assertGreaterEqual(result["quality"]["word_count"], 12)
        self.assertEqual(len(completions.calls), 6)


if __name__ == "__main__":
    unittest.main()
