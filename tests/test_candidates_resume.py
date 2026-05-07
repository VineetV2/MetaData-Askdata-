import json
import tempfile
import unittest
from pathlib import Path


class CandidateResumeTests(unittest.TestCase):
    def test_filter_completed_candidates_skips_existing_result_and_error_ids(self):
        from candidates import filter_completed_candidates

        questions = [
            {"question_id": 1, "db_id": "a"},
            {"question_id": 2, "db_id": "a"},
            {"question_id": 3, "db_id": "a"},
        ]

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "candidates.json"
            path.write_text(
                json.dumps(
                    {
                        "results": [{"question_id": 1}],
                        "errors": [{"question_id": 2}],
                    }
                ),
                encoding="utf-8",
            )

            remaining, previous_results, previous_errors = filter_completed_candidates(
                questions,
                path,
            )

        self.assertEqual([q["question_id"] for q in remaining], [3])
        self.assertEqual([r["question_id"] for r in previous_results], [1])
        self.assertEqual([e["question_id"] for e in previous_errors], [2])


if __name__ == "__main__":
    unittest.main()
