import tempfile
import unittest
import json
from pathlib import Path


class PipelineWorkerTests(unittest.TestCase):
    def test_group_questions_by_db_preserves_first_seen_db_order(self):
        from pipeline import group_questions_by_db

        questions = [
            {"question_id": 1, "db_id": "b"},
            {"question_id": 2, "db_id": "a"},
            {"question_id": 3, "db_id": "b"},
            {"question_id": 4, "db_id": "c"},
        ]

        grouped = group_questions_by_db(questions)

        self.assertEqual(list(grouped), ["b", "a", "c"])
        self.assertEqual([q["question_id"] for q in grouped["b"]], [1, 3])

    def test_merge_worker_outputs_preserves_requested_db_order(self):
        from pipeline import merge_worker_outputs

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "b.json"
            second = root / "a.json"
            first.write_text(
                '{"results": [{"question_id": 1}], "errors": [{"question_id": 9}]}',
                encoding="utf-8",
            )
            second.write_text(
                '{"results": [{"question_id": 2}], "errors": []}',
                encoding="utf-8",
            )

            merged = merge_worker_outputs([("b", first), ("a", second)])

        self.assertEqual([r["question_id"] for r in merged["results"]], [1, 2])
        self.assertEqual([e["question_id"] for e in merged["errors"]], [9])

    def test_filter_completed_questions_skips_existing_result_and_error_ids(self):
        from pipeline import filter_completed_questions

        questions = [
            {"question_id": 1, "db_id": "a"},
            {"question_id": 2, "db_id": "a"},
            {"question_id": 3, "db_id": "a"},
        ]

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "shard.json"
            path.write_text(
                json.dumps(
                    {
                        "results": [{"question_id": 1}],
                        "errors": [{"question_id": 2}],
                    }
                ),
                encoding="utf-8",
            )

            remaining, previous_results, previous_errors = filter_completed_questions(
                questions,
                path,
            )

        self.assertEqual([q["question_id"] for q in remaining], [3])
        self.assertEqual([r["question_id"] for r in previous_results], [1])
        self.assertEqual([e["question_id"] for e in previous_errors], [2])

    def test_few_shot_retriever_loads_cached_index(self):
        import faiss
        import numpy as np
        from pipeline import FewShotRetriever

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            index_path = root / "few_shot.faiss"
            pool_path = root / "few_shot_pool.json"

            index = faiss.IndexFlatIP(2)
            index.add(np.array([[1.0, 0.0]], dtype="float32"))
            faiss.write_index(index, str(index_path))
            pool_path.write_text(
                json.dumps([{"question": "q", "SQL": "select 1"}]),
                encoding="utf-8",
            )

            retriever = FewShotRetriever([])
            loaded = retriever.load_cache(index_path, pool_path)

        self.assertTrue(loaded)
        self.assertEqual(len(retriever.pool), 1)
        self.assertEqual(retriever._index.ntotal, 1)

    def test_build_final_schema_links_keeps_focused_columns(self):
        from pipeline import build_final_schema_links

        schema = {
            "schools": [
                {"column": "CDSCode", "type": "TEXT", "pk": True, "fk": False},
                {"column": "School", "type": "TEXT", "pk": False, "fk": False},
            ],
            "frpm": [
                {"column": "CDSCode", "type": "TEXT", "pk": False, "fk": True},
                {"column": "District Name", "type": "TEXT", "pk": False, "fk": False},
            ],
        }
        focused_columns = {"frpm.District Name"}
        combo_columns = {
            "focused_short": [{"table": "schools", "column": "School"}],
            "full_short": [],
        }

        links = build_final_schema_links(
            schema,
            focused_columns=focused_columns,
            combo_columns=combo_columns,
        )

        self.assertIn({"table": "frpm", "column": "District Name"}, links)
        self.assertIn({"table": "schools", "column": "School"}, links)

    def test_build_final_schema_links_adds_key_columns_for_linked_tables(self):
        from pipeline import build_final_schema_links

        schema = {
            "cards": [
                {"column": "id", "type": "INTEGER", "pk": True, "fk": False},
                {"column": "uuid", "type": "TEXT", "pk": False, "fk": False},
                {"column": "name", "type": "TEXT", "pk": False, "fk": False},
            ],
            "legalities": [
                {"column": "uuid", "type": "TEXT", "pk": False, "fk": True},
                {"column": "status", "type": "TEXT", "pk": False, "fk": False},
            ],
        }

        links = build_final_schema_links(
            schema,
            focused_columns=set(),
            combo_columns={
                "focused_short": [{"table": "legalities", "column": "status"}],
            },
        )

        self.assertIn({"table": "legalities", "column": "uuid"}, links)
        self.assertIn({"table": "legalities", "column": "status"}, links)


if __name__ == "__main__":
    unittest.main()
