import tempfile
import unittest
from pathlib import Path


class PipelineConfigPathTests(unittest.TestCase):
    def test_configure_paths_updates_pipeline_dataset_locations(self):
        import pipeline
        from config import load_config

        old_root = pipeline.MINIDEV_ROOT
        old_json = pipeline.MINIDEV_JSON
        old_train = pipeline.TRAIN_JSON

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg_path = root / "config.yaml"
            cfg_path.write_text(
                """
dataset:
  name: bird_dev
  db_root: bird/dev/dev_databases
  questions: bird/dev/dev.json
  train_questions: bird/train/train.json
artifacts:
  profile_root: artifacts/profiles
  index_root: artifacts/indexes
  run_root: runs/bird_dev
llm:
  backend: vllm
  api_base: http://127.0.0.1:8000/v1
  api_key: EMPTY
  models:
    - name: gpt-oss-20b
      model_id: openai/gpt-oss-20b
""",
                encoding="utf-8",
            )
            cfg = load_config(cfg_path)

            try:
                pipeline.configure_paths(cfg)

                self.assertEqual(pipeline.MINIDEV_ROOT, root.resolve() / "bird/dev/dev_databases")
                self.assertEqual(pipeline.MINIDEV_JSON, root.resolve() / "bird/dev/dev.json")
                self.assertEqual(pipeline.TRAIN_JSON, root.resolve() / "bird/train/train.json")
            finally:
                pipeline.MINIDEV_ROOT = old_root
                pipeline.MINIDEV_JSON = old_json
                pipeline.TRAIN_JSON = old_train

    def test_make_backend_from_config_uses_model_alias(self):
        from config import load_config, make_backend_from_config
        from llm import OpenAICompatibleBackend

        cfg = load_config("config.yaml")
        backend = make_backend_from_config(cfg, "gpt-oss-20b", cache=False)

        self.assertIsInstance(backend, OpenAICompatibleBackend)
        self.assertEqual(backend.model_id, "openai/gpt-oss-20b")


if __name__ == "__main__":
    unittest.main()
