import tempfile
import unittest
from unittest import mock
from pathlib import Path


class ConfigTests(unittest.TestCase):
    def test_load_config_resolves_paths_relative_to_config_file(self):
        from config import load_config

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg_path = root / "config.yaml"
            cfg_path.write_text(
                """
dataset:
  name: bird_dev
  db_root: data/bird/dev/dev_databases
  questions: data/bird/dev/dev.json
  train_questions: data/bird/train/train.json
artifacts:
  profile_root: artifacts/bird_dev/profiles
  index_root: artifacts/bird_dev/indexes
  run_root: runs/bird_dev
llm:
  backend: vllm
  api_base: http://127.0.0.1:8000/v1
  api_key: EMPTY
  timeout_seconds: 123
  models:
    - name: gpt-oss-20b
      model_id: openai/gpt-oss-20b
hf:
  token_env: HF_TOKEN
embeddings:
  backend: sentence_transformers
  model: BAAI/bge-base-en-v1.5
  device: cuda
  batch_size: 64
pipeline:
  faiss_top_k: 7
  candidate_workers: 2
  questions_limit: 3
""",
                encoding="utf-8",
            )

            cfg = load_config(cfg_path)

            self.assertEqual(cfg.dataset.name, "bird_dev")
            root = root.resolve()
            self.assertEqual(cfg.dataset.db_root, root / "data/bird/dev/dev_databases")
            self.assertEqual(cfg.dataset.questions, root / "data/bird/dev/dev.json")
            self.assertEqual(cfg.dataset.train_questions, root / "data/bird/train/train.json")
            self.assertEqual(cfg.artifacts.run_root, root / "runs/bird_dev")
            self.assertEqual(cfg.llm.timeout_seconds, 123)
            self.assertEqual(cfg.hf.token_env, "HF_TOKEN")
            self.assertEqual(cfg.embeddings.backend, "sentence_transformers")
            self.assertEqual(cfg.embeddings.model, "BAAI/bge-base-en-v1.5")
            self.assertEqual(cfg.embeddings.device, "cuda")
            self.assertEqual(cfg.embeddings.batch_size, 64)
            self.assertEqual(cfg.model("gpt-oss-20b").model_id, "openai/gpt-oss-20b")
            self.assertEqual(cfg.pipeline.faiss_top_k, 7)
            self.assertEqual(cfg.pipeline.candidate_workers, 2)
            self.assertEqual(cfg.pipeline.questions_limit, 3)

    def test_openai_compatible_api_key_can_come_from_environment(self):
        from config import load_config, make_backend_from_config

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg_path = root / "config.yaml"
            cfg_path.write_text(
                """
dataset:
  db_root: data/bird/dev/dev_databases
  questions: data/bird/dev/dev.json
  train_questions: data/bird/train/train.json
artifacts:
  run_root: runs/bird_dev
llm:
  backend: openai_compatible
  api_base: https://example.test/v1
  api_key_env: COMPAT_API_KEY
  timeout_seconds: 321
  models:
    - name: local-model
      model_id: provider/model
""",
                encoding="utf-8",
            )

            with mock.patch.dict("os.environ", {"COMPAT_API_KEY": "compat-test-key"}):
                cfg = load_config(cfg_path)
                backend = make_backend_from_config(cfg, "local-model", cache=False)

        self.assertEqual(cfg.llm.api_key_env, "COMPAT_API_KEY")
        self.assertEqual(backend.api_base, "https://example.test/v1")
        self.assertEqual(backend.api_key, "compat-test-key")
        self.assertEqual(backend.model_id, "provider/model")
        self.assertEqual(backend.timeout_seconds, 321)

    def test_default_config_points_at_bird_dev_files(self):
        from config import load_config

        cfg = load_config("config.yaml")

        self.assertTrue(cfg.dataset.db_root.exists())
        self.assertTrue(cfg.dataset.questions.exists())
        self.assertTrue(cfg.dataset.train_questions.exists())


if __name__ == "__main__":
    unittest.main()
