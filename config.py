from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    db_root: Path
    questions: Path
    train_questions: Path


@dataclass(frozen=True)
class ArtifactsConfig:
    profile_root: Path
    index_root: Path
    run_root: Path


@dataclass(frozen=True)
class ModelConfig:
    name: str
    model_id: str


@dataclass(frozen=True)
class LLMConfig:
    backend: str
    api_base: str = ""
    api_key: str = ""
    timeout_seconds: int = 600
    models: List[ModelConfig] = field(default_factory=list)


@dataclass(frozen=True)
class EmbeddingsConfig:
    backend: str = "openai"
    model: str = "text-embedding-3-small"
    device: str = "cpu"
    batch_size: int = 32


@dataclass(frozen=True)
class HFConfig:
    token_env: str = "HF_TOKEN"


@dataclass(frozen=True)
class PipelineConfig:
    faiss_top_k: int = 10
    candidate_workers: int = 1
    candidates_per_combo: int = 3
    final_candidates: int = 3
    max_correction_retries: int = 3
    questions_limit: Optional[int] = None


@dataclass(frozen=True)
class AppConfig:
    path: Path
    dataset: DatasetConfig
    artifacts: ArtifactsConfig
    llm: LLMConfig
    hf: HFConfig
    embeddings: EmbeddingsConfig
    pipeline: PipelineConfig

    def model(self, name: str) -> ModelConfig:
        for model in self.llm.models:
            if model.name == name:
                return model
        available = ", ".join(m.name for m in self.llm.models) or "(none)"
        raise KeyError(f"Unknown model alias {name!r}. Available: {available}")


def _load_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "PyYAML is required to read config files. Install with: pip install PyYAML"
        ) from exc
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    return data


def _resolve(base: Path, value: str) -> Path:
    p = Path(value).expanduser()
    if p.is_absolute():
        return p
    return (base / p).resolve()


def load_config(path: str | Path = "config.yaml") -> AppConfig:
    cfg_path = Path(path).expanduser().resolve()
    data = _load_yaml(cfg_path)
    base = cfg_path.parent

    dataset = data.get("dataset", {})
    artifacts = data.get("artifacts", {})
    llm = data.get("llm", {})
    hf = data.get("hf", {})
    embeddings = data.get("embeddings", {})
    pipeline = data.get("pipeline", {})

    models = [
        ModelConfig(name=str(item["name"]), model_id=str(item["model_id"]))
        for item in llm.get("models", [])
    ]

    return AppConfig(
        path=cfg_path,
        dataset=DatasetConfig(
            name=str(dataset.get("name", "bird_dev")),
            db_root=_resolve(base, dataset["db_root"]),
            questions=_resolve(base, dataset["questions"]),
            train_questions=_resolve(base, dataset["train_questions"]),
        ),
        artifacts=ArtifactsConfig(
            profile_root=_resolve(base, artifacts.get("profile_root", "artifacts/profiles")),
            index_root=_resolve(base, artifacts.get("index_root", "artifacts/indexes")),
            run_root=_resolve(base, artifacts.get("run_root", "runs")),
        ),
        llm=LLMConfig(
            backend=str(llm.get("backend", "openai")),
            api_base=str(llm.get("api_base", "")),
            api_key=str(llm.get("api_key", "")),
            timeout_seconds=int(llm.get("timeout_seconds", 600)),
            models=models,
        ),
        hf=HFConfig(
            token_env=str(hf.get("token_env", "HF_TOKEN")),
        ),
        embeddings=EmbeddingsConfig(
            backend=str(embeddings.get("backend", "openai")),
            model=str(embeddings.get("model", "text-embedding-3-small")),
            device=str(embeddings.get("device", "cpu")),
            batch_size=int(embeddings.get("batch_size", 32)),
        ),
        pipeline=PipelineConfig(
            faiss_top_k=int(pipeline.get("faiss_top_k", 10)),
            candidate_workers=int(pipeline.get("candidate_workers", 1)),
            candidates_per_combo=int(pipeline.get("candidates_per_combo", 3)),
            final_candidates=int(pipeline.get("final_candidates", 3)),
            max_correction_retries=int(pipeline.get("max_correction_retries", 3)),
            questions_limit=pipeline.get("questions_limit"),
        ),
    )


def apply_hf_token_env(cfg: AppConfig) -> None:
    """Mirror the configured HF token env var to names common HF tools read."""
    token = os.environ.get(cfg.hf.token_env, "")
    if not token:
        return
    os.environ.setdefault("HF_TOKEN", token)
    os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", token)


def make_backend_from_config(
    cfg: AppConfig,
    model_name: str,
    cache: bool = True,
):
    from llm import make_backend

    apply_hf_token_env(cfg)
    model = cfg.model(model_name)
    return make_backend(
        cfg.llm.backend,
        model_id=model.model_id,
        api_base=cfg.llm.api_base,
        api_key=cfg.llm.api_key,
        timeout_seconds=cfg.llm.timeout_seconds,
        cache=cache,
    )
