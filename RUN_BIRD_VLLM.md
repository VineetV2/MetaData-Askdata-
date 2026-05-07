# Running BIRD Dev With vLLM

This repo is configured for BIRD dev evaluation with train questions used only as few-shot examples.

## 1. Interactive Wulver Session

Request an interactive GPU session on Wulver. For a quick 20B test, one A100 80GB should be enough. For 120B, start with 4 A100 80GB and vLLM tensor parallelism.

The NJIT docs show Wulver GPU nodes have 4 NVIDIA A100 80GB GPUs per node on the `gpu` partition. Use the interactive command or Open OnDemand workflow your account supports. Avoid running model servers on login nodes.

## 2. Environment

```bash
cd /path/to/Metadata_paper
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install vllm
```

Set your Hugging Face token before starting vLLM or downloading local embedding models:

```bash
export HF_TOKEN=hf_...
huggingface-cli login --token "$HF_TOKEN"
```

`config.yaml` uses `hf.token_env: HF_TOKEN`, and the Python config helper mirrors it to `HUGGING_FACE_HUB_TOKEN` for tools that read that name.

The current FAISS/few-shot embedding code uses OpenAI embeddings:

```bash
export OPENAI_API_KEY=...
```

## 3. Start vLLM For Shared Profiles

Start one vLLM server before profile generation. A practical choice is to use `gpt-oss-120b` for the shared metadata if resources allow, or `gpt-oss-20b` for a faster/lighter setup.

Example using 120B:

```bash
source venv/bin/activate
python -m vllm.entrypoints.openai.api_server \
  --model openai/gpt-oss-120b \
  --tensor-parallel-size 4 \
  --host 127.0.0.1 \
  --port 8000
```

## 4. Shared Profiles

Generate BIRD dev profiles once. These are shared by both gpt-oss runs.

```bash
python profiles.py --config config.yaml --model gpt-oss-120b --all --overwrite
```

This writes profile JSONL files next to each SQLite DB under `data/bird/dev/dev_databases/<db_id>/`.

Stop the profile vLLM server after profiles complete if you want to free the GPUs before indexing.

## 5. Shared FAISS + LSH Indexes

Rebuild indexes from the BIRD dev profiles:

```bash
python indexes.py --config config.yaml --all --overwrite
```

## 6. Start vLLM For 20B

In one terminal inside the interactive GPU session:

```bash
source venv/bin/activate
python -m vllm.entrypoints.openai.api_server \
  --model openai/gpt-oss-20b \
  --host 127.0.0.1 \
  --port 8000
```

In another terminal on the same node, smoke test one question:

```bash
python pipeline.py \
  --config config.yaml \
  --model gpt-oss-20b \
  --db california_schools \
  --questions_per_db 1 \
  --out runs/bird_dev/gpt-oss-20b/schema_links_smoke.json
```

Then run all BIRD dev questions:

```bash
python pipeline.py \
  --config config.yaml \
  --model gpt-oss-20b \
  --all \
  --questions_per_db 1000000 \
  --out runs/bird_dev/gpt-oss-20b/schema_links.json

python candidates.py \
  --config config.yaml \
  --model gpt-oss-20b \
  --input runs/bird_dev/gpt-oss-20b/schema_links.json \
  --out runs/bird_dev/gpt-oss-20b/candidates.json
```

Stop the vLLM server after the 20B run finishes.

## 7. Start vLLM For 120B

In one terminal inside a multi-GPU interactive session:

```bash
source venv/bin/activate
python -m vllm.entrypoints.openai.api_server \
  --model openai/gpt-oss-120b \
  --tensor-parallel-size 4 \
  --host 127.0.0.1 \
  --port 8000
```

Run the same pipeline with the 120B alias:

```bash
python pipeline.py \
  --config config.yaml \
  --model gpt-oss-120b \
  --all \
  --questions_per_db 1000000 \
  --out runs/bird_dev/gpt-oss-120b/schema_links.json

python candidates.py \
  --config config.yaml \
  --model gpt-oss-120b \
  --input runs/bird_dev/gpt-oss-120b/schema_links.json \
  --out runs/bird_dev/gpt-oss-120b/candidates.json
```

## Notes

- `config.yaml` controls dataset paths, run roots, vLLM API base, and model aliases.
- `--questions_per_db 1000000` is used with `--all` to avoid the legacy MiniDev default of 4 questions per DB.
- If you want a smaller debug run, set `pipeline.questions_limit` in `config.yaml` or pass a small `--questions_per_db` value.
- The repo does not create Slurm scripts; commands above assume you already have an interactive compute session.
