# gemma-sae Makefile — local dev + Unity SLURM remote workflow
#
# Usage:
#   make populate-pilot   # Create DB with pilot slice
#   make push             # Sync code to Unity
#   make submit-pilot     # Submit pilot sampling jobs
#   make status           # Check SLURM job status
#   make pull             # Pull DB + logs back
#   make classify-pilot   # Run both classifiers on pilot data (API, local)

REMOTE := unity
REMOTE_BASE := /scratch4/workspace/zevwilson_umass_edu-culture-llm/llm-cultural-values
LOCAL_DIR := gemma-sae
REMOTE_DIR := $(REMOTE_BASE)/gemma-sae
DB := $(LOCAL_DIR)/data/culture.db

RSYNC_EXCLUDE := --exclude='__pycache__' --exclude='.git' --exclude='*.pyc' \
                 --exclude='.venv/' --exclude='wandb/' --exclude='*.db' \
                 --exclude='*.db-wal' --exclude='*.db-shm'

# ── Local DB management ──────────────────────────────────────────

.PHONY: populate-pilot populate-full

populate-pilot:
	cd $(LOCAL_DIR) && PYTHONPATH=. python -m db.populate --db data/culture.db --pilot

populate-full:
	cd $(LOCAL_DIR) && PYTHONPATH=. python -m db.populate --db data/culture.db

# ── Remote sync ──────────────────────────────────────────────────

.PHONY: push pull status

push:
	@echo "==> Pushing code to $(REMOTE)..."
	rsync -avz $(RSYNC_EXCLUDE) $(LOCAL_DIR)/ $(REMOTE):$(REMOTE_DIR)/
	@echo "==> Pushing DB..."
	rsync -avz $(DB) $(REMOTE):$(REMOTE_DIR)/data/culture.db

pull:
	@echo "==> Pulling DB from $(REMOTE)..."
	rsync -avz $(REMOTE):$(REMOTE_DIR)/data/culture.db $(DB)
	@echo "==> Pulling logs..."
	rsync -avz $(REMOTE):$(REMOTE_DIR)/logs/ $(LOCAL_DIR)/logs/ 2>/dev/null || true

status:
	ssh $(REMOTE) 'squeue -u $$USER -o "%.8i %.9P %.30j %.8T %.10M %.6D %R"'

# ── SLURM submission ─────────────────────────────────────────────

.PHONY: submit-pilot submit-pilot-dry

submit-pilot: push
	ssh $(REMOTE) 'cd $(REMOTE_BASE) && bash gemma-sae/slurm/launch_pilot.sh'

submit-pilot-dry:
	ssh $(REMOTE) 'cd $(REMOTE_BASE) && bash gemma-sae/slurm/launch_pilot.sh --dry-run'

# ── Classification (API, runs locally) ───────────────────────────

.PHONY: classify-pilot classify-openai classify-anthropic

classify-pilot: classify-openai classify-anthropic

classify-openai:
	cd $(LOCAL_DIR) && PYTHONPATH=. python -m classify.classify \
		--db data/culture.db --classifier gpt-4.1-mini --batch-mode

classify-anthropic:
	cd $(LOCAL_DIR) && PYTHONPATH=. python -m classify.classify \
		--db data/culture.db --classifier claude-haiku-4.5 --batch-mode

# ── Local validation (no GPU) ────────────────────────────────────

.PHONY: dry-run dry-run-all validate-classify

# Test DB queries, extraction, filtering — downloads tokenizer only, no model
dry-run:
	cd $(LOCAL_DIR) && PYTHONPATH=. python -m inference.sample \
		--model_id gemma3_27b_pt --model_hf_id google/gemma-3-27b-pt \
		--db data/culture.db --dry-run --n-samples 50

# Dry-run all model configs
dry-run-all:
	@echo "=== gemma3_27b_pt ==="
	cd $(LOCAL_DIR) && PYTHONPATH=. python -m inference.sample \
		--model_id gemma3_27b_pt --model_hf_id google/gemma-3-27b-pt \
		--db data/culture.db --dry-run --n-samples 50
	@echo "\n=== gemma3_12b_pt ==="
	cd $(LOCAL_DIR) && PYTHONPATH=. python -m inference.sample \
		--model_id gemma3_12b_pt --model_hf_id google/gemma-3-12b-pt \
		--db data/culture.db --dry-run --n-samples 50
	@echo "\n=== eurollm22b ==="
	cd $(LOCAL_DIR) && PYTHONPATH=. python -m inference.sample \
		--model_id eurollm22b --model_hf_id utter-project/EuroLLM-22B-2512 \
		--db data/culture.db --dry-run --n-samples 50
	@echo "\n=== hplt2c_eng ==="
	cd $(LOCAL_DIR) && PYTHONPATH=. python -m inference.sample \
		--model_id hplt2c_eng --model_hf_id HPLT/hplt2c_eng_checkpoints \
		--db data/culture.db --dry-run --n-samples 50 --lang eng

validate-classify:
	cd $(LOCAL_DIR) && PYTHONPATH=. python -m classify.classify \
		--db data/culture.db --classifier gpt-4.1-mini --validate

# ── Remote interactive validation (GPU, one model at a time) ─────

.PHONY: interactive remote-validate

# Open an interactive GPU session on Unity (30 min, smallest useful GPU)
interactive: push
	@echo "Requesting interactive GPU session on Unity..."
	@echo "Once inside, run: cd $(REMOTE_BASE) && PYTHONPATH=gemma-sae gemma-sae/.venv/bin/python -m inference.sample --model_id hplt2c_eng --model_hf_id HPLT/hplt2c_eng_checkpoints --db gemma-sae/data/culture.db --validate --n-samples 3"
	ssh -t $(REMOTE) 'srun --gres=gpu:1 --mem=32G -p gpu-preempt -t 00:30:00 --pty bash'

# Push + run --validate for the smallest model via srun (blocks until GPU allocated + test done)
remote-validate: push
	@echo "Running HPLT validation via srun (will queue for GPU)..."
	ssh $(REMOTE) 'srun --gres=gpu:1 --mem=32G -p gpu-preempt -t 00:10:00 \
		bash -c "cd $(REMOTE_BASE) && \
		module load cuda/12.6 && \
		export HF_HOME=/scratch4/workspace/zevwilson_umass_edu-culture-llm/.hf_cache && \
		export PYTHONPATH=gemma-sae && \
		export PYTHONUNBUFFERED=1 && \
		gemma-sae/.venv/bin/python -m inference.sample \
			--model_id hplt2c_eng \
			--model_hf_id HPLT/hplt2c_eng_checkpoints \
			--db gemma-sae/data/culture.db \
			--validate --n-samples 3 --temperature 0.8"'
