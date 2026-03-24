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

# ── Validate (quick smoke tests) ────────────────────────────────

.PHONY: validate-sample validate-classify

validate-sample:
	cd $(LOCAL_DIR) && PYTHONPATH=. python -m inference.sample \
		--model_id gemma3_12b_pt --model_hf_id google/gemma-3-12b-pt \
		--db data/culture.db --validate --n-samples 5

validate-classify:
	cd $(LOCAL_DIR) && PYTHONPATH=. python -m classify.classify \
		--db data/culture.db --classifier gpt-4.1-mini --validate
