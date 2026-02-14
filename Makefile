# ===========================================================================
#  Restaurant Search Engine — BERT Contrastive Training Pipeline
#  Usage: make <target>    |   make help
# ===========================================================================

PYTHON         := python3
SCRIPTS        := scripts
DATA_RAW       := data/raw
BIZ_JSON       := $(DATA_RAW)/yelp_dataset/yelp_academic_dataset_business.json
CKPT_DIR       := models/checkpoints
BEST_MODEL     := $(CKPT_DIR)/best_model
EMBED_MODEL    := models/embedding_model

# ── Tuneable defaults (override on CLI: make train EPOCHS=5) ──────────────
EPOCHS         := 3
BATCH_SIZE     := 32
LOSS_TYPE      := mnrl
POOLING        := mean
REPORT_TO      := tensorboard
ES_HOST        := http://localhost:9200
BIZ_INDEX      := yelp_businesses_raw
PAIRS_INDEX    := yelp_training_pairs
EVAL_SPLIT     := val

.PHONY: help setup \
        es_start es_stop \
        ingest_raw process_data \
        train evaluate export \
        lint format

# ── help ──────────────────────────────────────────────────────────────────
help:
	@printf "\n  Restaurant Search Engine — available targets\n\n"
	@printf "  make setup           Install Python dependencies\n"
	@printf "  make es_start        Spin up self-hosted ES (Docker Compose)\n"
	@printf "  make es_stop         Tear down ES container\n"
	@printf "  make ingest_raw      Stream Yelp business JSON → ES raw index\n"
	@printf "  make process_data    ES raw → (query, business_text) pairs index\n"
	@printf "  make train           Contrastive fine-tuning  [LOSS_TYPE=mnrl|triplet]\n"
	@printf "  make evaluate        Embedding quality metrics [EVAL_SPLIT=val|test]\n"
	@printf "  make export          Save embedding model      [POOLING=cls|mean]\n"
	@printf "  make lint            Lint scripts/ (ruff + black --check)\n"
	@printf "  make format          Auto-format scripts/ in place\n\n"
	@printf "  Tuneable vars: EPOCHS  BATCH_SIZE  LOSS_TYPE  POOLING  REPORT_TO\n"
	@printf "                 ES_HOST  BIZ_INDEX  PAIRS_INDEX\n\n"

# ── setup ─────────────────────────────────────────────────────────────────
setup:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

# ── ElasticSearch lifecycle ───────────────────────────────────────────────
es_start:
	docker compose up -d
	@printf "\nWaiting for ES to be ready…\n"
	@until curl -sf $(ES_HOST)/_cluster/health > /dev/null 2>&1; do \
	    printf "."; sleep 2; \
	done
	@printf "\nES is ready at $(ES_HOST)\n"

es_stop:
	docker compose down

# ── Data ingestion ─────────────────────────────────────────────────────────
ingest_raw: _require_biz_json
	$(PYTHON) $(SCRIPTS)/DataHandling/es_ingest_businesses.py \
	    --input_path $(BIZ_JSON) \
	    --index_name $(BIZ_INDEX) \
	    --es_host    $(ES_HOST)

process_data: _require_pairs_index_src
	$(PYTHON) $(SCRIPTS)/DataHandling/es_process.py \
	    --source_index $(BIZ_INDEX) \
	    --target_index $(PAIRS_INDEX) \
	    --es_host      $(ES_HOST)

# ── Training ──────────────────────────────────────────────────────────────
train: _require_pairs_index
	$(PYTHON) $(SCRIPTS)/Finetuning/train_contrastive.py \
	    --loss_type  $(LOSS_TYPE) \
	    --num_epochs $(EPOCHS) \
	    --batch_size $(BATCH_SIZE) \
	    --report_to  $(REPORT_TO) \
	    --model_dir  $(CKPT_DIR)

# ── Evaluation ────────────────────────────────────────────────────────────
evaluate: _require_best_model
	$(PYTHON) $(SCRIPTS)/Evaluation/evaluate.py \
	    --model_dir  $(BEST_MODEL) \
	    --output_dir $(BEST_MODEL) \
	    --split      $(EVAL_SPLIT) \
	    --es_host    $(ES_HOST)

# ── Export ────────────────────────────────────────────────────────────────
export: _require_best_model
	$(PYTHON) $(SCRIPTS)/export_embeddings.py \
	    --model_dir        $(BEST_MODEL) \
	    --output_dir       $(EMBED_MODEL) \
	    --pooling_strategy $(POOLING)

# ── Lint & format ──────────────────────────────────────────────────────────
lint:
	$(PYTHON) -m ruff check $(SCRIPTS)/
	$(PYTHON) -m black --check $(SCRIPTS)/

format:
	$(PYTHON) -m ruff check --fix $(SCRIPTS)/
	$(PYTHON) -m black $(SCRIPTS)/

# ── Prerequisite guards ────────────────────────────────────────────────────
_require_biz_json:
	@test -f "$(BIZ_JSON)" || { \
	    printf "\nERROR: Yelp dataset not found at $(BIZ_JSON)\n"; \
	    printf "Download from https://www.yelp.com/dataset\n\n"; \
	    exit 1; \
	}

_require_pairs_index_src:
	@curl -sf "$(ES_HOST)/$(BIZ_INDEX)" > /dev/null 2>&1 || { \
	    printf "\nERROR: ES index '$(BIZ_INDEX)' not found.\n"; \
	    printf "Run: make ingest_raw\n\n"; \
	    exit 1; \
	}

_require_pairs_index:
	@curl -sf "$(ES_HOST)/$(PAIRS_INDEX)" > /dev/null 2>&1 || { \
	    printf "\nERROR: ES index '$(PAIRS_INDEX)' not found.\n"; \
	    printf "Run: make process_data\n\n"; \
	    exit 1; \
	}

_require_best_model:
	@test -d "$(BEST_MODEL)" || { \
	    printf "\nERROR: Best model not found at $(BEST_MODEL)\n"; \
	    printf "Run: make train\n\n"; \
	    exit 1; \
	}
