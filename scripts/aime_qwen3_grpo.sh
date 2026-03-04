#!/usr/bin/env bash
set -xeuo pipefail

############################################
# Paths
############################################
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(git -C "$SCRIPT_DIR/.." rev-parse --show-toplevel 2>/dev/null || (cd "$SCRIPT_DIR/.." && pwd))"

# Allow local imports without requiring installation.
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

############################################
# Experiment / Weights & Biases
############################################
export WANDB_PROJECT="GRPO"
export EXP_NAME="Qwen3-4B-Instruct-2507-aime"
# Set this outside the script:
#   export WANDB_API_KEY="..."
# Optional:
#   export WANDB_MODE="offline"

############################################
# Model / Data
############################################
export MODEL_PATH="Qwen/Qwen3-4B-Instruct-2507"

TRAIN_PATH="$ROOT/data/rlmia_aime24_25.parquet"
VAL_PATH="$ROOT/data/valid_aime.parquet"

# Hydra list syntax as a single string
TRAIN_FILES="['$TRAIN_PATH']"
VAL_FILES="['$VAL_PATH']"

############################################
# Networking / Ray stability
############################################
export no_proxy="127.0.0.1,localhost"
export NO_PROXY="127.0.0.1,localhost"

# Put Ray sessions somewhere predictable (short paths help AF_UNIX)
RUN_DIR="$ROOT/checkpoints/$WANDB_PROJECT/$EXP_NAME"
mkdir -p "$RUN_DIR" "$RUN_DIR/ray_tmp" "$RUN_DIR/ray_spill"

# HuggingFace caches: avoid flaky /workspace/.hf_home mounts (stale file handle).
# Put everything under the run directory so Ray workers inherit it reliably.
mkdir -p "$RUN_DIR/hf_home" "$RUN_DIR/hf_home/hub" "$RUN_DIR/hf_home/datasets" "$RUN_DIR/hf_home/transformers"
export HF_HOME="$RUN_DIR/hf_home"
export HF_HUB_CACHE="$RUN_DIR/hf_home/hub"
export HF_DATASETS_CACHE="$RUN_DIR/hf_home/datasets"
export TRANSFORMERS_CACHE="$RUN_DIR/hf_home/transformers"
export XDG_CACHE_HOME="$RUN_DIR/hf_home/xdg"
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_DISABLE_SYMLINKS_WARNING=1

ln -sfn "$RUN_DIR/ray_tmp" /raytmp
ln -sfn "$RUN_DIR/ray_spill" /rayspill

export RAY_TMPDIR="/raytmp"
export RAY_OBJECT_SPILLING_CONFIG='{"type":"filesystem","params":{"directory_path":"/rayspill"}}'

# Disable dashboard/usage stats in this environment
export RAY_DISABLE_DASHBOARD=1
export RAY_DISABLE_USAGE_STATS=1

# Give Ray more time in containerized environments
export RAY_raylet_start_wait_time_s=240
export RAY_gcs_server_request_timeout_seconds=240

# NCCL stability knobs (often helps on cloud GPUs/containers)
export NCCL_DEBUG=WARN
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_CUMEM_ENABLE=0

# vLLM env
export VLLM_LOGGING_LEVEL=WARN
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=true
# Let vLLM pick engine/backend; do not force XFORMERS unless you must
unset VLLM_ATTENTION_BACKEND || true

############################################
# Fail-fast checks
############################################
python3 - <<'PY'
import shutil, sys
total, used, free = shutil.disk_usage("/")
free_gb = free / (1024**3)
if free_gb < 15:
    print(f"ERROR: low free disk space: {free_gb:.2f} GB free on /.", file=sys.stderr)
    sys.exit(1)
PY

command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L || true

############################################
# Start Ray explicitly (more stable than ray.init creating a cluster)
############################################
ray stop --force || true
pkill -9 -f "raylet|gcs_server|dashboard_agent|runtime_env_agent" || true
rm -rf /tmp/ray 2>/dev/null || true

# NOTE: If you want Ray to see exactly 8 GPUs, keep --num-gpus 8.
ray start \
  --head \
  --num-cpus 30 \
  --num-gpus 8 \
  --include-dashboard=false \
  --temp-dir=/raytmp

export RAY_ADDRESS="auto"

############################################
# Training config
############################################
LOGGER="['console','wandb']"

# Safer rollout settings to avoid GPU contention during startup
ROLLOUT_N=4
TP_SIZE=1
GPU_UTIL=0.30

TOTAL_EPOCHS=5
SAVE_FREQ=50
TEST_FREQ=1

# Batch/lengths
TRAIN_BSZ=32
VAL_BSZ=256
MAX_PROMPT_LEN=1024
MAX_RESP_LEN=4096

############################################
# Run training
############################################
cd "$ROOT"

# Enable full Hydra stack traces if something fails
export HYDRA_FULL_ERROR=1
export RAY_LOG_TO_STDERR=1

python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  +ray_kwargs.ray_init.include_dashboard=False \
  \
  data.train_files="$TRAIN_FILES" \
  data.val_files="$VAL_FILES" \
  data.shuffle=True \
  data.train_batch_size="${TRAIN_BSZ}" \
  data.val_batch_size="${VAL_BSZ}" \
  data.max_prompt_length="${MAX_PROMPT_LEN}" \
  data.max_response_length="${MAX_RESP_LEN}" \
  \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.entropy_coeff=0.001 \
  \
  actor_rollout_ref.actor.ppo_mini_batch_size=4 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384 \
  \
  actor_rollout_ref.actor.kl_loss_coef=0.0 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.use_kl_loss=False \
  \
  actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
  \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
  \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.tensor_model_parallel_size="${TP_SIZE}" \
  actor_rollout_ref.rollout.max_num_seqs=64 \
  actor_rollout_ref.rollout.enable_chunked_prefill=False \
  actor_rollout_ref.rollout.enable_prefix_caching=False \
  actor_rollout_ref.rollout.prompt_length="${MAX_PROMPT_LEN}" \
  actor_rollout_ref.rollout.response_length="${MAX_RESP_LEN}" \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
  actor_rollout_ref.rollout.gpu_memory_utilization="${GPU_UTIL}" \
  actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
  \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  \
  algorithm.kl_ctrl.kl_coef=0.0 \
  \
  trainer.critic_warmup=0 \
  trainer.logger="${LOGGER}" \
  trainer.project_name="${WANDB_PROJECT}" \
  trainer.experiment_name="${EXP_NAME}" \
  trainer.val_before_train=True \
  trainer.n_gpus_per_node=8 \
  trainer.nnodes=1 \
  trainer.save_freq="${SAVE_FREQ}" \
  trainer.test_freq="${TEST_FREQ}" \
  trainer.max_actor_ckpt_to_keep=2 \
  trainer.max_critic_ckpt_to_keep=2 \
  trainer.total_epochs="${TOTAL_EPOCHS}" \
  trainer.default_hdfs_dir=null \
  trainer.default_local_dir="${RUN_DIR}" \
  \
  "$@" 2>&1 | tee "${RUN_DIR}/train_$(date +%Y%m%d_%H%M%S).log"
