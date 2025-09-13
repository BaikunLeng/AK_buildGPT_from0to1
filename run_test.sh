set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"     # 如需指定别的 python，可在命令前加 PYTHON_BIN=/usr/local/bin/python3
SCRIPT="gpt_mini.py"

# 如需离线跑（不登录 W&B），取消下一行注释
# export WANDB_MODE=offline

log_dir="runs_logs"
mkdir -p "$log_dir"

run_one () {
  local dataset="$1"
  local max_iters="$2"
  local block_size="$3"
  local batch_size="$4"
  local n_embd="$5"
  local n_head="$6"
  local n_layer="$7"
  local lr="$8"

  timestamp="$(date +%Y%m%d-%H%M%S)"
  log_file="${log_dir}/${dataset}-${timestamp}.log"

  echo "===== Running dataset=${dataset}  iters=${max_iters}  blk=${block_size}  bs=${batch_size}  emb=${n_embd}  H=${n_head}  L=${n_layer}  lr=${lr} ====="
  MAX_ITERS="$max_iters" \
  BLOCK_SIZE="$block_size" \
  BATCH_SIZE="$batch_size" \
  N_EMBD="$n_embd" \
  N_HEAD="$n_head" \
  N_LAYER="$n_layer" \
  LR="$lr" \
  "$PYTHON_BIN" "$SCRIPT" --dataset "$dataset" 2>&1 | tee "$log_file"

  echo ">>> Log saved to: $log_file"
  echo
}

# 你可以按需调整每个数据集的超参（以下是建议值，保证能在笔记本上跑）
# tiny Shakespeare：训练步数多一点
run_one "tinyshakespeare"  "5000"  "32"  "16"  "64"  "4"  "4"  "1e-3"

# WikiText-2：更大更难，先少迭代验证
run_one "wikitext2"        "2000"  "64"  "16"  "64"  "4"  "4"  "1e-3"

# PTB：规模不大，配置同上即可
run_one "ptb"              "2000"  "64"  "16"  "64"  "4"  "4"  "1e-3"

echo "All runs finished. Check W&B dashboard (or ${log_dir} for logs)."