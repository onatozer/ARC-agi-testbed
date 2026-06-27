#!/bin/bash
# Launch TRM-DEQ on ARC-AGI on rashidinejad partition with 4 GPUs.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

RUN_NAME="trm-deq-arc-30m"
DATA_DIR="/scratch1/feinashl/data/arc1concept-aug-1000"
OUT_DIR="/scratch1/feinashl/eqlm_sudoku/${RUN_NAME}"
NUM_GPUS=3
LOG_DIR="${SCRIPT_DIR}/trm/logs"

mkdir -p "${LOG_DIR}"

if [ -d "${OUT_DIR}" ] && [ -f "${OUT_DIR}/last.pt" ]; then
    echo "Warning: ${OUT_DIR}/last.pt already exists. Will overwrite." >&2
fi

sbatch \
    --partition=rashidinejad \
    --gres=gpu:${NUM_GPUS} \
    --cpus-per-task=32 \
    --mem=256G \
    --time=7-00:00:00 \
    --job-name="${RUN_NAME}" \
    --output="${LOG_DIR}/${RUN_NAME}_%j.log" \
    --error="${LOG_DIR}/${RUN_NAME}_%j.log" \
    --wrap="
set -euo pipefail
cd ${REPO_ROOT}

module load cuda/12.4.0 2>/dev/null || true

export PYTHONUNBUFFERED=1
export PYTHONPATH=\"${REPO_ROOT}:\${PYTHONPATH:-}\"
export TRITON_CACHE_DIR=\"/tmp/triton_cache_\${SLURM_JOB_ID}\"
export PYTORCH_CUDA_ALLOC_CONF=\"expandable_segments:True\"

echo '=========================================='
echo \"SLURM_JOB_ID = \${SLURM_JOB_ID}\"
echo \"SLURM_JOB_NODELIST = \${SLURM_NODELIST}\"
echo '=========================================='

which python
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free \
    --format=csv,noheader || true

torchrun --standalone --nproc_per_node=${NUM_GPUS} \
    -m experiments.eqlm_sudoku.trm.train_trm_deq \
    --data_dir ${DATA_DIR} \
    --out_dir  ${OUT_DIR} \
    --evaluator arc \
    --hidden_size 512 \
    --L_layers 8 \
    --H_cycles 3 \
    --num_heads 8 \
    --expansion 4.0 \
    --halt_max_steps 16 \
    --global_batch_size 192 \
    --epochs 100000 \
    --eval_interval 1000 \
    --eval_max_batches 0 \
    --lr 1e-4 \
    --lr_warmup_steps 2000 \
    --weight_decay 0.1 \
    --puzzle_emb_weight_decay 0.1 \
    --puzzle_emb_lr 1e-2 \
    --deq_max_iter 8 \
    --deq_min_iter 4 \
    --deq_tol 1e-3 \
    --bptt_through 2 \
    --jacobian_reg_lambda 0.0 \
    --ema true \
    --ema_rate 0.999 \
    --log_interval 50

echo 'Job finished at' \$(date)
"

echo "Submitted ${RUN_NAME} on rashidinejad with ${NUM_GPUS} GPUs"
echo "Logs: ${LOG_DIR}/${RUN_NAME}_<jobid>.log"
echo "Output: ${OUT_DIR}"
