#!/usr/bin/env bash
# Launch train.py with torchrun on multiple NVIDIA GPUs, otherwise use one process.
# Run from Git Bash / WSL / Linux: bash run_distributed.sh --epochs 5 --batch-size 128 --lr 1e-3 --tracker csv
set -Eeuo pipefail

usage() {
  cat <<'HELP'
Usage: bash run_distributed.sh [launcher options] [train.py options]
Launcher options:
  --nproc-per-node N    Requested workers (default: auto-detect GPUs)
  --help                Show this message
Common train.py options (passed through):
  --epochs N            Number of epochs (5)
  --batch-size N        Batch size (128)
  --lr FLOAT            Learning rate (default: 0.001)
  --tracker csv|wandb   Tracker (default: csv; train.py must implement it)
  --optimiser SGD|Adam  Optimiser (if supported by train.py)
  --run-name NAME       Experiment name (if supported by train.py)

Example:
  bash run_distributed.sh --epochs 5 --batch-size 128 --lr 1e-3 --tracker csv
HELP
}

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"
requested_workers=""
train_args=()
while (($#)); do
  case "$1" in
    --help|-h) usage; exit 0 ;;
    --nproc-per-node|--nproc_per_node)
      if (($# < 2)); then echo "ERROR: $1 requires an integer value." >&2; exit 2; fi
      requested_workers="$2"; shift 2 ;;
    --nproc-per-node=*|--nproc_per_node=*) requested_workers="${1#*=}"; shift ;;
    --epochs|--batch-size|--lr|--tracker|--optimiser|--run-name)
      if (($# < 2)); then echo "ERROR: $1 requires a value." >&2; exit 2; fi
      train_args+=("$1" "$2"); shift 2 ;;
    --epochs=*|--batch-size=*|--lr=*|--tracker=*|--optimiser=*|--run-name=*)
      train_args+=("$1"); shift ;;
    *) train_args+=("$1"); shift ;;
  esac
done

if [[ -n "$requested_workers" && ! "$requested_workers" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: --nproc-per-node must be a positive integer." >&2; exit 2
fi

# Check common arguments. Extra flags are left for train.py's argument parser.
for ((i=0; i<${#train_args[@]}; i++)); do
  arg="${train_args[i]}"; key="${arg%%=*}"
  if [[ "$arg" == *=* ]]; then value="${arg#*=}"; else value="${train_args[i+1]-}"; fi
  case "$key" in
    --epochs|--batch-size)
      if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then echo "ERROR: $key must be a positive integer." >&2; exit 2; fi ;;
    --lr)
      if ! [[ "$value" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][-+]?[0-9]+)?$ ]] || ! python -c 'import sys, math; x=float(sys.argv[1]); sys.exit(0 if math.isfinite(x) and x>0 else 1)' "$value"; then
        echo "ERROR: --lr must be a positive finite number." >&2; exit 2
      fi ;;
    --tracker)
      if [[ "$value" != csv && "$value" != wandb ]]; then echo "ERROR: --tracker must be csv or wandb (not wandbn)." >&2; exit 2; fi ;;
  esac
done


if ! command -v python >/dev/null 2>&1; then
  echo "ERROR: Python is not on PATH. Activate your .venv first." >&2; exit 1
fi
if [[ ! -f train.py ]]; then
  echo "ERROR: train.py not found next to run_distributed.sh ($script_dir)." >&2; exit 1
fi
if ! gpu_count="$(python -c 'import torch; print(torch.cuda.device_count() if torch.cuda.is_available() else 0)' 2>&1)"; then
  echo "ERROR: Unable to import PyTorch. Activate the correct .venv and install torch." >&2
  echo "$gpu_count" >&2
  exit 1
fi
if [[ ! "$gpu_count" =~ ^[0-9]+$ ]]; then echo "ERROR: Unexpected GPU count: $gpu_count" >&2; exit 1; fi

if ((gpu_count >= 2)); then
  workers="${requested_workers:-$gpu_count}"
  if (( workers > gpu_count )); then
    echo "WARNING: Requested $workers workers but only $gpu_count CUDA GPUs found; using $gpu_count." >&2
    workers="$gpu_count"
  fi
else
  workers=1
  if [[ -n "$requested_workers" && "$requested_workers" != 1 ]]; then
    echo "WARNING: Requested $requested_workers workers but $gpu_count GPUs found; falling back to one process." >&2
  fi
fi

mkdir -p runs
timestamp="$(date +%Y%m%d_%H%M%S)"
logfile="runs/torchrun_${timestamp}_$$.log"
if ((gpu_count == 0)); then mode="CPU single-process";
elif ((workers == 1)); then mode="CUDA single-process";
else mode="CUDA multi-GPU torchrun"; fi

set +e
{
  echo "=========================================="
  echo "PyTorch training launch"
  echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
  echo "CUDA GPUs detected: $gpu_count"
  echo "World size / worker count: $workers"
  echo "Mode: $mode"
  echo "Arguments: ${train_args[*]:-(train.py defaults)}"
  echo "Log: $logfile"
  echo "=========================================="
  if ((workers >= 2)); then
    echo "Starting torchrun (train.py must implement DistributedDataParallel)."
    python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node="$workers" train.py "${train_args[@]}"
  else
    echo "Starting single-process training."
    python -u train.py "${train_args[@]}"
  fi
} 2>&1 | tee "$logfile"
status=${PIPESTATUS[0]}
set -e
if ((status != 0)); then
  echo "ERROR: Training exited with code $status. See $logfile" | tee -a "$logfile" >&2
  echo "Troubleshooting: CUDA OOM -> reduce --batch-size; missing module -> activate .venv/install dependencies; invalid option -> check python train.py --help." | tee -a "$logfile" >&2
fi
exit "$status"
