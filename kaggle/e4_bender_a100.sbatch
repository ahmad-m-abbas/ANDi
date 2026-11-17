#!/bin/bash
#SBATCH --job-name=e4-a100
#SBATCH --partition=A100medium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=1-00:00:00
#SBATCH --export=NIL
#SBATCH --chdir=/tmp

set -eo pipefail
export PATH=/usr/bin:/bin
HOME=$(getent passwd "$(id -un)" | cut -d: -f6)
export HOME="${HOME:?Could not determine the Bender home directory}"

exec /bin/bash -l -s -- "$@" <<'AMD_JOB'
set -eo pipefail
cd "$HOME/ANDi"

collection=${1:?Pass a W&B run name as the first script argument}
shift

module purge
unset PYTHONPATH PYTHONHOME
module unuse /software/easybuild-INTEL_A40/modules/all
module use /software/easybuild-AMD_A100/modules/all
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
source "$HOME/venvs/andi-a100/bin/activate"

andi_site_packages=$(python -c 'import sysconfig; print(sysconfig.get_path("purelib"))')
export PYTHONPATH="$andi_site_packages${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_BASE_URL=https://api.forge.coreweave.com
export MPLBACKEND=Agg

scratch=/dev/shm
minimum_free_kib=$((25 * 1024 * 1024))
if mountpoint -q /local/nvme; then
    nvme_scratch="/local/nvme/$(id -un)"
    if mkdir -p "$nvme_scratch" && [[ -w "$nvme_scratch" ]]; then
        nvme_free_kib=$(df -Pk "$nvme_scratch" | awk 'NR == 2 {print $4}')
        if (( nvme_free_kib >= minimum_free_kib )); then
            scratch="$nvme_scratch"
        fi
    fi
fi
free_kib=$(df -Pk "$scratch" | awk 'NR == 2 {print $4}')
if (( free_kib < minimum_free_kib )); then
    echo "Less than 25 GiB free in $scratch for the temporary E4 databases" >&2
    exit 1
fi
e4_tmp=$(mktemp -d "$scratch/andi-${SLURM_JOB_ID}.XXXXXX")
export TMPDIR="$e4_tmp"
trap 'rm -rf -- "$e4_tmp"' EXIT

echo "E4 temporary storage: $TMPDIR"
df -hT "$TMPDIR"
python -u -c 'import torch; print("GPU:", torch.cuda.get_device_name(0))'
python -u kaggle/e4_one_cell.py \
    --data "$HOME/ANDi/data/BraTS21" \
    --output "$HOME/ANDi/results/e4-bender" \
    --entity ahmad-abbas176 --project andi-e4 \
    --collection "$collection" \
    --variant consistency --consistency-lambda 0.1 \
    --epochs 232 --batch-size 128 --microbatch-size 2 \
    --filesystem-checkpoints \
    "$@"
AMD_JOB
