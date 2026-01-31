#!/bin/bash

#SBATCH --job-name=llava_finetune_medVQA
#SBATCH --time=8:00:00
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#SBATCH --output=logs/llava_finetune_%j.out
#SBATCH --error=logs/llava_finetune_%j.err


############ Variables #########
USER="s5140455"
PROJECT_ROOT="/scratch/${USER}"
SCRIPT_DIR="${PROJECT_ROOT}/Deep-Learning-Project/finetuning"
RAW_DATA_DIR="${PROJECT_ROOT}"
RESULTS_DIR="/scratch/${USER}/job_results/job_${SLURM_JOBID}"


########## Setting up environment ###############

module purge
module load CUDA/12.4.1
module load Python/3.10.4-GCCcore-11.3.0-bare 

# Activate env
source "${PROJECT_ROOT}/unsloth_env2/bin/activate"


# #from https://huggingface.co/microsoft/maira-2/discussions/6
# pip install git+https://github.com/huggingface/transformers.git@d7950bff82b18c823193d17d72188c5e46d06c83
pip install transformers==4.51.3

# pip show
# Isolate python to avoid dependency issues
export PYTHONPATH=""
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=0

# Useful if assigned an A100
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1


############## Setting up workspace in node's local storage ##############
mkdir -p $TMPDIR/training_dataset $TMPDIR/outputs $TMPDIR/lora_llava $TMPDIR/logs
mkdir -p "${RESULTS_DIR}" "${SCRIPT_DIR}/logs"

# HF caches in /home/ without the code below
export HF_HOME="$TMPDIR/.cache"
mkdir -p "$HF_HOME"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
export TMPDIR="$TMPDIR"


############## Dependency health check ####################
echo "Checking Dependencies & GPU"
python -c "import torch; import torchvision; import unsloth; import datasets; print(f'All packages found. Torch: {torch.__version__} | GPU: {torch.cuda.get_device_name(0)}')"

if [ $? -ne 0 ]; then
    echo "ERROR: Dependency check failed."
    exit 1
fi

############ Loading dataset #################
echo "Copying parquet files to local node storage."
cp "${RAW_DATA_DIR}/parquet/train.parquet" "${RAW_DATA_DIR}/parquet/test.parquet" $TMPDIR/training_dataset/
cp "${SCRIPT_DIR}/finetuning.py" $TMPDIR/

cd $TMPDIR || exit 1

############# Script execution ###################

echo "Verifying parquet files"
ls -lh training_dataset/

if [ ! -f "training_dataset/train.parquet" ]; then
    echo "ERROR: train.parquet not found"
    exit 1
fi

if [ ! -f "training_dataset/test.parquet" ]; then
    echo "ERROR: test.parquet not found"
    exit 1
fi

echo "Parquet files loaded successfully"

# # Copy parquet files to results directory
# echo "Copying parquet files to ${RESULTS_DIR}"
# mkdir -p "${RESULTS_DIR}/training_dataset"
# cp -r training_dataset/*.parquet "${RESULTS_DIR}/training_dataset/"

echo "Parquet files copied successfully"

echo "Finetuning"
python finetuning.py 2>&1 | tee logs/training.log

if [ $? -ne 0 ]; then
    echo "ERROR: Training failed"
    cp -r logs/* "${RESULTS_DIR}/logs/"
    exit 1
fi

############## Gather results and clean #############
echo "Creating results tarball."
tar czf $TMPDIR/results.tar.gz lora_llava outputs logs

echo "Moving results to ${RESULTS_DIR}"
cp $TMPDIR/results.tar.gz "${RESULTS_DIR}/"
cp -r lora_llava outputs logs "${RESULTS_DIR}/"

# Update adapters
cp -r lora_llava "${SCRIPT_DIR}/"

echo "Cleaning up $TMPDIR"
cd /tmp && rm -rf $TMPDIR

echo "Job completed successfully at $(date)"