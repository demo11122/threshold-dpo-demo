export HF_HOME="/home/ec2-user/SageMaker/threshold-dpo/hf_cache_files"
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export DATASET_NAME="yuvalkirstain/pickapic_v1"

accelerate launch ddpo_pytorch_main/train.py
