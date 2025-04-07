export HF_HOME="/home/ec2-user/SageMaker/threshold-dpo/hf_cache_files"

python evaluator.py --pretrained_model_name "runwayml/stable-diffusion-v1-5" --dataset_name "yuvalkirstain/pickapic_v1" --filtered_size 200