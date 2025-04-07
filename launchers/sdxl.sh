export HF_HOME="/home/ec2-user/SageMaker/threshold-dpo/hf_cache_files"
export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export VAE="madebyollin/sdxl-vae-fp16-fix"
export DATASET_NAME="yuvalkirstain/pickapic_v1"


# Effective BS will be (N_GPU * train_batch_size * gradient_accumulation_steps)
# Paper used 2048. Training takes ~30 hours / 200 steps

accelerate launch train.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --pretrained_vae_model_name_or_path=$VAE \
  --dataset_name=$DATASET_NAME \
  --train_batch_size=1 \
  --dataloader_num_workers=8 \
  --gradient_accumulation_steps=1 \
  --mixed_precision="fp16" \
  --max_train_steps=5000 \
  --lr_scheduler="constant_with_warmup" --lr_warmup_steps=200 \
  --learning_rate=1e-8 --scale_lr \
  --choice_model 'pickscore' \
  --cache_dir="datasets" \
  --dpo_type 'leveled_offset' \
  --checkpointing_steps 1000 \
  --beta_dpo 5000 \
   --sdxl  \
  --output_dir="tmp-sdxl-dpo"\
  --resolution 512
  
