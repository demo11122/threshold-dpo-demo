export HF_HOME="/home/sagemaker-user/threshold-dpo/hf_cache_files"
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export DATASET_NAME="yuvalkirstain/pickapic_v1"
# Effective BS will be (N_GPU * train_batch_size * gradient_accumulation_steps)
# Paper used 2048. Training takes ~24 hours / 2000 steps

accelerate launch train.py\
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME\
  --cache_dir="datasets"\
  --train_batch_size=1 \
  --dataloader_num_workers=4 \
  --gradient_accumulation_steps=2 \
  --max_train_steps=20000 \
  --mixed_precision "fp16" \
  --lr_scheduler="constant_with_warmup" --lr_warmup_steps=500 \
  --learning_rate=1e-8 --scale_lr \
  --checkpointing_steps 1250 \
  --beta_dpo 5000 \
  --choice_model 'pickscore' \
  --dpo_type 'leveled_offset' \
  --train_method 'dpo' \
  --report_to 'wandb' \
  --output_dir="tmp-sd15-draft" \
  --use_adafactor