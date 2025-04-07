export HF_HOME="/home/ec2-user/SageMaker/threshold-dpo/hf_cache_files"

python odpo/train.py \
model=gpt2-large \
datasets=[tldr] \
loss=odpo \
seed=0 \
loss.beta=0.5 \
model.name_or_path=[SFT CHECKPOINT] \
exp_name=tldr \
gradient_accumulation_steps=2 \
batch_size=16 \
max_prompt_length=512 \
max_length=1024 \
eval_batch_size=16 \
trainer=FSDPTrainer \
local_run_dir=[LOCAL DIR] \
sample_during_eval=false