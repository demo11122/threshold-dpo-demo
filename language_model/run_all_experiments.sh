export HF_HOME="/home/ec2-user/SageMaker/threshold-dpo/hf_cache_files"

examples=5000
temp=1.0
given_seed=1 
for loss_type in tdpo dpo odpo
do
  for beta in 0.1 0.2 0.3
  do
    echo "Running with loss=${loss_type}, loss.beta=${beta}..."
    
    if [ $loss_type = "tdpo" ] ; then

      python odpo/train.py \
      model=gpt2-large \
      datasets=[toxicity] \
      loss=${loss_type} \
      seed=$given_seed \
      exp_name=toxicity_${loss_type}_${beta}_o \
      temperature=$temp \
      gradient_accumulation_steps=1 \
      batch_size=8 \
      eval_batch_size=4 \
      n_examples=$examples \
      model.name_or_path=gpt2-large \
      model.archive=toxicity_sft_${examples}_0/policy.pt \
      loss.beta=${beta} \
      loss.beta_t=true \
      local_run_dir=odpo \
      sample_during_eval=false

      python odpo/train.py \
      model=gpt2-large \
      datasets=[toxicity] \
      loss=${loss_type} \
      seed=$given_seed \
      exp_name=toxicity_${loss_type}_${beta}_o \
      temperature=$temp \
      gradient_accumulation_steps=1 \
      batch_size=8 \
      eval_batch_size=4 \
      n_examples=$examples \
      model.name_or_path=gpt2-large \
      model.archive=toxicity_sft_${examples}_0/policy.pt \
      loss.beta=${beta} \
      loss.beta_t=false \
      local_run_dir=odpo \
      sample_during_eval=false
    else

    python odpo/train.py \
      model=gpt2-large \
      datasets=[toxicity] \
      loss=${loss_type} \
      seed=$given_seed \
      exp_name=toxicity_${loss_type}_${beta} \
      temperature=$temp \
      gradient_accumulation_steps=1 \
      batch_size=8 \
      eval_batch_size=4 \
      n_examples=$examples \
      model.name_or_path=gpt2-large \
      model.archive=toxicity_sft_${examples}_0/policy.pt \
      loss.beta=${beta} \
      local_run_dir=odpo \
      sample_during_eval=false
    fi
    echo "Completed run with loss=${loss_type}, beta=${beta}, log saved to logs/train_${loss_type}_beta_${beta}.log"
  done
done

