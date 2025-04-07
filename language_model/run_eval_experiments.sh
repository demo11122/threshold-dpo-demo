export HF_HOME="/home/ec2-user/SageMaker/threshold-dpo/hf_cache_files"

examples=10000
given_seed=1

for temp in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1
do
for loss_type in tdpo dpo odpo 
do
  for beta in 0.1 0.2 0.3
  do
    echo "Running with loss=${loss_type}, loss.beta=${beta}..."
    
    if [ $loss_type = "tdpo" ] ; then
      python odpo/train.py \
      model=gpt2-large \
      datasets=[imdb] \
      loss=${loss_type} \
      seed=$given_seed \
      n_eval_model_samples=512 \
      'function'=eval \
      exp_name=imdb_${loss_type}_${beta}_b \
      temperature=$temp \
      gradient_accumulation_steps=1 \
      batch_size=64 \
      eval_batch_size=64 \
      n_examples=$examples \
      model.name_or_path=gpt2-large \
      model.archive=imdb_b_${loss_type}_${examples}_${beta}/policy.pt \
      loss.beta=${beta} \
      local_run_dir=odpo \
      loss.beta_t=true \
      sample_during_eval=false

      python odpo/train.py \
      model=gpt2-large \
      datasets=[imdb] \
      loss=${loss_type} \
      seed=$given_seed \
      n_eval_model_samples=512 \
      'function'=eval \
      exp_name=imdb_${loss_type}_${beta}_o \
      temperature=$temp \
      gradient_accumulation_steps=1 \
      batch_size=64 \
      eval_batch_size=64 \
      n_examples=$examples \
      model.name_or_path=gpt2-large \
      model.archive=imdb_o_${loss_type}_${examples}_${beta}/policy.pt \
      loss.beta=${beta} \
      loss.beta_t=false \
      local_run_dir=odpo \
      sample_during_eval=false
    else

    python odpo/train.py \
      model=gpt2-large \
      datasets=[imdb] \
      loss=${loss_type} \
      'function'=eval \
      n_eval_model_samples=512 \
      seed=$given_seed \
      exp_name=imdb_${loss_type}_${beta} \
      temperature=$temp \
      gradient_accumulation_steps=1 \
      batch_size=64 \
      eval_batch_size=64 \
      n_examples=$examples \
      model.name_or_path=gpt2-large \
      model.archive=imdb_${loss_type}_${examples}_${beta}/policy.pt \
      loss.beta=${beta} \
      local_run_dir=odpo \
      sample_during_eval=false
    fi
    echo "Completed run with loss=${loss_type}, beta=${beta}, log saved to logs/train_${loss_type}_beta_${beta}.log"
  done
done
done