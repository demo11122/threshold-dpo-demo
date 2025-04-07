"""
Evaluate and run all tests for each version of the model we evaluate how it performs on all scoring functions.
"""
from itertools import product
from diffusers import StableDiffusionPipeline, UNet2DConditionModel, StableDiffusionXLPipeline
import torch
import pandas as pd
import numpy as np
from datasets import load_dataset
import argparse
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluator script after you trained the models")

    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
    "--filtered_size",
        type= int,
        default=3,
        help="How many example prompts from test dataset used to evaluate the models",
    )
    args = parser.parse_args()
    return args


finetune_methods = ["beta_tdpo","o_tdpo","odpo", "dpo" ]#, "ddpo", "draft"] # exclude these for now
reward_models = ["aes", "hps", "pickscore", "clip"]

# helper functions

def txt_saver(scores_list, filename):
    data =np.array(scores_list)
    print(data.shape)
    np.savetxt(filename, data, delimiter=",")

def get_filtered_prompts(all_prompts, size: int = 200, seed: int = 1):
    np.random.seed(seed)
    selected_ind = np.random.permutation(len(all_prompts))[:size]
    filtered_prompts = [all_prompts[ind] for ind in selected_ind]
    return filtered_prompts


def load_reward_model(eval_reward: str):
    if eval_reward == 'hps':
        from utils.hps_utils import Selector
    elif eval_reward == 'clip':
        from utils.clip_utils import Selector
    elif eval_reward == 'pickscore':
        from utils.pickscore_utils import Selector
    elif eval_reward == 'aes':
        from utils.aes_utils import Selector
    
    selector = Selector('cuda:0')
    return selector

def load_dpo_model(train_reward: str, finetune_method: str):
    """
    This function is a helper to load the models finetuned with de
    """
    if finetune_method=="beta_tdpo":
        post_fix= f"bdpo-{train_reward}"
    elif finetune_method=="o_tdpo":
        post_fix= f"tdpo-{train_reward}"
    elif finetune_method=="odpo":
        post_fix= f"odpo-{train_reward}"
    elif finetune_method=="dpo":
        post_fix= f"dpo-{train_reward}"
    elif finetune_method=="ddpo":
        post_fix= f"ddpo-{train_reward}"
        
    if finetune_method!="ddpo":
        finetuned_unet = UNet2DConditionModel.from_pretrained(
                                    #  'mhdang/dpo-sd1.5-text2image-v1',
                                    # 'mhdang/dpo-sdxl-text2image-v1',
                                    f'tmp-sd15-{post_fix}/checkpoint-10000', # path to the checkpoint
                                    subfolder='unet',
                                    torch_dtype=torch.float16
        ).to('cuda:0')
    else:
        raise NotImplementedError
    
    return finetuned_unet

args = parse_args()

dataset = load_dataset(
            args.dataset_name,
            None,
            cache_dir="datasets",
            data_dir=None,
        )

torch.set_grad_enabled(False)

pretrained_model_name= args.pretrained_model_name_or_path
# pretrained_model_name = "CompVis/stable-diffusion-v1-4"
# pretrained_model_name = "stabilityai/stable-diffusion-xl-base-1.0"
gs = (5 if 'stable-diffusion-xl' in pretrained_model_name else 20)

eval_prompts= get_filtered_prompts(all_prompts= dataset['test']['caption'], size=args.filtered_size)
# construct the pipeline
if 'stable-diffusion-xl' in pretrained_model_name:
    pipe = StableDiffusionXLPipeline.from_pretrained(
        pretrained_model_name, torch_dtype=torch.float16,
        variant="fp16", use_safetensors=True
    ).to("cuda:0")
else:
    pipe = StableDiffusionPipeline.from_pretrained(pretrained_model_name,
                                                   torch_dtype=torch.float16)
pipe = pipe.to('cuda:0')

pipe.set_progress_bar_config(disable=True)
pipe.safety_checker = None

scoring_models=[]
for reward_model in tqdm(reward_models, desc="Loading scoring models"):
    selector= load_reward_model(reward_model)
    scoring_models.append(selector)
    
    
all_models= product(finetune_methods, reward_models)
scores_all=[]
for finetune_method,reward_model in tqdm(all_models, 
                                         desc="Waiting for scores", 
                                         position=0):

    # load the finetuned model and include it in the pipeline
    finetuned_unet=load_dpo_model(reward_model, finetune_method)
    pipe.unet= finetuned_unet
    # initialize the generator
    generator = torch.Generator(device='cuda:0')
    generator = generator.manual_seed(0)
    
    ims=[]
    for prompt in tqdm(eval_prompts, desc= "generating images", position=0):
        im = pipe(prompt=prompt, generator=generator, guidance_scale=gs).images[0]
        ims.append(im)
    
    scores_model=[]
    for eval_reward in tqdm(reward_models, desc="using different scoring functions", position=0):
        scores_reward=[]
        selector= scoring_models[reward_models.index(eval_reward)]
        for image, prompt in tqdm(zip(ims, eval_prompts),desc="scoring the images", position=0):
            score= selector.score([image], prompt)
            print(score)
            scores_reward.append(score[0])
            
        scores_model.append(scores_reward)
    
    txt_saver(scores_model, f"results/scores_{finetune_method}_{reward_model}_pickapic.txt")
    scores_all.append(scores_model)


scores_all_index=[]
for finetune_method,reward_model in all_models:
    temp_name= f"{finetune_method}_{reward_model}"
    scores_all_index.append(temp_name)


# save to txt scores
# txt_saver(scores_all, "results/scores_all_pickapic.txt")

# save scores_all_index
file_path = "results/scores_all_index_pickapic.txt"
with open(file_path, "w") as file:
    for item in scores_all_index:
        file.write(item + "\n")

