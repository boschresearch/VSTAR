
import argparse, os, sys, glob, yaml, math, random, json
sys.path.append('.')
sys.path.append('./scripts/evaluation/')

import datetime, time
import numpy as np
from omegaconf import OmegaConf
from collections import OrderedDict
from tqdm import trange, tqdm
from einops import rearrange, repeat
from functools import partial
import torch
from pytorch_lightning import seed_everything

from funcs import load_model_checkpoint, load_prompts, load_image_batch, get_filelist, save_videos
from funcs import batch_ddim_sampling
from utils.utils import instantiate_from_config, encode_attribute_multiple

import torchvision
from pathlib import Path
from PIL import Image
import torch.nn.functional as F
import cv2
import scipy as sp
from scipy import stats
from dataclasses import dataclass
import pyrallis

# My utils
from utils.attention_utils import *
from utils.vis_utils import *
from utils.test_list import *

@dataclass
class GenerateConfig:
    frames: int
    savedir: str = "./z_submit_results"
    postfix: str = ""
    seed: int = 17
    num_rounds: int = 1
    ablate_id: str = "0,1,2,3,4"
    until_time: int = 13
    use_short_pick_list: bool = False
    run_baselines: bool = True
    run_reg_wihtout_recap: bool = False

    # sampling
    num_timesteps: int = 25
    cfg_scale: float = 12

    # prompt
    negative_prompt: str = ""

    # split
    data_from: int = 0
    cur_split_id: int = 0
    num_per_split: int = 2

    def __post_init__(self):
        os.makedirs(self.savedir, exist_ok=True)



def img_callback(pred_x0, i):
    video = model.decode_first_stage_2DAE(pred_x0).clip(-1.0, 1.0)
    video = (video / 2 + 0.5).clamp(0, 1)  # -1,1 -> 0,1
    save_path_inter = f"step{i}.jpg"
    save_path_inter = os.path.join(save_dir_latest, save_path_inter)
    save_image_grid(video, save_path_inter, rescale=False, n_rows=8, )



@pyrallis.wrap()
def run(run_config: GenerateConfig):
    ddim_steps = run_config.num_timesteps
    unconditional_guidance_scale = run_config.cfg_scale
    config = 'configs/inference_t2v_512_v2.0.yaml'
    ckpt = 'checkpoints/base_512_v2/model.ckpt'
    savedir = run_config.savedir
    fps = 28
    height, width = 320, 512
    gpu_num = 1
    mode = "base"
    n_samples = 1
    bs = 1
    savefps = 8  # 10
    global frames
    frames = run_config.frames  # 16 #64 #-1
    args_dict = {
        "ckpt_path": ckpt,
        "config": config,
        "mode": mode,
        "fps": fps,
        "width": width,
        "height": height,
        "n_samples": n_samples,
        "bs": bs,
        "ddim_steps": ddim_steps, "ddim_eta": 1.0,
        "unconditional_guidance_scale": unconditional_guidance_scale,
        "savedir": savedir, "frames": frames,
        "savefps": savefps,
    }

    args = OmegaConf.create(args_dict)
    print(args)

    ## step 1: model config
    ## -----------------------------------------------------------------
    config = OmegaConf.load(args.config)
    # data_config = config.pop("data", OmegaConf.create())
    model_config = config.pop("model", OmegaConf.create())
    global model
    model = instantiate_from_config(model_config)
    model = model.cuda()
    assert os.path.exists(args.ckpt_path), f"Error: checkpoint [{args.ckpt_path}] Not Found!"
    model = load_model_checkpoint(model, args.ckpt_path)
    model.eval()

    ## sample shape
    assert (args.height % 16 == 0) and (args.width % 16 == 0), "Error: image size [h,w] should be multiples of 16!"
    ## latent noise shape
    h, w = args.height // 8, args.width // 8
    frames = model.temporal_length if args.frames < 0 else args.frames
    channels = model.channels
    print("Frames: ", frames)

    ## saving folders
    now = datetime.datetime.now().strftime("%Y-%m-%d")
    save_dir = os.path.join(args.savedir, now + f'-{mode}' + run_config.postfix)
    os.makedirs(save_dir, exist_ok=True)
    print("Create save_dir: ", save_dir)

    # Set seed here
    seed_list = [run_config.seed + i * 177 for i in range(run_config.num_rounds)]
    print("Seed list: ", seed_list)
    indices_list = None
    global save_dir_latest


    def run_generation(prompt, attribute_list, attention_store, save_dir, seed_list, use_delta_attention,
                       ablation_dict, use_ref_delta_attention=False):
        ## step 3: run over samples
        ## -----------------------------------------------------------------
        start = time.time()
        n_rounds = len(seed_list)

        for idx in range(0, n_rounds):
            attention_store.reset()
            cur_prompt = prompt
            cur_seed = seed_list[idx]
            seed_everything(cur_seed)
            save_prompt = "-".join((cur_prompt.replace("/", "").split(" ")[:15]))

            num_attribute = len(attribute_list)
            if num_attribute > 0:
                save_dir_cur = f"embed{num_attribute}"
            else:
                save_dir_cur = "prompt"

            # Temp Test
            if use_delta_attention:
                save_dir_cur += f"_deltaAttn_f{frames}_{post_fix_folder}"
            else:
                save_dir_cur += f"_f{frames}"

            save_dir_latest_parent = os.path.join(save_dir, save_prompt, save_dir_cur)
            global save_dir_latest
            save_dir_latest = os.path.join(save_dir_latest_parent, f"{cur_seed}")
            attention_store.set_save_dir(os.path.join(save_dir_latest, "attention"))

            # print(f'Work on prompt {idx + 1} / {n_rounds}... Seed={cur_seed}')
            print(cur_prompt)
            batch_size = args.bs
            noise_shape = [batch_size, channels, frames, h, w]
            fps = torch.tensor([args.fps] * batch_size).to(model.device).long()

            x_T = None
            print(f'----> saved in {save_dir_latest}')

            if isinstance(cur_prompt, str):
                prompts = [cur_prompt]

            if len(attribute_list) == 0:
                print("Use normal prompt embedding.")
                text_emb = model.get_learned_conditioning(prompts)  # (1,77,1024)
            else:
                print("Use attrbites embeddings.")
                # text_emb = encode_attribute(model, positive_attribute, negative_attribute, frames)
                text_emb = encode_attribute_multiple(model, attribute_list, frames, interpolation_mode,
                                                     indices_list=indices_list)

            if args.mode == 'base':
                cond = {"c_crossattn": [text_emb], "fps": fps}
            else:
                raise NotImplementedError

            ## inference
            batch_samples = batch_ddim_sampling(
                model, cond, noise_shape, args.n_samples,
                args.ddim_steps, args.ddim_eta, args.unconditional_guidance_scale,
                verbose=True, img_callback=img_callback,
                x_T=x_T,
            )

            ## b,samples,c,t,h,w
            file_names = [f"{cur_seed}"]
            save_videos(batch_samples, save_dir_latest_parent, file_names, fps=args.savefps, ext_name="gif")
            final_frame_save_dir = os.path.join(save_dir_latest, 'final_video')
            save_image_batch(batch_samples[0, 0], final_frame_save_dir)

            # Save config
            config_cur = {
                "seed": cur_seed,
                "prompt": cur_prompt,
                # "negative_prompt": n_prompt,
                "attribute_list": attribute_list,
                "ablation_dict": ablation_dict,
            }
            with open(os.path.join(save_dir_latest, f"{save_dir_cur}.json"), "w") as outfile:
                json.dump(config_cur, outfile, indent=4)
            print()
        print(f"Saved in {args.savedir}. Time used: {(time.time() - start):.2f} seconds")

    data_start = run_config.data_from + run_config.cur_split_id * run_config.num_per_split
    data_end = data_start + run_config.num_per_split
    print(f'---> Start = {data_start}, End = {data_end}')
    for prompt_id in range(data_start, data_end):
        if run_config.use_short_pick_list:
            prompt = short_pick_list[prompt_id]["prompt"]
            attribute_list = short_pick_list[prompt_id]["subprompts"]
        else:
            prompt = gpt_prompt_list[prompt_id]["prompt"]
            attribute_list = gpt_prompt_list[prompt_id]["subprompts"]
        interpolation_mode = "linear"
        # Attention store related
        keep_timestep_list = []
        save_timestep_list = [*range(1, 26)]
        save_maps = True
        save_npy = False

        attention_store = AttentionStore(
            base_width=64, base_height=40,
            keep_timestep_list=keep_timestep_list,
            save_timestep_list=save_timestep_list,
            save_maps=save_maps, save_npy=save_npy
        )

        use_ref_delta_attention = False
        use_delta_attention = False
        register_attention_control(model, attention_store)
        if run_config.run_baselines:
            print("Use delta attention: ", use_delta_attention)
            # print("-------> Run base model.")
            run_generation(prompt, [], attention_store, save_dir, seed_list, use_delta_attention,
                           ablation_dict=None, use_ref_delta_attention=False)
            # print("-------> Run Multi Prompt.")
            # run_generation(prompt, attribute_list, attention_store, save_dir, seed_list, use_delta_attention,
            #                ablation_dict=None, use_ref_delta_attention=False)


        use_delta_attention = True
        ablate_id_list = run_config.ablate_id.split(',')
        print(run_config.ablate_id, ablate_id_list)
        for ablation_cur_id in ablate_id_list:
        #for i, ablation_dict in enumerate(ablation_dict_list):
            # if str(i) not in ablate_id_list:
            #     continue
            if ablation_cur_id is None:
                break
            else:
                print(len(ablation_dict_list), int(ablation_cur_id))
            ablation_dict = ablation_dict_list[int(ablation_cur_id)]
            register_attention_control_vstar(model, attention_store, ablation_dict)
            post_fix_folder = ""
            ablation_dict.update({"until_time":run_config.until_time})

            for i, k in enumerate(ablation_dict["regularize_res_list"]):
                diag = ablation_dict[f'diag_{k}']
                scale = ablation_dict[f'scale_{k}']
                if i != 0:
                    post_fix_folder += '_'
                post_fix_folder += f"res{k}-std{diag}"
                if scale != 1.0:
                    post_fix_folder += f"-scale{scale}"

            until_time = ablation_dict["until_time"]
            post_fix_folder += f"-until{until_time}"
            print("-------> Run Attention Ablation: ", post_fix_folder)
            if run_config.run_reg_wihtout_recap:
                run_generation(prompt, [], attention_store, save_dir, seed_list, use_delta_attention,
                               ablation_dict=ablation_dict, use_ref_delta_attention=False)
            else:
                run_generation(prompt, attribute_list, attention_store, save_dir, seed_list, use_delta_attention,
                               ablation_dict=ablation_dict, use_ref_delta_attention=False)


if __name__ == '__main__':
    run()