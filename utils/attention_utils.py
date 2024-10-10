import os, abc

import PIL.Image

from lvdm.common import default, exists
import numpy as np
from einops import rearrange, repeat
import torch
from torch import einsum
from lvdm.modules.attention import TemporalTransformer, SpatialTransformer, CrossAttention
from utils.vis_utils import *
import scipy as sp
from scipy import stats

def move_to(obj_to_move, device='cpu'):
    if isinstance(obj_to_move, dict):
        res = {}
        for k, v in obj_to_move.items():
            res[k] = move_to(v, device)
        return res
    elif isinstance(obj_to_move, list):
        res = []
        for v in obj_to_move:
            res.append(move_to(v, device))
        return res
    elif torch.is_tensor(obj_to_move):
        return obj_to_move.to(device)
    else:
        raise TypeError("Invalid type for move_to")


def register_attention_control_vstar(model, controller, ablation_dict=None):
    def ca_forward(self, place_in_unet, ablation_dict=None):
        def forward(x, context=None, mask=None, time_index=None):
            h = self.heads

            q = self.to_q(x)
            context = default(context, x)
            ## considering image token additionally
            if context is not None and self.img_cross_attention:
                context, context_img = context[:, :self.text_context_len, :], context[:, self.text_context_len:, :]
                k = self.to_k(context)
                v = self.to_v(context)
                k_ip = self.to_k_ip(context_img)
                v_ip = self.to_v_ip(context_img)
            else:
                k = self.to_k(context)
                v = self.to_v(context)

            q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))
            sim = torch.einsum('b i d, b j d -> b i j', q, k) * self.scale
            if self.relative_position:
                len_q, len_k, len_v = q.shape[1], k.shape[1], v.shape[1]
                k2 = self.relative_position_k(len_q, len_k)
                sim2 = einsum('b t d, t s d -> b t s', q, k2) * self.scale  # TODO check
                sim += sim2
            del k

            if exists(mask):
                ## feasible for causal attention mask only
                max_neg_value = -torch.finfo(sim.dtype).max
                mask = repeat(mask, 'b i j -> (b h) i j', h=h)
                sim.masked_fill_(~(mask > 0.5), max_neg_value)

            def process_sim(cur_sim, time_index, ablation_dict):
                aspect_ratio = 40.0 / 64.0  # h/w
                h_w = cur_sim.shape[0] // h
                frames = cur_sim.shape[-1]
                cur_width = int(math.sqrt(h_w / aspect_ratio))
                soft_max_sim = cur_sim.softmax(dim=-1)
                regularize_res_list = ablation_dict.get("regularize_res_list", [64, 32])

                if cur_width not in regularize_res_list:
                    return soft_max_sim

                cur_height = int(cur_width * aspect_ratio)

                # compute score
                base_ratio = 1.0
                scale_max = ablation_dict[f"scale_{cur_width}"]
                diag_std = ablation_dict[f"diag_{cur_width}"]
                attention_delta = create_diag_offset_matrix(frames, diag_std, mean=0.0)
                attention_delta = attention_delta / attention_delta[0, 0]

                max_values, _ = torch.max(cur_sim, dim=-1, keepdim=True)  # * cur_sim.max()
                scale_factor = scale_max * max_values * base_ratio

                attention_delta = torch.from_numpy(attention_delta).to(cur_sim.device).unsqueeze(0)  # (1,L,L)
                attention_delta = attention_delta.to(cur_sim.dtype)
                attention_delta = attention_delta * scale_factor
                new_sim = cur_sim + attention_delta

                new_sim = new_sim.softmax(dim=-1)
                return new_sim

            if time_index <= ablation_dict["until_time"] and place_in_unet != 'init':
                sim = process_sim(sim, time_index, ablation_dict)
            else:
                # attention, what we cannot get enough of
                sim = sim.softmax(dim=-1)

            # New: store in the controller
            attn2 = rearrange(sim, '(b h) k c -> h b k c', h=h).mean(0)  # -> (h*w,L,L)
            controller(attn2, place_in_unet)

            out = torch.einsum('b i j, b j d -> b i d', sim, v)
            if self.relative_position:
                v2 = self.relative_position_v(len_q, len_v)
                out2 = einsum('b t s, t s d -> b t d', sim, v2)  # TODO check
                out += out2
            out = rearrange(out, '(b h) n d -> b n (h d)', h=h)

            ## considering image token additionally
            if context is not None and self.img_cross_attention:
                k_ip, v_ip = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (k_ip, v_ip))
                sim_ip = torch.einsum('b i d, b j d -> b i j', q, k_ip) * self.scale
                del k_ip
                sim_ip = sim_ip.softmax(dim=-1)
                out_ip = torch.einsum('b i j, b j d -> b i d', sim_ip, v_ip)
                out_ip = rearrange(out_ip, '(b h) n d -> b n (h d)', h=h)
                out = out + self.image_cross_attention_scale * out_ip
            del q

            return self.to_out(out)

        return forward

    num_temporal_attention = 0
    cur_temporal_name = None
    for name, layer in model.named_modules():
        if isinstance(layer, TemporalTransformer):
            cur_temporal_name = name
            if "input_blocks" in name:
                place_in_unet = "down"
            elif "init_attn" in name:
                place_in_unet = "init"
            elif "middle_block" in name:
                place_in_unet = "mid"
            elif "output_blocks" in name:
                place_in_unet = "up"
            else:
                raise ValueError("Unknown position of attention in UNet!")

        if (cur_temporal_name is not None and name.startswith(cur_temporal_name)) and (
                name.endswith("transformer_blocks.0.attn1") or name.endswith("transformer_blocks.0.attn2")):
            layer.forward = ca_forward(layer, place_in_unet, ablation_dict)
            num_temporal_attention += 1
    controller.num_att_layers = num_temporal_attention
    print("Total attention layers: ", num_temporal_attention)

def register_attention_control(model, controller):
    def ca_forward(self, place_in_unet):
        def forward(x, context=None, mask=None,time_index=None):
            h = self.heads
            
            q = self.to_q(x)
            context = default(context, x)
            ## considering image token additionally
            if context is not None and self.img_cross_attention:
                context, context_img = context[:,:self.text_context_len,:], context[:,self.text_context_len:,:]
                k = self.to_k(context)
                v = self.to_v(context)
                k_ip = self.to_k_ip(context_img)
                v_ip = self.to_v_ip(context_img)
            else:
                k = self.to_k(context)
                v = self.to_v(context)

            q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))
            sim = torch.einsum('b i d, b j d -> b i j', q, k) * self.scale
            if self.relative_position:
                len_q, len_k, len_v = q.shape[1], k.shape[1], v.shape[1]
                k2 = self.relative_position_k(len_q, len_k)
                sim2 = einsum('b t d, t s d -> b t s', q, k2) * self.scale # TODO check 
                sim += sim2
            del k

            if exists(mask):
                ## feasible for causal attention mask only
                max_neg_value = -torch.finfo(sim.dtype).max
                mask = repeat(mask, 'b i j -> (b h) i j', h=h)
                sim.masked_fill_(~(mask>0.5), max_neg_value)

            # attention, what we cannot get enough of
            sim = sim.softmax(dim=-1)
            
            # New: store in the controller
            attn2 = rearrange(sim, '(b h) k c -> h b k c', h=h).mean(0) # -> (h*w,L,L)
            # print("---->", place_in_unet)
            controller(attn2, place_in_unet)
            
            out = torch.einsum('b i j, b j d -> b i d', sim, v)
            if self.relative_position:
                v2 = self.relative_position_v(len_q, len_v)
                out2 = einsum('b t s, t s d -> b t d', sim, v2) # TODO check
                out += out2
            out = rearrange(out, '(b h) n d -> b n (h d)', h=h)

            ## considering image token additionally
            if context is not None and self.img_cross_attention:
                k_ip, v_ip = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (k_ip, v_ip))
                sim_ip =  torch.einsum('b i d, b j d -> b i j', q, k_ip) * self.scale
                del k_ip
                sim_ip = sim_ip.softmax(dim=-1)
                out_ip = torch.einsum('b i j, b j d -> b i d', sim_ip, v_ip)
                out_ip = rearrange(out_ip, '(b h) n d -> b n (h d)', h=h)
                out = out + self.image_cross_attention_scale * out_ip
            del q

            return self.to_out(out)

        return forward
    
    num_temporal_attention = 0
    cur_temporal_name = None
    for name, layer in model.named_modules():
        if isinstance(layer, TemporalTransformer):
            cur_temporal_name = name
            if "input_blocks" in name:
                place_in_unet = "down"
            elif "init_attn" in name:
                place_in_unet = "init"
            elif "middle_block" in name:
                place_in_unet = "mid"
            elif "output_blocks" in name:
                place_in_unet = "up"
            else:
                raise ValueError("Unknown position of attention in UNet!")
                
        if (cur_temporal_name is not None and name.startswith(cur_temporal_name)) and (
            name.endswith("transformer_blocks.0.attn1") or name.endswith("transformer_blocks.0.attn2")): 
                #print(place_in_unet)
                layer.forward = ca_forward(layer, place_in_unet)
                num_temporal_attention += 1
    controller.num_att_layers = num_temporal_attention
    print("Total attention layers: ", num_temporal_attention)


class AttentionControl(abc.ABC):

    def step_callback(self, x_t):
        return x_t

    def between_steps(self):
        return

    @property
    def num_uncond_att_layers(self):
        return 0

    @abc.abstractmethod
    def forward(self, attn, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, attn, place_in_unet: str):
        attn = self.forward(attn, place_in_unet)
        return attn

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0


class AttentionStore(AttentionControl):
    @staticmethod
    def get_empty_store():
        return {"init":[], "down": [], "mid": [], "up": []}

    def forward(self, attn, place_in_unet: str):
        key = f"{place_in_unet}"
        #if attn.shape[1] <= ((self.max_size) ** 2 )* self.image_ratio:  # avoid memory overhead
        self.step_store[key].append(attn)
        self.attention_counter += 1
        if self.attention_counter == self.num_att_layers:
            self.between_steps()
        return attn

    def between_steps(self):
        self.attention_store = self.step_store
        self.step_store = self.get_empty_store()
        self.attention_counter = 0
        self.all_time_step += 1
        #print(self.all_time_step, self.cur_time_step)
        if self.all_time_step % 2 ==0: # Not saving unconditional attention maps
            return

        self.cur_time_step += 1

        if self.cur_time_step in self.keep_timestep_list or \
                (self.save_maps and self.cur_time_step in self.save_timestep_list):
            cpu_store = move_to(self.attention_store, device='cpu')
            if self.cur_time_step in self.keep_timestep_list:
                self.keep_timestep_dict[self.cur_time_step] = cpu_store

            if self.save_maps and self.cur_time_step in self.save_timestep_list:
                self.save_attention_local(cpu_store, self.cur_time_step)
                self.save_attention_local_res(cpu_store, self.cur_time_step)

            del cpu_store


    def get_average_attention(self):
        average_attention = {key: [item for item in self.step_store[key]] for key in self.step_store}
        return average_attention
            
    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}
        self.keep_timestep_dict = {}
        self.all_time_step = 0
        self.cur_time_step = 0
        self.attention_counter = 0

    def set_save_dir(self, save_dir):
        self.save_dir = save_dir
        os.makedirs(self.save_dir,exist_ok=True)
        parent = os.path.dirname(save_dir)
        self.separate_attention_dir = os.path.join(parent, 'separate_attention')
        os.makedirs(self.separate_attention_dir, exist_ok=True)
        if self.save_npy:
            self.npy_dir = os.path.join(parent,'z_npy')
            os.makedirs(self.npy_dir, exist_ok=True)

    def save_attention_local(self, cpu_store, chosen_timestep):
        visualize_list = ['init', 'down', 'mid', 'up']
        visualize_res_dict = {
            "init": [64],
            "down": [64, 32, 16],
            "mid": [8],
            "up": [64, 32, 16],
        }
        full_title = f"Timestep {chosen_timestep}"
        if self.save_npy:
            npz_dict = {}

        for k in visualize_list:
            attention_image_list = []
            image_titles = []

            for res in visualize_res_dict[k]:
                vis_width = [res]
                vis_keys = [k]  # ['down', 'mid', 'up'] #["init"]
                summed_sa = aggregate_attention_dict(
                    cpu_store, vis_width=vis_width, vis_keys=vis_keys,
                    vis_all=False, required_width=64, required_height=40
                )
                summed_sa = summed_sa.mean(0)
                summed_sa_norm = min_max_norm(summed_sa)
                mask = summed_sa_norm.cpu().numpy()
                if self.save_npy:
                    npz_dict[f"{k}_{res}"] = mask

                # Visualization
                image = vis_attention_colorcap(mask,output_res=256,color_mode="jet")
                save_dir_cur = os.path.join(self.separate_attention_dir, f't{chosen_timestep}-{k}-{res}.png')
                image.save(save_dir_cur)
                attention_image_list.append(image)
                image_titles.append(f"{k} - {res}")
            save_dir_cur = os.path.join(self.save_dir, f't{chosen_timestep}-{k}.jpg')
            _ = show_images(attention_image_list, titles=image_titles, full_title=full_title,
                            show_image=False, save_dir=save_dir_cur, verbose=False)
        if self.save_npy:
            npy_name = os.path.join(self.npy_dir, f'{chosen_timestep}.npz')
            np.savez(npy_name, **npz_dict)


    def save_attention_local_res(self, cpu_store, chosen_timestep):
        visualize_list = ['down', 'mid', 'up']
        visualize_res_list = [64,32,16,8]
        if self.save_npy:
            npz_dict = {}

        for vis_width in visualize_res_list:
            summed_sa = aggregate_attention_dict(
                cpu_store, vis_width=[vis_width], vis_keys=visualize_list,
                vis_all=False, required_width=64, required_height=40
            )
            summed_sa = summed_sa.mean(0)
            summed_sa_norm = min_max_norm(summed_sa)
            mask = summed_sa_norm.cpu().numpy()
            if self.save_npy:
                npz_dict[f"{vis_width}"] = mask

            # Visualization
            image = vis_attention_colorcap(mask,output_res=256,color_mode="jet")
            save_dir_cur = os.path.join(self.separate_attention_dir, f't{chosen_timestep}-{vis_width}.png')
            image.save(save_dir_cur)

        # _ = show_images(attention_image_list, titles=image_titles, full_title=full_title,
        #                 show_image=False, save_dir=save_dir_cur, verbose=False)
        if self.save_npy:
            npy_name = os.path.join(self.npy_dir, f'all_{chosen_timestep}.npz')
            np.savez(npy_name, **npz_dict)



    def __init__(self, base_width=64, base_height=40,
                 keep_timestep_list=None, save_timestep_list=None,
                 save_maps=False,save_npy=False,):
        super(AttentionStore, self).__init__()
        self.base_width = base_width
        self.base_height = base_height
        self.save_timestep_list = default(save_timestep_list,[])
        self.keep_timestep_list = default(keep_timestep_list,[])

        self.step_store = self.get_empty_store()
        self.keep_timestep_dict = {}
        self.attention_store = {}
        self.attention_counter = 0
        self.cur_time_step = 0
        self.all_time_step = 0
        self.num_att_layers = 34
        self.save_maps = save_maps
        self.save_npy = save_npy
        print("Keep attention maps at: ", self.keep_timestep_list)
        if self.save_maps:
            print("Store attention maps at: ", self.save_timestep_list)



#######################################################
# Matrix Related
######################################################
# https://math.stackexchange.com/questions/1392491/measure-of-how-much-diagonal-a-matrix-is
def cal_diagonalness(A):
    d = A.shape[0]
    j = np.ones(d)
    n = j.dot(A.dot(j.T))
    r = np.arange(d) + 1
    r2 = r ** 2
    Sx = r.dot(A.dot(j.T))
    Sy = j.dot(A.dot(r.T))
    Sx2 = r2.dot(A.dot(j.T))
    Sy2 = j.dot(A.dot(r2.T))
    Sxy = r.dot(A.dot(r.T))
    r = (n*Sxy - Sx*Sy) / (((n*Sx2 - Sx ** 2) ** 0.5) * ((n*Sy2 - Sy ** 2) ** 0.5))
    return r


def create_diag_offset_matrix(frames, std, mean=0.0) -> np.ndarray:
    dia_matrix = np.zeros((frames, frames))
    dia_value = stats.norm.pdf([0], mean, std)[0]
    np.fill_diagonal(dia_matrix, (dia_value))
    dia_matrix_new = dia_matrix

    for i in range(frames - 1):
        cur_id = i + 1
        offset_value = stats.norm.pdf([cur_id], mean, std)[0]
        offset_matrix = np.array([offset_value] * (frames - cur_id))
        up_a = np.diag(offset_matrix, cur_id)
        down_a = np.diag(offset_matrix, int(-1 * cur_id))
        dia_matrix_new = dia_matrix_new + up_a + down_a

    return dia_matrix_new

def create_diag_matrix(frames) -> np.ndarray:
    dia_matrix = np.zeros((frames, frames))
    np.fill_diagonal(dia_matrix, (1.0))
    return dia_matrix

def visualize_diag_offset(frames, std, apply_norm=True,mean=0.0,verbose=False, vis_res=256, return_matrix=True) -> PIL.Image.Image:
    new_matrix = create_diag_offset_matrix(frames,std, mean)
    if verbose:
        print("diagonalness: ", cal_diagonalness(new_matrix) )

    if apply_norm:
        mask = min_max_norm(new_matrix)
    else:
        mask = new_matrix
    image = vis_attention_colorcap(mask, output_res=vis_res, color_mode="jet")
    if return_matrix:
        return image, new_matrix
    else:
        return image
