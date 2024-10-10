import math
import torch
from einops import  rearrange
import numpy as np
from typing import List
import matplotlib.pyplot as plt
import PIL
import cv2
from PIL import Image
import os
import torchvision


def min_max_norm(x, min_v=None, max_v=None):
    if min_v is not None and max_v is not None:
        y = (x - min_v) / (max_v - min_v)
    else:
        y = (x - x.min()) / (x.max() - x.min())
    return y

def aggregate_attention(attention_map_list, required_width=64, required_height=40) -> torch.Tensor:
    out = []
    aspect_ratio = required_height / required_width #h / w
    for attn_map in attention_map_list:
        cur_width = int(math.sqrt(attn_map.shape[0] / aspect_ratio))
        cur_height = int(cur_width * aspect_ratio)
        attn_map = rearrange(attn_map, '(h w) k t -> k t h w',h=cur_height)
        attn_map = torch.nn.functional.interpolate(attn_map, size=(required_height, required_width), mode='bilinear')
        attn_map = rearrange(attn_map, 'k t h w -> (h w) k t')
        out.append(attn_map)
    out = torch.cat(out, dim=0)  # [x,16,16]
    return out

def aggregate_attention_dict(attention_map_dict, vis_width:  List[int], vis_keys: List[str],
                             vis_all: bool=False,
                             required_width=64, required_height=40) -> torch.Tensor:
    """
        attention_map_dict: dict of attention maps
        vis_width: width of attention to be visualized, [64, 32, 16, 8]
        vis_keys: a list of attention levels to be visualized, ['init', 'down', 'mid', 'up']
        vis_all: if True, visualize all, ignore vis_width
    """
    out = []
    aspect_ratio = required_height / required_width #h / w
    for k in vis_keys:
        attention_map_list = attention_map_dict[k] # a list of attention
        for attn_map in attention_map_list:
            cur_width = int(math.sqrt(attn_map.shape[0] / aspect_ratio))
            if not vis_all and cur_width not in vis_width:
                continue
            cur_height = int(cur_width * aspect_ratio)
            attn_map = rearrange(attn_map, '(h w) k t -> k t h w',h=cur_height)
            attn_map = torch.nn.functional.interpolate(attn_map, size=(required_height, required_width), mode='bilinear')
            attn_map = rearrange(attn_map, 'k t h w -> (h w) k t')
            out.append(attn_map)
    out = torch.cat(out, dim=0)  # [x,16,16]
    return out

def vis_attention_colorcap(mask, output_res=128,color_mode="jet") -> PIL.Image.Image:
    """
    mask: range(0,1)
    """
    color_map_dict = {
        "jet": cv2.COLORMAP_JET,
        "hot": cv2.COLORMAP_HOT,
    }
    if isinstance(mask, torch.Tensor):
        heatmap = cv2.applyColorMap(np.uint8(255 * mask.cpu().numpy()), color_map_dict[color_mode])
    else:
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), color_map_dict[color_mode])
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(heatmap)
    image = image.resize((output_res, output_res), Image.Resampling.NEAREST)

    return image

def show_images(images, cols=1, titles=None, full_title=None, show_image=False, save_dir=None,verbose=False):
    """Display a list of images in a single figure with matplotlib.

    Parameters
    ---------
    images: List of np.arrays compatible with plt.imshow.

    cols (Default = 1): Number of columns in figure (number of rows is
                        set to np.ceil(n_images/float(cols))).

    titles: List of titles corresponding to each image. Must have
            the same length as titles.
    """
    assert ((titles is None) or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None:
        titles = ['Image %d' % i for i in range(1, n_images + 1)]
    fig = plt.figure(constrained_layout=True)
    for n, (image, title) in enumerate(zip(images, titles)):
        if isinstance(image, PIL.Image.Image):
            image = np.asarray(image)
        a = fig.add_subplot(cols, int(np.ceil(n_images / float(cols))), n + 1)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
        a.set_axis_off()
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    cur_w, cur_h = fig.get_size_inches()
    y_ratio_dict = {
        1: 1.05,
        2: 0.88,
        3: 0.76,
    }
    if full_title is not None:
        plt.suptitle(full_title, fontsize=14, y=y_ratio_dict[n_images])
    # fig.tight_layout() # rect=[0, 0.03, 1, 0.95]
    # plt.subplots_adjust(tosave_image_batchp=0.85)
    if save_dir is not None:
        _ = plt.savefig(
            save_dir,
            bbox_inches='tight', transparent=True, pad_inches=0.01
        )
        if verbose:
            print("Save to: ", save_dir)
    plt.close("all")
    if show_image:
        plt.show()
        
        
def save_image_batch(video_tensor, save_dir, ext_type="png"):
    # video_tensor: torch tensor, (c,t,h,w)
    video = rearrange(video_tensor, 'c t h w -> t h w c')
    video = video.detach().cpu()
    video = torch.clamp(video.float(), -1., 1.)
    video = (video + 1.0) / 2.0
    os.makedirs(save_dir, exist_ok=True)

    for i,frame in enumerate(video):
        f = (frame.numpy() * 255).astype(np.uint8)
        pil_img = Image.fromarray(f)
        img_path = os.path.join(save_dir, f'frame_{i}.{ext_type}')
        pil_img.save(img_path)

def save_image_grid(video: torch.Tensor, path: str, rescale=False, n_rows=8,):
    images = video[0] # (3, 16, h, w)
    images = rearrange(images, "c t h w -> t c h w")
    x = torchvision.utils.make_grid(images, nrow=n_rows)
    x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
    x = (x * 255).to(torch.uint8).cpu().numpy().astype(np.uint8)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    im = Image.fromarray(x)
    im.save(path)