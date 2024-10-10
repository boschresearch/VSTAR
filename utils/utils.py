import os
import importlib
import numpy as np
import cv2
import torch
import torch.distributed as dist


def encode_attribute_multiple(model, attribute_list, num_frames, interpolation_mode="linear", indices_list=None):
    num_attribute = len(attribute_list)
    embeddings = model.get_learned_conditioning(attribute_list)  # (num_attribute,77,1024)
    embedding_list = []
    if indices_list is None:
        indices = torch.linspace(0, num_frames - 1, num_attribute, dtype=torch.int8)
    else:
        indices = torch.tensor(indices_list).to(torch.int8)
    print(indices.cpu().numpy())
    for i in range(num_attribute - 1):
        interval = indices[i + 1] - indices[i] + 1
        print('Interval is ', interval.item())
        pos_embedding = embeddings[i:i + 1, :, :].repeat(interval, 1, 1)
        neg_embedding = embeddings[i + 1:i + 2, :, :].repeat(interval, 1, 1)
        beta = torch.linspace(0, 1.0, interval, device=embeddings.device).view(interval, 1, 1)

        if interpolation_mode == "linear":
            mixed_embeddings = pos_embedding * (1.0 - beta) + neg_embedding * beta  # (16, 77, 1024)
        elif interpolation_mode == "same":
            mixed_embeddings = pos_embedding
        else:
            raise ValueError("interpolation mode is not supported!")
        if i != num_attribute - 2:
            embedding_list.append(mixed_embeddings[:-1])
        else:
            embedding_list.append(mixed_embeddings)
        # print(beta, embedding_list[-1].shape)

    mixed_embeddings = torch.cat(embedding_list, dim=0)
    print(mixed_embeddings.shape)
    assert mixed_embeddings.shape[0] == num_frames
    return mixed_embeddings

def count_params(model, verbose=False):
    total_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"{model.__class__.__name__} has {total_params*1.e-6:.2f} M params.")
    return total_params


def check_istarget(name, para_list):
    """ 
    name: full name of source para
    para_list: partial name of target para 
    """
    istarget=False
    for para in para_list:
        if para in name:
            return True
    return istarget


def instantiate_from_config(config):
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def load_npz_from_dir(data_dir):
    data = [np.load(os.path.join(data_dir, data_name))['arr_0'] for data_name in os.listdir(data_dir)]
    data = np.concatenate(data, axis=0)
    return data


def load_npz_from_paths(data_paths):
    data = [np.load(data_path)['arr_0'] for data_path in data_paths]
    data = np.concatenate(data, axis=0)
    return data   


def resize_numpy_image(image, max_resolution=512 * 512, resize_short_edge=None):
    h, w = image.shape[:2]
    if resize_short_edge is not None:
        k = resize_short_edge / min(h, w)
    else:
        k = max_resolution / (h * w)
        k = k**0.5
    h = int(np.round(h * k / 64)) * 64
    w = int(np.round(w * k / 64)) * 64
    image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LANCZOS4)
    return image


def setup_dist(args):
    if dist.is_initialized():
        return
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(
        'nccl',
        init_method='env://'
    )