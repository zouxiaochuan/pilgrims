from Pyro5.nameserver import NameServer
import Pyro5.api as api
import json
import sys
import os
import importlib
import torch


def get_nameserver(host, port) -> NameServer:
    return api.locate_ns(host=host, port=port)
    pass


def chunk_parameter(total_size, num_chunks, idx):
    chunk_size = total_size // num_chunks
    start_index = idx * chunk_size
    end_index = (idx + 1) * chunk_size
    if end_index > total_size:
        end_index = total_size
        pass
    return start_index, end_index


def load_config(config_file):
    with open(config_file, 'r') as f:
        config = json.load(f)
        pass
    return config
    pass


def relative_import_module_and_get_class(module_path, module_name, class_name):
    parent_path = os.path.dirname(os.path.abspath(module_path))
    if parent_path not in sys.path:
        sys.path.append(parent_path)
        pass
    relative_module_name = os.path.basename(module_path) + '.' + module_name
    module = importlib.import_module(relative_module_name)
    cls = getattr(module, class_name)
    return cls
    pass

def batch_to_device(batch, device):
    b = dict()

    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            b[k] = v.to(device)
            pass
        elif isinstance(v, dict):
            b[k] = batch_to_device(v, device)
            pass
        else:
            b[k] = v
            pass
        pass
    return b
    pass


def masked_nd_assign(tensor, mask, dim, value):
    mask1 = mask[:, :, None].expand(tensor.shape)
    mask2 = torch.zeros(tensor.shape[-1], device=tensor.device, dtype=torch.bool)
    mask2[dim] = True
    mask2 = mask2.expand(tensor.shape)

    tensor[torch.logical_and(mask1, mask2)] = value
    return tensor


def multi_masked_assign(tensor, masks, value):
    
    masks = [m.expand(tensor.shape) for m in masks]
    mask = torch.all(torch.stack(masks, dim=0), dim=0)

    tensor[mask] = value

    pass
