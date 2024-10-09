import numpy as np
import torch


def load_batch_dict_to_gpu(batch_dict, device):
    for key, val in batch_dict.items():
        if isinstance(val, np.ndarray):
            if key in ['idx', 'name']:
                continue
            batch_dict[key] = ndarray_to_tensor(val, device)
    return batch_dict


def ndarray_to_tensor(data, device):
    if np.issubdtype(data.dtype, np.float64):
        return torch.from_numpy(data).double().to(device)
    elif np.issubdtype(data.dtype, np.float32):
        return torch.from_numpy(data).float().to(device)
    elif np.issubdtype(data.dtype, np.int64):
        return torch.from_numpy(data).long().to(device)
    elif np.issubdtype(data.dtype, np.int32):
        return torch.from_numpy(data).int().to(device)
    else:
        raise NotImplementedError


def tensor_to_ndarray(data):
    if torch.is_tensor(data):
        return data.detach().to('cpu').numpy()
    return data
