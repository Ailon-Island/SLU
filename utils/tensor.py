from torch import Tensor


def dict2device(tensor_dict, device):
    for key in tensor_dict:
        if isinstance(tensor_dict[key], Tensor):
            tensor_dict[key] = tensor_dict[key].to(device)
    return tensor_dict



