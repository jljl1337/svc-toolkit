import torch

def get_available_device():
    device_list = []

    if torch.cuda.is_available():
        device_list.append(('GPU', 'cuda'))

    elif torch.backends.mps.is_available():
        device_list.append(('MPS', 'mps'))

    device_list.append(('CPU', 'cpu'))

    return device_list