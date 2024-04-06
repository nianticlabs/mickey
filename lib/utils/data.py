import torch

def data_to_model_device(data, model):
    '''Move all tensors in data dictionary to the same device as model'''

    try:
        device = next(model.parameters()).device
    except:
        # in case the model has no parameters (baseline models)
        device = 'cpu'

    for k, v in data.items():
        if torch.is_tensor(v):
            data[k] = v.to(device)

    return data
