import torch

def update_code_dim(num_filters_in_final_layer, img_size, num_conv_layers):
    """
    Getting c_dim in the code

    """
    c_dim = [num_filters_in_final_layer, img_size // 2**num_conv_layers, img_size // 2**num_conv_layers]
    return c_dim

def idx2onehot(idx, n):

    assert torch.max(idx).item() < n
    if idx.dim() == 1:
        idx = idx.unsqueeze(1)

    onehot = torch.zeros(idx.size(0), n)
    onehot.scatter_(1, idx, 1)

    return onehot
