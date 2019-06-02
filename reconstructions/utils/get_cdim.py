def update_code_dim(num_filters_in_final_layer, img_size, num_conv_layers):
    """
    Getting c_dim in the code

    """
    c_dim = [num_filters_in_final_layer, img_size // 2**num_conv_layers, img_size // 2**num_conv_layers]
    return c_dim
