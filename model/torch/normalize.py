

def inverse_minmax(input_vals, scale, feature_min, feature_max, data_min):
    """
    Inverse min max scaled tensor
    """
    range_min, range_max = feature_min, feature_max 
    output_tensor = (input_vals - range_min)/scale + data_min 
    return output_tensor

def minmax(input_vals, scale, feature_min, feature_max, data_min):
    """
    Minmax scaling based on scikit-learn minmaxscaler
    """
    # scale = (range_max - range_min) / (torch.max(input_tensor,0)[0] - torch.min(input_tensor,0)[0])
    range_min, range_max = feature_min, feature_max
    output_tensor = scale * input_vals + range_min - data_min * scale #torch.min(input_tensor, 0)[0] * scale
    return output_tensor