import torch


def minmax_tensor(scaler, input_tensor):
    """
    Minmax scaling based on scikit-learn minmaxscaler
    """
    # scale = (range_max - range_min) / (torch.max(input_tensor,0)[0] - torch.min(input_tensor,0)[0])
    range_min, range_max = scaler['feature_range'][:]
    
    scale = torch.from_numpy(scaler['scale_'][:])
    data_min = torch.from_numpy(scaler['data_min_'][:])
    output_tensor = scale * input_tensor + range_min - data_min * scale #torch.min(input_tensor, 0)[0] * scale
    return output_tensor

def inverse_minmax_tensor(scaler, input_tensor):
    """
    Inverse min max scaled tensor
    """
    range_min, range_max = scaler['feature_range'][:] 
    scale = torch.from_numpy(scaler['scale_'][:])
    data_min = torch.from_numpy(scaler['data_min_'][:])
    output_tensor = (input_tensor - range_min)/scale + data_min 
    return output_tensor

