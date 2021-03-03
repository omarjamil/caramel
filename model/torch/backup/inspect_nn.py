import numpy as np
import torch

def trace_out(model_state, query_node):
    """
    Trace the output node back throught the neural network
    back to input via maximum weights
    """
    print("query node: {0}".format(query_node))
    layers_keys = list(model_state.keys())[::-1] # Reverse to have out first
    max_node = None
    flayer = None
    weight_sorted_nodes = None
    for layer in layers_keys:
        print(model_state[layer].shape)
        if "weight" in layer:
            weights = model_state[layer].data.numpy()
            if max_node is not None:
                weight_sorted_nodes = np.argsort(weights[:])[::-1]
                max_node = weight_sorted_nodes[:3]
            else:
                weight_sorted_nodes = np.argsort(weights[:])[::-1]
                max_node = weight_sorted_nodes[:3]
            flayer = layer      
    # print(flayer,max_node)

if __name__ == "__main__":
    model = "/project/spice/radiation/ML/CRM/data/models/torch/q_qadv_t_tadv_swtoa_lhf_shf_qtphys_009_lyr_183_in_090_out_0210_hdn_030_epch_01000_btch_021501AQ_mae_levs.tar"
    model = torch.load(model, map_location=torch.device('cpu'))
    model_state = model['model_state_dict']
    for i in range(1):
        query_node = i
        trace_out(model_state, query_node) 
