from src.helper import read_list_from_file
import os
import torch
def load_everything(path_to_save_folder,train_run_label):
    path_full = os.path.join(path_to_save_folder,train_run_label)
    losses = read_list_from_file("losses",path_full)
    step_losses= read_list_from_file("step_losses",path_full)
    perplexities = read_list_from_file("perplexities",path_full)
    all_perplex = read_list_from_file("all_perplex",path_full)
    model = model = torch.load(path_full+"/"+"model_full", weights_only=False)
    return (losses, step_losses, perplexities,all_perplex,model)
