import sys, os
import torch

# Import files from the local folder
root_path = os.path.abspath('.')
sys.path.append(root_path)
from opt import opt


def load_model(model_name):
    return



if __name__ == "__main__":
    input_path = opt['input_path']
    store_path = opt['store_path']
    model = opt['model']
    scale = opt['scale']

