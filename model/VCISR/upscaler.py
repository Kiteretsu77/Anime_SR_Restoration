from torch import nn as nn
import torch
from torch.nn import functional as F
from torchvision.transforms import ToTensor
from torchvision.utils import save_image
import numpy as np
import os, sys, cv2


# Import files from the local folder
root_path = os.path.abspath('.')
sys.path.append(root_path)
from opt import opt
from model.weight_utils import check_weight_path
from model.VCISR.RRDB import RRDBNet



class VCISR_upscaler(object):
    
    def __init__(self, scale, weight_path = "pretrained/2x_VCISR_RRDB.pth"):
        
        # Load the model here
        self.model = RRDBNet(3, 3, scale)
        
        # Read and Clean the weight
        check_weight_path(weight_path, "VCISR")
        checkpoint_g = torch.load(weight_path)
        
        # Load the weight
        self.model.load_state_dict(checkpoint_g['model_state_dict'])
        self.model = self.model.eval().cuda()
        
        # Other setting
        self.model_name = "VCISR"
    
    
    
    def __call__(self, input, store_path=None):
        ''' Super-Resolve the image with input_path
        Args:
            input_path (str/numpy):     The input path or numpy
            store_path (str):           The store path (default: None) [If this is None, we will return the super-resolved result back]
        Returns:
            gen_hr (numpy):     The generated HR image in numpy form
        '''
        
        # Read image and Transform
        if type(input) is str:
            img_lr = cv2.imread(input)
            img_lr = cv2.cvtColor(img_lr, cv2.COLOR_BGR2RGB)
        else:
            img_lr = input
        
        # Automatically do 4x crop
        h, w, _ = img_lr.shape
        if h % 4 != 0:
            img_lr = img_lr[:4*(h//4),:,:]
        if w % 4 != 0:
            img_lr = img_lr[:,:4*(w//4),:]

        # Tensor Transform
        img_lr = ToTensor()(img_lr).unsqueeze(0).cuda()     # Use tensor format

        # Inference
        gen_hr = self.model(img_lr)
        
        
        # Store the generated image
        if store_path is not None:
            save_image(gen_hr, store_path) 
        else:
            return np.uint8(np.transpose( torch.clamp(255.0*gen_hr.squeeze(), 0, 255).cpu().detach().numpy(), (1, 2, 0)))
        
        # Empty the cache every time you finish processing one image
        # torch.cuda.empty_cache() 