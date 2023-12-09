# -*- coding: utf-8 -*-
import os

opt = {}

# File Setting
opt['input_path'] = ""      # The path to the input folder/images (can detect automatically)
opt['store_path'] = ""     # The path to stored the super-resolved images/videos
opt['model'] = ""           # Model name:  Bicubic | Anime4K | Real-CUGAN | Real-ESRGAN | 
opt['scale'] = 4            # The scaling factor needed   (Supported scale factor to each model: )


# Miscellaneous Setting




# GPU Setting
opt['CUDA_VISIBLE_DEVICES'] = '0'           #   '0/1'
os.environ['CUDA_VISIBLE_DEVICES'] = opt['CUDA_VISIBLE_DEVICES']  


