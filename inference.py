import sys, os, cv2
import torch

# Import files from the local folder
root_path = os.path.abspath('.')
sys.path.append(root_path)
from opt import opt
from model.RealESRGAN.upscaler import RealESRGAN_upscaler
from model.RealCuGAN.upscaler import RealCuGAN_upscaler
from model.VCISR.upscaler import VCISR_upscaler

supported_img_extension = ['jpg', 'png']
supported_video_extension = ['mp4']



def load_model(model_name, scale):
    ''' This is a function to load multiple model
    Args:
        model_name (str):       The model name got from opt.py
        scale (int):            The scale factor of Super-Resolution
    Returns:
        SR_instance (obj):      The instance of the class we obtain
    '''
    
    if model_name == "Real-ESRGAN":
        return RealESRGAN_upscaler(scale)
    
    elif model_name == "Real-CuGAN":
        return RealCuGAN_upscaler(scale)
    
    elif model_name == "VCISR":
        return VCISR_upscaler(scale)
    
    else:
        raise NotImplementedError("We don't support such model now")



def process_img(SR_instance, input_path, store_path):
    ''' Super-Resolve single image file
    Args:
        SR_instance (object):       The instance object for the Super Resolution Class
        input_path (str):           The input path
        store_path (str):           The store path
    '''
    
    # Prepare the directory
    if os.path.exists(store_path):
        os.remove(store_path)
    dir_path = os.path.dirname(store_path)
    os.makedirs(dir_path, exist_ok=True)        # Create the parent folder it doesn't exists
    
    # Inference
    SR_instance(input_path, store_path)
    
    print("The processed image is successfully stored in " + store_path)
    


def process_video():
    ''' Super-Resolve single video file
    Args:
        SR_instance (object):       The instance object for the Super Resolution Class
        input_path (str):           The input path
        store_path (str):           The store path
    '''
    return



if __name__ == "__main__":
    
    # Prepare setting
    input_path = opt['input_path']
    store_path = opt['store_path']
    model_name = opt['model_name']
    scale = opt['scale']


    # Init the model    
    SR_instance = load_model(model_name, scale)

    
    # Process the input based on the form
    if not os.path.isdir(input_path):       # Single file input
        input_extension = input_path.split('.')[-1]
        output_extension = store_path.split('.')[-1]
        
        if input_extension in supported_img_extension: # If the input path is single image
            # Check if the output format is correct
            if output_extension not in supported_img_extension:
                raise ValueError('The output format does not match the input format.')
            process_img(SR_instance, input_path, store_path)
        
        elif input_extension in supported_video_extension: # If the input path is single video
            # Check if the output format is correct
            if output_extension not in supported_video_extension:
                raise ValueError('The output format does not match the input format.')

        else:
            raise NotImplementedError("This single image input format is not what we support!")
    
    else: # If the input path is a folder
        # Check if the output format is correct
        print("We will recursively read and process all files in this folder")

    
    print("Finish processing all input files!")