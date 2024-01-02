import sys, os, cv2, shutil
from tqdm import tqdm
from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter
from moviepy.editor import VideoFileClip
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
    


def process_video(SR_instance, input_path, store_path, rescale_factor=1):
    ''' Super-Resolve single video file
    Args:
        SR_instance (object):       The instance object for the Super Resolution Class
        input_path (str):           The input path to video file
        store_path (str):           The store path
        rescale_factor (int):       This is used for the case of rescale if this is set
    '''
    # Default setting
    encode_params = ['-crf', '23', '-preset', 'medium', '-tune', 'animation'] 
    
    # Read the video path
    objVideoReader = VideoFileClip(filename=input_path)
    scale = opt['scale']
    
    # Obtain basic video information
    width, height = objVideoReader.reader.size
    original_fps = objVideoReader.reader.fps
    nframes = objVideoReader.reader.nframes
    has_audio = objVideoReader.audio
    
    # Create a tmp file
    if os.path.exists("tmp"):
        shutil.rmtree("tmp")
    os.makedirs("tmp")
    if os.path.exists(store_path):
        os.remove(store_path)
    
    
    # Create a video writer
    output_size = (width * scale * rescale_factor, height * scale * rescale_factor)
    if has_audio:
        objVideoReader.audio.write_audiofile("tmp/output_audio.mp3")    # Hopefully, mp3 format is supported for all input video 
        writer = FFMPEG_VideoWriter(store_path, output_size, original_fps, ffmpeg_params=encode_params, audiofile="tmp/output_audio.mp3")
    else:
        writer = FFMPEG_VideoWriter(store_path, output_size, original_fps, ffmpeg_params=encode_params)
    
    
    # Setup Progress bar
    progress_bar = tqdm(range(0, nframes), initial=0, desc="Frame",)
    
    
    # Iterate frames from the video and super-resolve individually
    for idx, frame in enumerate(objVideoReader.iter_frames(fps=original_fps)):
        
        # Rescale the video frame at the beginning if we want a different output resolution
        if rescale_factor != 1:
            frame = cv2.resize(frame, (int(width*rescale_factor), int(height*rescale_factor))) # interpolation=cv2.INTER_LANCZOS4
        
        # Inference
        img_SR = SR_instance(frame)
        
        # Write into the frame
        writer.write_frame(img_SR)
        
        progress_bar.update(1)

    writer.close()
        



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
            process_video(SR_instance, input_path, store_path)
        else:
            raise NotImplementedError("This single image input format is not what we support!")
    
    else: # If the input path is a folder
        # Check if the output format is correct
        print("We will recursively read and process all files in this folder")

    
    print("Finish processing all input files!")