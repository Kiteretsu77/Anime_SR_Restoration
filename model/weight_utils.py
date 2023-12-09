import os, shutil, cv2, requests


weight_path_url = {
    "Real-CuGAN" : "https://drive.google.com/u/0/uc?id=1hc1Xh_1qBkU4iGzWxkThpUa5_W9t7GZ_&export=download",
    "Real-ESRGAN" : "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
    "VCISR": "https://drive.google.com/u/0/uc?id=1y59iJ86xujt3990tzyyBAq73VcPKJ3ri&export=download",
}


def check_weight_path(weight_path, model_name):
    ''' Check if the needed model weight exists in the local directory; Else, download one and put in the designed position
    Args:
        weight_path (str):      The local directory where we need to store the path
        model_name (str):       The model name
    '''
    
    # Check if the needed weight path exists in the local directory
    if os.path.exists(weight_path):
        return  
    
    # The path doesn't exist in the local folder, we will download one here
    if model_name not in weight_path_url:
        raise NotImplementedError("We don't have record for the pretrained weight of "+model_name+" now.")
    
    
    url = weight_path_url[model_name]
    r = requests.get(url, allow_redirects=True)
    open(weight_path, 'wb').write(r.content)
    print("Finish downloading pretrained weight for "+model_name+"!")