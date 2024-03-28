# Anime Super-Resolution and Restoration

This repository is a collection of existing Anime Super-Resolution and Restoration models for the convenience of researchers and engineers who want to restore anime images. This repo is a non-accelerated version of my Fast Anime VSR (https://github.com/Kiteretsu77/FAST_Anime_VSR/tree/main).
**This repo is continuously developing! Feel free to leave your suggestions!**\
:star: If you like Anime_SR_Restoration, please help star this repo. Thanks! :hugs:



## :book:Table Of Contents
- [Update](#update)
- [Supported Model](#support)
- [Installation](#installation)
- [Inference](#inference)


## <a name="update"></a>Update
- **2023.12.09**: This repo is released.


## <a name="support"></a> Model supported now:
1. **Real-CUGAN**:   The original model weight provided by BiliBili (from https://github.com/bilibili/ailab/tree/main)
2. **Real-ESRGAN**:  Using Anime version RRDB with 6 Blocks (full model has 23 blocks) (from https://github.com/xinntao/Real-ESRGAN/blob/master/docs/model_zoo.md#for-anime-images--illustrations)
3. **VCISR**:        A model I trained with my paper using a private Anime training datasets (https://github.com/Kiteretsu77/VCISR-official)



## <a name="installation"></a> Installation (Environment Preparation)

```shell
git clone git@github.com:Kiteretsu77/Anime_SR_Restoration.git
cd Anime_SR_Restoration

# Create conda env
conda create -n ASRR python=3.10
conda activate ASRR

# Install Pytorch we use:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install Other packages:
pip install -r requirements.txt

```


## <a name="inference"></a> Inference:
1. Setup opt.py for the input and output path
2. Execute the following:
    ```shell
    python inference.py
    ```






## License
This project is released under the [GPL 3.0 license](LICENSE).

## Contact
If you have any questions, please feel free to contact with me at hikaridawn412316@gmail.com.

