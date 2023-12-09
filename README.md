# Anime_SR_Restoration

This repository is a collection of existing Anime Super-Resolution and Restoration model for the convenience of researcher and engineers who wants to restore anime images.
:star: If you like Anime_SR_Restoration, please help star this repo. Thanks! :hugs:


## :book:Table Of Contents
- [Update](#update)
- [Installation](#installation)
- [Inference](#inference)


## <a name="update"></a>Update
- **2023.12.09**: This repo is released.


## <a name="installation"></a> Installation (Environment Preparation)

```shell
git clone git@github.com:Kiteretsu77/VCISR-official.git
cd ASRR

# Create conda env
conda create -n ASRR python=3.10
conda activate ASRR

# Install Pytorch we use torch.compile in our repository by default
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

```





## <a name="inference"></a> Inference:
1. Setup opt.py for the input and output path
2. Execute the following:
    ```shell
    python inferece.py
    ```






## License
This project is released under the [GPL 3.0 license](LICENSE).

## Contact
If you have any questions, please feel free to contact with me at hikaridawn412316@gmail.com.

