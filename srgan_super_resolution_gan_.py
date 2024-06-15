# %pwd
!git clone https://github.com/entbappy/SRGAN-Super-Resolution-GAN.git
# %cd SRGAN-Super-Resolution-GAN

# %ls
!pip install --upgrade scikit-image
# %pwd

#test
!python main.py --mode test_only --LR_path test_data --generator_path pretrained_models/SRGAN.pt

#train
!python main.py --LR_path Data/train_LR --GT_path Data/train_HR