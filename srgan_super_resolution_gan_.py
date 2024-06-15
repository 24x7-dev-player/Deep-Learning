# %pwd
!git clone https://github.com/entbappy/SRGAN-Super-Resolution-GAN.git

# Commented out IPython magic to ensure Python compatibility.
# %cd SRGAN-Super-Resolution-GAN

# Commented out IPython magic to ensure Python compatibility.
# %ls

!pip install --upgrade scikit-image

# Commented out IPython magic to ensure Python compatibility.
# %pwd

#test
!python main.py --mode test_only --LR_path test_data --generator_path pretrained_models/SRGAN.pt

#train
!python main.py --LR_path Data/train_LR --GT_path Data/train_HR