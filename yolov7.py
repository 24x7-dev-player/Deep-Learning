# Download YOLOv7 repository and install requirements
!git clone https://github.com/WongKinYiu/yolov7
%cd yolov7
!pip install -r requirements.txt


%cd /content
!curl -L "https://github.com/entbappy/Branching-tutorial/raw/master/data_yolov7.zip" > data_yolov7.zip; unzip data_yolov7.zip; rm data_yolov7.zip


#Prepare Image Path
import os
train_img_path = "/content/images/train"
val_img_path = "/content/images/val"
%cd /content

#Training images
with open('train.txt', "a+") as f:
  img_list = os.listdir(train_img_path)
  for img in img_list:
    f.write(os.path.join(train_img_path,img+'\n'))
  print("Done")
  
# Validation Image
with open('val.txt', "a+") as f:
  img_list = os.listdir(val_img_path)
  for img in img_list:
    f.write(os.path.join(val_img_path,img+'\n'))
  print("Done")
%cp /content/yolov7/data/coco.yaml /content/yolov7/data/custom.yaml


# download COCO starting checkpoint
%cd /content/yolov7
!wget "https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt"
%cp /content/yolov7/cfg/training/yolov7.yaml /content/yolov7/cfg/training/custom_yolov7.yaml


#training
!python train.py --batch 16 --cfg cfg/training/custom_yolov7.yaml --epochs 50 --data /content/yolov7/data/custom.yaml --weights 'yolov7.pt' --device 0

#inference
# Run
!python detect.py --weights /content/yolov7/runs/train/exp/weights/best.pt  --source /content/yolov7/Hello.jpg

#display inference on ALL test images

import glob
from IPython.display import Image, display

i = 0
limit = 10000 # max images to print
for imageName in glob.glob('/content/yolov7/runs/detect/exp/*.jpg'): #assuming JPG
    if i < limit:
      display(Image(filename=imageName))
      print("\n")
    i = i + 1