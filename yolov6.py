# clone YOLOv6 repository
!git clone https://github.com/meituan/YOLOv6
%cd YOLOv6
!pip install -r requirements.txt


#custom data
%cd /content
!curl -L "https://github.com/entbappy/Branching-tutorial/raw/master/data_yolov6.zip" > data_yolov6.zip; unzip data_yolov6.zip; rm data_yolov6.zip


# Downaod pretrain weights
%cd /content/YOLOv6
# !wget https://github.com/meituan/YOLOv6/releases/download/0.1.0/yolov6s.pt
!wget https://github.com/meituan/YOLOv6/releases/download/0.4.0/yolov6s.pt


#train
%cd YOLOv6
!python tools/train.py --batch 16 --conf configs/yolov6s_finetune.py --data /content/dataset.yaml --device 0 --epochs 50


#evaluation
!python tools/eval.py --data /content/dataset.yaml  --weights runs/train/exp1/weights/best_ckpt.pt --device 0

#inference
from IPython.display import Image
Image(filename = "/content/Hello.jpg", width=1000)

!python tools/infer.py --weights runs/train/exp1/weights/best_ckpt.pt --source /content/Hello.jpg --device 0 --yaml /content/dataset.yaml

from IPython.display import Image
Image(filename = "/content/YOLOv6/runs/inference/exp/Hello.jpg", width=1000)