# Pip install method (recommended)
!pip install ultralytics==8.0.28

from IPython import display
display.clear_output()
import ultralytics
ultralytics.checks()

from ultralytics import YOLO
from IPython.display import display, Image

%cd {HOME}
!yolo task=segment mode=predict model=yolov8s-seg.pt conf=0.25 source='https://media.roboflow.com/notebooks/examples/dog.jpeg' save=true

%cd {HOME}
Image(filename='runs/segment/predict/dog.jpeg', height=600)

#Python SDK
model = YOLO(f'{HOME}/yolov8s-seg.pt')
results = model.predict(source='https://media.roboflow.com/notebooks/examples/dog.jpeg', conf=0.25)
results[0].boxes.xyxy
results[0].boxes.conf
ROOT_PATH = "/content/drive/MyDrive/My Courses/Yolov8-seg"
%cd "/content/drive/MyDrive/My Courses/Yolov8-seg"


#custom training
!yolo task=segment mode=train model=yolov8s-seg.pt data=data.yaml epochs=100 imgsz=640 save=true
!ls runs/segment/train/
Image(filename=f'runs/segment/train/results.png', width=600)
Image(filename=f'runs/segment/train/train_batch0.jpg', width=600)


#Validate Custom Model
!yolo task=segment mode=val model=runs/segment/train/weights/best.pt data=data.yaml

#Inference
!yolo task=segment mode=predict model=runs/segment/train/weights/best.pt conf=0.25 source=cell_data/test/images save=true

import glob
from IPython.display import Image, display

for image_path in glob.glob(f'runs/segment/predict/*.jpg')[:3]:
      display(Image(filename=image_path, height=600))
      print("\n")