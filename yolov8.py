#Installation
!pip install ultralytics

#Detection
!yolo task=detect mode=predict model=yolov8n.pt source= "/content/dog.jpeg" save=True

#segmentation
!yolo task=segment mode=predict model=yolov8n-seg.pt source= "/content/dog.jpeg" save=True

#classification
!yolo task=classify mode=predict model=yolov8n-cls.pt source= "/content/cat.jpg" save=True

# Python sdk
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.predict("/content/dog.jpeg")