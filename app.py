import cv2, threading
import tensorflow as tf
from ultralytics import YOLO
import matplotlib.pyplot as plt

model = YOLO("./models/banman-yolov8s-orig-02.pt", task="detect")

# class_names = {
#     "class0": 
# }

results = model(
    "./test-images/joshua-olsen-G6elMNRLFew-unsplash.jpg",
    imgsz=224
)
for result in results:
    img = result.plot()
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()
