from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2

model = YOLO("./models/banman-yolov8s-orig-01.pt")

results = model("./test-images/Raw_Banana.8142f28b-ebcb-11ed-830f-346f24e2fa38.jpg")

for result in results:
    img = result.plot()
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()
