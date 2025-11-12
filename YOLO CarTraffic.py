from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
import cv2

videodata = cv2.VideoCapture("Datasets/cartraffic03.mp4")
model = YOLO("yolov8n.pt")

colors = {
    "person": (0, 0, 255),
    "bicycle": (0, 128, 255),
    "car": (225, 0, 0),
    "motorcycle": (0, 255, 128),
    "airplane": (128, 0, 255),
    "bus": (0, 200, 200),
    "train": (200, 100, 0),
    "truck": (0, 150, 0),
    "boat": (128, 128, 0),
    "traffic light": (255, 128, 0),
    "fire hydrant": (128, 64, 0),
    "stop sign": (200, 0, 128),
    "parking meter": (0, 64, 128),
    "bench": (64, 0, 128),
    "bird": (255, 0, 128),
    "cat": (128, 0, 64),
    "dog": (0, 128, 64),
    "horse": (64, 128, 0),
    "sheep": (128, 128, 64),
    "cow": (64, 64, 192),
    "elephant": (0, 64, 255),
    "bear": (64, 0, 255),
    "zebra": (255, 64, 64),
    "giraffe": (255, 128, 192),
    "backpack": (192, 64, 128),
    "umbrella": (128, 192, 64),
    "handbag": (64, 192, 128),
    "tie": (192, 128, 64),
    "suitcase": (0, 192, 64),
    "frisbee": (0, 64, 192),
    "skis": (192, 0, 64),
    "snowboard": (64, 192, 0),
    "sports ball": (192, 0, 192),
    "kite": (0, 192, 192),
    "baseball bat": (128, 64, 192),
    "baseball glove": (64, 128, 192),
    "skateboard": (192, 64, 64),
    "surfboard": (64, 192, 64),
    "tennis racket": (192, 192, 0),
    "bottle": (64, 64, 128),
    "wine glass": (128, 64, 64),
    "cup": (64, 128, 128),
    "fork": (128, 0, 0),
    "knife": (0, 128, 0),
    "spoon": (0, 0, 128),
    "bowl": (128, 128, 0),
    "banana": (255, 200, 0),
    "apple": (200, 50, 50),
    "sandwich": (150, 75, 0),
    "orange": (255, 100, 0),
    "broccoli": (0, 150, 50),
    "carrot": (255, 140, 0),
    "hot dog": (200, 100, 50),
    "pizza": (200, 50, 0),
    "donut": (150, 0, 150),
    "cake": (255, 100, 150),
    "chair": (100, 150, 200),
    "sofa": (150, 100, 200),
    "potted plant": (0, 200, 100),
    "bed": (100, 0, 200),
    "dining table": (200, 100, 150),
    "toilet": (100, 200, 150),
    "tv": (50, 100, 200),
    "laptop": (100, 50, 150),
    "mouse": (150, 50, 100),
    "remote": (50, 150, 50),
    "keyboard": (50, 150, 150),
    "cell phone": (150, 150, 50),
    "microwave": (100, 100, 50),
    "oven": (100, 50, 100),
    "toaster": (50, 100, 100),
    "sink": (75, 125, 175),
    "refrigerator": (125, 75, 175),
    "book": (175, 125, 75),
    "clock": (125, 175, 75),
    "vase": (75, 175, 125),
    "scissors": (175, 75, 125),
    "teddy bear": (125, 75, 125),
    "hair drier": (75, 125, 75),
    "toothbrush": (125, 125, 125)
}

while videodata.isOpened():
    success, frame = videodata.read()
    if success == True:
        h, w = frame.shape[:2]
        aspect_ratio = w / h
        new_h = int(aspect_ratio * 640)
        new_w = 640
        frame = cv2.resize(frame, (new_h, new_w))

        results = model.track(frame)
        boxes = results[0].boxes.xyxy
        names = results[0].names
        annotator = Annotator(frame, line_width=2, font_size=1)
        classes = results[0].boxes.cls

        for box, cls in zip(boxes, classes):
            x1, y1, x2, y2 = box
            label = names[int(cls)]
            color = colors.get(label, (255, 255, 255))
            annotator.box_label((x1, y1, x2, y2), label, color)
       
        cv2.imshow("YOLO",frame)
        if cv2.waitKey(1) == 13:
            break