import cv2
import numpy as np
from gui_buttons import Buttons

# Initialize Buttons

button = Buttons()
button.add_button("person", 20, 20)
button.add_button("cell phone", 20, 100)
button.add_button("keyboard", 20, 180)
button.add_button("remote", 20, 260)


net = cv2.dnn.readNet("dnn_model/yolov4-tiny.weights", "dnn_model/yolov4-tiny.cfg")
wts_path = "dnn_model/yolov4-tiny.weights"
cfg_path = "dnn_model/yolov4-tiny.cfg"
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1288)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


def click_button(event, x, y, flags, params):
    global button_person
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, y)
        button.button_click(x, y)

        # Create window


cv2.namedWindow("Frame")

cv2.setMouseCallback("Frame", click_button)

model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(416, 416), scale=1 / 255)
classes = []
with open("dnn_model/classes.txt", "r") as file_object:
    for class_name in file_object.readlines():
        class_name = class_name.strip()
        classes.append(class_name)

while True:
    ret, frame = cap.read()
    (class_ids, scores, bboxes) = model.detect(frame)
    active_buttons = button.active_buttons_list()

    for class_id, score, bbox in zip(class_ids, scores, bboxes):
        (x, y, w, h) = bbox
        class_name = classes[class_id]
        if class_name in active_buttons:
            cv2.putText(frame, class_name, (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 2, (200, 0, 50), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (200, 0, 50), 3)
    button.display_buttons(frame)
    print("Class ids : ", class_ids)
    print("score :", scores)
    print("bboxes", bboxes)
    cv2.imshow("Frame", frame)
    cv2.waitKey(1)
