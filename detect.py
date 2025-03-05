import time
import cv2
import numpy as np
from threading import Thread
from gtts import gTTS
import os
from pygame import mixer

mixer.init()
print(os.getcwd())

def get_output_layers(net):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers

def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = str(classes[class_id])
    color = COLORS[class_id]
    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)
    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return label, confidence

cap = cv2.VideoCapture(0) # Open the default camera
time.sleep(2) # Wait for the camera to warm up

classes = None

with open("coco.names", 'r') as f:
    classes = [line.strip() for line in f.readlines()]

COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg") # Read the model

# def speak(objects):
#     text = "object found's are "
#     for obj in objects:
#         text += obj[0] + ', '
#     language = "en-US"
#     obj = gTTS(text=text, lang=language, slow=False)
#     obj.save("speak.mp3")
#     mixer.music.load("C:/Users/hp/Downloads/speak.mp3")
#     mixer.music.play()
#     time.sleep(6)
#     os.remove("speak.mp3")

def speak(objects):
    text = "object found's are "
    for obj in objects:
        text += obj[0] + ', '
    language = "en-US"
    obj = gTTS(text=text, lang=language, slow=False)
    obj.save("speak.mp3")
    
    # Acquire a lock before loading and playing the audio
    mixer.music.load("speak.mp3")
    mixer.music.play()
    
    # Wait for the audio playback to finish
    while mixer.music.get_busy():
        time.sleep(0.1)
    
    # Release the lock
    mixer.music.stop()
    mixer.music.unload()
    
    # Acquire the lock before removing the file
    with file_access_lock:
        try:
            os.remove("speak.mp3")
        except OSError as e:
            print("Error while removing file:", e)



while True:
    ret, frame = cap.read() # Read a frame from the camera
    if not ret:
        break

    Width = frame.shape[1]
    Height = frame.shape[0]
    scale = 0.00392

    blob = cv2.dnn.blobFromImage(frame, scale, (416,416), (0,0,0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(get_output_layers(net))

    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4

    object_found = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])
                object_found.append(list(map(str, draw_prediction(frame, class_id, confidence, round(x), round(y), round(x+w), round(y+h)))))

    if object_found:
        Thread(target=speak, args=(object_found,)).start()

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    for i in indices:
        i = int(i)
        box = boxes[i]
        x = round(box[0])
        y = round(box[1])
        w = round(box[2])
        h = round(box[3])
        draw_prediction(frame, class_ids[i], confidences[i], x, y, x+w, y+h)

    cv2.imshow("object detection", frame) # Show the output frame
    if cv2.waitKey(1) & 0xFF == ord('q'): # Exit condition
        os.remove("speak.mp3")
        break

       
cap.release()
cv2.destroyAllWindows()
