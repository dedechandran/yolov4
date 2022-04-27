import cv2
import numpy as np
import base64 

CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.3

LABELS = open("app/yolo/obj.names").read().strip().split("\n")

np.random.seed(4)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")

model = cv2.dnn.readNet("app/yolo/yolov4-custom_best.weights","app/yolo/yolov4-custom.cfg")

def detect_object(img):
    height, width = img.shape[:2]
    layer_names = model.getLayerNames()
    layer_names = [layer_names[i - 1] for i in model.getUnconnectedOutLayers()]
    blob = cv2.dnn.blobFromImage(img,1 / 255.0, (416, 416), swapRB=True, crop=False)
    model.setInput(blob)
    outs = model.forward(layer_names)

    class_ids=[]
    confidences=[]
    boxes=[]

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > CONFIDENCE_THRESHOLD:
                centerX = int(detection[0] * width)
                centerY = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(centerX - (w / 2))
                y = int(centerY - (h / 2))
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, int(w), int(h)])

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD,
	NMS_THRESHOLD)

    result = []
    # ensure at least one detection exists
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            color = [int(c) for c in COLORS[class_ids[i]]]
            label = LABELS[class_ids[i]]
            confidence = confidences[i]

            detected_object = {
                "color": color,
                "label": label,
                "confidence": confidence,
                "boundingBox": {
                    "xAxis": x,
                    "yAxis": y,
                    "width": w,
                    "height": h
                }
            }
            result.append(detected_object)
    print(result)
    return result


