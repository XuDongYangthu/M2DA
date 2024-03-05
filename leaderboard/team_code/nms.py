import numpy as np

def apply_nms(detected_boxes, iou_threshold):
    if len(detected_boxes) == 0:
        return []
    
    boxes = sorted(detected_boxes, key=lambda x: x[2], reverse=True)
    selected_boxes = []
    
    while boxes:
        current_box = boxes.pop(0)
        selected_boxes.append(current_box)
        
        boxes = [box for box in boxes if iou(box, current_box) < iou_threshold]
    
    return selected_boxes

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    unionArea = boxAArea + boxBArea - interArea
    
    iou = interArea / float(unionArea)
    return iou

def find_peak_box_with_nms(data, iou_threshold=0.1):
    det_data = np.zeros((22, 22, 7))
    det_data[1:21, 1:21] = data
    det_data[19:21, 1:21, 0] -= 0.1
    detected_boxes = []
    
    for i in range(1, 21):
        for j in range(1, 21):
            if det_data[i, j, 0] > 0.9 or (
                det_data[i, j, 0] > 0.4
                and all(det_data[i, j, 0] > det_data[i + di, j + dj, 0] for di in range(-1, 2) for dj in range(-1, 2) if not (di == 0 and dj == 0))
            ):
                box = [
                    i + det_data[i, j, 1] * 3.5,  # Recalculate x-coordinate of the left top corner
                    j + det_data[i, j, 2] * 3.5,  # Recalculate y-coordinate of the left top corner
                    i + det_data[i, j, 1] * 3.5 + det_data[i, j, 4] * 3.5,  # Recalculate x-coordinate of the right bottom corner
                    j + det_data[i, j, 2] * 3.5 + det_data[i, j, 5] * 2.0   # Recalculate y-coordinate of the right bottom corner
                ]
                detected_boxes.append((box[0], box[1], box[2], box[3], det_data[i, j, 0], i-1, j-1))
    
    # nms
    boxes_after_nms = apply_nms(detected_boxes, iou_threshold)
    res = []
    for i in boxes_after_nms:
        tmp = i[5:7]
        res.append(tmp)
    box_info = {"car": [], "bike": [], "pedestrian": []}
    for instance in res:
        i, j = instance
        box = np.array(det_data[i + 1, j + 1, 4:6])
        if box[0] > 2.0:
            box_info["car"].append((i, j))
        elif box[0] / box[1] > 1.5:
            box_info["bike"].append((i, j))
        else:
            box_info["pedestrian"].append((i, j))
    return res, box_info    