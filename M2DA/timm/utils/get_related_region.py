import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch

# def get_region(raw_image, image):
#     gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#     threshold_value = 100
#     _, binary_image = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY)
#     contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     if contours:
#         max_contour = max(contours, key=cv2.contourArea)

#         # import ipdb; ipdb.set_trace()
#         # 计算坐标的边界框
#         x, y, w, h = cv2.boundingRect(max_contour)

#         cropped_image = raw_image[y:y+h, x:x+w]
#         canvas = np.zeros_like(raw_image.cpu().detach())
#         canvas[y:y+h, x:x+w] = cropped_image.cpu().detach()
#         return canvas
#     return np.zeros_like(raw_image.cpu().detach())


# def get_region(raw_image, image):
#     gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#     threshold_value = 100
#     _, binary_image = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY)
#     contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     if contours:
#         max_contour = max(contours, key=cv2.contourArea)

#         # 计算坐标的边界框
#         x, y, w, h = cv2.boundingRect(max_contour)
#         cropped_image = raw_image[y:y+h, x:x+w]
#         # import ipdb; ipdb.set_trace()
        
#         cropped_image = torch.nn.functional.interpolate(
#             cropped_image.unsqueeze(0).permute(0,3,1,2), size=(128, 128)
#         ).squeeze(0).permute(1,2,0)
        
#         return cropped_image
#     raw_image = torch.nn.functional.interpolate(
#         raw_image.unsqueeze(0).permute(0,3,1,2), size=(128, 128)
#     ).squeeze(0).permute(1,2,0)
#     return raw_image

def get_region(raw_image, image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    threshold = 100

    # 找到大于阈值的像素的坐标
    row_indices, col_indices = np.where(gray_image > threshold)

    if np.any(row_indices) and np.any(col_indices):
        min_row, max_row = np.min(row_indices), np.max(row_indices)
        min_col, max_col = np.min(col_indices), np.max(col_indices)
        
        cropped_image = raw_image[min_row:max_row, min_col:max_col]
        if cropped_image.shape[1] == 0 or cropped_image.shape[0] == 0:
            cropped_image = raw_image
        cropped_image = torch.nn.functional.interpolate(
            cropped_image.unsqueeze(0).permute(0,3,1,2), size=(128, 128)
        ).squeeze(0).permute(1,2,0)
        
        return cropped_image.tolist()
    raw_image = torch.nn.functional.interpolate(
        raw_image.unsqueeze(0).permute(0,3,1,2), size=(128, 128)
    ).squeeze(0).permute(1,2,0)
    return raw_image.tolist()