from __future__ import print_function
import os
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import cv2
from face_detection.face_detection import Face
import time

def main():
    image_path = './images/test.jpeg'
    img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)
    im_height, im_width, _ = img_raw.shape
    
    faceboxes = Face()
    start_time = time.time()
    boxes = faceboxes.face_detection(img_raw)
    used_time = time.time() - start_time
    print('used_time: ', used_time)
    vis_thres = 0.9
    # show image
    for b in boxes:
        x1, y1, x2, y2, score = int(b[0]), int(b[1]), int(b[2]), int(b[3]), b[4]
        if score < vis_thres:
            continue
        
        w = x2 - x1 + 1
        h = y2 - y1 + 1

        size = int(max([w, h])* 1.0)
        cx = x1 + w//2
        cy = y1 + h//2
        x1 = cx - size//2
        x2 = x1 + size
        y1 = cy - size//2
        y2 = y1 + size

        dx = max(0, -x1)
        dy = max(0, -y1)
        x1 = max(0, x1)
        y1 = max(0, y1)

        edx = max(0, x2 - im_width)
        edy = max(0, y2 - im_height)
        x2 = min(im_width, x2)
        y2 = min(im_height, y2)
        cropped = img_raw[y1:y2, x1:x2]
        if (dx > 0 or dy > 0 or edx > 0 or edy > 0):
            print('copyMakeBorder')
            print('dy, edy, dx, edx: ', dy, edy, dx, edx)
            cropped = cv2.copyMakeBorder(cropped, dy, edy, dx, edx, cv2.BORDER_CONSTANT, value=[0,0,0])
        cropped = cv2.resize(cropped, (96, 96))
        #cv2.imshow('cropped', cropped)
        #cv2.waitKey(0)

        text = "{:.4f}".format(score)
        cv2.rectangle(img_raw, (x1, y1), (x2, y2), (0, 0, 255), 2)

        cv2.putText(img_raw, text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
    cv2.imshow('res', img_raw)
    cv2.waitKey(0)

if __name__ == '__main__':
    main()