from __future__ import print_function
import os
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import cv2
import time
from face_detection.face_detection import Face
from models.mobilenetv3 import mobilenetv3
from torchvision import transforms
from PIL import Image

transform = transforms.Compose([
    transforms.Resize(96),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

def test_inference(model, bgr_image):
    #img = Image.open(image_path).convert('RGB')
    img = Image.fromarray(cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB))
    img = img.resize((96, 96), Image.ANTIALIAS)
    img = transform(img)
    img = torch.unsqueeze(img, 0)
    img = img.cuda().float()

    output = model(img)
    softmax_output = torch.softmax(output, dim=-1)

    mask_prob = softmax_output[0][1]
    nomask_porb = softmax_output[0][0]
    return mask_prob, nomask_porb


def main():
    # load model
    faceboxes = Face()
    
    checkpoint = torch.load('./checkpoint/mask_detection.pth.tar')
    model = mobilenetv3().cuda()
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    model.cuda()

    image_path = './images/test.jpg'
    img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)
    im_height, im_width, _ = img_raw.shape

    start_time = time.time()
    boxes = faceboxes.face_detection(img_raw)

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

        # mask model inference
        mask_prob, nomask_porb = test_inference(model, cropped)
        
        if mask_prob >= 0.5:
            show_text = "mask {:.2f}".format(mask_prob)
            rectangle_color = (0, 255, 0)
        else:
            show_text = "unmask {:.2f}".format(nomask_porb)
            rectangle_color = (0, 0, 255)

        cv2.rectangle(img_raw, (x1, y1), (x2, y2), rectangle_color, 2)

        cv2.putText(img_raw, show_text, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, rectangle_color)

    used_time = time.time() - start_time
    print('used_time: ', used_time)

    cv2.imshow('res', img_raw)
    cv2.waitKey(0)

if __name__ == '__main__':
    main()