# FaceMaskDetection-pytorch
face mask detection, including face detection and mask recognition



## 1.dataset

Here we mainly use dataset `RMFD` and   Baidu open dataset. In total 14000s for training, and 3500s for testing. 

RMFD dataset url: https://github.com/X-zhangyang/Real-World-Masked-Face-Dataset



## 2.training

Running script:

```bash
sh train.sh
```



Model is efficient and high-accuracy, with real-time speed on the CPU devices.

|                   | Parameters | Accuracy |
| ----------------- | ---------- | -------- |
| MobileNetv3 large | 12M        | 99.0%    |
| MobileNetv3 small | 3.8M       | 98.6%    |

model download:

链接： https://pan.baidu.com/s/1fP9q4nELXm6dQQirbPy6cA 提取码：kjkx



examples:

![image](https://github.com/liguiyuan/FaceMaskDetection-pytorch/blob/master/images/test_result.jpg)



## 3.Deploy

We use `ncnn` to deploy it on Android devices.

convert pytorch model to onnx model:

```bash
cd tools
python pytorch2onnx.py

# simpllifier
python -m onnxsim face_mask.onnx face_mask_sim.onnx
```



convert onnx to ncnn:

```bash
./onnx2ncnn face_mask_sim.onnx face_mask_sim.param face_mask_sim.bin
```



## 4.References

https://github.com/zisianw/FaceBoxes.PyTorch

https://github.com/kuan-wang/pytorch-mobilenet-v3

https://github.com/X-zhangyang/Real-World-Masked-Face-Dataset

