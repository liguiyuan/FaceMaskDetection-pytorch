# FaceMaskDetection-pytorch
face mask detection, including face detection and mask recognition



## 1.dataset

Here we mainly use dataset `RMFD` and   Baidu open dataset. In total 14000s for training, and 3500s for testing. 

RMFD dataset url: https://github.com/X-zhangyang/Real-World-Masked-Face-Dataset



### 2.training

Running script:

```bash
sh train.sh
```



results:

|                   | Parameters | Accuracy |
| ----------------- | ---------- | -------- |
| MobileNetv3 large | 12M        | 99.0%    |
| MobileNetv3 small | 3.7M       | 98.6%    |



### 3.Deploy

We use `ncnn` to deploy it on Android devices.

TODO!

