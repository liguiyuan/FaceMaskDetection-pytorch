import torch
import onnx
import sys

sys.path.append('../')
from models.mobilenetv3 import mobilenetv3

def main():
    checkpoint = torch.load('../checkpoint/mask_detection_98.pth.tar')
    net = mobilenetv3().cuda()
    net.load_state_dict(checkpoint['state_dict'])
    net.eval()

    dummy_input = torch.randn(1, 3, 96, 96, device='cuda')
    input_names = ['data']
    output_names = ['fc']
    torch.onnx.export(net, dummy_input, 'face_mask.onnx', 
        export_params=True, verbose=True, input_names=input_names, output_names=output_names)

if __name__ == '__main__':
    main()