from PIL import Image
import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import cv2
import time
import imageio
# import pydensecrf.densecrf as dcrf
import torchvision.transforms as transforms
# from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral, create_pairwise_gaussian
from Networks.net import MODEL as net
from skimage import morphology

# print(torch.cuda.current_device())
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# ids = torch.cuda.device_count()
device = torch.device('cuda:0')       # CUDA:0


model = net(in_channel=2)

model_path = "./models/model.pth"
use_gpu = torch.cuda.is_available()

if use_gpu:
    print('GPU Mode Acitavted')
    model = model.cuda()
    model.cuda()

    model.load_state_dict(torch.load(model_path))
    print(model)
else:
    print('CPU Mode Acitavted')
    state_dict = torch.load(model_path, map_location='cpu')

    model.load_state_dict(state_dict)

def fusion_gray():

    path1 = './ir/IR.bmp'

    path2 = './vi/VIS.bmp'
    img1 = Image.open(path1).convert('L')
    img2 = Image.open(path2).convert('L')

    img1_read = np.array(img1)
    img2_read = np.array(img2)
    h = img1_read.shape[0]
    w = img1_read.shape[1]

    img1_org = img1
    img2_org = img2

    tran = transforms.ToTensor()

    img_a = tran(img1_org)
    img_b = tran(img2_org)

    window_size = 8

    h_pad = (h // window_size + 1) * window_size - h
    w_pad = (w // window_size + 1) * window_size - w

    img_a = torch.cat([img_a, torch.flip(img_a, [1])], 1)[:, :h + h_pad, :]
    img_a = torch.cat([img_a, torch.flip(img_a, [2])], 2)[:, :, :w + w_pad]
    img_b = torch.cat([img_b, torch.flip(img_b, [1])], 1)[:, :h + h_pad, :]
    img_b = torch.cat([img_b, torch.flip(img_b, [2])], 2)[:, :, :w + w_pad]

    input_img = torch.cat((img_a, img_b), 0).unsqueeze(0)
    if use_gpu:
        input_img = input_img.cuda()
    else:
        input_img = input_img

    model.eval()
    out = model(input_img)

    out = out[..., :h * 1, :w * 1]

    out= np.squeeze(out.detach().cpu().numpy())


    out = (out * 255).astype(np.uint8)

    imageio.imwrite('./result/result.bmp', out)



if __name__ == '__main__':

    fusion_gray()
