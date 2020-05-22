# predict
import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from unet import UNet
from utils.data_vis import plot_img_and_mask
from utils.dataset import BasicDataset


def predict_img(net, full_img, device,  out_threshold=0.5):
    net.eval()
    img = torch.from_numpy(BasicDataset.preprocess(full_img))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        dat = enumerate(testLoader).next()
        depth, mask = net(img) #depth, mask
        probs1 = torch.sigmoid(depth)
        probs2 = torch.sigmoid(mask)
        probs1 = probs1.squeeze(0)
        probs2 = probs2.squeeze(0)
        tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(full_img.size[1]),
                transforms.ToTensor()
            ]
        )

        probs1 = tf(probs1.cpu())
        probs2 = tf(probs2.cpu())
        full_depth = probs1.squeeze().cpu().numpy()
        full_mask = probs2.squeeze().cpu().numpy()
    full_depth = full_depth - np.min(full_depth)
    full_depth = full_depth / np.max(full_depth)
    return full_depth , full_mask# > out_threshold


def get_output_filenames(args):
    in_files = args.input
    out_files = []

    if not args.output:
        for f in in_files:
            pathsplit = os.path.splitext(f)
            out_files.append("{}_OUT{}".format(pathsplit[0], pathsplit[1]))
    elif len(in_files) != len(args.output):
        logging.error("Input files and output files are not of the same length")
        raise SystemExit()
    else:
        out_files = args.output

    return out_files


def mask_to_image(mask):
    return Image.fromarray((mask * 255).astype(np.uint8))

