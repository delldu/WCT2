"""Image/Video Photo Style Package."""  # coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2022(18588220928@163.com) All Rights Reserved.
# ***
# ***    File Author: Dell, 2022年 09月 08日 星期四 01:39:22 CST
# ***
# ************************************************************************************/
#

__version__ = "1.0.0"

import os
from tqdm import tqdm
import torch
import torch.nn.functional as F

import todos
from . import photo_style

import pdb


def get_model():
    """Create model."""

    model_path = "models/image_photo_style.pth"
    cdir = os.path.dirname(__file__)
    checkpoint = model_path if cdir == "" else cdir + "/" + model_path

    model = photo_style.WCT2()
    todos.model.load(model, checkpoint)

    device = todos.model.get_device()
    model = model.to(device)
    model.eval()

    print(f"Running on {device} ...")
    model = torch.jit.script(model)

    todos.data.mkdir("output")
    if not os.path.exists("output/image_photo_style.torch"):
        model.save("output/image_photo_style.torch")

    return model, device


def model_forward(model, device, content_tensor, style_tensor, multi_times=8):
    # zeropad for model
    H, W = content_tensor.size(2), content_tensor.size(3)
    if H % multi_times != 0 or W % multi_times != 0:
        content_tensor = todos.data.zeropad_tensor(content_tensor, times=multi_times)

    output_tensor = todos.model.two_forward(model, device, content_tensor, style_tensor)

    return output_tensor[:, :, 0:H, 0:W]


def image_predict(input_files, style_file, output_dir):
    # Create directory to store result
    todos.data.mkdir(output_dir)

    # load model
    model, device = get_model()

    # load files
    image_filenames = todos.data.load_files(input_files)
    style_tensor = todos.data.load_tensor(style_file)

    # start predict
    progress_bar = tqdm(total=len(image_filenames))
    for filename in image_filenames:
        progress_bar.update(1)

        # orig input
        content_tensor = todos.data.load_tensor(filename)
        B, C, H, W = content_tensor.shape

        predict_tensor = model_forward(model, device, content_tensor, style_tensor)
        output_file = f"{output_dir}/{os.path.basename(filename)}"

        SB, SC, SH, SW = style_tensor.shape
        if SH != H or SW != W:
            style_tensor = F.interpolate(style_tensor, size=(H, W), mode="bilinear", align_corners=False)
        todos.data.save_tensor([content_tensor, style_tensor, predict_tensor], output_file)
