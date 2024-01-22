import os
import torch
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from decord import VideoReader
from decord import cpu
from uniformer import uniformer_small
from kinetics_class_index import kinetics_classnames
from transforms import (
    GroupNormalize, GroupScale, GroupCenterCrop, 
    Stack, ToTorchFormatTensor
)

import gradio as gr
from huggingface_hub import hf_hub_download


def get_index(num_frames, num_segments=16, dense_sample_rate=8):
    sample_range = num_segments * dense_sample_rate
    sample_pos = max(1, 1 + num_frames - sample_range)
    t_stride = dense_sample_rate
    start_idx = 0 if sample_pos == 1 else sample_pos // 2
    offsets = np.array([
        (idx * t_stride + start_idx) %
        num_frames for idx in range(num_segments)
    ])
    return offsets + 1


def load_video(video_path):
    vr = VideoReader(video_path, ctx=cpu(0))
    num_frames = len(vr)
    frame_indices = get_index(num_frames, 16, 16)

    # transform
    crop_size = 224
    scale_size = 256
    input_mean = [0.485, 0.456, 0.406]
    input_std = [0.229, 0.224, 0.225]

    transform = T.Compose([
        GroupScale(int(scale_size)),
        GroupCenterCrop(crop_size),
        Stack(),
        ToTorchFormatTensor(),
        GroupNormalize(input_mean, input_std) 
    ])

    images_group = list()
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy())
        images_group.append(img)
    torch_imgs = transform(images_group)
    
    # The model expects inputs of shape: B x C x T x H x W
    TC, H, W = torch_imgs.shape
    torch_imgs = torch_imgs.reshape(1, TC//3, 3, H, W).permute(0, 2, 1, 3, 4)

    return torch_imgs
    

def inference(video):
    vid = load_video(video)
    
    prediction = model(vid)
    prediction = F.softmax(prediction, dim=1).flatten()

    return {kinetics_id_to_classname[str(i)]: float(prediction[i]) for i in range(400)}
    

# Device on which to run the model
# Set to cuda to load on GPU
device = "cpu"
model_path = hf_hub_download(repo_id="Sense-X/uniformer_video", filename="uniformer_small_k400_16x8.pth")
# Pick a pretrained model 
model = uniformer_small()
state_dict = torch.load(model_path, map_location='cpu')
model.load_state_dict(state_dict)

# Set to eval mode and move to desired device
model = model.to(device)
model = model.eval()

# Create an id to label name mapping
kinetics_id_to_classname = {}
for k, v in kinetics_classnames.items():
    kinetics_id_to_classname[k] = v

inputs = gr.inputs.Video()
label = gr.outputs.Label(num_top_classes=5)

title = "UniFormer-S"
description = "Gradio demo for UniFormer: To use it, simply upload your video, or click one of the examples to load them. Read more at the links below."
article = "<p style='text-align: center'><a href='https://arxiv.org/abs/2201.04676' target='_blank'>[ICLR2022] UniFormer: Unified Transformer for Efficient Spatiotemporal Representation Learning</a> | <a href='https://github.com/Sense-X/UniFormer' target='_blank'>Github Repo</a></p>"

gr.Interface(
    inference, inputs, outputs=label, 
    title=title, description=description, article=article, 
    examples=[['hitting_baseball.mp4'], ['hoverboarding.mp4'], ['yoga.mp4']]
    ).launch(enable_queue=True)
