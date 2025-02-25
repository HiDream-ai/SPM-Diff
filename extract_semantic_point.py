import argparse
import json
import os
import os.path as osp
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision.transforms import PILToTensor
from tqdm import tqdm
from PIL import Image, ImageDraw
from dift_util import SDFeaturizer
import time

def read_frames_from_folder(frames_folder):
    frames = []
    for filename in sorted(os.listdir(frames_folder)):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            frame_path = os.path.join(frames_folder, filename)
            frame = Image.open(frame_path).convert('RGB')
            frames.append(np.array(frame))
    return np.stack(frames)


def progagate_human_keypoint(file_path):

    with open(file_path, 'r') as file:

        pred_tracks = torch.zeros((len(file.readlines()), 2))

        file.seek(0)
        idx = 0
        for line in file:
            x, y = map(float, line.strip().split(','))
            pred_tracks[idx, 0] = int(x)
            pred_tracks[idx, 1] = int(y)
            idx =idx+1

    tap_dict = {'pred_tracks': pred_tracks.cpu(), }
    return tap_dict



def extract_dift_feature(image, dift_model,out_dir):
    if isinstance(image, Image.Image):
        image = image
    else:
        image = Image.open(image).convert('RGB')
    prompt = f'a photo of an upper body garment'

    img_tensor = (PILToTensor()(image) / 255.0 - 0.5) * 2
    dift_feature = dift_model.forward(img_tensor, prompt=prompt,out_dir = out_dir, ensemble_size=8)

    return dift_feature


def extract_point_embedding(tap_dict, image_path, model_id,out_dir):
    
    num_point = tap_dict['pred_tracks'].shape[0]
    init_embedding = torch.zeros((num_point, 1280))
    init_count = torch.zeros(num_point)

    dift_model = SDFeaturizer(sd_id=model_id)
    
    img = Image.open(image_path).convert('RGB')
    x, y = img.size
    img_dift = extract_dift_feature(img,dift_model=dift_model,out_dir = out_dir)
    img_dift = nn.Upsample(size=(y, x), mode='bilinear')(img_dift)

    point_all = tap_dict['pred_tracks']

    for point_idx, point in enumerate(point_all):
        point_x, point_y = int(np.round(point[0])), int(np.round(point[1]))

        if point_x >= 0 and point_y >= 0:
            point_embedding = img_dift[0, :, point_y//2, point_x//2]
            init_embedding[point_idx] += point_embedding.cpu()
            init_count[point_idx] += 1

    for point_idx in range(init_count.shape[0]):
        if init_count[point_idx] != 0:
            init_embedding[point_idx] /= init_count[point_idx]

    tap_dict['point_embedding'] = init_embedding
    
    return tap_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--save_dir', type=str, default='point_embeds_feature_cloth')
    parser.add_argument('--image_type', type=str, choices=["cloth", "model"])
    parser.add_argument('--dataroot', type=str, default='datasets/vitonhd/')
    parser.add_argument('--phase', type=str, default='test')
    
    args = parser.parse_args()

    folder_path = os.path.join(args.dataroot,args.phase)
    for filename in sorted(os.listdir(os.path.join(folder_path,args.image_type))):

        image_path = os.path.join(folder_path,args.image_type,filename)
        file_path = os.path.join(folder_path, 'point',args.image_type,filename.replace('.jpg','.txt'))
        tap_dict = progagate_human_keypoint(file_path)
        

        tap_dict = extract_point_embedding(
            tap_dict, image_path, model_id='sd1.5',out_dir=args.save_dir)

        torch.save(tap_dict, os.path.join(folder_path, args.save_dir,filename.replace('.jpg','.pth')))
        
        
