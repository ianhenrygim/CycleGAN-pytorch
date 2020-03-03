from torchvision.utils import save_image
from parser import CustomParser
from model import CycleGAN
from data import get_dataloader
import os
import torch

opt = CustomParser().get_parser()
opt.isTrain = False

dataloader = get_dataloader(opt)
model = CycleGAN(opt)
model.load_networks(epoch=200)

for i, batch in enumerate(dataloader):
    model.test(batch)
    images = model.get_current_images()

    for key, value in images.items():
        save_image((value.data + 1.0) * 0.5, os.path.join(opt.output_path, f'{key}_{i+1}_.png'))

        