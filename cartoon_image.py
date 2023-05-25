#!/usr/bin/env python
# coding: utf-8

# In[12]:


import torch.nn as nn
import torch.nn.functional as F
from torch import sigmoid

class ResidualBlock(nn.Module):
  def __init__(self):
    super(ResidualBlock, self).__init__()
    self.conv_1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
    self.conv_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
    self.norm_1 = nn.BatchNorm2d(256)
    self.norm_2 = nn.BatchNorm2d(256)

  def forward(self, x):
    output = self.norm_2(self.conv_2(F.relu(self.norm_1(self.conv_1(x)))))
    return output + x #ES

class Generator(nn.Module):
    def __init__(self):
      super(Generator, self).__init__()
      self.conv_1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=1, padding=3)
      self.norm_1 = nn.BatchNorm2d(64)
      
      # down-convolution #
      self.conv_2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)
      self.conv_3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
      self.norm_2 = nn.BatchNorm2d(128)
      
      self.conv_4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1)
      self.conv_5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
      self.norm_3 = nn.BatchNorm2d(256)
      
      # residual blocks #
      residualBlocks = []
      for l in range(8):
        residualBlocks.append(ResidualBlock())
      self.res = nn.Sequential(*residualBlocks)
      
      # up-convolution #
      self.conv_6 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1)
      self.conv_7 = nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
      self.norm_4 = nn.BatchNorm2d(128)

      self.conv_8 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1)
      self.conv_9 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
      self.norm_5 = nn.BatchNorm2d(64)
      
      self.conv_10 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=7, stride=1, padding=3)

    def forward(self, x):
      x = F.relu(self.norm_1(self.conv_1(x)))
      
      x = F.relu(self.norm_2(self.conv_3(self.conv_2(x))))
      x = F.relu(self.norm_3(self.conv_5(self.conv_4(x))))
      
      x = self.res(x)
      x = F.relu(self.norm_4(self.conv_7(self.conv_6(x))))
      x = F.relu(self.norm_5(self.conv_9(self.conv_8(x))))

      x = self.conv_10(x)

      x = sigmoid(x)

      return x








import sys
import requests
from io import BytesIO
import torch
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

class ArtifyGAN:
    def __init__(self, path):
        self.path = path

    def generator(self):
        G = Generator()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        file_path = '../data/checkpoint_epoch079.pth'
        path1 = os.path.abspath(os.path.dirname(__file__))
        modelpath = os.path.join(path1, "data/checkpoint_epoch079.pth")
        
        checkpoint = torch.load(modelpath, map_location=device)
        G_inference1 = Generator()
        G_inference1.load_state_dict(checkpoint['g_state_dict'])

        return G_inference1

    def load_image(self, path):
        try:
            img = cv2.imread(path)
            # Convert to BGR format (assuming original image is in RGB format)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            # Resize the image to 256x256
            img = cv2.resize(img, (256, 256))

            # Convert the image to a tensor with shape (1, 256, 256, 3)
            img_tensor = np.expand_dims(img, axis=0)
            img_tensor = np.transpose(img_tensor, (0, 3, 1, 2))
            img_tensor = torch.from_numpy(img_tensor).float()
            return img_tensor
        
        except FileNotFoundError:
            print("Image not found. Please provide a valid image path.")
            sys.exit(1)

    def generate_cartoon(self, img_tensor, G_inference1):
        result_images = G_inference1(img_tensor)
        plt.figure(figsize=(4, 5))
        plt.axis('off')
        output_image= np.transpose(result_images[0].detach().numpy(), (1, 2, 0))
        output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)  # Convert image from RGB to BGR

        cv2.imshow("Generated Cartoon Image", output_image)
        cv2.waitKey()
        cv2.destroyWindow("Generated Cartoon Image")

    def generate_cartoon_image(self):
        G_inference1 = self.generator()

        if G_inference1 is None:
            print("Failed to load generator weights. Please check the generator model.")
            sys.exit(1)

        img_tensor = self.load_image(self.path)
        self.generate_cartoon(img_tensor, G_inference1)





if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Please provide the image path as a command-line argument.")
        sys.exit(1)

    image_path = sys.argv[1]
    artify_gan = ArtifyGAN(image_path)
    artify_gan.generate_cartoon_image()




