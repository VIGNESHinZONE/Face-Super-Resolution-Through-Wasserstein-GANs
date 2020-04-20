
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.utils as utils

def matplotlib_imshow(img,tor = True):
  
  fig = plt.figure(figsize=(12,12))
  plt.axis("off")
  plt.imshow(np.transpose(utils.make_grid(img[:36], padding=2, normalize=True).cpu(),(1,2,0)))
  return fig

def plot_image(low,inter,predicted,high):
  fig, ax = plt.subplots(nrows=4, ncols=1, figsize=(12, 9))
  
  #fig = plt.figure(figsize=(16,16))
  ax[0].axis("off")
  ax[1].axis("off")
  ax[2].axis("off")
  ax[3].axis("off")
  ax[0].title.set_text('Low Resolution 16 x 16')
  ax[1].title.set_text('Bilinear Interpolation 64 x 64')
  ax[2].title.set_text('Predicted Images 64 x 64')
  ax[3].title.set_text('High Resolution 64 x 64')
  
  ax[0].imshow(np.transpose(utils.make_grid(low, padding=1, normalize=True).cpu(),(1,2,0)))
  ax[1].imshow(np.transpose(utils.make_grid(inter, padding=1, normalize=True).cpu(),(1,2,0)))
  ax[2].imshow(np.transpose(utils.make_grid(predicted, padding=1, normalize=True).cpu(),(1,2,0)))
  ax[3].imshow(np.transpose(utils.make_grid(high, padding=1, normalize=True).cpu(),(1,2,0)))
  fig.tight_layout()
  return fig

