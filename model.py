import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
  def __init__(self,dim = 64 , bn = True):

    super(ResidualBlock,self).__init__()
    if bn:
      self.model = nn.Sequential(nn.Conv2d(dim,dim,3,1,padding=1),nn.BatchNorm2d(dim),nn.LeakyReLU()
                               ,nn.Conv2d(dim,dim,3,1,padding=1),nn.BatchNorm2d(dim),nn.LeakyReLU())
    else:
      self.model = nn.Sequential(nn.Conv2d(dim,dim,3,1,padding=1),nn.BatchNorm2d(dim)
                               ,nn.Conv2d(dim,dim,3,1,padding=1),nn.BatchNorm2d(dim))
      
  def forward(self,x):
    return x + self.model(x)


    

class Generator(nn.Module):
  def __init__(self,in_cn = 3,bn = True, n_layers = 16):
    

    super(Generator,self).__init__()
    self.conv1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=9, stride=1, padding=4),nn.ReLU())
    modules = []
    for i in range(n_layers):
      modules.append(ResidualBlock(64))
    self.res_blocks = nn.Sequential(*modules)

    modules1 = []
    modules1.append(nn.Conv2d(64,64,3,1,padding=1))
    if bn:
      modules1.append(nn.BatchNorm2d(64))
    self.conv2 = nn.Sequential(*modules1)

    self.conv3 = nn.Sequential(nn.Conv2d(64,256,3,1,padding=1),nn.PixelShuffle(2),nn.ReLU())
    self.conv4 = nn.Sequential(nn.Conv2d(64,256,3,1,padding=1),nn.PixelShuffle(2),nn.ReLU())
    self.conv5 = nn.Sequential(nn.Conv2d(64, 3, kernel_size=9, stride=1, padding=4), nn.Tanh())

  def forward(self,x):
    y = self.conv1(x)
    z = self.res_blocks(y)
    z = self.conv2(z)
    x = torch.add(z,y)
    x = self.conv3(x)
    x = self.conv4(x)
    x = self.conv5(x)
    return x



class Discriminator(nn.Module):
  def __init__(self,in_cn = 3,bn = 'batch_norm' ):
    super(Discriminator,self).__init__()
    dim = [64,64,128,128,256,256,512,512]
    shape = [32,32,16,16,8,8,4]
    self.conv1 = nn.Sequential(nn.Conv2d(3,64,3,1,padding=1),nn.LeakyReLU())
    modules = []
    for i in range(len(dim)-1):
      
      modules.append(nn.Conv2d(dim[i],dim[i+1],3,2-(i%2),padding=1))
      if bn == 'batch_norm':
        modules.append(nn.BatchNorm2d(dim[i+1]))
      elif bn == 'instance_norm':
        modules.append(nn.InstanceNorm2d(dim[i+1]))
      elif bn == 'layer_norm':
        modules.append(nn.InstanceNorm2d(shape[i]))
      modules.append(nn.LeakyReLU())
    modules.append(nn.Conv2d(512,1,4,4,padding=0))
    self.conv2 = nn.Sequential(*modules)
    
  def forward(self,x):
    x = self.conv1(x)
    x = self.conv2(x)
    return x


