import glob
import argparse
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as utils
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import grad
from torch import autograd
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim.lr_scheduler import StepLR


import numpy as np
import matplotlib.pyplot as plt
import random

from model import Generator,Discriminator
from Dataloader import get_dataloader
from utils import matplotlib_imshow



class WGAN_GP():
    def __init__(self,generator , discriminator , Dataloader , Dataloader_val, critic_iters =5 ,lr = 5e-4, gamma = 10, use_cuda = True , logdir = "None" ):
        self.G_Net = generator
        self.D_Net = discriminator
        self.G_optim = optim.Adam(self.G_Net.parameters(), lr=lr, betas=(0.5, 0.9))
        self.D_optim = optim.Adam(self.D_Net.parameters(), lr=lr, betas=(0.5, 0.9))
        self.Dataloader = Dataloader
        self.d_scheduler = StepLR(self.D_optim, step_size=1, gamma=0.99)
        self.g_scheduler = StepLR(self.G_optim, step_size=1, gamma=0.99)
        self.Dataloader_val = Dataloader_val
        self.fixed_validation_noise = torch.rand(64,3,16,16)

        self.critic_iters = critic_iters
        self.gamma = gamma
        self.use_cuda = use_cuda
        self.device = 'cpu'

        self.fixed_vector_1 = torch.rand(36,3,16,16)

        self.writer = SummaryWriter(logdir)
        self.total_epochs = 0
        self.steps = 0

        if self.use_cuda:
            self.G_Net.cuda()
            self.D_Net.cuda()

            self.fixed_vector_1 = self.fixed_vector_1.cuda()
            self.fixed_validation_noise = self.fixed_validation_noise.cuda()
            self.device = 'cuda'

      def train(self, n_epochs , main_path):
          for epoch in range(1,n_epochs + 1):
            print('Starting epoch {}...'.format(epoch) )
            print('Time - ',datetime.now())
            self.train_epoch(epoch,n_epochs)
            self.d_scheduler.step()
            self.g_scheduler.step()
            self.total_epochs += 1
            if epoch % 5 == 0 or epoch == n_epochs:
              torch.save(self.G_Net.state_dict(), main_path+ 'results/' + 'WGAN'+'_generator_{}.pt'.format(self.total_epochs))
              torch.save(self.D_Net.state_dict(), main_path+ 'results/' +'WGAN'+'_discriminator_{}.pt'.format(self.total_epochs))

      def train_epoch(self,epoch,n_epochs):
          for i, (data,_) in enumerate(self.Dataloader):

            for p in self.D_Net.parameters():
              p.requires_grad = True
            self.D_Net.zero_grad()

            real_cpu = data.to(self.device)
            b_size = real_cpu.size(0)
            generated_data = self.G_Net(self._sample_(b_size))
            grad_penalty = self.gradient_penalty(real_cpu,generated_data)
            d_loss = self.D_Net(generated_data).mean()-self.D_Net(real_cpu).mean()+ grad_penalty
            d_loss.backward()
            self.D_optim.step()
            W_D = -1*d_loss + grad_penalty

            if i % self.critic_iters == 0 and i != 0:
              
              for p in self.D_Net.parameters():
                p.requires_grad = False
              self.G_Net.zero_grad()

              generated_data = self.G_Net(self._sample_(b_size))
              g_loss = -self.D_Net(generated_data).mean()
              g_loss.backward()
              self.G_optim.step()

            if i % 500 == 0 and i!=0 :
              print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f \t Wassertian Distance: %.4f'% (epoch, n_epochs, i, len(self.Dataloader),d_loss.item(), g_loss.item(), W_D.item()))

              self.writer.add_scalar('Discriminator training loss ',d_loss.item(),self.steps)
              self.writer.add_scalar('Generator training loss ',g_loss.item(),self.steps)
              self.writer.add_scalar('Wassertian Distance ', W_D.item(),self.steps)

              self.writer.add_figure('Training 1',matplotlib_imshow(self.G_Net(self.fixed_vector_1).detach()),self.steps)
              self.validate(epoch)
              self.steps+=1
              
      def validate(self,epoch,n_epochs):
          with torch.no_grad():
            
            discrimnator_value = 0.0
            W_D_overall = 0.0
            for i,(data,_) in enumerate(self.Dataloader_val):
              real_cpu = data.to(self.device)
              b_size = real_cpu.size(0)
              generated_data = self.G_Net(self._sample_(b_size))
              d_val = self.D_Net(real_cpu).mean()
              
              d_loss = self.D_Net(generated_data).mean()- d_val 
              W_D_overall += (-1*d_loss ).item()
              discrimnator_value += d_loss.item()
            
            generated_data = self.G_Net(self.fixed_validation_noise)
            g_loss = -self.D_Net(generated_data).mean()
            discrimnator_value/= len(self.Dataloader_val)
            W_D_overall /= len(self.Dataloader_val)

            
            print('[%d/%d][%d]\tValLoss_D: %.4f\tValLoss_G: %.4f \t Val Wassertian Distance: %.4f'% (epoch, n_epochs, len(self.Dataloader_val),discrimnator_value, g_loss.item(), W_D_overall))
            
            self.writer.add_scalar('Discriminator Validation loss ',discrimnator_value,self.steps)
            self.writer.add_scalar('Generator Validation loss ', g_loss.item(), self.steps)
            self.writer.add_scalar('Wassertian Distance Validation ', W_D_overall, self.steps)

              
      def gradient_penalty(self, data, generated_data, gamma=10):
          batch_size = data.size(0)
          epsilon = torch.rand(batch_size, 1, 1, 1)
          epsilon = epsilon.expand_as(data)

          if self.use_cuda:
            epsilon = epsilon.cuda()

          interpolation = epsilon * data.data + (1 - epsilon) * generated_data.data
          interpolation.requires_grad=True

          if self.use_cuda:
            interpolation = interpolation.cuda()

          interpolation_logits = self.D_Net(interpolation)
          grad_outputs = torch.ones(interpolation_logits.size())

          if self.use_cuda:
            grad_outputs = grad_outputs.cuda()

          gradients = autograd.grad(outputs=interpolation_logits, inputs=interpolation, grad_outputs=grad_outputs, create_graph=True, retain_graph=True)[0]
          gradients = gradients.view(batch_size, -1)
          gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
          return self.gamma * ((gradients_norm - 1) ** 2).mean()

      def _sample_(self,n_samples):
          z = torch.rand(n_samples,3,16,16)
          if self.use_cuda:
            z = z.cuda()
            return z

    



  











  










