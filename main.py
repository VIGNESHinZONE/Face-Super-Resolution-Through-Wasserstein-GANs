import glob
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import random
from model import Generator,Discriminator
from Dataloader import get_dataloader_resolution
from utils import plot_image
from Face_Resolution_model import WGAN_GP
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',required=False,help='name')
    opt = parser.parse_args()    
    seed = 1234
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)    
    image_size = 64
    batch_size = 64
    n_epochs = 1
    lr = 1e-4
    critic_iters = 5
    gamma = 10
    main_path = "../home/mvenkataraman_ph/Face-Super-Resolution-Through-Wasserstein-GANs/"
    train_loader , validation_loader = get_dataloader_resolution(main_path + "root",image_size,batch_size,seed)
    log_dir = main_path + "runs/logs7"    
    genarator = Generator()
    discriminator = Discriminator(bn="instance_norm")
    wgan = WGAN_GP(genarator,discriminator,train_loader , validation_loader , critic_iters = critic_iters , lr = lr, gamma = gamma,logdir= log_dir)
    wgan.train(n_epochs,main_path)



  











  










