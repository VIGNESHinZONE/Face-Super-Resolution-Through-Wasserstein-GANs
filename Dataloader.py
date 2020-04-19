import numpy as np
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as utils
from torch.utils.data.sampler import SubsetRandomSampler

def get_dataloader(root_dir,image_size,batch_size,seed,validation_split = 0.05,shuffle_dataset = True):
    dataset = dset.ImageFolder(root=root_dir,
                            transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))

    
    
    
    random_seed= seed
    # Creating data indices for training and validation splits:
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset :
        np.random.seed(random_seed)
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                           sampler=train_sampler , num_workers = 5)
    validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                sampler=valid_sampler , num_workers = 5)

    return train_loader, validation_loader


class Image_Dataset(Dataset):

  def __init__(self,root_dir,image_size_1=16,image_size_2=64):
    self.dataset_16 = dset.ImageFolder(root=root_dir,
                           transform=transforms.Compose([
                               transforms.Resize(image_size_1),
                               transforms.CenterCrop(image_size_1),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
    self.dataset_64 = dset.ImageFolder(root=root_dir,
                           transform=transforms.Compose([
                               transforms.Resize(image_size_2),
                               transforms.CenterCrop(image_size_2),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
    

  def __len__(self):
    return len(self.dataset_16)

  def __getitem__(self,idx):
    return self.dataset_16[idx][0],self.dataset_64[idx][0]

def get_dataloader_resolution(root_dir,image_size,batch_size,seed,validation_split = 0.05,shuffle_dataset = True):
    
    dataset = Image_Dataset(root_dir)

    
    
    
    random_seed= seed
    # Creating data indices for training and validation splits:
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset :
        np.random.seed(random_seed)
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                           sampler=train_sampler , num_workers = 5)
    validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                sampler=valid_sampler , num_workers = 5)

    return train_loader, validation_loader