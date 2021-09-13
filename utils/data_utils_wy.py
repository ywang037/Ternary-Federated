from utils.config import Args
import torch
import numpy as np
import torch.utils.data
from torch.utils.data.dataset import Dataset
from torchvision import datasets, transforms
# from torch.utils import data

# set path to the data source
DATA_PATH = "./toYuanWANG/fl_seagate/images_FL_nonIID"

# set worker names
WORKERS = ['alice', 'bob', 'carol']

# ------------------------------------------
# from here, copied and modified from Seagate's notebook
# ------------------------------------------

# define customized dataset class to do transforms
class MyDataset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        
    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y
        
    def __len__(self):
        return len(self.subset)

def get_datasets_raw():
    """ input: path to the data folder, and list of workers/clients
        return: untransformed test dataset and training datasets of each client
    """
    local_ds = {}
    local_train_ds = {}
    local_test_ds = {}
    
    for worker in WORKERS:
        # load local dataset for each worker
        local_ds[worker] = datasets.ImageFolder(root = DATA_PATH + '/' + worker)
        print(worker, "->", len(local_ds[worker]), "train images loaded")

        # split each local dataset into train/test
        local_train_ds[worker], local_test_ds[worker] = torch.utils.data.random_split(local_ds[worker], 
                                                                                    [int(0.8*len(local_ds[worker])), 
                                                                                    len(local_ds[worker])-int(0.8*len(local_ds[worker]))])
        print("split dataset in train:", len(local_train_ds[worker]), " and test:", len(local_test_ds[worker]))

    # merge all local test datasets in a global testing dataset    
    central_train_ds = torch.utils.data.ConcatDataset(list(local_train_ds.values()))    
    print("\ncentralized training dataset is composed of", len(central_train_ds), "images")

    # merge all local test datasets in a global testing dataset    
    test_ds = torch.utils.data.ConcatDataset(list(local_test_ds.values()))    
    print("testing dataset is composed of", len(test_ds), "images")

    return local_test_ds, central_train_ds, test_ds

def get_datasets():
    """ input: list of workers, loaded untransformed datasets
        return: transformed datasets, to be used as input of Dataloader
    """
    # Data augmentation and normalization for training, only normalization for testing
    input_size = 256
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # fisrt, load untransformed datasets
    local_train_ds, central_train_ds, test_ds = get_datasets_raw()

    # then, do data augmentation to get datasets with transformed input images
    local_train_ds_transformed = {}
    for worker in WORKERS:
        local_train_ds_transformed[worker] = MyDataset(local_train_ds[worker], data_transforms['train'])
        # print(worker, "->", len(local_train_ds_transformed[worker]), "local training dataset transformed")
    
    central_train_ds_transformed = MyDataset(central_train_ds, data_transforms['train'])
    # print(len(central_train_ds_transformed), "centralized training dataset transformed")

    test_ds_transformed = MyDataset(test_ds, data_transforms['test'])
    # print(len(test_ds_transformed), "testing dataset transformed")

    return local_train_ds_transformed, test_ds_transformed, central_train_ds_transformed

# ----------------------------------------------------------------
# above this line, copied and modified from Seagate's notebook
# ----------------------------------------------------------------

def seagate_dataloader(args=Args):
    local_train_ds_transformed, test_ds_transformed, central_train_ds_transformed = get_datasets()
    # client_train_loaders = {}
    # for worker in WORKERS:
    #     client_train_loaders[worker] = torch.utils.data.DataLoader(local_train_ds_transformed[worker], batch_size=64, shuffle=True) 
    client_train_loaders = [torch.utils.data.DataLoader(local_train_ds_transformed[worker], batch_size=args.batch_size, shuffle=True) for worker in WORKERS]
    central_train_loader = torch.utils.data.DataLoader(central_train_ds_transformed, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_ds_transformed, batch_size=100, shuffle=False)
    return client_train_loaders, test_loader, central_train_loader

