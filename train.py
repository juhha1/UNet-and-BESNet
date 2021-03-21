import os, torch, tqdm, logging
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from models.PyTorch import UNet, BESNet
from utils.dataset import PetDataset, MRDataset
from options.train_options import get_opt

def train_pet(net, 
          device, 
          num_epoch = 100, 
          batch_size = 8, 
          lr = 1e-3,
          image_height = 128,
          image_width = 128,
          train_rate = 0.9,
          save_net = True,
          alpha = None,
          beta = None,
          bece_loss = True):
    # Data directory configurations
    base_dir = os.path.join('data', 'oxford-iiit-pet')
    # bring train/test image file names - it is selected by the oxford-iiit-pet
    train_filenames = [line.split()[0] for line in open(os.path.join(base_dir, 'annotations', 'trainval.txt'), 'r').readlines()]
    test_filenames = [line.split()[0] for line in open(os.path.join(base_dir, 'annotations', 'test.txt'), 'r').readlines()]
    # exclude files with invalid format - there are invalid data (2d image)
    list_invalid_files = set(['Abyssinian_34', 'Egyptian_Mau_167', 'Egyptian_Mau_191', 'Egyptian_Mau_177', 'Egyptian_Mau_139', 'Egyptian_Mau_145', 'staffordshire_bull_terrier_22', 'Egyptian_Mau_129', 'staffordshire_bull_terrier_2', 'Egyptian_Mau_14', 'Abyssinian_5', 'Egyptian_Mau_186'])
    train_filenames = [f for f in train_filenames if f not in list_invalid_files]
    test_filenames = [f for f in test_filenames if f not in list_invalid_files]

    # load dataset and dataloader
    trans_image = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((image_height, image_width)),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) # imagenet
    ])
    trans_mask = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((image_height, image_width)),
        transforms.Normalize(0, 1 / 255.)
    ])
    # dataset returns edges only for the BESNet
    return_edge = True if isinstance(net, BESNet) else False
    dataset = PetDataset(base_dir, train_filenames, transform_image = trans_image, transform_mask = trans_mask, return_edge = return_edge)
    dataset_test = PetDataset(base_dir, test_filenames, transform_image = trans_image, transform_mask = trans_mask, return_edge = return_edge)
    # train/val/test split
    train_size = int(len(dataset) * train_rate)
    val_size = len(dataset) - train_size
    dataset_train, dataset_val = random_split(dataset, [train_size, val_size])
    # make dataloader
    dataloaders = {
        'train': DataLoader(dataset_train, batch_size = batch_size, shuffle = True, num_workers = 8),
        'val': DataLoader(dataset_val, batch_size = batch_size, shuffle = False, num_workers = 8),
        'test': DataLoader(dataset_test, batch_size = batch_size, shuffle = False, num_workers = 8)
    }
    if isinstance(net, BESNet):
        train_besnet(net, dataloaders, num_epoch, lr, device, save_net, alpha, beta, bece_loss, 'BESNet-pet')
    else:
        train_unet(net, dataloaders, num_epoch, lr, device, save_net, 'UNet-pet')

def train_mri(net, 
          device, 
          num_epoch = 100, 
          batch_size = 8, 
          lr = 1e-3,
          image_height = 256,
          image_width = 256,
          train_rate = 1,
          save_net = True,
          alpha = None,
          beta = None,
          bece_loss = True):
    # Data directory configurations
    base_dir = os.path.join('data', 'lgg-mri-segmentation', 'kaggle_3m')
    
    # load dataset and dataloader
    trans_image = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((image_height, image_width))
    ])
    trans_mask = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((image_height, image_width))
    ])
    # dataset returns edges only for the BESNet
    return_edge = True if isinstance(net, BESNet) else False
    list_mr_files_train = [f for f in glob.glob(os.path.join(base_dir, '*', r'*[0-9].tif')) if '_CS_' not in f]
    list_mr_files_test = glob.glob(os.path.join(base_dir, '*_CS_*', r'*[0-9].tif'))
    dataset = MRDataset(list_mr_files_train, transform_image = trans_image, transform_mask = trans_mask, return_edge = return_edge)
    dataset_test = MRDataset(list_mr_files_test, transform_image = trans_image, transform_mask = trans_mask, return_edge = return_edge)
    # train/val/test split
    train_size = int(len(dataset) * train_rate)
    val_size = len(dataset) - train_size
    dataset_train, dataset_val = random_split(dataset, [train_size, val_size])
    # make dataloader
    dataloaders = {
        'train': DataLoader(dataset_train, batch_size = batch_size, shuffle = True, num_workers = 8),
        'val': DataLoader(dataset_val, batch_size = batch_size, shuffle = False, num_workers = 8),
        'test': DataLoader(dataset_test, batch_size = batch_size, shuffle = False, num_workers = 8)
    }
    if isinstance(net, BESNet):
        train_besnet(net, dataloaders, num_epoch, lr, device, save_net, alpha, beta, bece_loss, 'BESNet-mri')
    else:
        train_unet(net, dataloaders, num_epoch, lr, device, save_net, 'UNet-mri')

def train_unet(net, dataloaders, num_epoch, lr, device, save_net, model_name):
    cp_dir = 'checkpoints'
    
    net = net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr = lr)
    criterion = F.binary_cross_entropy
    for epoch in range(num_epoch):
        net.train()
        pbar = tqdm.tqdm(total = len(dataloaders['train']), position = 1, desc = f'Epoch {epoch}/{num_epoch}')
        for image, mask in dataloaders['train']:
            # Load data
            image = image.to(device)
            mask = mask.to(device)
            # Forward
            pred = net(image)
            # Get Loss
            loss = criterion(pred, mask)
            # weight updates
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_postfix({'error': np.round(loss.item(), 2)})
            pbar.update(1)
        if save_net:
            os.makedirs(cp_dir, exist_ok = True)
            torch.save(net.state_dict(), os.path.join(cp_dir, f'{model_name}_cp@epoch{epoch+1}.pth'))

# define bece loss for BESNet
def mdp_bece_loss(output, target, bdp):
    bx = alpha * torch.max(beta - bdp, torch.zeros_like(bdp))
    loss = target * torch.log(output + torch.finfo(torch.float32).eps)
    loss += (1 - target) * torch.log(1 - output + torch.finfo(torch.float32).eps)
    return -loss.mean()
            
def train_besnet(net, dataloaders, num_epoch, lr, device, save_net, alpha, beta, bece_loss, model_name):
    cp_dir = 'checkpoints'
    
    net = net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr = lr)
    criterion_bdp = F.binary_cross_entropy
    criterion_mdp = mdp_bece_loss if bece_loss else F.binary_cross_entropy
    for epoch in range(num_epoch):
        net.train()
        pbar = tqdm.tqdm(total = len(dataloaders['train']), position = 1, desc = f'Epoch {epoch}/{num_epoch}')
        for image, mask, edge in dataloaders['train']:
            # Load data
            image = image.to(device)
            mask = mask.to(device)
            edge = edge.to(device)
            # Forward
            bdp, mdp = net(image)
            # Get Loss
            bdp_loss = criterion_bdp(bdp, edge)
            mdp_loss = criterion_mdp(mdp, mask, bdp) if bece_loss else criterion_mdp(mdp, mask)
            loss = bdp_loss + mdp_loss
            # weight updates
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_postfix({'bdp_error': np.round(bdp_loss.item(), 2), 'mdp_error': np.round(mdp_loss.item(), 2), 'total_error': np.round(loss.item(), 2)})
            pbar.update(1)
        if save_net:
            os.makedirs(cp_dir, exist_ok = True)
            torch.save(net.state_dict(), os.path.join(cp_dir, f'{model_name}_cp@epoch{epoch+1}.pth'))

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    opt = get_opt()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = BESNet(3) if opt.net == 'besnet' else UNet(3)
    save_net = opt.save_net
    num_epoch = opt.num_epoch
    batch_size = opt.batch_size
    lr = opt.lr
    image_height = opt.height
    image_width = opt.width
    alpha = opt.alpha
    beta = opt.beta
    bece_loss = opt.bece_loss
    if opt.data == 'pet':
        train_pet(net, device, num_epoch, batch_size, lr, image_height, image_width, train_rate = 0.9, save_net = save_net, alpha = alpha, beta = beta, bece_loss = bece_loss)
    elif opt.data == 'mri':
        train_mri(net, device, num_epoch, batch_size, lr, image_height, image_width, train_rate = 0.9, save_net = save_net, alpha = alpha, beta = beta, bece_loss = bece_loss)