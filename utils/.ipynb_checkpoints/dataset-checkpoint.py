import os, glob
import torch, torchvision
from torch.utils.data import Dataset
from PIL import Image

class PetDataset(Dataset):
    def __init__(self, base_dir, filenames, transform_image, transform_mask, return_edge = True):
        
        self.base_dir = base_dir
        self.filenames = filenames
        self.transform_image = transform_image
        self.transform_mask = transform_mask
        self.return_edge = return_edge
    def __len__(self):
        return len(self.filenames)
    def __getitem__(self, idx):
        image_file = os.path.join(self.base_dir, 'images', f'{self.filenames[idx]}.jpg')
        image = self.load(image_file, self.transform_image)
        mask_file = os.path.join(self.base_dir, 'masks', f'{self.filenames[idx]}.png')
        mask = self.load(mask_file, self.transform_mask)
        if self.return_edge:
            edge_file = os.path.join(self.base_dir, 'edges', f'{self.filenames[idx]}.png')
            edge = self.load(edge_file, self.transform_mask)
            return image, mask, edge
        else:
            return image, mask
    def load(self, file, trans):
        image = Image.open(file)
        return trans(image)
    
class MRDataset(Dataset):
    def __init__(self, list_mr_files, transform_image, transform_mask, return_edge = True):
        self.list_mr_files = list_mr_files
        self.transform_image = transform_image
        self.transform_mask = transform_mask
        self.return_edge = return_edge
    def __len__(self):
        return len(self.list_mr_files)
    def __getitem__(self, idx):
        image_file = self.list_mr_files[idx]
        image = self.load(image_file, self.transform_image)
        mask_file = image_file[:-4] + '_mask.tif'
        mask = self.load(mask_file, self.transform_mask)
        if self.return_edge:
            edge_file = image_file[:-4] + '_edge.tif'
            edge = self.load(edge_file, self.transform_mask)
            return image, mask, edge
        else:
            return image, mask
    def load(self, file, trans):
        image = Image.open(file)
        return trans(image)
        