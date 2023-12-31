from torch.utils.data import Dataset
from PIL import Image
import os
import torchvision.transforms as transforms
from glob import glob
from sklearn.model_selection import train_test_split
import cv2
from randaugment import RandAugment 

def SplitDataset(img_dir:str, val_size:float=0.1, seed=42):
    fake_images = glob(f'{img_dir}/fake_images/*.png')
    real_images = glob(f'{img_dir}/real_images/*.png')
    labels = [1] * len(fake_images) + [0] * len(real_images)

    X_train, X_val, y_train, y_val = train_test_split(fake_images + real_images, labels, test_size=val_size, random_state=seed, shuffle=True)

    return X_train, X_val, y_train, y_val

class CustomDataset(Dataset):
    def __init__(self, X, y):
        '''
        X: list of image path
        y: list of label (0->real, 1->fake)
        '''
        
        self.X = X
        self.y = y
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224,224)),
            transforms.Normalize([0.485, 0.456,0.406],[0.229,0.224,0.225])
        ])

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        impath = self.X[index]
        fname = os.path.basename(impath)
        img = Image.open(impath).convert('RGB')
        img = self.transforms(img)
        target = self.y[index]

        return img, target, fname
    
class CustomDataset2(Dataset):
    def __init__(self, X, y):
        '''
        X: list of image path
        y: list of label (0->real, 1->fake)
        '''
        
        self.X = X
        self.y = y
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256,256)),
            transforms.RandomCrop(200),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            transforms.RandomRotation(30),
            transforms.Resize((224, 224)),
            transforms.Normalize([0.485, 0.456,0.406],[0.229,0.224,0.225])
        ])
        
    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        impath = self.X[index]
        fname = os.path.basename(impath)
        img = Image.open(impath).convert('RGB')
        img = self.transforms(img)
        target = self.y[index]
        return img, target, fname
    
class CustomDataset3(Dataset):
    def __init__(self, X, y):
        '''
        X: list of image path
        y: list of label (0->real, 1->fake)
        '''
        
        self.X = X
        self.y = y
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256,256)),
            transforms.RandomCrop(224),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            transforms.RandomHorizontalFlip(p = 1),
            transforms.RandomRotation(15),
            transforms.Resize((224, 224)),
            transforms.Normalize([0.485, 0.456,0.406],[0.229,0.224,0.225])
        ])
        
    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        impath = self.X[index]
        fname = os.path.basename(impath)
        img = Image.open(impath).convert('RGB')
        img = self.transforms(img)
        target = self.y[index]
        return img, target, fname

class CustomDatasetFlip(Dataset):
    def __init__(self, X, y):
        '''
        X: list of image path
        y: list of label (0->real, 1->fake)
        '''
        
        self.X = X
        self.y = y
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224,224)),
            transforms.RandomHorizontalFlip(p = 1),
            transforms.Normalize([0.485, 0.456,0.406],[0.229,0.224,0.225])
        ])
        
    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        impath = self.X[index]
        fname = os.path.basename(impath)
        img = Image.open(impath).convert('RGB')
        img = self.transforms(img)
        target = self.y[index]
        return img, target, fname
    
class CustomDatasetRot(Dataset):
    def __init__(self, X, y):
        '''
        X: list of image path
        y: list of label (0->real, 1->fake)
        '''
        
        self.X = X
        self.y = y
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(p = 0.5),
            transforms.RandomRotation(15),
            transforms.Resize((224,224)),
            transforms.Normalize([0.485, 0.456,0.406],[0.229,0.224,0.225])
        ])
        
    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        impath = self.X[index]
        fname = os.path.basename(impath)
        img = Image.open(impath).convert('RGB')
        img = self.transforms(img)
        target = self.y[index]
        return img, target, fname
    
class CustomDatasetCrop(Dataset):
    def __init__(self, X, y):
        '''
        X: list of image path
        y: list of label (0->real, 1->fake)
        '''
        
        self.X = X
        self.y = y
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256,256)),
            transforms.RandomCrop(200),
            transforms.RandomHorizontalFlip(p = 0.5),
            transforms.RandomRotation(30),
            transforms.Resize((224,224)),
            transforms.Normalize([0.485, 0.456,0.406],[0.229,0.224,0.225])
        ])
        
    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        impath = self.X[index]
        fname = os.path.basename(impath)
        img = Image.open(impath).convert('RGB')
        img = self.transforms(img)
        target = self.y[index]
        return img, target, fname
    
class CustomDatasetJit(Dataset):
    def __init__(self, X, y):
        '''
        X: list of image path
        y: list of label (0->real, 1->fake)
        '''
        
        self.X = X
        self.y = y
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256,256)),
            transforms.RandomCrop(200),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            transforms.RandomHorizontalFlip(p = 0.5),
            transforms.RandomRotation(15),
            transforms.Resize((224,224)),
            transforms.Normalize([0.485, 0.456,0.406],[0.229,0.224,0.225])
        ])
        
    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        impath = self.X[index]
        fname = os.path.basename(impath)
        img = Image.open(impath).convert('RGB')
        img = self.transforms(img)
        target = self.y[index]
        return img, target, fname
    

#---------------------------------Test DataSet------------------------------
class TestDataset(Dataset):
    def __init__(self, X):
        '''
        X: list of image path
        '''
        self.X = X
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224,224)),
            transforms.Normalize([0.485, 0.456,0.406],[0.229,0.224,0.225])
        ])

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        impath = self.X[index]
        fname = os.path.basename(impath)
        img = Image.open(impath).convert('RGB')
        img = self.transforms(img)

        return img, fname

if __name__ == '__main__':
    pass

        