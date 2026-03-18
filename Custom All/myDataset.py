from torch.utils.data import Dataset, DataLoader, random_split
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import os

class myDataset(Dataset):
    def __init__(self, directory, transform = None):
        self.data = ImageFolder(directory, transform = transform)
        self.directory = directory

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    @property
    def classes(self):
        return self.data.classes

    def getNoImagesInClass(self):
        classDict = {'0':0,
                    '1':0,
                    '2':0,
                    '3':0,
                    '4':0,
                    '5':0,
                    '6':0,
                    }
        
        for folder_name in os.listdir(self.directory):
            folder_path = os.path.join(self.directory, folder_name)
            allImages = os.listdir(folder_path)
            classDict[folder_name] = len(allImages)
        return classDict
    
train_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.Grayscale(3),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.GaussianBlur(kernel_size=5, sigma=(0.1,2)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[.229,.224,.225]),
])
val_transform = transforms.Compose([ 
    transforms.Resize((224,224)),
    transforms.Grayscale(3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[.229,.224,.225]),
])
test_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.Grayscale(3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[.229,.224,.225]),
])