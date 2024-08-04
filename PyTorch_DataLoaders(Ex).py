import pandas as pd
from PIL import Image
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import matplotlib.pyplot as plt


#for custom data
class CustomDataset(Dataset):
    def __init__(self, csv_file, root_dir, split, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.data_frame = self.data_frame[self.data_frame['data set'] == split]

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.data_frame.iloc[idx, 1])
        image = Image.open(img_name).convert('RGB')
        label = self.data_frame.iloc[idx, 2]
        class_id = int(self.data_frame.iloc[idx, 0])
        
        if self.transform:
            image = self.transform(image)
            
        return image, class_id, label

# transform done seperately
transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
])

csv_file_path = r'D:\Others\SportDataset\sports.csv' 

train_dataset = CustomDataset(csv_file=csv_file_path, root_dir=r"D:\Others\SportDataset", split='train', transform=transform)
valid_dataset = CustomDataset(csv_file=csv_file_path, root_dir=r"D:\Others\SportDataset", split='valid', transform=transform)
test_dataset = CustomDataset(csv_file=csv_file_path, root_dir=r"D:\Others\SportDataset", split='test', transform=transform)



#Dataloader
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True)



#Display
images, class_ids, labels = next(iter(train_dataloader))
img = images[0].permute(1, 2, 0)  
label = labels[0]
plt.imshow(img)
plt.title(f"Class ID: {class_ids[0]}, Label: {label}")
plt.show()

