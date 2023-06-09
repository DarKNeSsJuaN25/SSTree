import numpy as np
import pandas as pd
from PIL import Image
from glob import glob
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from SSTree2 import SSTree

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class AnimalDataset(Dataset):
    def __init__(self,img_paths,img_labels):
        self.img_paths = img_paths
        self.img_labels = img_labels
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        if len(self.img_paths) != len(self.img_labels):
            raise ValueError("Paths and Labels lengths do not match")
        
    def __len__(self):
        return len(self.img_paths)
        
    def __getitem__(self,index):
        PIL_IMAGE = Image.open(self.img_paths[index])
        img = self.transform(PIL_IMAGE)
        return img, self.img_paths[index]

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        model = torch.hub.load('hankyul2/EfficientNetV2-pytorch', 'efficientnet_v2_s_in21k', pretrained=True)
        self.net = nn.Sequential(*list(model.children()))[:2]
        self.sal = nn.Sequential(*list(list(model.children())[2].children()))[:2]        
        
    def forward(self, x):
        x = self.net(x)
        x = self.sal(x)
        return x.reshape([x.shape[0],-1])


def getDataset(rootFolder):
    paths  = []
    labels = []
    for category, label in [("cat", 0), ("dog", 1), ("wild", 2)]:
        for img_path in glob(f"{rootFolder}/train/{category}/*") + glob(f"{rootFolder}/val/{category}/*"):
            paths.append(img_path)
            labels.append(label)

    dataset = AnimalDataset(paths, labels)
    return dataset


def insertData2SSTree(dataset, sstree, batch_size):
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    net = Encoder()
    for param in net.parameters():
        param.requires_grad = False

    eps = 1e-8
    net.eval()
    net.to(device)
    for data, paths in tqdm(loader):
        data = data.to(device)
        embeddings = net(data)
        embeddings = embeddings.cpu().numpy()
        for embedding, path in zip(embeddings, paths):
            norm = np.linalg.norm(embedding)
            sstree.insert(embedding/(norm+eps), path)

def main():
    dataset = getDataset("afhq")
    tree = SSTree(M=100, m=50)
    insertData2SSTree(dataset, tree, batch_size=200)
    tree.save("tree.ss")


if __name__ == "__main__":
    main()