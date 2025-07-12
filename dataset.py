import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np

PALETTE2ID = {
    (128,  64, 128): 0, (244,  35, 232): 1, ( 70,  70,  70): 2,
    (102, 102, 156): 3, (190, 153, 153): 4, (153, 153, 153): 5,
    (250, 170,  30): 6, (220, 220,   0): 7, (107, 142,  35): 8,
    (152, 251, 152): 9, ( 70, 130, 180): 10, (220,  20,  60): 11,
    (255,   0,   0): 12, (  0,   0, 142): 13, (  0,   0,  70): 14,
    (  0,  60, 100): 15, (  0,  80, 100): 16, (  0,   0, 230): 17,
    (119,  11,  32): 18,
}

COLORS = np.array([
    (128,  64,128), (244,  35,232), ( 70,  70, 70), (102,102,156), (190,153,153),
    (153,153,153), (250,170, 30), (220,220,  0), (107,142, 35), (152,251,152),
    ( 70,130,180), (220, 20, 60), (255,  0,  0), (  0,  0,142), (  0,  0, 70),
    (  0, 60,100), (  0, 80,100), (  0,  0,230), (119, 11, 32), (255, 255, 0)
], dtype=np.uint8)

# 🗺️ Dataset ufficiale con gtFine
class CityscapesFineDataset(Dataset):
    def __init__(self, root, split='train', transform=None, resize=(256,512)):
        self.img_dir = os.path.join(root, 'leftImg8bit', split)
        self.label_dir = os.path.join(root, 'gtFine', split)
        self.transform = transform
        self.resize = resize
        self.images = []
        self.labels = []

        for city in os.listdir(self.img_dir):
            for fn in os.listdir(os.path.join(self.img_dir, city)):
                if fn.endswith('_leftImg8bit.png'):
                    self.images.append(os.path.join(self.img_dir, city, fn))
                    self.labels.append(
                        os.path.join(self.label_dir, city,
                                     fn.replace('_leftImg8bit.png', '_gtFine_labelTrainIds.png'))
                    )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert('RGB')
        lbl = Image.open(self.labels[idx])
        img = img.resize((self.resize[1], self.resize[0]), Image.BILINEAR)
        lbl = lbl.resize((self.resize[1], self.resize[0]), Image.NEAREST)
        if self.transform:  
            img = self.transform(img)
        lbl = torch.from_numpy(np.array(lbl)).long()
        return img, lbl