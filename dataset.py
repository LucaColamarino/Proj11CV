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
    (  0, 60,100), (  0, 80,100), (  0,  0,230), (119, 11, 32)
], dtype=np.uint8)


def rgb_to_id(mask_rgb, palette2id=PALETTE2ID, default=255):
    h, w, _ = mask_rgb.shape
    mask_id = np.full((h, w), default, dtype=np.uint8)
    for rgb, cls_id in palette2id.items():
        mask_id[(mask_rgb == rgb).all(axis=-1)] = cls_id
    return mask_id

class CityscapesDataset(Dataset):
    def __init__(self, root, split='train', transform=None, resize=(256, 512)):
        self.img_dir = os.path.join(root, split, 'img')
        self.label_dir = os.path.join(root, split, 'label')
        self.files = sorted(os.listdir(self.img_dir))
        self.transform = transform
        self.resize = resize

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.files[idx])
        label_path = os.path.join(self.label_dir, self.files[idx])
        image = Image.open(img_path).convert('RGB').resize((self.resize[1], self.resize[0]), Image.BILINEAR)
        label_rgb = Image.open(label_path).resize((self.resize[1], self.resize[0]), Image.NEAREST)
        label_id = rgb_to_id(np.array(label_rgb))


        if self.transform:
            image = self.transform(image)

        return image, torch.from_numpy(label_id).long()

    def __len__(self):
        return len(self.files)
