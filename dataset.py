import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np

PALETTE2ID = {
    (128,  64, 128): 0,   # road
    (244,  35, 232): 1,   # sidewalk
    ( 70,  70,  70): 2,   # building
    (102, 102, 156): 3,   # wall
    (190, 153, 153): 4,   # fence
    (153, 153, 153): 5,   # pole
    (250, 170,  30): 6,   # traffic light
    (220, 220,   0): 7,   # traffic sign
    (107, 142,  35): 8,   # vegetation
    (152, 251, 152): 9,   # terrain
    ( 70, 130, 180): 10,  # sky
    (220,  20,  60): 11,  # person
    (255,   0,   0): 12,  # rider
    (  0,   0, 142): 13,  # car
    (  0,   0,  70): 14,  # truck
    (  0,  60, 100): 15,  # bus
    (  0,  80, 100): 16,  # train
    (  0,   0, 230): 17,  # motorcycle
    (119,  11,  32): 18,  # bicycle
}

def rgb_to_id(mask_rgb, palette2id=PALETTE2ID, default=255):
    h, w, _ = mask_rgb.shape
    mask_id = np.full((h, w), default, dtype=np.uint8)
    for rgb, cls_id in palette2id.items():
        matches = (mask_rgb == rgb).all(axis=-1)
        mask_id[matches] = cls_id
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
        image = Image.open(img_path).convert('RGB')
        label_rgb = np.array(Image.open(label_path))
        label_id = rgb_to_id(label_rgb)
        label = Image.fromarray(label_id)

        # Resize both
        image = image.resize(self.resize, Image.BILINEAR)
        label = label.resize(self.resize, Image.NEAREST)

        if self.transform:
            image = self.transform(image)

        return image, torch.from_numpy(np.array(label)).long()

    def __len__(self):
        return len(self.files)
