import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np
from torchvision import transforms

PALETTE2ID = {
    (128,  64, 128): 0, (244,  35, 232): 1, ( 70,  70,  70): 2,
    (102, 102, 156): 3, (190, 153, 153): 4, (153, 153, 153): 5,
    (250, 170,  30): 6, (220, 220,  0): 7, (107, 142,  35): 8,
    (152, 251, 152): 9, ( 70, 130, 180): 10, (220,  20,  60): 11,
    (255,   0,   0): 12, (  0,   0, 142): 13, (  0,   0,  70): 14,
    (  0,  60, 100): 15, (  0,  80, 100): 16, (  0,   0, 230): 17,
    (119,  11,  32): 18
}

# Mapping 19 classi Cityscapes -> 7 macro-classi
CITYSCAPES_19_TO_7 = {
    0: 0,   # road → road
    1: 1, 2: 1, 3: 1,   # sidewalk, parking, rail track → flat(no road)
    4: 2, 5: 2,         # person, rider → human
    6: 3, 7: 3, 8: 3, 9: 3, 10: 3, 11: 3, 12: 3, 13: 3,  # vehicle group
    14: 4, 15: 4, 16: 4, 17: 4,       # construction group
    18: 5,  # pole → object
    # Vegetation, terrain, sky (macro background)
    # Aggiungi eventuali ID extra se li hai nel tuo PALETTE2ID (es. vegetation, terrain, sky)
}
DEFAULT_BG_CLASS = 6  # Usalo per i pixel non mappati







COLORS = np.array([
    (128,  64,128), (244,  35,232), ( 70,  70, 70), (102,102,156), (190,153,153),
    (153,153,153), (250,170, 30), (220,220,  0), (107,142, 35), (152,251,152),
    ( 70,130,180), (220, 20, 60), (255,  0,  0), (  0,  0,142), (  0,  0, 70),
    (  0, 60,100), (  0, 80,100), (  0,  0,230), (119, 11, 32), (255, 255, 255) # Add white for unknown obstacles
], dtype=np.uint8)

# Official Dataset with gtFine
class CityscapesFineDataset(Dataset):
    def __init__(self, root, split='train', transform=None, resize=(256,512)):
        self.img_dir = os.path.join(root, 'leftImg8bit', split)
        self.label_dir = os.path.join(root, 'gtFine', split)
        self.transform = transform # This will now only handle ToTensor and Normalize for image
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
        try:
            img = Image.open(self.images[idx]).convert('RGB')
            lbl = Image.open(self.labels[idx])

            # Resize images first to target training size
            img = img.resize((self.resize[1], self.resize[0]), Image.BILINEAR)
            lbl = lbl.resize((self.resize[1], self.resize[0]), Image.NEAREST)

            # Data Augmentations (applied to both image and label)
            
            # Random Horizontal Flip
            if torch.rand(1) < 0.5: # 50% probability
                img = transforms.functional.hflip(img)
                lbl = transforms.functional.hflip(lbl)

            # Random Rotation (up to 10 degrees)
            if torch.rand(1) < 0.5: # 50% probability
                angle = (torch.rand(1) * 20 - 10).item() # Random angle between -10 and +10
                # Fill value for image (0,0,0) for black, for mask (255) for ignore_index
                img = transforms.functional.rotate(img, angle, fill=(0,0,0))
                lbl = transforms.functional.rotate(lbl, angle, fill=255, interpolation=Image.NEAREST)

            # Convert label to tensor (numpy array first, then torch tensor)
            # Ensure label is long type as it contains class IDs
            lbl = torch.from_numpy(np.array(lbl)).long()
            # Rimappa le classi a 7 macro-classi
            lbl_7 = torch.full_like(lbl, DEFAULT_BG_CLASS)
            for orig_id, group_id in CITYSCAPES_19_TO_7.items():
                lbl_7[lbl == orig_id] = group_id
            lbl = lbl_7


            if self.transform:
                img = self.transform(img) # This applies ToTensor and Normalize to the image only

            return img, lbl
        except Exception as e:
            print(f"Error loading image or label at index {idx}: {self.images[idx]}, {self.labels[idx]} - {e}")
            # If an error occurs, we can either skip this sample or raise an exception
            raise

class LostAndFoundDataset(Dataset):
    def __init__(self, root, split='train', transform=None, resize=(256,512)):
        self.img_dir = os.path.join(root, 'leftImg8bit', split)
        self.label_dir = os.path.join(root, 'gtCoarse', split)
        self.transform = transform
        self.resize = resize
        self.images = []
        self.labels = []

        for scene in os.listdir(self.img_dir):
            for fn in os.listdir(os.path.join(self.img_dir, scene)):
                if fn.endswith('_leftImg8bit.png'):
                    self.images.append(os.path.join(self.img_dir, scene, fn))
                    self.labels.append(
                        os.path.join(self.label_dir, scene,
                                     fn.replace('_leftImg8bit.png', '_gtCoarse_labelTrainIds.png'))
                    )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert('RGB')
        lbl = Image.open(self.labels[idx])

        # Resize
        img = img.resize((self.resize[1], self.resize[0]), Image.BILINEAR)
        lbl = lbl.resize((self.resize[1], self.resize[0]), Image.NEAREST)

        # Augmentations simili a Cityscapes
        if torch.rand(1) < 0.5:
            img = transforms.functional.hflip(img)
            lbl = transforms.functional.hflip(lbl)
        if torch.rand(1) < 0.5:
            angle = (torch.rand(1) * 20 - 10).item()
            img = transforms.functional.rotate(img, angle, fill=(0,0,0))
            lbl = transforms.functional.rotate(lbl, angle, fill=255, interpolation=Image.NEAREST)

        lbl = torch.from_numpy(np.array(lbl)).long()
        if self.transform:
            img = self.transform(img)

        return img, lbl
