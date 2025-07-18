import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np
from torchvision import transforms

PALETTE2ID = {
    (128,  64, 128): 0,  # road
    (244,  35, 232): 1,  # sidewalk
    ( 70,  70,  70): 2,  # building
    (102, 102, 156): 3,  # wall
    (190, 153, 153): 4,  # fence
    (153, 153, 153): 5,  # pole
    (250, 170,  30): 6,  # traffic light
    (220, 220,   0): 7,  # traffic sign
    (107, 142,  35): 8,  # vegetation
    (152, 251, 152): 9,  # terrain
    ( 70, 130, 180): 10, # sky
    (220,  20,  60): 11, # person
    (255,   0,   0): 12, # rider
    (  0,   0, 142): 13, # car
    (  0,   0,  70): 14, # truck
    (  0,  60, 100): 15, # bus
    (  0,  80, 100): 16, # train
    (  0,   0, 230): 17, # motorcycle
    (119,  11,  32): 18  # bicycle
}

# Mapping 19 classi Cityscapes -> 7 macro-classi
CITYSCAPES_19_TO_7_MACRO = {
    # Macro Class 0: Road
    0: 0, # road
    # Macro Class 1: Flat (non-road surfaces)
    1: 1, # sidewalk
    # Assuming parking and rail track (original IDs 2, 3) are now included in flat/non-road background.
    # If they are, map them: 2:1, 3:1
    # If not mapped here, they will become UNKNOWN_OBSTACLE_ID later.
    # Macro Class 2: Human (person, rider)
    11: 2, # person
    12: 2, # rider
    # Macro Class 3: Vehicle Group (car, truck, bus, train, motorcycle, bicycle)
    13: 3, # car
    14: 3, # truck
    15: 3, # bus
    16: 3, # train
    17: 3, # motorcycle
    18: 3, # bicycle
    # Macro Class 4: Construction Group (building, wall, fence)
    2: 4,  # building
    3: 4,  # wall
    4: 4,  # fence
    # Macro Class 5: Objects (pole, traffic light, traffic sign)
    5: 5,  # pole
    6: 5,  # traffic light
    7: 5,  # traffic sign

    # Macro Class 6: Background (vegetation, terrain, sky)
    8: 6,  # vegetation
    9: 6,  # terrain
    10: 6, # sky
}
DEFAULT_BG_CLASS = 7  # Usalo per i pixel non mappati

OLD_COLORS = np.array([
    (128,  64, 128),  # 0 road
    (244,  35, 232),  # 1 sidewalk
    (153, 153, 153),  # 2 building
    (153, 153, 153),  # 3 wall
    (153, 153, 153),  # 4 fence
    (107, 142,  35),  # 5 pole
    (107, 142,  35),  # 6 traffic light
    (107, 142,  35),  # 7 traffic sign
    (107, 142,  35),  # 8 vegetation
    (107, 142,  35),  # 9 terrain
    (107, 142,  35),# 10 sky
    (220,  20,  60), # 11 person
    (220,  20,  60),   # 12 rider
    (  0,   0, 142), # 13 car
    (  0,   0, 142),   # 14 truck
    (  0,   0, 142),# 15 bus
    (  0,   0, 142),# 16 train
    (  0,   0, 142), # 17 motorcycle
    (  0,   0, 142)   # 18 bicycle
], dtype=np.uint8)
COLORS = np.array([
    # Macro Class 0: Road (e.g., Dark Purple, similar to original road)
    (128,  64, 128),
    # Macro Class 1: Flat (Non-Road) (e.g., Light Pink/Magenta, similar to original sidewalk)
    (244,  35, 232),
    # Macro Class 2: Human (e.g., Red)
    (220,  20,  60), # Person color from original Cityscapes
    # Macro Class 3: Vehicle Group (e.g., Dark Blue, similar to original car)
    (  0,   0, 142),
    # Macro Class 4: Construction Group (e.g., Brown/Gray)
    (153, 153, 153), # Pole color, or pick a building-like color
    # Macro Class 5: Other Objects
    (250, 170,  30), # Traffic light color from original Cityscapes 
    # Macro Class 6: Background (e.g., Green for vegetation/sky)
    (107, 142,  35), # Vegetation color from original Cityscapes
    # UNKNOWN_OBSTACLE_ID (ID 6) - White for unknown
    (  255,   255, 255), # Blue for unknown obstacles
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
            for orig_id, group_id in CITYSCAPES_19_TO_7_MACRO.items():
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
