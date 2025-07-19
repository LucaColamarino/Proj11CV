import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np
from torchvision import transforms

# Original Cityscapes labelTrainIds palette to ID mapping
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

# Mapping 19 Cityscapes classes -> 7 macro-classes (IDs 0-6)
# Any original ID not explicitly in this dict will fall to DEFAULT_BG_CLASS (7)
CITYSCAPES_19_TO_7_MACRO = {
    # Macro Class 0: Road
    0: 0, # road
    # Macro Class 1: Flat (non-road surfaces)
    1: 1, # sidewalk
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

# DEFAULT_BG_CLASS is now the ID for the UNKNOWN_OBSTACLE_ID (7)
DEFAULT_BG_CLASS = 7

# Define colors for your 7 macro classes (0-6) + 1 unknown class (ID 7)
# Total 8 colors for model_output_classes = 8
COLORS = np.array([
    # Macro Class 0: Road
    (128,  64, 128),
    # Macro Class 1: Flat (Non-Road)
    (244,  35, 232),
    # Macro Class 2: Human
    (220,  20,  60),
    # Macro Class 3: Vehicle Group
    (  0,   0, 142),
    # Macro Class 4: Construction Group
    (153, 153, 153),
    # Macro Class 5: Objects
    (250, 170,  30),
    # Macro Class 6: Background
    (107, 142,  35),
    # UNKNOWN_OBSTACLE_ID (ID 7) - White for unknown
    (255, 255, 255), # Using white as (255, 255, 255)
], dtype=np.uint8)


# Helper function to map RGB to ID (for visualizing _gtCoarse_color.png from Lost & Found)
# This function uses the original PALETTE2ID for Cityscapes-like colors.
def rgb_to_id(rgb_mask, palette2id, default=255):
    h, w, _ = rgb_mask.shape
    id_mask = np.full((h, w), default, dtype=np.uint8)
    for rgb_tuple, class_id in palette2id.items():
        matches = np.all(rgb_mask == np.array(rgb_tuple).reshape(1, 1, 3), axis=-1)
        id_mask[matches] = class_id
    return id_mask


class CityscapesFineDataset(Dataset):
    def __init__(self, root, split='train', transform=None, resize=(256,512), unknown_obstacle_id=7):
        self.img_dir = os.path.join(root, 'leftImg8bit', split)
        self.label_dir = os.path.join(root, 'gtFine', split)
        self.transform = transform
        self.resize = resize
        self.unknown_obstacle_id = unknown_obstacle_id

        # Create a numpy array for efficient mapping
        self.mapping_array = np.full(256, self.unknown_obstacle_id, dtype=np.uint8)
        for orig_id, group_id in CITYSCAPES_19_TO_7_MACRO.items():
            if 0 <= orig_id < 255:
                self.mapping_array[orig_id] = group_id
        self.mapping_array[255] = 255 # Explicitly ensure ignore_index (255) remains 255

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

            img = img.resize((self.resize[1], self.resize[0]), Image.BILINEAR)
            lbl = lbl.resize((self.resize[1], self.resize[0]), Image.NEAREST)

            if torch.rand(1) < 0.5:
                img = transforms.functional.hflip(img)
                lbl = transforms.functional.hflip(lbl)
            if torch.rand(1) < 0.5:
                angle = (torch.rand(1) * 20 - 10).item()
                img = transforms.functional.rotate(img, angle, fill=(0,0,0))
                lbl = transforms.functional.rotate(lbl, angle, fill=255, interpolation=Image.NEAREST)

            lbl_np = np.array(lbl, dtype=np.uint8)
            lbl_macro_np = self.mapping_array[lbl_np]
            
            lbl = torch.from_numpy(lbl_macro_np).long()

            if self.transform:
                img = self.transform(img)

            return img, lbl
        except Exception as e:
            print(f"Error loading image or label at index {idx}: {self.images[idx]}, {self.labels[idx]} - {e}")
            raise


class LostAndFoundDataset(Dataset):
    def __init__(self, root, split='train', transform=None, resize=(256,512), unknown_obstacle_id=7): # Pass 7 here
        self.img_dir = os.path.join(root, 'leftImg8bit', split)
        self.label_dir = os.path.join(root, 'gtCoarse', split)
        self.transform = transform
        self.resize = resize
        self.unknown_obstacle_id = unknown_obstacle_id

        # Create a numpy array for efficient mapping
        self.mapping_array = np.full(256, self.unknown_obstacle_id, dtype=np.uint8)
        for orig_id, group_id in CITYSCAPES_19_TO_7_MACRO.items():
            if 0 <= orig_id < 255:
                self.mapping_array[orig_id] = group_id
        self.mapping_array[255] = 255 

        self.images = []
        self.labels = []
        for scene in os.listdir(self.img_dir):
            for fn in os.listdir(os.path.join(self.img_dir, scene)):
                if fn.endswith('_leftImg8bit.png'):
                    self.images.append(os.path.join(self.img_dir, scene, fn))
                    # For Lost & Found, gtCoarse_color.png is common.
                    # If your gtCoarse is _labelTrainIds.png, this path is correct.
                    # If it's _color.png, you'll need to adjust the loading in __getitem__.
                    self.labels.append(
                        os.path.join(self.label_dir, scene,
                                     fn.replace('_leftImg8bit.png', '_gtCoarse_labelTrainIds.png')) # Assuming labelTrainIds
                    )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert('RGB')
        lbl_path = self.labels[idx]
        
        # Determine if label is ID-based or RGB-based. Lost & Found often has _color.png
        if lbl_path.endswith('_gtCoarse_labelTrainIds.png'):
            lbl = Image.open(lbl_path)
            lbl_is_rgb = False
        elif lbl_path.endswith('_gtCoarse_color.png'):
            lbl = Image.open(lbl_path).convert('RGB')
            lbl_is_rgb = True
        else:
            # Fallback if neither is found, or raise an error
            # For Lost & Found, you might primarily deal with _color.png
            print(f"Warning: Unexpected label file format for LostAndFoundDataset: {lbl_path}. Assuming ID-based.")
            lbl = Image.open(lbl_path) # Attempt to open as is
            lbl_is_rgb = False


        # Resize
        img = img.resize((self.resize[1], self.resize[0]), Image.BILINEAR)
        lbl = lbl.resize((self.resize[1], self.resize[0]), Image.NEAREST)

        # Augmentations
        if torch.rand(1) < 0.5:
            img = transforms.functional.hflip(img)
            lbl = transforms.functional.hflip(lbl)
        if torch.rand(1) < 0.5:
            angle = (torch.rand(1) * 20 - 10).item()
            img = transforms.functional.rotate(img, angle, fill=(0,0,0))
            # Fill value for label rotation: 255 if ID-based, (0,0,0) if RGB-based
            lbl = transforms.functional.rotate(lbl, angle, fill=255 if not lbl_is_rgb else (0,0,0), interpolation=Image.NEAREST)

        # Convert label to numpy array and apply fast numpy mapping
        if lbl_is_rgb:
            # If it's an RGB mask, first convert it to Cityscapes original IDs using PALETTE2ID
            # Note: PALETTE2ID must contain mappings for Lost & Found's gtCoarse_color.png RGB values
            # to their corresponding original Cityscapes labelTrainIds (0-18)
            lbl_np = rgb_to_id(np.array(lbl), palette2id=PALETTE2ID, default=255)
        else:
            lbl_np = np.array(lbl, dtype=np.uint8)
            
        lbl_macro_np = self.mapping_array[lbl_np] # Apply fast element-wise mapping to macro classes
        
        lbl = torch.from_numpy(lbl_macro_np).long()
        
        if self.transform:
            img = self.transform(img)

        return img, lbl