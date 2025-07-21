import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np
from torchvision import transforms

# Default background class for unknown obstacles
DEFAULT_UK_CLASS = 7

# Define colors for visualization
COLORS = np.array([
    (128, 64, 128),   # 0 - Road - purple
    (244, 35, 232),   # 1 - Flat - pink
    (220, 20, 60),    # 2 - Human - red
    (0, 0, 142),      # 3 - Vehicle - blue
    (153, 153, 153),  # 4 - Construction - gray
    (250, 170, 30),   # 5 - Objects - orange
    (107, 142, 35),   # 6 - Vegetation - green
    (255, 255, 255),  # 7 - OoD - white
], dtype=np.uint8)

CITYSCAPES_19_TO_7_MACRO = {
    0: 0,
    1: 1,
    11: 2, 12: 2,
    13: 3, 14: 3, 15: 3, 16: 3, 17: 3, 18: 3,
    2: 4, 3: 4, 4: 4,
    5: 5, 6: 5, 7: 5,
    8: 6, 9: 6, 10: 6
}

# Build a mapping array for converting 19-class labels to 7-class labels
def build_mapping_array(unknown_obstacle_id=7):
    mapping_array = np.full(256, unknown_obstacle_id, dtype=np.uint8)
    for orig_id, group_id in CITYSCAPES_19_TO_7_MACRO.items():
        if 0 <= orig_id < 255: mapping_array[orig_id] = group_id
    mapping_array[255] = 255
    return mapping_array

# Base class for segmentation datasets
class BaseSegmentationDataset(Dataset):
    def __init__(self, img_dir, label_dir, mapping_array, transform=None,
                 resize=(256, 512), ood_mode=False, unknown_obstacle_id=7):
        self.img_dir, self.label_dir = img_dir, label_dir
        self.transform, self.resize = transform, resize
        self.ood_mode, self.unknown_obstacle_id = ood_mode, unknown_obstacle_id
        

        self.mapping_array = mapping_array
        self.images, self.labels = [], []

        for scene in os.listdir(img_dir):
            for fn in os.listdir(os.path.join(img_dir, scene)):
                if fn.endswith('_leftImg8bit.png'):
                    self.images.append(os.path.join(img_dir, scene, fn))
                    label_suffix = '_gtFine_labelTrainIds.png' if 'gtFine' in label_dir else '_gtCoarse_labelTrainIds.png'
                    self.labels.append(os.path.join(label_dir, scene, fn.replace('_leftImg8bit.png', label_suffix)))

    def __len__(self): return len(self.images)

    def __getitem__(self, idx):
        # Image loading
        img = Image.open(self.images[idx]).convert('RGB')
        lbl = Image.open(self.labels[idx])

        img = img.resize((self.resize[1], self.resize[0]), Image.BILINEAR)
        lbl = lbl.resize((self.resize[1], self.resize[0]), Image.NEAREST)

        # Augmentations before transformation
        if torch.rand(1) < 0.5:
            img = transforms.functional.hflip(img)
            lbl = transforms.functional.hflip(lbl)
        if torch.rand(1) < 0.5:
            angle = (torch.rand(1) * 20 - 10).item()
            img = transforms.functional.rotate(img, angle, fill=(0, 0, 0))
            lbl = transforms.functional.rotate(lbl, angle, fill=255, interpolation=Image.NEAREST)

        # Label Conversion into macro-classes
        lbl_np = np.array(lbl, dtype=np.uint8)
        lbl_macro_np = self.mapping_array[lbl_np]
        
        if self.ood_mode:
            lbl_macro_np[lbl_np == 255] = self.unknown_obstacle_id
        lbl = torch.from_numpy(lbl_macro_np).long()

        # Transforming image into tensor
        if self.transform:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)
        return img, lbl

