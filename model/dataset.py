#######  Dataset ############

import labelme
from PIL import Image
import numpy as np
import torch.utils.data
import torchvision.transforms as T
import glob
import os.path
import torch
from typing import Tuple


ANNOTATION_PATH = "data"
CLASS_LABELS = ["bg", "sample", "defect"]


class CTDataset(torch.utils.data.Dataset):

    def __init__(self, root, transforms=None) -> None:
        """
        root:       directory with labelme jsons
        transforms: albumentation transforms/augs
        """
        super().__init__()
        self.root = root
        self.files = glob.glob(os.path.join(self.root, "*.json"))
        self.transforms = transforms

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        lf = labelme.LabelFile(self.files[index])
        image = labelme.utils.img_data_to_arr(lf.imageData)
        masks = {}
        for shape in lf.shapes:
            label = shape["label"].lower()
            assert label in ["sample", "defect"], label
            masks[label] = labelme.utils.shape_to_mask(image.shape, shape["points"])
        mask_array = np.zeros_like(image)
        # Be careful, first goes sample, then defect overlays
        for i, label in enumerate(CLASS_LABELS):
            if label in masks:
                mask_array[masks[label]] = i
        
        if self.transforms:
            transformed = self.transforms(image=image, mask=mask_array)
            image = transformed["image"]
            mask_array = transformed["mask"]
        
        image_tensor = torch.from_numpy(image)
        image_tensor.unsqueeze_(0)
        mask_tensor = torch.from_numpy(mask_array).to(dtype=torch.long)

        return image_tensor, mask_tensor

    def __len__(self):
        return len(self.files)
