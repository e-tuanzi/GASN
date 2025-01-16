import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import json
import cv2


class FSCDataset(Dataset):
    def __init__(self, dir, part="train", transform=None):
        self.image_dir = os.path.join(dir, "images_384_VarV2")
        self.pt_dir = os.path.join(dir, "FSC147_H_EMB")
        self.gt_dir = os.path.join(dir, "gt_density_map_adaptive_384_VarV2")
        ann_file = os.path.join(dir, "annotation_FSC147_384.json")
        dataset_split_file = os.path.join(dir, "Train_Test_Val_FSC_147.json")
        with open(ann_file) as f:
            self.annotations = json.load(f)
        with open(dataset_split_file) as f:
            dataset_split = json.load(f)
        self.image_list = dataset_split[part]
        self.transform = transform

    def __getitem__(self, item):
        image_file = self.image_list[item]
        ann = self.annotations[image_file]
        bgr_image = cv2.imread(os.path.join(self.image_dir, image_file))
        image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        boxes = self.transform.box2xyxy(ann['box_examples_coordinates'], image.shape[:2]).squeeze(0)
        dots = np.array(ann['points'])
        pt_path = os.path.join(self.pt_dir, image_file.split(".jpg")[0] + ".pt")
        image_embedding = torch.load(pt_path).squeeze(0)
        density_path = os.path.join(self.gt_dir, image_file.split(".jpg")[0] + ".npy")
        density = np.load(density_path).astype('float32')
        gt_density = self.transform.resize_gt(density)
        return image_embedding, boxes, len(dots), gt_density

    def __len__(self):
        return len(self.image_list)

