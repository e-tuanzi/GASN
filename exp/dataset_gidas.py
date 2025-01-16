import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import json
import cv2


class GIDASDataset(Dataset):
    def __init__(self, dir, transform=None):
        self.image_dir = os.path.join(dir, "images")
        self.pt_dir = os.path.join(dir, "GIDAS_H_EMB")
        self.gt_dir = os.path.join(dir, "gt_density")
        ann_file = os.path.join(dir, "annotation_GIDAS.json")
        with open(ann_file) as f:
            self.annotations = json.load(f)
        # some of dataset
        # with open(os.path.join(dir, "c_mung_bean.txt"), 'r') as f:
        #     res = f.readlines()
        #     for i in res:
        #         line = i.strip('\n')
        #         image_list.append(line)
        # self.image_list = image_list
        # one data
        self.image_list = ["103.jpg"]
        # all data
        # self.image_list = os.listdir(self.image_dir)
        self.image_list.sort(key=lambda x: int(x.split('.')[0]))
        self.transform = transform

    def __getitem__(self, item):
        image_file = self.image_list[item]
        ann = self.annotations[image_file]
        bgr_image = cv2.imread(os.path.join(self.image_dir, image_file))
        image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        resized_image = self.transform.transform.apply_image(image)
        boxes = ann['example_boxes']
        boxes = np.array(boxes)
        boxes = self.transform.transform.apply_boxes(boxes[None, :], image.shape[:2])
        boxes_torch = torch.as_tensor(boxes, dtype=torch.float, device=self.transform.device)
        boxes_torch = boxes_torch[None, :].squeeze(0)
        dots = np.array(ann['points'])
        pt_path = os.path.join(self.pt_dir, image_file.split(".jpg")[0] + ".pt")
        image_embedding = torch.load(pt_path).squeeze(0)
        density_path = os.path.join(self.gt_dir, image_file.split(".jpg")[0] + ".npy")
        density = np.load(density_path).astype('float32')
        gt_density = self.transform.resize_gt(density)
        return resized_image, image_embedding, boxes_torch, len(dots), gt_density

    def __len__(self):
        return len(self.image_list)

