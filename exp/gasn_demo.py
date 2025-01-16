import argparse
import logging

import cv2
from config import parser
from tqdm import tqdm
import torch
import numpy as np
from utils import Visualizer

from model.gasn import Gasn
from par_segment_anything.utils.transforms import FSCTransform
from par_segment_anything import sam_model_registry
from par_segment_anything.automatic_mask_generator import SamAutomaticMaskGenerator
import matplotlib.pyplot as plt
from dataset_gidas import GIDASDataset, DataLoader


def mask_filter(anns, max_area, min_area):
    if not anns:
        return []
    max_area *= 0.86
    min_area *= 0.3
    return [ann for ann in sorted(anns, key=lambda x: x['area'], reverse=True) if min_area <= ann['area'] <= max_area]


def show_anns(anns):
    if not anns:
        return
    ax = plt.gca()
    ax.set_autoscale_on(False)
    img = np.ones((anns[0]['segmentation'].shape[0], anns[0]['segmentation'].shape[1], 4))
    contor_mask = np.zeros((anns[0]['segmentation'].shape[0], anns[0]['segmentation'].shape[1]), dtype=np.uint8)
    color_contor = np.array([0, 0, 1, 0.6])
    img[:, :, 3] = 0
    for ann in anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.5]])
        img[m] = color_mask

        m_8 = m.astype(np.uint8)
        contours, _ = cv2.findContours(m_8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        max_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(contor_mask, [max_contour], -1, 255, thickness=2)
    contor_mask = np.where(contor_mask > 0)
    img[contor_mask] = color_contor
    ax.imshow(img)


def object_prompts_slide_window(boxes, gt_density, output):
    point_prompt = []
    max_area = float('-inf')
    min_area = float('inf')
    sum_window = 0
    sum_window_step = 0

    for box in boxes:
        x1, y1, x2, y2 = map(lambda x: int(x / 4), box)
        r_x1, r_y1, r_x2, r_y2 = map(int, box)
        w, h = abs(x2 - x1), abs(y2 - y1)
        r_w, r_h = r_x2 - r_x1, r_y2 - r_y1
        temp_area = r_w * r_h

        max_area = max(max_area, temp_area)
        min_area = min(min_area, temp_area)

        temp_window = min(w, h)
        sum_window += temp_window
        temp_window_step = max(3, int(sum_window / 5))
        sum_window_step += temp_window_step

    adaptive_window = int(sum_window / 3)
    adaptive_window_step = int(sum_window_step / 3)

    for y in range(0, gt_density.shape[1], adaptive_window_step):
        for x in range(0, gt_density.shape[2], adaptive_window_step):
            step = 0
            temp_gt = output[:, y + step:y + adaptive_window - step, x + step:x + adaptive_window - step].squeeze()
            temp_conf = torch.sum(temp_gt)
            if temp_conf > 0.8:
                point_prompt.append([x + int(adaptive_window / 2), y + int(adaptive_window / 2)])
    return point_prompt, (max_area, min_area)


def main(args: argparse.Namespace) -> None:
    cnt = 0
    visualizer = Visualizer()
    gidas_dataset = GIDASDataset(args.dataset, transform=FSCTransform())
    gidas_dataloader = DataLoader(gidas_dataset, batch_size=1, num_workers=0)

    gasn = Gasn(args.model_path, args.type, args.backbone)
    sam = sam_model_registry[args.type](checkpoint=args.backbone)
    sam.to(device=args.device)

    for image, image_embedding, boxes, dots, gt_density in tqdm(gidas_dataloader):
        output = gasn._train_predict(image_embedding, boxes)
        boxes_np = boxes.detach().cpu().squeeze().numpy()
        point_prompt, (max_area, min_area) = object_prompts_slide_window(boxes_np, gt_density, output)
        bgr_image = cv2.cvtColor(image.squeeze().detach().cpu().numpy(), cv2.COLOR_RGB2BGR)
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

        if len(point_prompt) > 0:
            for point in point_prompt:
                x = int(point[0] * 4)
                y = int(point[1] * 4)
                color_3 = (0, 0, 255)
                cv2.circle(bgr_image, (x, y), 6, color_3, -1)
            cv2.imshow("image", bgr_image)
            visualizer.show_feature(gt_density[:, 0:170], display="gt")
            visualizer.show_feature(output[:, 0:170], display="output")
            cv2.waitKey()
            input_prompt = np.array(point_prompt, dtype=float)
            # x / W 归一化
            input_prompt[:, 0] *= 4 / rgb_image.shape[1]
            input_prompt[:, 1] *= 4 / rgb_image.shape[0]
            mask_generator = SamAutomaticMaskGenerator(
                model=sam, points_per_side=None, point_grids=[input_prompt]
            )
            masks = mask_generator.generate(rgb_image)
            filter_anns = mask_filter(masks, max_area, min_area)
            plt.figure(figsize=(6, 5))
            plt.imshow(rgb_image)
            show_anns(filter_anns)
            plt.axis('off')
            plt.show()
            gasn_pred_cnt = len(filter_anns)
            gt_cnt = dots.item()
            err = abs(gasn_pred_cnt - gt_cnt)
            print("%d: %d,%d,%d,%d\n" % (cnt, gasn_pred_cnt, gt_cnt, len(point_prompt), err))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = parser.parse_args()
    main(args)
