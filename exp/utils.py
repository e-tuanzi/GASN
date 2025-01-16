from copy import deepcopy

import cv2
import numpy as np
import torch
import torch.nn as nn





class Visualizer:
    def __init__(self, image=None):
        self.color_map = cv2.COLORMAP_JET
        self.size = (256, 256)
        self.size = (256, 168)
        self.image = image

    def draw_fsc147(self, image: np.ndarray, bboxes: list, dots: list) -> np.ndarray:
        bboxes_xyxy = []
        for bbox in bboxes:
            # 单个bbox由矩形的四个顶点组成，从左上角顺时针排序，而矩形都不是旋转矩形
            # 可以直接通过左上角和右下角两点构成矩形
            x1 = bbox[0][0]
            y1 = bbox[0][1]
            x2 = bbox[2][0]
            y2 = bbox[2][1]
            rect = [x1, y1, x2, y2]
            bboxes_xyxy.append(rect)
        for rect in bboxes_xyxy:
            x1, y1, x2, y2 = rect
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        for point in dots:
            # 由于cv2.circle参数需要整形，不需要浮点型，这里要对参数做个变换
            x = int(point[0])
            y = int(point[1])
            cv2.circle(image, (x, y), 1, (255, 0, 0), -1)
        return image

    def show_feature(self, feature, device="cuda", display=["image_embedding"], count=0):
        if device == "cuda":
            feature = feature.detach().cpu().numpy()
        if "image_embedding" in display:
            self._show_image_embedding(feature)
        if "match" in display:
            self._show_match(feature)
        if "gt" in display:
            self._show_gt(feature, count)
        if "output" in display:
            self._show_output(feature, count)

    def _show_image_embedding(self, image_embedding):

        batch_num = image_embedding.shape[0]
        map_num = image_embedding.shape[1]
        for batch_id in range(batch_num):
            count = 0
            all_heat = None
            line_heat = None
            for map_id in range(map_num):
                heatmap = self._draw_heatmap(image_embedding[batch_id][map_id])
                if line_heat is not None:
                    line_heat = cv2.hconcat([line_heat, heatmap])
                else:
                    line_heat = heatmap
                count += 1
                if count % 3 == 0:
                    if all_heat is not None:
                        all_heat = cv2.vconcat([all_heat, line_heat])
                    else:
                        all_heat = line_heat
                    line_heat = None
                if count == 9:
                    break
            cv2.imshow("image_embedding", all_heat)
            # cv2.waitKey(0)

    def _show_match(self, match):
        batch_num = match.shape[0]
        map_num = match.shape[1]
        count = 0
        for batch_id in range(batch_num):
            all_heat = None
            line_heat = None
            print("map num", map_num)
            for map_id in range(map_num):
                # heatmap = self._draw_heatmap(combined[example_id][map_id])
                heatmap = self._draw_heatmap(match[batch_id][map_id])
                cv2.imshow("match", heatmap)
                # cv2.imwrite("../data/att_{}.tif".format(map_id), heatmap)
                # if line_heat is not None:
                #     line_heat = cv2.hconcat([line_heat, heatmap])
                # else:
                #     line_heat = heatmap
                # count += 1
                # if count % 3 == 0:
                #     if all_heat is not None:
                #         all_heat = cv2.vconcat([all_heat, line_heat])
                #     else:
                #         all_heat = line_heat
                #     line_heat = None
                cv2.waitKey(0)
            # cv2.imshow("match", all_heat)

    def _show_gt(self, gt, count=0):
        batch_num = gt.shape[0]
        for batch_id in range(batch_num):
            heatmap = self._draw_heatmap(gt[batch_id])
            cv2.imshow("gt", heatmap)
            # cv2.imwrite("../data/gt_{}.tif".format(count), heatmap)
        # cv2.waitKey(s0)


    def _show_output(self, output, count=0):
        output = output.squeeze()
        heatmap = self._draw_heatmap(output)
        cv2.imshow("output", heatmap)
        # cv2.imwrite("../data/output_{}.tif".format(count), heatmap)
        # cv2.waitKey(0)

    def _draw_heatmap(self, feature):
        heatmap = None
        heatmap = cv2.normalize(
            feature, heatmap, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U
        )
        heatmap = cv2.resize(heatmap, self.size, interpolation=cv2.INTER_NEAREST)  # 改变特征图尺寸
        heatmap = cv2.applyColorMap(heatmap, self.color_map)  # 变成伪彩图
        # heatmap = np.asarray(heatmap, np.float64)
        # self.image = np.asarray(self.image, np.float64)
        # overlapping = cv2.addWeighted(self.image, 0.5, heatmap, 0.5, 0)
        return heatmap

def weights_normal_init(model, dev=0.01):
    if isinstance(model, list):
        for m in model:
            weights_normal_init(m, dev)
    else:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, dev)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, dev)

def weights_xavier_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('relu'))
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)