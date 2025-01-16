# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import List, Tuple, Type
from exp.utils import Visualizer
import torch
from torch import nn
from torch.nn import functional as F


class PromptMatchModule:
    def __init__(self) -> None:
        super().__init__()
        self._box_scale = 16.0  # 1024 / 64
        # self._sample_scales = [(x+1)/16 for x in range(64)]
        self._sample_scales = [0.9, 1.0, 1.1]

    def _scale_boxes(
            self, box_coords: torch.Tensor, original_size: Tuple[int, ...]
    ) -> torch.Tensor:
        feat_h, feat_w = original_size
        # box_coords: BxNx4  (x1, y1, x2, y2)
        box_coords = box_coords / self._box_scale
        box_coords[:, :, 0:2] = torch.floor(box_coords[:, :, 0:2])  # 向下取整
        box_coords[:, :, 2:4] = torch.ceil(box_coords[:, :, 2:4])  # 向上取整
        box_coords[:, :, 2:4] = box_coords[:, :, 2:4] + 1  # make the end indices exclusive
        box_coords[:, :, 0:2] = torch.clamp_min(box_coords[:, :, 0:2], 0)
        box_coords[:, :, 3] = torch.clamp_max(box_coords[:, :, 3], feat_h)
        box_coords[:, :, 2] = torch.clamp_max(box_coords[:, :, 2], feat_w)
        return box_coords

    @staticmethod
    def _crop_embedding(boxes: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        # boxes: Nx4
        # embedding: CxHxW
        box_hs = boxes[:, 3] - boxes[:, 1]
        box_ws = boxes[:, 2] - boxes[:, 0]
        max_h = math.ceil(max(box_hs))
        max_w = math.ceil(max(box_ws))
        # box_embeddings: NxCxHxW
        box_embeddings = None
        for box_id in range(0, boxes.shape[0]):
            x1, y1 = int(boxes[box_id, 0]), int(boxes[box_id, 1])
            x2, y2 = int(boxes[box_id, 2]), int(boxes[box_id, 3])
            if x1 > x2:
                temp = x1
                x1 = x2
                x2 = temp
            if y1 > y2:
                temp = y1
                y1 = y2
                y2 = temp
            crop_embedding = embedding[:, y1:y2, x1:x2].unsqueeze(0)
            if crop_embedding.shape[1] != max_h or crop_embedding.shape[2] != max_w:
                crop_embedding = F.interpolate(crop_embedding, size=(max_h, max_w), mode="bilinear")
            if box_embeddings is not None:
                box_embeddings = torch.cat((box_embeddings, crop_embedding), dim=0)
            else:
                box_embeddings = crop_embedding
        return box_embeddings

    @staticmethod
    def _scale_embedding(
            embedding: torch.Tensor, original_size: Tuple[int, ...], scale: float
    ) -> Tuple[torch.Tensor, Tuple[int, ...]]:
        # embedding: NxCxHxW
        h, w = original_size
        h_new = math.ceil(h * scale)
        w_new = math.ceil(w * scale)
        if h_new < 1:  # use original size if scaled size is too small
            h_new = h
        if w_new < 1:
            w_new = w
        # embedding_scaled: NxCxHxW
        embedding_scaled = F.interpolate(embedding, size=(h_new, w_new), mode="bilinear")
        return embedding_scaled, (h_new, w_new)

    @staticmethod
    def _match_embedding(
            sample_embedding: torch.Tensor, image_embedding: torch.Tensor, pad_size
    ) -> torch.Tensor:
        # sample_embedding: NxCxHxW
        # image_embedding: CxHxW
        pad_h, pad_w = pad_size
        embedding_pad = F.pad(
            image_embedding,
            ((int(pad_w / 2)), int((pad_w - 1) / 2), int(pad_h / 2), int((pad_h - 1) / 2)),
        )
        # embedding_matched: NxHxW
        # sample_embedding = torch.repeat_interleave(sample_embedding, image_embedding.shape[0], 0)
        embedding_matched = F.conv2d(embedding_pad, sample_embedding)
        return embedding_matched

    def _multiscale_match(
            self, sample_embedding: torch.Tensor, image_embedding: torch.Tensor
    ) -> torch.Tensor:
        """
        Convolving example features over image features
        """
        # sample_embedding: NxCxHxW
        # image_embedding: CxHxW
        # mm_embedding: (NxS)xHxW
        mm_embedding = None  # multiscale matched embedding
        for scale in self._sample_scales:
            embedding_size = (sample_embedding.shape[2], sample_embedding.shape[3])
            sample_embedding_scaled, pad_size = self._scale_embedding(
                sample_embedding, embedding_size, scale
            )
            embedding_matched = self._match_embedding(
                sample_embedding_scaled, image_embedding, pad_size
            )
            embedding_matched = torch.reshape(embedding_matched, (-1, 64, 64))
            if mm_embedding is not None:
                mm_embedding = torch.cat((mm_embedding, embedding_matched), dim=0)
            else:
                mm_embedding = embedding_matched
        return mm_embedding

    def predict(
            self, image_embeddings: torch.Tensor, box_coords: torch.Tensor
    ) -> torch.Tensor:
        # boxes_scaled:
        # batch_density_map: BxCxHxW
        boxes_scaled = self._scale_boxes(box_coords, (64, 64))
        batch_density_map = None
        for batch_id in range(box_coords.shape[0]):
            examples_features = self._crop_embedding(
                boxes_scaled[batch_id], image_embeddings[batch_id]
            )
            density_map = self._multiscale_match(examples_features, image_embeddings[batch_id])
            density_map = density_map.unsqueeze(0)
            if batch_density_map is not None:
                batch_density_map = torch.cat((batch_density_map, density_map), dim=0)
            else:
                batch_density_map = density_map
        batch_density_map = torch.mean(batch_density_map, dim=1, keepdim=True)
        # batch_density_map = torch.max(batch_density_map, 1, keepdim=True)
        return batch_density_map


class ResidualBlock(torch.nn.Module):
    def __init__(self, channels, kernel_size, padding):
        super(ResidualBlock, self).__init__()
        self.channels = channels
        self.conv1 = torch.nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding)
        self.conv2 = torch.nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding)

    def forward(self, x):
        y = F.relu(self.conv1(x))
        y = self.conv2(y)
        return F.relu(x + y)


class DensityMapGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
        self.up_sampling = nn.UpsamplingBilinear2d(scale_factor=2)
        self.res1 = ResidualBlock(256, 7, 3)
        self.conv2 = nn.Conv2d(256, 128, 5, padding=2)
        self.res2 = ResidualBlock(128, 5, 2)
        self.conv3 = nn.Conv2d(128, 64, 3, padding=1)
        self.res3 = ResidualBlock(64, 3, 1)
        self.conv4 = nn.Conv2d(64, 32, 1)
        self.res4 = ResidualBlock(32, 1, 0)
        self.conv5 = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        output = self.res1(x)
        output = self.up_sampling(output)
        output = self.relu(self.conv2(output))
        output = self.res2(output)
        output = self.up_sampling(output)
        output = self.relu(self.conv3(output))
        output = self.res3(output)
        output = self.relu(self.conv4(output))
        output = self.res4(output)
        output = self.relu(self.conv5(output))
        return output


class PromptDrivenDensityNetwork(nn.Module):
    def __init__(self, model_path, device):
        super().__init__()
        self.vis = Visualizer()
        self.pmm = PromptMatchModule()
        self.dmg = DensityMapGenerator()
        self.dmg.load_state_dict(torch.load(model_path))
        self.dmg.to(device=device)

    def predict(self, image_embedding, boxes):
        density_map = self.pmm.predict(image_embedding, boxes)
        # print("density_map: ", density_map.shape)
        # self.vis.show_feature(density_map[:, :, 0:42, :], display="match")
        attention_embedding = density_map * image_embedding
        # print("attention_embedding: ", attention_embedding.shape)
        # self.vis.show_feature(attention_embedding[:, :, 0:42, :], display="match")
        output = self.dmg(attention_embedding).squeeze(1)
        return output

    def forward(self, image_embedding, boxes):
        density_map = self.pmm.predict(image_embedding, boxes)
        # self.vis.show_feature(density_map[:, :, 0:42, :], display="match")
        attention_embedding = density_map * image_embedding
        output = self.dmg(attention_embedding).squeeze(1)
        return output