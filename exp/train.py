import argparse
import json
import logging
import math
import os
import random
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from config import parser
from tqdm import tqdm
from utils import weights_normal_init

from dataset_fsc import FSCDataset, DataLoader
from model.pddn import PromptMatchModule, DensityMapGenerator
from par_segment_anything.utils.transforms import FSCTransform

parser.add_argument("-ckpt", "--checkpoint", type=str, default="./ckpt", help="Path to output logs")
parser.add_argument("-ep", "--epochs", type=int, default=50, help="number of training epochs")
parser.add_argument("-lr", "--learning-rate", type=float, default=1e-5, help="learning rate")
parser.add_argument("-bs", "--batchsize", type=int, default=8, help="number of training epochs")
args = parser.parse_args()

train_dataset = FSCDataset(R"D:\datasets\FSC147_384_V2", part="train", transform=FSCTransform())
val_dataset = FSCDataset(R"D:\datasets\FSC147_384_V2", part="val", transform=FSCTransform())
train_dataloader = DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True, num_workers=0)
val_dataloader = DataLoader(val_dataset, batch_size=args.batchsize, num_workers=0)

match_net = PromptMatchModule()
density_decoder = DensityMapGenerator()
weights_normal_init(density_decoder, dev=0.001)
density_decoder.to(device=args.device)
optimizer = optim.Adam(density_decoder.parameters(), lr=args.learning_rate)
criterion = nn.MSELoss().cuda()
transform = FSCTransform()


def train():
    train_mae, train_rmse, train_loss, cnt = 0, 0, 0, 0
    qbar = tqdm(train_dataloader)
    for image_embedding, boxes, dots, gt_density in qbar:
        density_map = match_net.predict(image_embedding, boxes)
        image_embedding = density_map * image_embedding
        image_embedding.requires_grad = True
        optimizer.zero_grad()
        output = density_decoder(image_embedding).squeeze(1)
        loss = criterion(output, gt_density)
        loss.backward()
        optimizer.step()
        for batch_id in range(output.shape[0]):
            pred_cnt = torch.sum(output[batch_id]).item()
            gt_cnt = torch.sum(gt_density[batch_id]).item()
            cnt_err = abs(pred_cnt - gt_cnt)
            cnt += 1
            train_loss += loss.item()
            train_mae += cnt_err
            train_rmse += cnt_err ** 2
        qbar.set_description(
            'actual-predicted: error: {:6.1f}. Current MAE: {:5.2f}, RMSE: {:5.2f}'.format(
                cnt_err, train_mae / cnt, (train_rmse / cnt) ** 0.5))
    train_loss = train_loss / len(train_dataset)
    train_mae = (train_mae / len(train_dataset))
    train_rmse = (train_rmse / len(train_dataset)) ** 0.5
    return train_loss, train_mae, train_rmse


def eval():
    density_decoder.eval()
    cnt, SAE, SSE = 0, 0, 0
    print("Evaluation on {} data".format(args.part))
    qbar = tqdm(val_dataloader)
    for image_embedding, boxes, dots, gt_density in qbar:
        density_map = match_net.predict(image_embedding, boxes)
        image_embedding = density_map * image_embedding
        output = density_decoder(image_embedding).squeeze(1)
        for batch_id in range(output.shape[0]):
            pred_cnt = torch.sum(output[batch_id]).item()
            gt_cnt = torch.sum(dots[batch_id]).item()
            err = abs(pred_cnt - gt_cnt)
            cnt += 1
            SAE += err
            SSE += err ** 2
        qbar.set_description(
            'actual-predicted: error: {:6.1f}. Current MAE: {:5.2f}, RMSE: {:5.2f}'.format(err, SAE / cnt,
                                                                                           (SSE / cnt) ** 0.5))
    print('On {} data, MAE: {:6.2f}, RMSE: {:6.2f}'.format(args.part, SAE / cnt, (SSE / cnt) ** 0.5))
    return SAE / cnt, (SSE / cnt) ** 0.5


def main():
    best_mae, best_rmse = 1e7, 1e7
    stats = list()
    for epoch in range(0, args.epochs):
        train_loss, train_mae, train_rmse = train()
        val_mae, val_rmse = eval()
        stats.append((train_loss, train_mae, train_rmse, val_mae, val_rmse))
        stats_file = os.path.join(args.log_dir, "pddn_xxxx" + ".txt")
        with open(stats_file, 'w') as f:
            for s in stats:
                f.write("%s\n" % ','.join([str(x) for x in s]))
        epoch_name = str((epoch // 50) * 50 + 50)
        if best_mae >= val_mae:
            best_mae = val_mae
            best_rmse = val_rmse
            model_name = args.checkpoint + '/' + "pddn_xxxx" + epoch_name + ".pth"
            torch.save(density_decoder.state_dict(), model_name)
        print(
            "Epoch {}, Avg. Epoch Loss: {} Train MAE: {} Train RMSE: {} Val MAE: {} Val RMSE: {} Best Val MAE: {} Best Val RMSE: {} ".format(
                epoch + 1, stats[-1][0], stats[-1][1], stats[-1][2], stats[-1][3], stats[-1][4], best_mae, best_rmse))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
