import json
import cv2
import torch
from config import parser
from tqdm import tqdm
from par_segment_anything import SamPredictor, sam_model_registry
import h5py
import scipy.io as sio
import numpy as np
import os
import glob
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.spatial import KDTree
import scipy
import warnings

warnings.filterwarnings('ignore')

parser.add_argument("--output", type=str, default=r"D:\datasets\GIDAS_V3", help="Path to output logs")
args = parser.parse_args()


def fsc_preprocess():
    image_dir = os.path.join("D:\datasets\FSC147_384_V2", "images_384_VarV2")
    dataset_split_file = os.path.join(args.dataset, "Train_Test_Val_FSC_147.json")
    with open(dataset_split_file) as f:
        dataset_split = json.load(f)
    sam = sam_model_registry[args.type](checkpoint=args.ckpt)
    sam.to(device=args.device)
    predictor = SamPredictor(sam)
    images_list = dataset_split["train"]
    print(len(images_list))
    pbar = tqdm(images_list)
    for image_id in pbar:
        bgr_image = cv2.imread(os.path.join(image_dir, image_id))
        image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        predictor.set_image(image)
        image_embedding = predictor.get_image_embedding()
        torch.save(image_embedding, os.path.join(r"D:\datasets\FSC147_384_V3\FSC147_H_EMB", image_id[:-4] + ".pt"))


def gidas_preprocess():
    image_dir = os.path.join("D:\datasets\GIDAS_V3", "images")
    sam = sam_model_registry[args.type](checkpoint=args.ckpt)
    sam.to(device=args.device)
    predictor = SamPredictor(sam)
    images_list = os.listdir(image_dir)
    print(len(images_list))
    pbar = tqdm(images_list)
    for image_id in pbar:
        bgr_image = cv2.imread(os.path.join(image_dir, image_id))
        image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        predictor.set_image(image)
        image_embedding = predictor.get_image_embedding()
        torch.save(image_embedding, os.path.join("D:\datasets\GIDAS_V3\GIDAS_H_EMB", image_id[:-4] + ".pt"))


def generate_density_mat():
    image_dir = os.path.join("D:\datasets\GIDAS_V3", "images")
    images_list = os.listdir(image_dir)
    ann_file = os.path.join("D:\datasets\GIDAS_V3", "annotation_GIDAS.json")
    with open(ann_file) as f:
        ann = json.load(f)
    for image in images_list:
        image_info = ann[image]
        point_data = image_info["points"]
        print(len(point_data))
        data_inner = {'location': point_data, 'number': len(point_data)}
        image_info = np.zeros((1,), dtype=object)
        image_info[0] = data_inner
        name = image.split('.')[0] + '.mat'
        sio.savemat(os.path.join('D:\datasets\GIDAS_V3\label_mat', name), {'image_info': image_info})


def gaussian_filter_density(gt):
    print(gt.shape)
    density = np.zeros(gt.shape, dtype=np.float32)
    gt_count = np.count_nonzero(gt)
    if gt_count == 0:
        return density
    pts = np.array(list(zip(np.nonzero(gt)[1], np.nonzero(gt)[0])))
    leafsize = 2048
    tree = KDTree(pts.copy(), leafsize=leafsize)
    distances, locations = tree.query(pts, k=4)
    print('generate density...')
    for i, pt in enumerate(pts):
        pt2d = np.zeros(gt.shape, dtype=np.float32)
        pt2d[pt[1], pt[0]] = 1.
        if gt_count > 1:
            sigma = (distances[i][1] + distances[i][2] + distances[i][3]) * 0.03
        else:
            sigma = np.average(np.array(gt.shape)) / 2. / 2.  # case: 1 point
        density += scipy.ndimage.filters.gaussian_filter(pt2d, sigma, mode='constant')
    print('done.')
    print('density:', density.sum())
    print("-------------------------")
    return density


def generate_density_map():
    #  set the root to the dataset
    root = r'D:\datasets\GIDAS_V2'
    path_sets = os.path.join(root, 'images')
    img_paths = []
    for img_path in glob.glob(os.path.join(path_sets, '*.jpg')):
        img_paths.append(img_path)
    ann_file = os.path.join("D:\datasets\GIDAS_V3", "annotation_GIDAS.json")
    with open(ann_file) as f:
        ann = json.load(f)
    for p, img_path in enumerate(img_paths):
        (_path, filename) = os.path.split(img_path)
        image_info = ann[filename]
        point_data = image_info["points"]
        print("gt num", len(point_data))
        mat = sio.loadmat(img_path.replace('.jpg', '.mat').replace('images', 'label_mat'))
        img = plt.imread(img_path)
        k = np.zeros((img.shape[0], img.shape[1]))
        gt = mat["image_info"][0, 0][0, 0][0]
        for i in range(0, len(gt)):
            if int(gt[i][1]) < img.shape[0] and int(gt[i][0]) < img.shape[1]:
                k[int(gt[i][1]), int(gt[i][0])] = 1
        k = gaussian_filter_density(k)

        name = os.path.basename(os.path.realpath(img_path)).split('.')[0]
        np.save("D:\datasets\GIDAS_V3\gt_density\{}.npy".format(name), k)

        plt.subplot(121)
        plt.imshow(k)
        plt.subplot(122)
        plt.imshow(img)
        plt.savefig(r"D:\datasets\GIDAS_V3\gt_image\tem{}.jpg".format(p))
        with h5py.File(img_path.replace('.jpg', '.h5').replace('images', 'h5'), 'w') as hf:
            hf['density'] = k
    plt.clf()

if __name__ == "__main__":
    # # generate FSC147 image embedding
    # fsc_preprocess()
    # # generate GIDAS image embedding
    # gidas_preprocess()
    # # generate GIDAS density map step 1, transform file type to mat
    # # it need only to run once
    # generate_density_mat()
    # # generate GIDAS density map step 2
    # # if set different sigma, it must be rerun
    # generate_density_map()
    pass
