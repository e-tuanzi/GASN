import os
import cv2
from par_segment_anything.utils.transforms import FSCTransform
import json
import numpy as np

dataset_path = r"D:\datasets\GIDAS_V2"
image_dir = os.path.join(dataset_path, "images")
image_path = os.listdir(image_dir)
image_path.sort(key=lambda x: int(x.split('.')[0]))
json_path = os.path.join(dataset_path, "annotation_GIDAS.json")

def classes_info(name):
    image_id = int(name.split('.')[0])
    classes = None
    if 1 <= image_id <= 12 or 113 <= image_id <= 114:
        classes = "red_bean"            # 红豆
    if 13 <= image_id <= 24:
        classes = "mung_bean"           # 绿豆
    if 25 <= image_id <= 36 or 115 <= image_id <= 116:
        classes = "soybean"             # 黄豆
    if 37 <= image_id <= 48:
        classes = "red_kidney_bean"     # 红芸豆
    if 49 <= image_id <= 60:
        classes = "black_soybean"       # 黑豆
    if 61 <= image_id <= 72:
        classes = "coix_seed"           # 薏仁米
    if 73 <= image_id <= 84:
        classes = "buckwheat"           # 荞麦
    if 85 <= image_id <= 96 or 111 <= image_id <= 112:
        classes = "oat"                 # 燕麦
    if 97 <= image_id <= 108 or 109 <= image_id <= 110:
        classes = "rice"                # 大米
    return classes

with open(json_path, "r", encoding='utf-8') as f:
    ann = json.load(f)
transformer = FSCTransform()
box_count = 0
point_count = 0
for name in image_path:
    if name.endswith('.jpg'):
        print(name)
        image = cv2.imread(os.path.join(image_dir, name))
        info = ann[name]
        boxes = info["example_boxes"]
        points = info["points"]
        box_count += len(boxes)
        point_count += len(points)
        print("classes:", classes_info(name), "points num:", len(points))
        dw_boxes = []
        dw_points = []
        for item in boxes:
            x1 = item[0]
            y1 = item[1]
            x2 = item[2]
            y2 = item[3]
            rect = [x1, y1, x2, y2]
            dw_rect = transformer.transform.apply_boxes(np.array(rect), image.shape[:2])
            dw_boxes.append(dw_rect[0])
        for item in points:
            point = item
            dw_point = transformer.transform.apply_coords(np.array(point), image.shape[:2])
            dw_points.append(dw_point)
        print("------------------------------------------------------------")
        rs_image = transformer.transform.apply_image(image)
        for rect in dw_boxes:
            x1, y1, x2, y2 = rect
            cv2.rectangle(rs_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 4)
        for point in dw_points:
            # 由于cv2.circle参数需要整形，不需要浮点型，这里要对参数做个变换
            x = int(point[0])
            y = int(point[1])
            color = np.random.choice(range(256), size=3)
            color = tuple([int(x) for x in color])
            cv2.circle(rs_image, (x, y), 4, color, 3)
        cv2.imshow("image", rs_image)
        cv2.waitKey()
