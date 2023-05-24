import os

import cv2
import numpy as np
from tqdm import tqdm

# 定义数据集路径
dataset_path = './test_data'

# 初始化SIFT特征提取器
sift = cv2.SIFT_create()

# 定义存储特征的列表
features = []

# 读取数据集中的所有图片，并提取SIFT特征
for file_name in tqdm(os.listdir(dataset_path), desc='Extracting features'):
    # 读取图像
    img = cv2.imread(os.path.join(dataset_path, file_name))
    # 转换成灰度图像
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 直方图均衡化
    gray = cv2.equalizeHist(gray)
    # 提取SIFT特征，返回关键点和描述符
    keypoints, descriptor = sift.detectAndCompute(gray, None)
    # 将特征添加到列表中
    features.append(descriptor)

# 读取查询图像，并提取SIFT特征
query_img = cv2.imread('./ashmolean_000000.jpg')
query_gray = cv2.cvtColor(query_img, cv2.COLOR_BGR2GRAY)
query_keypoints, query_descriptor = sift.detectAndCompute(query_gray, None)

# 创建FLANN匹配器
flann = cv2.FlannBasedMatcher({'algorithm': 0, 'trees': 5}, {'checks': 50})

# 计算查询图像与数据集中所有图像的SIFT特征距离
distances = []
for feature in tqdm(features, desc='Calculating distances'):
    matches = flann.knnMatch(feature, query_descriptor, k=2)
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)
    # 如果没有匹配的特征，则距离为1
    distance = 1
    if len(good_matches) != 0:
        distance = 1 - len(good_matches) / len(matches)
    distances.append(distance)

# 根据SIFT特征距离排序，找出与查询图像最相似的图像
indices = np.argsort(distances)[:5]

# 显示结果，将5张图显示在不同的窗口
for index in indices:
    img = cv2.imread(os.path.join(dataset_path, os.listdir(dataset_path)[index]))
    cv2.imshow('result', img)
    cv2.waitKey(0)

cv2.destroyAllWindows()
