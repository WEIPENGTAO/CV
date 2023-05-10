import os

import cv2
import numpy as np
from sklearn.cluster import KMeans
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

# 将特征列表转换为numpy数组
features = np.concatenate(features)

# 使用K均值聚类构建词袋模型
kmeans = KMeans(n_clusters=100, n_init=10)  # 设置聚类中心数量，n_init表示聚类的初始尝试次数
kmeans.fit(features)

# 定义存储词袋特征的列表
bag_of_features = []

# 计算每张图像的词袋特征
for file_name in tqdm(os.listdir(dataset_path), desc='Calculating bag-of-features'):
    # 读取图像
    img = cv2.imread(os.path.join(dataset_path, file_name))
    # 转换成灰度图像
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 直方图均衡化
    gray = cv2.equalizeHist(gray)
    # 提取SIFT特征，返回关键点和描述符
    keypoints, descriptor = sift.detectAndCompute(gray, None)
    # 使用K均值聚类将描述符映射到词袋中心
    labels = kmeans.predict(descriptor)
    # 统计每个词袋中心的频次
    hist, _ = np.histogram(labels, bins=np.arange(101))
    # 归一化特征向量
    hist = hist.astype(float)
    hist /= np.sum(hist)
    # 将特征向量添加到列表中
    bag_of_features.append(hist)

# 将词袋特征列表转换为numpy数组
bag_of_features = np.array(bag_of_features)

# 读取查询图像，并提取SIFT特征
query_img = cv2.imread('./ashmolean_000000.jpg')
query_gray = cv2.cvtColor(query_img, cv2.COLOR_BGR2GRAY)
query_keypoints, query_descriptor = sift.detectAndCompute(query_gray, None)
# 使用K均值聚类将查询图像的描述符映射到词袋中心
query_labels = kmeans.predict(query_descriptor)
# 统计每个词袋中心的频次
query_hist, _ = np.histogram(query_labels, bins=np.arange(101))
# 归一化特征向量
query_hist = query_hist.astype(float)
query_hist /= np.sum(query_hist)

# 计算查询图像与数据集中每张图像的相似度
scores = np.sum(np.minimum(bag_of_features, query_hist), axis=1)

# 获取相似度最高的前5张图像的索引
top5 = np.argsort(scores)[::-1][:5]

# 显示查询图像
cv2.imshow('Query image', query_img)
cv2.waitKey(0)

# 显示相似度最高的前5张图像
for i in range(5):
    img = cv2.imread(os.path.join(dataset_path, os.listdir(dataset_path)[top5[i]]))
    cv2.imshow('Top {}'.format(i + 1), img)
    cv2.waitKey(0)

cv2.destroyAllWindows()
