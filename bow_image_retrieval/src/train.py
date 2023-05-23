# python train.py -dataset 数据集路径 -save 保存模型的路径

import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm
import joblib
import argparse


# 保存K均值聚类模型和词袋特征
class Model:
    def __init__(self, kmeans, bag_of_features, images_path):
        self.kmeans = kmeans
        self.bag_of_features = bag_of_features
        self.num_words = kmeans.n_clusters
        self.image_paths = images_path


# 解析命令行参数
def parse_arguments():
    parser = argparse.ArgumentParser(description='Train the model')
    parser.add_argument('-dataset', type=str, help='Path to the dataset')
    parser.add_argument('-save', type=str, help='Path to save the model')
    return parser.parse_args()


def main(dataset_path, save_path):
    # 定义数据集路径
    train_path = dataset_path
    images_list = os.listdir(train_path)

    # 读取数据集中的前1500张图片
    images_list = images_list[:1500]
    images_path = [os.path.join(train_path, image_path) for image_path in images_list]

    # 初始化SIFT特征提取器
    sift = cv2.SIFT_create()

    # 存储所有描述符的列表
    descriptors_list = []  # 特征描述符

    # 读取数据集中的所有图片，并提取SIFT特征
    for image_path in tqdm(images_path, desc='Extracting features'):
        img = cv2.imread(image_path)  # 读取图像
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转换成灰度图像
        gray = cv2.equalizeHist(gray)  # 直方图均衡化
        keypoints, descriptors = sift.detectAndCompute(gray, None)  # 提取SIFT特征，返回关键点和描述符
        descriptors_list.append(descriptors)  # 将特征添加到列表中

    # 将特征列表转换为numpy数组
    descriptor_list = np.concatenate(descriptors_list)

    # 使用K均值聚类构建词袋模型
    kmeans = KMeans(n_clusters=100, n_init=5)  # 设置聚类中心数量，n_init表示聚类的初始尝试次数
    kmeans.fit(descriptor_list)

    # 定义存储词袋特征的列表
    bag_of_features = []

    # 计算每张图像的词袋特征
    for descriptors in tqdm(descriptors_list, desc='Calculating bag-of-features'):
        labels = kmeans.predict(descriptors)  # 使用K均值聚类将描述符映射到词袋中心
        hist, _ = np.histogram(labels, bins=np.arange(101))  # 统计每个词袋中心的频次
        hist = hist.astype(float)  # 归一化特征向量
        hist /= np.sum(hist)
        bag_of_features.append(hist)  # 将特征向量添加到列表中

    # 将词袋特征列表转换为numpy数组
    bag_of_features = np.array(bag_of_features)

    # 保存模型
    model = Model(kmeans, bag_of_features, images_path)
    joblib.dump(model, save_path)


if __name__ == '__main__':
    args = parse_arguments()
    main(args.dataset, args.save)
