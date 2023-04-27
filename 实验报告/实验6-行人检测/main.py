import os

import numpy as np
from PIL import Image
# 初始化HOGDescriptor类
from skimage.feature import hog
from sklearn.svm import LinearSVC
from tqdm import tqdm

# 获取训练和测试数据
train_pos_dir = './INRIADATA/normalized_images/train/pos/'
train_neg_dir = './INRIADATA/normalized_images/train/neg/'
test_pos_dir = './INRIADATA/normalized_images/test/pos/'
test_neg_dir = './INRIADATA/original_images/test/neg/'

train_pos_list = os.listdir(train_pos_dir)
train_neg_list = os.listdir(train_neg_dir)
test_pos_list = os.listdir(test_pos_dir)
test_neg_list = os.listdir(test_neg_dir)

# 提取训练和测试数据的HOG特征
train_hog_list = []
train_label_list = []
test_hog_list = []
test_label_list = []

img_size = (64, 128)

# 加载正样本
for img_path in tqdm(train_pos_list, desc='Extracting positive train HOG features'):
    img = Image.open(os.path.join(train_pos_dir, img_path)).convert('L')
    img = img.resize(img_size)
    hog_feature = hog(img, feature_vector=True)
    train_hog_list.append(hog_feature)
    train_label_list.append(1)

for img_path in tqdm(train_neg_list, desc='Extracting negative train HOG features'):
    img = Image.open(os.path.join(train_neg_dir, img_path)).convert('L')
    img = img.resize(img_size)
    hog_feature = hog(img, feature_vector=True)
    train_hog_list.append(hog_feature)
    train_label_list.append(0)

for img_path in tqdm(test_pos_list, desc='Extracting positive test HOG features'):
    img = Image.open(os.path.join(test_pos_dir, img_path)).convert('L')
    img = img.resize(img_size)
    hog_feature = hog(img, feature_vector=True)
    test_hog_list.append(hog_feature)
    test_label_list.append(1)

for img_path in tqdm(test_neg_list, desc='Extracting negative test HOG features'):
    img = Image.open(os.path.join(test_neg_dir, img_path)).convert('L')
    img = img.resize(img_size)
    hog_feature = hog(img, feature_vector=True)
    test_hog_list.append(hog_feature)
    test_label_list.append(0)

# 训练 SVM 分类器
clf = LinearSVC()
clf.fit(train_hog_list, train_label_list)

# 测试
result = clf.predict(test_hog_list)

accuracy = np.mean(result == test_label_list)
print('Accuracy:', accuracy)

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# 计算ROC曲线和AUC值
fpr, tpr, _ = roc_curve(test_label_list, clf.decision_function(test_hog_list))
roc_auc = auc(fpr, tpr)

# 画ROC曲线
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()
