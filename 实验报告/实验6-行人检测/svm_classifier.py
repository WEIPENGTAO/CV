import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from joblib import dump
from skimage.feature import hog
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from tqdm import tqdm


def extract_hog_features(data_dir, img_size):
    """
    提取 HOG 特征
    :param data_dir:
    :param img_size:
    :return:
    """
    hog_list = []
    label_list = []
    img_list = os.listdir(data_dir)
    for img_path in tqdm(img_list, desc='Extracting HOG features'):
        img = Image.open(os.path.join(data_dir, img_path)).convert('L')
        img = img.resize(img_size)
        hog_feature = hog(img, feature_vector=True)
        hog_list.append(hog_feature)
        label_list.append(1 if 'pos' in data_dir else 0)
    return hog_list, label_list


def plot_roc_curve(fpr, tpr, auc_score, save_path):
    """
    绘制 ROC 曲线并保存
    :param fpr: 假正例率
    :param tpr: 真正例率
    :param auc_score: AUC 值
    :param save_path: 保存路径
    :return: None
    """
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(auc_score))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.title('ROC curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.legend(loc='lower right')
    plt.savefig(save_path)
    plt.show()


# 得到最优参数的分类器
def get_best_svm_classifier(train_hog_list, train_label_list, param_grid):
    svm_classifier = SVC()

    # 创建GridSearchCV对象，将SVM模型和参数网格传入
    grid_search = GridSearchCV(svm_classifier, param_grid=param_grid, cv=5)

    # 对模型进行训练
    grid_search.fit(train_hog_list, train_label_list)

    # 输出最佳参数组合和模型的评分
    print('Best parameters:', grid_search.best_params_)
    print('Best score:', grid_search.best_score_)

    return grid_search.best_estimator_


if __name__ == '__main__':
    # 获取训练和测试数据
    train_pos_dir = './INRIADATA/normalized_images/train/pos/'
    train_neg_dir = './INRIADATA/normalized_images/train/neg/'
    test_pos_dir = './INRIADATA/normalized_images/test/pos/'
    test_neg_dir = './INRIADATA/original_images/test/neg/'

    img_size = (64, 128)

    # 提取训练和测试数据的HOG特征
    train_hog_list, train_label_list = extract_hog_features(train_pos_dir, img_size)
    train_hog_list_neg, train_label_list_neg = extract_hog_features(train_neg_dir, img_size)
    train_hog_list += train_hog_list_neg
    train_label_list += train_label_list_neg

    test_hog_list, test_label_list = extract_hog_features(test_pos_dir, img_size)
    test_hog_list_neg, test_label_list_neg = extract_hog_features(test_neg_dir, img_size)
    test_hog_list += test_hog_list_neg
    test_label_list += test_label_list_neg

    param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    }

    svm_classifier = get_best_svm_classifier(train_hog_list, train_label_list, param_grid)

    dump(svm_classifier, 'svm_classifier.joblib')

    # 测试
    result = svm_classifier.predict(test_hog_list)

    accuracy = np.mean(result == test_label_list)
    print('Accuracy:', accuracy)

    # 计算 ROC 曲线和 AUC 值
    fpr, tpr, _ = roc_curve(test_label_list, svm_classifier.decision_function(test_hog_list))
    roc_auc = auc(fpr, tpr)

    # 画ROC曲线
    plot_roc_curve(fpr, tpr, roc_auc, 'svm_roc_curve.png')
