import cv2
import numpy as np

# 加载图像
image1 = cv2.imread('./data/image1.jpg')
image2 = cv2.imread('./data/image2.jpg')

# 转换成灰度图像
gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
# 直方图均衡化
gray_image1 = cv2.equalizeHist(gray_image1)
gray_image2 = cv2.equalizeHist(gray_image2)

# 创建SIFT对象
sift = cv2.SIFT_create()

# 检测关键点并计算描述符
keypoints1, descriptors1 = sift.detectAndCompute(gray_image1, None)
keypoints2, descriptors2 = sift.detectAndCompute(gray_image2, None)

# 在图像上绘制关键点
for kp in keypoints1:
    x, y = kp.pt
    cv2.circle(image1, (int(x), int(y)), 3, (0, 255, 0), -1)
for kp in keypoints2:
    x, y = kp.pt
    cv2.circle(image2, (int(x), int(y)), 3, (0, 255, 0), -1)

# 创建FLANN匹配器
matcher = cv2.BFMatcher()

# 使用knnMatch进行关键点匹配
matches = matcher.knnMatch(descriptors1, descriptors2, k=2)

# 进行筛选，保留优秀的匹配
good_matches = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good_matches.append(m)

# 绘制匹配结果
matching_result = cv2.drawMatches(
    image1, keypoints1,
    image2, keypoints2,
    [m for m, n in matches], None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

# RANSAC算法参数
ransac_reproj_threshold = 5.0  # 重投影阈值，用于判断内点和外点

# 从匹配对中提取关键点的坐标
src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

# 使用cv2.findHomography函数计算单应性矩阵
M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_reproj_threshold)

# 对匹配对进行RANSAC筛选，保留内点
good_matches_ransac = [m for m, msk in zip(good_matches, mask) if msk[0] == 1]

# 绘制匹配结果
matching_result = cv2.drawMatches(
    image1, keypoints1,
    image2, keypoints2,
    good_matches_ransac, None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

# 获取图像1到图像2的投影映射关系（单应性矩阵）
h, w = image1.shape[:2]
warped_image1 = cv2.warpPerspective(image1, M, (w, h))

# 将两个图像进行拼接
result = np.concatenate((warped_image1, image2), axis=1)

# 显示拼接结果
cv2.imshow("Stitched Image", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
