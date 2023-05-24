import cv2

# 加载分类器
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 加载图像
img = cv2.imread('./big/img_1289.jpg')

# 转换为灰度图像
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 检测人脸
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

# 绘制矩形框显示检测结果
for (x, y, w, h) in faces:
    cv2.rectangle(img, pt1=(x, y), pt2=(x + w, y + h), color=(0, 255, 0), thickness=2)

# 显示检测结果
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
