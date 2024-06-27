import cv2 as cv
fig = cv.imread(r'D:\yolov5-master\form\img.png')


#图像反转
L = 255
fig1 = L - fig
cv.imshow('image',fig)
cv.imshow('image negative',fig1)
cv.waitKey(0)
cv.destroyAllWindows()


# # 镜像翻转
# from PIL import Image
#
# image = Image.open(r"D:\yolov5-master\dataset\train\images\normal_24.jpg")
# mirrored_image = image.transpose(Image.FLIP_LEFT_RIGHT)
# mirrored_image.show()
# mirrored_image.save(r'D:\研究生\写论文\帕金森\英文论文二改\mirrored_image.jpg')

# import cv2
# import numpy as np
#
# img = cv2.imread(r'D:\yolov5-master\dataset\test\images\normal_30.jpg', 0)
# cv2.imshow("img", img)
# h, w = img.shape
# for i in range(h):
#     for j in range(w):
#         if 100 < img[i][j] < 180:
#             img[i][j] = 150
#         else:
#             img[i][j] = 25
#
# cv2.imshow("img_test", img)
# cv2.waitKey()

# import cv2
#
# img = cv2.imread("test41.png", 0)
# img1 = cv2.imread("test42.png", 0)
# img2 = img1 - img
# cv2.namedWindow("img", 0)
# cv2.resizeWindow('img', 400, 300)
# cv2.namedWindow("img1", 0)
# cv2.resizeWindow('img1', 400, 300)
# cv2.namedWindow("img_test", 0)
# cv2.resizeWindow('img_test', 400, 300)
# cv2.imshow("img", img)
# cv2.imshow("img1", img1)
# cv2.imshow("img_test", img2)
# cv2.waitKey()