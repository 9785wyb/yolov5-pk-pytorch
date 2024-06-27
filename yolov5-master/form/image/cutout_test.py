import cv2
from torchvision import transforms
from cutout import Cutout

# 执行cutout
img = cv2.imread(r'D:\yolov5-master\dataset\train\images\normal_26.jpg')
img = transforms.ToTensor()(img)
cut = Cutout(length=90)
img = cut(img)

# cutout图像写入本地
img = img.mul(255).byte()
img = img.numpy().transpose((1, 2, 0))
cv2.imwrite('normal_26.jpg', img)