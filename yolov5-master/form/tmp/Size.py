import os

def getfilesize_single(filepath):
    byte_size = os.path.getsize(filepath)  # 单位是Byte (1KB = 1024Byte, 1MB = 1024KB)
    # print(byte_size)
    pt_size = round(byte_size / 1024 / 1024, 3)  # 换算后，单位是MB
    return pt_size


if __name__ == '__main__':
    weightpath = r'D:\yolov5-master\runs\train\exp13\weights\best.pt'
    # weightpath = r'D:\yolov5-master\runs\train\exp10\weights\best.pt'
    print(f'{getfilesize_single(weightpath)} MB')
