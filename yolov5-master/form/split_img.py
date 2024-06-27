"""
    分数据集
"""
import os
import shutil

path_list = []
# with open(r"D:\yolov5-master\form\tmp\train.txt", "r") as f:
with open(r"D:\yolov5-master\form\tmp\data_1\test.txt", "r", encoding='utf-8') as f:
# with open(r"D:\yolov5-master\form\tmp\val.txt", "r") as f:
    for line in f.readlines():
        line = line[22:]
        line = line.strip('\n')  #去掉列表中每一个元素的换行符
        # line = line.replace('jpg','txt') # 标签
        path_list.append(line)
        # print(line)
parent_folder = r"D:\NingXia\YOLO\image_json_1"

# 提取文件保存目录
destination_folder = r"D:\yolov5-master\new_dataset_1\data\test\images"
# destination_folder = r"D:\mydata\dataset\test\labels"
# destination_folder = r"D:\mydata\dataset\val\images"
# destination_folder = r"D:\mydata\dataset\val\labels"
# destination_folder = r"D:\mydata\dataset\train\images"
# destination_folder = r"D:\mydata\dataset\train\labels"
# # 遍历当前子文件夹中的所有文件  替换 txt
# for file_name in path_list:
#     # 只处理图片文件
#     if file_name.endswith(('jpg', 'jpeg', 'png', 'gif')):# 提取jpg、jpeg等格式的文件到指定目录
#     # if file_name.endswith(('.txt')):#提取json格式的文件到指定目录
#          # 构造源文件路径和目标文件路径
#
#         # print(source_path)
#         files = file_name.split("/")[-1].replace("png", "txt")
#         print(files)
#         source_path = os.path.join(parent_folder, files)
#         destination_path = os.path.join(destination_folder, files)
#         print(destination_path)
#         # print(file_name)
#         # print("file:"+files)
#         # print("destination_path:"+destination_path)
#         # 复制文件到目标文件夹
#         shutil.copyfile(source_path, destination_path)

# 替换图片
for file_name in path_list:
    # 只处理图片文件
    if file_name.endswith(('jpg', 'jpeg', 'png', 'gif')):# 提取jpg、jpeg等格式的文件到指定目录
    # if file_name.endswith(('.txt')):#提取json格式的文件到指定目录
         # 构造源文件路径和目标文件路径

        # print(source_path)
        files = file_name.split("/")[-1]
        print(files)
        source_path = os.path.join(parent_folder, files)
        destination_path = os.path.join(destination_folder, files)
        print(destination_path)
        # print(file_name)
        # print("file:"+files)
        # print("destination_path:"+destination_path)
        # 复制文件到目标文件夹
        shutil.copyfile(source_path, destination_path)



