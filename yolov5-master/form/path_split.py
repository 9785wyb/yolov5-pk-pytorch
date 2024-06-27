import os
import shutil

#文件存放目录
source_folder = r"D:\NingXia\YOLO\image_json_1"
#提取文件保存目录
destination_folder = r"D:\yolov5-master\new_dataset_1\txt_save"
# 自动创建输出目录
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

# 遍历所有子文件夹
for parent_folder, _, file_names in os.walk(source_folder):
    # 遍历当前子文件夹中的所有文件
    for file_name in file_names:
        # 只处理图片文件
        # if file_name.endswith(('jpg', 'jpeg', 'png', 'gif')):#提取jpg、jpeg等格式的文件到指定目录
        if file_name.endswith(('.txt')):#提取json格式的文件到指定目录
            # 构造源文件路径和目标文件路径
            source_path = os.path.join(parent_folder, file_name)
            destination_path = os.path.join(destination_folder, file_name)
            # 复制文件到目标文件夹
            shutil.copy(source_path, destination_path)
