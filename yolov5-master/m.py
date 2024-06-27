import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import warnings
from pathlib import Path

class ConfusionMatrix:
    def __init__(self, nc):
        self.matrix = np.zeros((nc, nc))

    def update_matrix(self, data):
        for i, j, value in data:
            self.matrix[i, j] = value

    def plot(self, normalize=False, save_dir='', names=()):
        if normalize:
            array = self.matrix / (self.matrix.sum(0).reshape(1, -1) + 1E-9)
            fmt = '.2f'  # 使用浮点格式
        else:
            array = self.matrix.astype(int)
            fmt = 'd'  # 使用整数格式

        fig, ax = plt.subplots(1, 1, figsize=(12, 9), tight_layout=True)
        nc, nn = self.matrix.shape[0], len(names)
        sn.set(font_scale=1.5 if nc < 50 else 1.2)  # 调整 font_scale 以增大字体
        labels = (0 < nn < 99) and (nn == nc)
        ticklabels = names if labels else 'auto'

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            sn.heatmap(array,
                       ax=ax,
                       annot=nc < 30,
                       annot_kws={'size': 15},  # 调整注释字体大小
                       cmap='Blues',
                       fmt=fmt,  # 设置注释格式
                       square=True,
                       vmin=0.0,
                       xticklabels=ticklabels,
                       yticklabels=ticklabels).set_facecolor((1, 1, 1))
        ax.set_xlabel('True', fontsize=15)
        ax.set_ylabel('Predicted', fontsize=15)
        ax.set_title('Confusion Matrix', fontsize=18)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        fig.savefig(Path(save_dir) / 'confusion_matrix1.png', dpi=250)
        plt.close(fig)

# 设置类别名称
names = ['HC', 'PD']

# 创建ConfusionMatrix对象
conf_matrix = ConfusionMatrix(nc=2)

# 更新混淆矩阵数据 (行是实际类别，列是预测类别)
data = [
    (0, 0, 49),  # 49 个样本被正确预测为 HC
    (1, 0, 5),   # 5 个 PD 样本被错误预测为 HC
    (1, 1, 54),  # 54 个样本被正确预测为 PD
    (0, 1, 4)    # 4 个 HC 样本被错误预测为 PD
]

conf_matrix.update_matrix(data)

# 绘制混淆矩阵
conf_matrix.plot(normalize=False, save_dir='./runs', names=names)
