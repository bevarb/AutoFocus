import matplotlib.pyplot as plt
from seaborn import heatmap
import seaborn as sns
import numpy as np
import pandas as pd
from IPython.display import Latex
import math
from scipy import optimize
class vis_data():
    '''读取tensor数据，
    绘制散点图：
    绘制拟合曲线：
    '''
    def __init__(self, x, y, list, base):
        super(vis_data, self).__init__()
        self.base = base
        self.predict_data = [i + base for i in x]
        self.target_data = [i + base for i in y]
        self.list = list
        self.len = len(self.target_data)


    def __plot_scatter__(self):
        '''绘制散点图，
        list: 列表里的为需要特殊标记的
        '''
        for i in range(len(self.target_data)-1):
            if i == self.list[0]:
                plt.scatter(self.target_data[i], self.predict_data[i], s=20, color='orange', label="not in train")
            elif i == self.list[1]:
                plt.scatter(self.target_data[i], self.predict_data[i], s=20, color='orange')
            else:
                plt.scatter(self.target_data[i], self.predict_data[i], s=20, color='black')
        plt.scatter(self.target_data[len(self.target_data)-1],
                    self.predict_data[len(self.target_data)-1],
                    s=20, color='black', label='predict')
        plt.scatter(self.target_data, self.target_data, s=24, color='red', label='target')
        plt.xlabel('Target Value')
        plt.ylabel('Predict Value')
        plt.xticks([i[0] for i in self.target_data], rotation=45)
        plt.legend()
        plt.show()

    def __plot_error__(self, flag=False):
        self.predict_error = []
        self.predict_mean = []
        self.target = []
        yerr = [[], []]  # 左边为下置信度，右边为上置信度
        for i in range(self.len):
            # temp = self.target_data[i] - self.predict_data[i]
            if i in self.list:
                plt.errorbar(self.target_data[i][0], [self.predict_data[i].min(), self.predict_data[i].max()],
                             fmt='o-',  # 数据点的标记样式以及相互之间联系的方式
                             ecolor='b',  # 误差棒的线条颜色
                             elinewidth=3,  # 误差棒的线条粗细
                             ms=5,  # 数据点的大小
                             mfc='blue',  # 数据点的颜色
                             mec='blue',  # 数据点边缘的颜色
                             capsize=3,  # 误差棒边界横杠的颜色
                             capthick=2
                             )
                # plt.scatter(self.target_data[i], temp, color="y")
            else:
                plt.errorbar(self.target_data[i][0], [self.predict_data[i].min(), self.predict_data[i].max()],
                             fmt='o-g',  # 数据点的标记样式以及相互之间联系的方式
                             ecolor='black',  # 误差棒的线条颜色
                             elinewidth=3,  # 误差棒的线条粗细
                             ms=5,  # 数据点的大小
                             mfc='forestgreen',  # 数据点的颜色
                             mec='forestgreen',  # 数据点边缘的颜色
                             capsize=3  # 误差棒边界横杠的颜色
                             )
            # self.predict_error.append(temp)
            yerr[0].append(round(self.predict_data[i].min(), 3))
            yerr[1].append(round(self.predict_data[i].max(), 3))
            self.predict_mean.append(self.predict_data[i].mean())
            self.target.append(self.target_data[i][0])
        #plt.scatter(target, self.predict_mean, color="r", label="mean")
        plt.plot(self.target, self.predict_mean, color="coral", marker='o')
        if flag == True:
            plt.plot(self.target, self.target, color="r", marker='o')
        plt.xlabel("Target Value")
        plt.ylabel("Predict Value")
        plt.legend(labels=['mean', 'target', 'type1', 'type2'])
        plt.xticks([i[0] for i in self.target_data], rotation=45)
        plt.show()

    def __plot_box__(self):
        '''绘制箱线图，

        '''
        for i in range(self.len-1):
            if i in self.list:
                plt.boxplot(self.predict_data[i], positions=[i],
                            widths=0.8,
                            patch_artist=True,
                            boxprops={'color': 'orange', 'facecolor': 'orange'})
            else:
                plt.boxplot(self.predict_data[i], positions=[i],
                            widths=0.8,
                            patch_artist=True,
                            boxprops={'color': 'wheat', 'facecolor': 'pink'}
                            )
        # plt.boxplot(self.predict_data, notch=True,)
#        plt.scatter(self.target_data, self.target_data, s=24, color='red', label='target')
        plt.xlabel('Target Value')
        plt.ylabel('Predict Value')
        plt.yticks(size=6)
        plt.xticks([i for i in range(15)], self.target, rotation=45, size=6)
        plt.annotate('type1', xy=(self.base + 0, self.base + 1.7),
                     xytext=(self.base + 1, self.base + 3),
                     size=12,
                     arrowprops=dict(facecolor='g', shrink=0.05)
        )
        plt.annotate('type2', xy=(self.base + 10.5, self.base + 5.8),
                     xytext=(self.base + 8.5, self.base + 6.5),
                     size=12,
                     arrowprops=dict(facecolor='g', shrink=0.05)
        )
        plt.title('Boxplot for Predict Result')
        plt.show()
    def __plot_data__(self, data, index):
        columns = [tar[0] for tar in self.target_data]
        columns.append("ALL")
        df = pd.DataFrame(data, index=index,
                          columns=columns)
        df.iloc[:, 0: len(self.target_data)].T.plot(marker="o")
        plt.ylim([self.base + 0, self.base + 0.6])
        plt.xlabel("Different target values")
        plt.ylabel("Parameter values")
        plt.xticks([i[0] for i in self.target_data], rotation=45)
        plt.title("Different Parameter results")
        plt.show()
        #print(df)

    def __plot_optimizer__(self, a, b, rr):
        plt.scatter(self.target_data, self.predict_data, s=20, color='black', label="Predicted data")
        X = np.arange(self.base + 0, self.base + 7.5, 0.5)
        Y = a * X + b
        plt.plot(X, Y, color="r", label="Fitted")
        plt.xlabel('Target Value')
        plt.ylabel('Predict Value')
        plt.xticks([i[0] for i in self.target_data], rotation=45)
        plt.legend()
        plt.title('Fitted equation: y = %.3f x + %.3f   ' % (a, b) + r"  ${R^2=%.3f}$" % rr)
        plt.show()









