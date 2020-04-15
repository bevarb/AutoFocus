import numpy as np
from scipy import optimize
class get_value():
    def __init__(self, x, y):
        self.base = 2837
        self.predict_data = [i+self.base for i in x]
        self.target_data = [i+self.base for i in y]
        self.len = len(self.target_data)
        self.set_len = len(self.target_data[0])
        self.total = self.len * self.set_len

    def __calculate_base__(self):
        '''计算每组均值，方差'''
        self.mean = []
        self.std = []
        for i in range(self.len):
            mean = round(self.predict_data[i].mean(), 2)
            std = round(self.predict_data[i].std(), 2)
            self.mean.append(mean)
            self.std.append(std)
        return self.mean, self.std

    def __calculate_mse__(self):
        '''计算MSE和RMSE'''
        self.mse = []
        self.rmse = []
        for i in range(self.len):
            mse = np.sum(np.square(self.predict_data[i] - self.target_data[i])) / self.set_len
            rmse = np.sqrt(mse)
            self.mse.append(round(mse, 2))
            self.rmse.append(round(rmse, 2))
        all_mse = round(np.sum(self.mse) / self.len, 2)
        all_rmse = round(np.sum(self.rmse) / self.len, 2)
        self.mse.append(all_mse)
        self.rmse.append(all_rmse)

        return self.mse, self.rmse
    def __calculate_mae__(self):
        self.mae = []
        for i in range(self.len):
            mae = np.sum(np.abs(self.predict_data[i] - self.target_data[i])) / self.set_len
            self.mae.append(round(mae, 2))
        all_mae = round(np.sum(self.mae) / self.set_len, 2)
        self.mae.append(all_mae)
        return self.mae

    def __calculate_rr__(self):

        def f_1(x, A, B):
            return A * x + B
        # 直线拟合与绘制
        #print(np.array(self.target_data))
        x = np.array(self.target_data).reshape((1, self.total))[0]
        y = np.array(self.predict_data).reshape((1, self.total))[0]
        # print(x, x.dtype)
        a, b = optimize.curve_fit(f_1, x, y)[0]
        y_tar = x * a + b
        y_mean = np.ones(self.total) * (np.sum(y) / self.total)
        RSS = np.sum(np.square(y - y_tar))
        TSS = np.sum(np.square(y - y_mean))
        rr = round(1 - RSS / TSS, 3)
        return a, b, rr





