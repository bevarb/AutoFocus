import pandas as pd
import numpy as np
from visualize.plot_fig import vis_data
from visualize.get_value import get_value
import matplotlib.pyplot as plt
df = pd.read_excel(r'C:\Users\雪山飞狐\Desktop\easy\test.xlsx')
df = df.drop(columns=["Unnamed: 0"], axis=1)
data = []
target = []
for i in range(15):
    data.append(np.array(df.iloc[i*10:(i+1)*10, 0]))
    target.append(np.array(df.iloc[i*10:(i+1)*10, 1]))
vis_data = vis_data(data, target, [1, 11], 2837.0)
vis_data.__plot_scatter__()
vis_data.__plot_error__()
vis_data.__plot_box__()
# # vis_data.__plot_heatmap__()

get_value = get_value(data, target)
mean, std = get_value.__calculate_base__()
mse, rmse = get_value.__calculate_mse__()
mae = get_value.__calculate_mae__()
data = [# mean,
        std, mse, rmse, mae]
# vis_data.__plot_data__(data, ["Std", "MSE", "RMSE", "MAE"])
a, b, rr = get_value.__calculate_rr__()
vis_data.__plot_optimizer__(a, b, rr)