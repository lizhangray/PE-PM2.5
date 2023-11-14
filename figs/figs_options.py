import matplotlib.pyplot as plt


# 散点图？WL 算法
parameters = {
    'axes.labelsize': 18,
    'axes.titlesize': 18,
    'font.size': 16,
}
# # 损失, bw?
# parameters = {
#     'axes.labelsize': 24,  
#     'axes.titlesize': 24,
#     'font.size': 20,
# }

# 直方图：[28,28,24]
# 散点图：[24,24,20]

plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号
plt.rcParams.update(parameters)