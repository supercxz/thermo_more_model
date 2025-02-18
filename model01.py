#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import re
import warnings
warnings.filterwarnings("ignore")


# In[ ]:


data_high_bound = 30
data_low_bound = -20
need_to_normalized = False

data = pd.read_excel('./dataset/rt_df_thermo1.xlsx')
print(data.head())

print(f"max = {data['the thermal expansion'].max()}")
print(f"min = {data['the thermal expansion'].min()}")

in_range_count = data[(data['the thermal expansion'] >= -20) & (data['the thermal expansion'] <= 30)].shape[0]

# 计算总数
total_count = data.shape[0]

# 计算比例
proportion = in_range_count / total_count
print(proportion)


# In[ ]:


# 函数：解析化学成分及其比例
def parse_normalized_formulas(formula):
    """
    将Normalized_Formulas解析为化学成分及其比例。
    返回一个字典，键为化学成分，值为比例。
    """
    elements = re.findall(r'([A-Z][a-z]*)(\d*\.?\d+)', formula)
    return {element: float(ratio) for element, ratio in elements}


# In[ ]:


# 提取所有化学成分
all_elements = set()
for formula in data['Normalized_Formulas']:
    parsed = parse_normalized_formulas(formula)
    all_elements.update(parsed.keys())


# In[ ]:


all_elements


# In[ ]:


# 确保列顺序一致
all_elements = sorted(all_elements)

# 创建新列：每个元素作为一列，未出现的元素填充为0
for element in all_elements:
    data[element] = data['Normalized_Formulas'].apply(
        lambda x: parse_normalized_formulas(x).get(element, 0)
    )


# In[ ]:


data.head()


# In[ ]:


df = data.copy()
df = df.drop('formula', axis=1)
df = df.drop('Normalized_Formulas', axis=1)
df = df.drop('ID', axis=1)
df.head()


# In[ ]:


import class_plotpicture as pl
# 绘制目标特征的条形图
pl.plot_prediction_feature(df, 'the thermal expansion', 'ImageOfThermal')


# # 脏数据清理

# In[ ]:


# 删除 'the thermal expansion' 列中大于 data_high_bound 的行
df_cleaned = df[df['the thermal expansion'] <= data_high_bound]

# 删除 'the thermal expansion' 列中小于 data_low_bound 的行
df_cleaned = df_cleaned[df_cleaned['the thermal expansion'] >= data_low_bound]


# In[ ]:


import class_plotpicture as pl

# 绘制目标特征的条形图
pl.plot_prediction_feature(df_cleaned, 'the thermal expansion', 'ImageOfThermal')


# # 大致符合正态分布

# In[ ]:


# 绘制数据的相关性：热力图
pl.plot_headmap(df_cleaned, 'the thermal expansion', 'ImageOfThermal')


# In[ ]:


# 绘制数据的相关性：热力图
pl.plot_headmap(df_cleaned, 'the thermal expansion', 'ImageOfThermal', num=10)


# # 划分数据，进行训练和测试

# In[ ]:


all_features = df_cleaned.drop('the thermal expansion', axis=1)
all_labels = df_cleaned['the thermal expansion']
print(f'全部的特征：{all_features.shape}')
print(f'全部的标签：{all_labels.shape}')


# In[ ]:


from sklearn.preprocessing import StandardScaler
import numpy as np
# 为标准化特征做准备。但实际使用使用标准化后的特征，取决于代码最开始的 need_to_normalized
scaler = StandardScaler()
all_labels_scaler = scaler.fit_transform(np.array(all_labels).reshape(-1, 1))


# In[ ]:


from sklearn.model_selection import train_test_split
# 将总的数据集分开。这里根据是否需要对特征进行标准化
if need_to_normalized:
    X_train, X_test, y_train, y_test = train_test_split(all_features, all_labels_scaler, test_size=0.2, random_state=42)
else:
    X_train, X_test, y_train, y_test = train_test_split(all_features, all_labels, test_size=0.2, random_state=42)
print(f'训练集的特征：{X_train.shape}, 标签：{y_train.shape}')
print(f'测试集的特征：{X_test.shape}, 标签：{y_test.shape}')


# # 超参数优化

# In[ ]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


# In[ ]:


# 统计不同的值对随机森林的变化
cllo_mse = []
cllo_r2 = []
best_idx, best_mse, best_r2 = 0, 100, 0
# 先查找 n_estimators ，其他参数默认
for i in range(1, 301):
    print(f'第{i+1}个')
    clf = RandomForestRegressor(random_state=42, n_estimators=i)
    # 在训练集上拟合模型
    clf.fit(X_train,y_train)
    # 对测试集进行预测
    label_pred = clf.predict(X_test)
    # 计算MSE(平均误差)和精确度
    mse = mean_squared_error(y_test, label_pred)
    r2 = r2_score(y_test, label_pred)
    cllo_mse.append(mse)
    cllo_r2.append(r2)
    if best_mse > mse:
        best_mse = mse
        best_idx = i
        best_r2 = r2
print(f'在随机森林调优n_estimators过程中，最好的效果：mse:{best_mse:.5f}, r2:{best_r2:.5f}, n_estimators:{best_idx}')

import matplotlib.pyplot as plt
plt.figure(figsize=(10,6))
plt.subplot(121)
plt.plot(cllo_mse)
plt.title('在调节n_estimators参数时MSE的变化')
plt.subplot(122)
plt.plot(cllo_r2)
plt.title('在调节n_estimators参数时R2的变化')
plt.show()


# In[ ]:


# 统计不同的值对随机森林的变化
cllo_mse = []
cllo_r2 = []
best_idx, best_mse, best_r2 = 0, 100, 0
# 先查找 n_estimators ，其他参数默认
for i in range(1, 301):
    print(f'第{i+1}个')
    clf = RandomForestRegressor(random_state=42, n_estimators=2, max_features=i)
    # 在训练集上拟合模型
    clf.fit(X_train,y_train)
    # 对测试集进行预测
    label_pred = clf.predict(X_test)
    # 计算MSE(平均误差)和精确度
    mse = mean_squared_error(y_test, label_pred)
    r2 = r2_score(y_test, label_pred)
    cllo_mse.append(mse)
    cllo_r2.append(r2)
    if best_mse > mse:
        best_mse = mse
        best_idx = i
        best_r2 = r2
print(f'在随机森林调优max_features过程中，最好的效果：mse:{best_mse:.5f}, r2:{best_r2:.5f}, max_features:{best_idx}')

import matplotlib.pyplot as plt
plt.figure(figsize=(10,6))
plt.subplot(121)
plt.plot(cllo_mse)
plt.title('在调节max_features参数时MSE的变化')
plt.subplot(122)
plt.plot(cllo_r2)
plt.title('在调节max_features参数时R2的变化')
plt.show()


# In[ ]:


# 统计不同的值对随机森林的变化
cllo_mse = []
cllo_r2 = []
best_idx, best_mse, best_r2 = 0, 100, 0
# 先查找 n_estimators ，其他参数默认
for i in range(1, 301):
    print(f'第{i+1}个')
    clf = RandomForestRegressor(random_state=42, n_estimators=2, max_features=38, min_samples_leaf=i)
    # 在训练集上拟合模型
    clf.fit(X_train,y_train)
    # 对测试集进行预测
    label_pred = clf.predict(X_test)
    # 计算MSE(平均误差)和精确度
    mse = mean_squared_error(y_test, label_pred)
    r2 = r2_score(y_test, label_pred)
    cllo_mse.append(mse)
    cllo_r2.append(r2)
    if best_mse > mse:
        best_mse = mse
        best_idx = i
        best_r2 = r2
name = 'min_samples_leaf'
print(f'在随机森林调优{name}过程中，最好的效果：mse:{best_mse:.5f}, r2:{best_r2:.5f}, {name}:{best_idx}')

import matplotlib.pyplot as plt
plt.figure(figsize=(10,6))
plt.subplot(121)
plt.plot(cllo_mse)
plt.title(f'在调节{name}参数时MSE的变化')
plt.subplot(122)
plt.plot(cllo_r2)
plt.title(f'在调节{name}参数时R2的变化')
plt.show()


# In[ ]:


# 统计不同的值对随机森林的变化
cllo_mse = []
cllo_r2 = []
best_idx, best_mse, best_r2 = 0, 100, 0
# 先查找 n_estimators ，其他参数默认
for i in range(1, 301):
    print(f'第{i+1}个')
    clf = RandomForestRegressor(random_state=42, n_estimators=2, max_features=38, min_samples_leaf=1, max_depth=i)
    # 在训练集上拟合模型
    clf.fit(X_train,y_train)
    # 对测试集进行预测
    label_pred = clf.predict(X_test)
    # 计算MSE(平均误差)和精确度
    mse = mean_squared_error(y_test, label_pred)
    r2 = r2_score(y_test, label_pred)
    cllo_mse.append(mse)
    cllo_r2.append(r2)
    if best_mse > mse:
        best_mse = mse
        best_idx = i
        best_r2 = r2
name = 'max_depth'
print(f'在随机森林调优{name}过程中，最好的效果：mse:{best_mse:.5f}, r2:{best_r2:.5f}, {name}:{best_idx}')

import matplotlib.pyplot as plt
plt.figure(figsize=(10,6))
plt.subplot(121)
plt.plot(cllo_mse)
plt.title(f'在调节{name}参数时MSE的变化')
plt.subplot(122)
plt.plot(cllo_r2)
plt.title(f'在调节{name}参数时R2的变化')
plt.show()


# In[ ]:


# 创建随机森林分类器对象
clf = RandomForestRegressor(random_state=42, n_estimators=2, max_features=38, min_samples_leaf=1, max_depth=29)
# 在训练集上拟合模型
clf.fit(X_train, y_train)
train_pred = clf.predict(X_train)
# 对测试集进行预测
label_pred = clf.predict(X_test)
# 计算MSE(平均误差)和精确度
mse = mean_squared_error(y_test, label_pred)
r2 = r2_score(y_test, label_pred)
# 输出模型评估结果和目标方程
print(f'MSE:{mse:.5f}')
print(f"R2: {r2:.5f}")


# In[ ]:


import matplotlib.pyplot as plt
# 先将数据反归一化
salered_train_label_pred = scaler.inverse_transform(train_pred.reshape(-1,1))
salered_train_labels = scaler.inverse_transform(y_train)
plt.scatter(salered_train_labels, salered_train_label_pred, color='blue', label='Predicted vs True')
# 绘制Y=X的直线，表示完美预测
plt.plot([min(salered_train_labels), max(salered_train_label_pred)], [min(salered_train_labels), max(salered_train_label_pred)], 'r--', label='Perfect Prediction (Y=X)')
# 添加图例
plt.legend()
# 设置坐标轴标签
plt.xlabel('真确的值')
plt.ylabel('预测的值')
# 设置标题
plt.title('随机森林的训练集预测结果')
# 显示图形
plt.show()

# 预测集
salered_test_labels = scaler.inverse_transform(y_test)
salered_label_pred = scaler.inverse_transform(label_pred.reshape(-1,1))
plt.scatter(salered_test_labels, salered_label_pred, color='blue', label='Predicted vs True')
# 绘制Y=X的直线，表示完美预测
plt.plot([min(salered_test_labels), max(salered_label_pred)], [min(salered_test_labels), max(salered_label_pred)], 'r--', label='Perfect Prediction (Y=X)')
# 添加图例
plt.legend()
# 设置坐标轴标签
plt.xlabel('真确的值')
plt.ylabel('预测的值')
# 设置标题
plt.title('随机森林的测试集预测结果')
# 显示图形
plt.show()


# # 查看预测的偏差值

# In[ ]:


big_num = 0
small_num = 0
for i in range(len(label_pred)):
    if label_pred[i].mean() >= y_test[i].mean():
        big_num += 1
    else:
        small_num += 1
print(f'预测值大于原值的个数：{big_num}, 预测值小于原值的个数：{small_num}')

# 计算偏移量
value = []
for i in range(len(label_pred)):
    value.append(salered_test_labels[i] - salered_label_pred[i])

bins = np.arange(-70, 71, 10)  # 从-1到1，每隔0.1一个区间
counts, _ = np.histogram(value, bins=bins)
print(counts)
bin_centers = (bins[:-1] + bins[1:]) / 2
# 绘制柱形图
plt.figure(figsize=(10, 6))  # 设置图形大小
plt.bar(bin_centers, counts, width=5, color='skyblue', edgecolor='black')  # 宽度设置为0.1与区间宽度相匹配

# 添加标题和轴标签
plt.title('误差偏移量')
plt.xlabel('偏移量区间')
plt.ylabel('统计个数')

# 显示网格
plt.grid(True)
plt.show()

