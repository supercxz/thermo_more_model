import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# 这两行代码解决 plt 中文显示的问题
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
# Seaborn库的中文配置
rc = {'font.sans-serif': 'SimHei',
      'axes.unicode_minus': False# 设置字体样式 字体负号显示
}
# 字体大小在标题中显示0.8倍
sns.set(style="whitegrid", rc=rc, font_scale=0.8)  # Set the style
# 设置字体大小：应用于图内的字体，如热力图
plt.rcParams['font.size'] = 8

# 绘制要预测的特征的分布
def plot_prediction_feature(data, label, name):
    # 查看价格主要分布区域
    # 直方图+折线图
    plt.figure(figsize=(10, 6))
    sns.distplot(data[label], color='g', bins=100, hist_kws={'alpha': 0.4})
    plt.title(f'{label}分布')
    plt.savefig(f'./{name}/{label}分布直方图.jpg')
    plt.show()
    # 散点图
    plt.figure(figsize=(10, 6))  # Set the figure size
    sns.scatterplot(data[label])
    plt.title(f'{label}分布')
    plt.savefig(f'./{name}/{label}分布散点图.jpg')
    plt.show()

# 绘制数据的热力图（相关系数为皮尔逊），数据填充后（其实和填充前并没有多大区别）
# front_features：在绘制前几特征热力图的同时，会将其直方图和散点图的相关性也画出。
def plot_headmap(data, label, name, num=0, front_features = False):
    df_num = data.select_dtypes(include=['float64', 'int64'])
    plt.figure(figsize=(24, 8))
    # corr()是相关矩阵，括号里面没有填参数时默认是皮尔逊相关系数。
    corr_matrix = df_num.corr()
    # 相关性可以是正相关，这意味着它们具有直接关系，并且一个特征的增加会导致另一个特征的增加。
    # 负相关也是可能的，这表明这两个特征彼此呈反比关系，这意味着一个特征的上升将导致另一个特征的下降。
    corr_with_price = corr_matrix[label].sort_values(ascending=False)
    print(corr_with_price)
    # annot：‌是否在每个单元格中显示数值。‌fmt：‌用于格式化注解文本的格式字符串。‌

    if num == 0:
        sns.heatmap(corr_matrix, annot=True, fmt=".2f")  # corr_matrix为所有数值特征的皮尔逊相关矩阵。
        plt.title('全部特征与输出特征的热力图')
        plt.savefig(f'./{name}/{label}绘制各个特征的热力图.jpg')
        plt.show()
    else:
        corr_fea = corr_matrix.nlargest(num+1, label)[label]
        # cols：获取前11个的列名
        cols = corr_matrix.nlargest(num+1, label)[label].index
        # 绘制前十个相关性特征和价格的热力图
        sns.heatmap(df_num[cols].corr(), annot=True)
        plt.title(f'前{num}个特征与输出特征的热力图')
        plt.savefig(f'./{name}/{label}绘制前{num}个特征的热力图.jpg')
        plt.show()
        if front_features == True:
            sns.pairplot(data[cols[:7]], size=2.5)
            plt.savefig(f'./{name}/{label}绘制各个特征之间变化图.jpg')
            plt.show()


def plot_sorded_feature(clf, feature_data, title,name):
    # 绘制特征重要性条形图
    feature_importance = clf.feature_importances_
    #feature_data = pd.DataFrame(feature_data)
    feature_names = feature_data.columns.tolist()
    sorted_idx = feature_importance.argsort()
    plt.barh(range(len(feature_importance)), feature_importance[sorted_idx])
    plt.yticks(range(len(feature_importance)), [feature_names[i] for i in sorted_idx], fontsize=10)
    plt.xlabel('特征重要性')
    plt.ylabel('特征名称')
    plt.title(title)
    plt.savefig(f'./{name}的图片/{title}特征重要性条形图', dpi=300)
    plt.show()