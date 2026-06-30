import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# ==================== 1. 加载数据并合并 ====================
# 假设每个类别的数据文件路径如下（请根据实际路径修改）
cf = pd.read_excel(r"data\L1\cf12.xls").assign(Y="CF")
eo = pd.read_excel(r"data\L1\eo14.xls").assign(Y="EO")
fwc = pd.read_excel(r"data\L1\fwc10.xls").assign(Y="FWC")
fwe = pd.read_excel(r"data\L1\fwe10.xls").assign(Y="FWE")
nc = pd.read_excel(r"data\L1\nc1.xls").assign(Y="NC")
rl = pd.read_excel(r"data\L1\rl10.xls").assign(Y="RL")
ro = pd.read_excel(r"data\L1\ro10.xls").assign(Y="RO")
normal = pd.read_excel(r"data\L1\normal.xls").assign(Y="Normal")

# 合并所有数据
data = pd.concat([cf, eo, fwc, fwe, nc, rl, ro, normal], ignore_index=True)

# ==================== 2. 定义特征和标签 ====================
features = ["PO_net", "TWI", "VC", "FWE", "VE", "FWC", "TCA", "TO_feed", "TCI", "TEO", "TR_dis"]
x = data[features]
y = data["Y"]

# ==================== 3. 分层划分训练集和测试集（70%-30%） ====================
xtrain, xtest, ytrain, ytest = train_test_split(
    x, y,
    test_size=0.3,
    random_state=42,  # 固定随机种子保证可复现
    stratify=y       # 按类别分层划分
)

# 合并训练集的x和y以便后续处理
train_data = pd.concat([xtrain, ytrain], axis=1)

# ==================== 4. 构造不平衡训练集（Normal:每个故障类别=10:1） ====================
# 分离Normal样本和其他故障样本
normal_samples = train_data[train_data["Y"] == "Normal"]
fault_samples = train_data[train_data["Y"] != "Normal"]

# 设置比例（Normal:每个故障类别=10:1）
ratio = 1

# 计算每个故障类别需要保留的样本数量
n_fault_to_keep_per_class = len(normal_samples) // ratio

# 从每个故障类别中随机抽取指定数量的样本
fault_sampled = fault_samples.groupby("Y", group_keys=False).apply(
    lambda x: x.sample(min(len(x), n_fault_to_keep_per_class), random_state=42)
)

# 合并Normal和抽样后的故障样本
unbalanced_train = pd.concat([normal_samples, fault_sampled], axis=0)

# 重新拆分特征和标签
xtrain_unbalanced = unbalanced_train[features]
ytrain_unbalanced = unbalanced_train["Y"]

# ==================== 5. 验证数据集分布 ====================
print("\n========== 数据集分布统计 ==========")
print("原始数据集类别分布:")
print(y.value_counts())

print("\n训练集（分层划分后）类别分布:")
print(ytrain.value_counts())

print("\n测试集（分层划分后）类别分布:")
print(ytest.value_counts())

print("\n不平衡训练集（Normal:每个故障类别=10:1）类别分布:")
print(ytrain_unbalanced.value_counts())

# ==================== 6. 不平衡训练集类别分布可视化 ====================
plt.figure(figsize=(12, 8), dpi=600)  # 增大画布尺寸和分辨率
plt.rcParams['font.family'] = 'Times New Roman'

# 设置全局字体大小
plt.rcParams['axes.titlesize'] = 22  # 主标题
plt.rcParams['axes.labelsize'] = 20  # 坐标轴标签
plt.rcParams['xtick.labelsize'] = 20  # X轴刻度
plt.rcParams['ytick.labelsize'] = 20  # Y轴刻度
# 获取类别顺序（按样本数从高到低）
class_order = ytrain_unbalanced.value_counts().index
# 绘制柱状图（使用原Set2配色）
sns.countplot(x="Y",
              data=unbalanced_train,
              hue="Y",
              palette='Set2',  # 保持原配色
              order=class_order,
              legend=False,
              saturation=0.8)  # 控制颜色饱和度
# 添加数据标签
for p in plt.gca().patches:
    plt.gca().annotate(f'{int(p.get_height())}',
                      (p.get_x() + p.get_width() / 2., p.get_height()),
                      ha='center', va='center',
                      fontsize=20,
                      color='black',
                      xytext=(0, 5),
                      textcoords='offset points')
plt.title("Class Distribution in Unbalanced Training Set\n(Normal:Fault = 20:1)",pad=5)
plt.xlabel("Class", labelpad=2)
plt.ylabel("Count", labelpad=2)
plt.xticks(rotation=45)
# 设置深色背景的网格（增强Set2配色的对比度）
sns.set_style("whitegrid")
plt.gca().set_facecolor('#f0f0f0')
plt.tight_layout()
plt.savefig('class_distribution.png', dpi=600, bbox_inches='tight', facecolor='#f0f0f0')  # 保持背景一致
plt.show()
# ==================== 6. 保存处理后的数据集（可选） ====================
unbalanced_train.to_csv("train.csv", index=False)
pd.concat([xtest, ytest], axis=1).to_csv("test.csv", index=False)
print("\n处理完成！不平衡训练集和测试集已保存为CSV文件。")