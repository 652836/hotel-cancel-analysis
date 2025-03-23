# 酒店预订取消率异动归因与用户分层策略分析报告

## 摘要

本报告分析了酒店预订数据集中的取消率影响因素，发现预订提前时间、客户类型、分销渠道和房间类型一致性是影响取消率的主要因素。新客户取消率显著高于重复客户，提前预订时间越长取消率越高，通过旅行社/旅游运营商渠道的订单取消率明显高于直接预订，而Groups市场细分的取消率更是高达61.06%。本报告提供了详细的数据分析过程和Python代码，并基于分析结果提出了优化建议框架。

## 1. 数据探索与清洗

首先，进行数据加载和探索性分析，了解数据结构并检查缺失值。

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# 加载数据
df = pd.read_csv('hotel_bookings.csv')

# 查看数据基本信息
print(f"数据行数: {df.shape[0]}")
print(f"数据列数: {df.shape[1]}")
print("\n前5行数据示例:")
print(df.head())

# 检查缺失值
missing_values = df.isnull().sum()
print("\n缺失值统计:")
print(missing_values[missing_values > 0])

# 数据类型检查
print("\n数据类型:")
print(df.dtypes)
```

数据缺失情况分析显示，children、agent和company字段存在缺失值。由于我们的分析重点不在这些字段，我们可以根据需要进行处理：

```python
# 处理缺失值
df['children'] = df['children'].fillna(0)
df['agent'] = df['agent'].fillna(0)
df['company'] = df['company'].fillna(0)

# 检查"NULL"值（字符串形式的缺失值）
for col in df.columns:
    null_count = (df[col] == 'NULL').sum()
    if null_count > 0:
        print(f"{col}: {null_count} 'NULL'值")

# 将字符串'NULL'转换为数值型缺失值并处理
for col in ['agent', 'company']:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    df[col] = df[col].fillna(0)
```

## 2. 整体取消率分析

计算总体取消率，并按酒店类型进行初步分析：

```python
# 计算整体取消率
total_bookings = len(df)
canceled_bookings = df[df['is_canceled'] == 1].shape[0]
overall_cancellation_rate = (canceled_bookings / total_bookings) * 100

print(f"整体取消率: {overall_cancellation_rate:.2f}%")

# 按酒店类型分析取消率
hotel_types = df['hotel'].unique()
for hotel_type in hotel_types:
    hotel_df = df[df['hotel'] == hotel_type]
    hotel_canceled = hotel_df[hotel_df['is_canceled'] == 1].shape[0]
    hotel_rate = (hotel_canceled / len(hotel_df)) * 100
    print(f"{hotel_type} 酒店取消率: {hotel_rate:.2f}%")
```

分析结果显示整体取消率为37.04%，其中城市酒店(City Hotel)的取消率(41.73%)明显高于度假酒店(Resort Hotel)(27.76%)。

## 3. 用户属性维度分析

### 3.1 新客户与重复客户分析

根据`is_repeated_guest`字段，区分新客户与重复客户，分析其取消率差异：

```python
# 新客户与重复客户定义
# 新客户：is_repeated_guest=0
# 重复客户：is_repeated_guest=1

# 新客户与重复客户取消率分析
new_guest_df = df[df['is_repeated_guest'] == 0]
repeat_guest_df = df[df['is_repeated_guest'] == 1]

new_guest_canceled = new_guest_df[new_guest_df['is_canceled'] == 1].shape[0]
repeat_guest_canceled = repeat_guest_df[repeat_guest_df['is_canceled'] == 1].shape[0]

new_guest_rate = (new_guest_canceled / len(new_guest_df)) * 100
repeat_guest_rate = (repeat_guest_canceled / len(repeat_guest_df)) * 100

print(f"新客户数量: {len(new_guest_df)}")
print(f"重复客户数量: {len(repeat_guest_df)}")
print(f"新客户取消率: {new_guest_rate:.2f}%")
print(f"重复客户取消率: {repeat_guest_rate:.2f}%")
print(f"取消率差异(新客户 - 重复客户): {new_guest_rate - repeat_guest_rate:.2f}%")

# 卡方检验 - 判断新/重复客户与取消率关系是否显著
contingency_table = pd.crosstab(df['is_repeated_guest'], df['is_canceled'])
chi2, p, dof, expected = chi2_contingency(contingency_table)
print(f"卡方值: {chi2:.4f}")
print(f"p值: {p:.10f}")
print(f"是否显著 (95% 水平): {'显著' if p < 0.05 else '不显著'}")
```

分析结果表明：
- 新客户占总体的96.81%（115580人），重复客户仅占3.19%（3810人）
- 新客户取消率为37.79%，而重复客户取消率仅为14.49%，差异达23.30个百分点
- 卡方检验结果显示，这一差异在统计上是显著的（p < 0.05）

### 3.2 客户类型分析

分析不同客户类型(`customer_type`)的取消率差异：

```python
# 客户类型与取消率关系分析
customer_types = df['customer_type'].unique()
for ctype in customer_types:
    type_df = df[df['customer_type'] == ctype]
    type_canceled = type_df[type_df['is_canceled'] == 1].shape[0]
    type_rate = (type_canceled / len(type_df)) * 100
    print(f"{ctype} 客户取消率: {type_rate:.2f}%")

# 卡方检验 - 判断客户类型与取消率关系是否显著
contingency_table = pd.crosstab(df['customer_type'], df['is_canceled'])
chi2, p, dof, expected = chi2_contingency(contingency_table)
print(f"卡方值: {chi2:.4f}")
print(f"p值: {p:.10f}")
print(f"是否显著 (95% 水平): {'显著' if p < 0.05 else '不显著'}")
```

分析结果显示:
- Transient（临时）客户取消率最高，为40.75%
- Group（团体）客户取消率最低，仅为10.23%
- Contract（合约）客户和Transient-Party（临时团体）客户的取消率分别为30.96%和25.43%
- 卡方检验结果显示，不同客户类型的取消率差异在统计上是显著的（p < 0.05）

## 4. 行为特征维度分析

### 4.1 提前预订时长分析

分析提前预订时长(`lead_time`)与取消率的关系：

```python
# 将lead_time进行分组
def group_lead_time(lead_time):
    if lead_time <= 7:
        return "0-7天"
    elif lead_time <= 30:
        return "8-30天"
    elif lead_time <= 90:
        return "31-90天"
    elif lead_time <= 180:
        return "91-180天"
    else:
        return "180天以上"

# 添加lead_time分组字段
df['lead_time_group'] = df['lead_time'].apply(group_lead_time)

# 分析各lead_time组的取消率
lead_time_groups = df['lead_time_group'].unique()
for group in sorted(lead_time_groups, key=lambda x: ["0-7天", "8-30天", "31-90天", "91-180天", "180天以上"].index(x)):
    group_df = df[df['lead_time_group'] == group]
    group_canceled = group_df[group_df['is_canceled'] == 1].shape[0]
    group_rate = (group_canceled / len(group_df)) * 100
    print(f"{group} 预订取消率: {group_rate:.2f}%，订单数: {len(group_df)}")

# 卡方检验 - 判断lead_time分组与取消率关系是否显著
contingency_table = pd.crosstab(df['lead_time_group'], df['is_canceled'])
chi2, p, dof, expected = chi2_contingency(contingency_table)
print(f"卡方值: {chi2:.4f}")
print(f"p值: {p:.10f}")
print(f"是否显著 (95% 水平): {'显著' if p < 0.05 else '不显著'}")

# 可视化lead_time与取消率的关系
plt.figure(figsize=(10, 6))
group_rates = df.groupby('lead_time_group')['is_canceled'].mean() * 100
group_rates = group_rates.reindex(["0-7天", "8-30天", "31-90天", "91-180天", "180天以上"])
group_rates.plot(kind='bar', color='skyblue')
plt.title('提前预订时长与取消率关系')
plt.xlabel('提前预订时长')
plt.ylabel('取消率 (%)')
plt.ylim(0, 100)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('lead_time_cancellation.png')
```

分析结果表明：
- 提前预订时间越长，取消率越高
- 提前预订0-7天的订单取消率最低，仅为9.63%
- 提前预订180天以上的订单取消率最高，达57.01%
- 卡方检验结果显示，提前预订时长与取消率的关系在统计上是显著的（p < 0.05）

### 4.2 预订修改次数分析

分析预订修改次数(`booking_changes`)与取消率的关系：

```python
# 预订修改次数与取消率关系分析
booking_changes = df['booking_changes'].unique()
booking_changes_sorted = sorted(booking_changes)

for changes in booking_changes_sorted:
    changes_df = df[df['booking_changes'] == changes]
    changes_canceled = changes_df[changes_df['is_canceled'] == 1].shape[0]
    changes_rate = (changes_canceled / len(changes_df)) * 100
    print(f"修改 {changes} 次的订单取消率: {changes_rate:.2f}%，订单数: {len(changes_df)}")

# 将修改次数分组，便于分析
def group_booking_changes(changes):
    if changes == 0:
        return "0次"
    elif changes == 1:
        return "1次"
    elif changes <= 3:
        return "2-3次"
    else:
        return "4次及以上"

df['booking_changes_group'] = df['booking_changes'].apply(group_booking_changes)

# 分析各预订修改次数组的取消率
changes_groups = df['booking_changes_group'].unique()
for group in sorted(changes_groups, key=lambda x: ["0次", "1次", "2-3次", "4次及以上"].index(x)):
    group_df = df[df['booking_changes_group'] == group]
    group_canceled = group_df[group_df['is_canceled'] == 1].shape[0]
    group_rate = (group_canceled / len(group_df)) * 100
    print(f"{group} 修改的订单取消率: {group_rate:.2f}%，订单数: {len(group_df)}")

# 卡方检验 - 判断预订修改次数与取消率关系是否显著
contingency_table = pd.crosstab(df['booking_changes_group'], df['is_canceled'])
chi2, p, dof, expected = chi2_contingency(contingency_table)
print(f"卡方值: {chi2:.4f}")
print(f"p值: {p:.10f}")
print(f"是否显著 (95% 水平): {'显著' if p < 0.05 else '不显著'}")
```

分析结果显示：
- 未修改过预订信息(0次)的订单取消率最高，达40.85%
- 修改过1次的订单取消率显著下降至14.23%
- 总体来看，进行过预订修改的订单取消率明显低于未修改的订单
- 卡方检验结果表明，预订修改次数与取消率的关系在统计上是显著的（p < 0.05）

## 5. 渠道与市场维度分析

### 5.1 分销渠道分析

分析分销渠道(`distribution_channel`)与取消率的关系：

```python
# 分销渠道与取消率关系分析
channels = df['distribution_channel'].unique()
for channel in channels:
    channel_df = df[df['distribution_channel'] == channel]
    channel_canceled = channel_df[channel_df['is_canceled'] == 1].shape[0]
    channel_rate = (channel_canceled / len(channel_df)) * 100
    print(f"{channel} 渠道取消率: {channel_rate:.2f}%，订单数: {len(channel_df)}")

# 卡方检验 - 判断分销渠道与取消率关系是否显著
contingency_table = pd.crosstab(df['distribution_channel'], df['is_canceled'])
chi2, p, dof, expected = chi2_contingency(contingency_table)
print(f"卡方值: {chi2:.4f}")
print(f"p值: {p:.10f}")
print(f"是否显著 (95% 水平): {'显著' if p < 0.05 else '不显著'}")

# 可视化分销渠道与取消率的关系
plt.figure(figsize=(10, 6))
channel_rates = df.groupby('distribution_channel')['is_canceled'].mean() * 100
channel_rates.plot(kind='bar', color='lightgreen')
plt.title('分销渠道与取消率关系')
plt.xlabel('分销渠道')
plt.ylabel('取消率 (%)')
plt.ylim(0, 100)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('channel