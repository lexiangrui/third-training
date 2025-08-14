#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 基站坐标
bs_positions = {
    'BS1': (0, 500),
    'BS2': (-433.0127, -250),
    'BS3': (433.0127, -250)
}

# 基站颜色
bs_colors = {
    'BS1': 'red',
    'BS2': 'blue', 
    'BS3': 'green'
}

# 第0个时刻用户坐标（从taskflow_用户位置.csv第一行数据提取）
user_positions = {
    'U1': (599.2650255, -402.8801163),
    'U2': (-515.0375859, -565.0239406),
    'U3': (-403.9704089, -122.5778481),
    'U4': (431.0012707, 657.1032969),
    'U5': (-779.3680745, 71.78877133),
    'U6': (-265.6328344, 142.2265352),
    'e1': (689.1041058, -179.2860946),
    'e2': (88.20024832, -122.7957428),
    'e3': (32.03990796, -658.3138079),
    'e4': (-280.3671097, -415.2223557),
    'e5': (658.8096535, 133.5655329),
    'e6': (140.8728867, 98.50893562),
    'e7': (-272.8789199, 608.4558131),
    'e8': (-137.6224923, 55.75231957),
    'e9': (196.1279227, -671.9030457),
    'e10': (-527.4675443, 187.6982008),
    'e11': (19.83680664, -673.4884098),
    'e12': (-370.3823901, -545.7228436),
    'm1': (-275.9527776, 2.836368426),
    'm2': (-401.3428842, -238.2648794),
    'm3': (-22.19557637, 693.0499743),
    'm4': (517.7982798, -423.4327102),
    'm5': (381.1499739, 452.7072094),
    'm6': (219.0678903, -341.7804075),
    'm7': (29.36252765, 721.3040714),
    'm8': (155.8450295, 446.8812243),
    'm9': (-619.0862103, 104.8892921),
    'm10': (-627.035874, -372.2334707),
    'm11': (-171.0723502, 746.8177075),
    'm12': (-507.6291019, 474.0251552),
    'm13': (207.7941701, 73.26590059),
    'm14': (646.3297831, -284.4737417),
    'm15': (-592.2377614, 115.3274163),
    'm16': (243.4899468, 395.5635363),
    'm17': (-439.1397469, -79.58221056),
    'm18': (-50.53223938, 618.6388068),
    'm19': (-7.714158309, -664.1064541),
    'm20': (30.5064091, 229.5971586),
    'm21': (143.1998943, -277.5168044),
    'm22': (704.0516735, 376.6117461),
    'm23': (253.8117186, -61.95069202),
    'm24': (289.0084625, -642.1978992),
    'm25': (-187.8796203, 136.8958945),
    'm26': (-650.1493557, 298.9117323),
    'm27': (-29.54883038, 339.8668374),
    'm28': (201.0501528, -216.0152039),
    'm29': (363.7227268, 468.63205),
    'm30': (-373.9201271, 508.2319189)
}

# 第0个窗口用户-基站分配关系
user_bs_allocation = {
    'U1': 'BS3', 'U2': 'BS2', 'U3': 'BS2', 'U4': 'BS1', 'U5': 'BS2', 'U6': 'BS3',
    'e1': 'BS1', 'e2': 'BS2', 'e3': 'BS2', 'e4': 'BS1', 'e5': 'BS3', 'e6': 'BS1',
    'e7': 'BS1', 'e8': 'BS1', 'e9': 'BS2', 'e10': 'BS1', 'e11': 'BS1', 'e12': 'BS3',
    'm1': 'BS1', 'm2': 'BS2', 'm3': 'BS2', 'm4': 'BS2', 'm5': 'BS2', 'm6': 'BS1',
    'm7': 'BS2', 'm8': 'BS3', 'm9': 'BS2', 'm10': 'BS2', 'm11': 'BS3', 'm12': 'BS1',
    'm13': 'BS3', 'm14': 'BS2', 'm15': 'BS3', 'm16': 'BS2', 'm17': 'BS3', 'm18': 'BS1',
    'm19': 'BS2', 'm20': 'BS3', 'm21': 'BS2', 'm22': 'BS2', 'm23': 'BS3', 'm24': 'BS2',
    'm25': 'BS3', 'm26': 'BS3', 'm27': 'BS3', 'm28': 'BS2', 'm29': 'BS2', 'm30': 'BS3'
}

# 创建图形
plt.figure(figsize=(14, 10))
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']  # 支持中文显示
plt.rcParams['axes.unicode_minus'] = False
# 设置matplotlib后端，避免显示问题
import matplotlib
matplotlib.use('Agg')

# 绘制基站
for bs_name, pos in bs_positions.items():
    plt.scatter(pos[0], pos[1], c=bs_colors[bs_name], s=200, marker='s', 
               edgecolors='black', linewidth=1, label=f'{bs_name}基站', zorder=5)
    plt.annotate(bs_name, (pos[0], pos[1]), xytext=(5, 5), 
                textcoords='offset points', fontsize=12, fontweight='bold')

# 绘制用户并连线到对应基站
user_types = {
    'URLLC': ['U1', 'U2', 'U3', 'U4', 'U5', 'U6'],
    'eMBB': [f'e{i}' for i in range(1, 13)],
    'mMTC': [f'm{i}' for i in range(1, 31)]
}

user_markers = {'URLLC': '^', 'eMBB': 'o', 'mMTC': 's'}
user_sizes = {'URLLC': 80, 'eMBB': 60, 'mMTC': 40}

# 绘制用户点和连线
for user_type, users in user_types.items():
    for user in users:
        if user in user_positions and user in user_bs_allocation:
            user_pos = user_positions[user]
            allocated_bs = user_bs_allocation[user]
            bs_pos = bs_positions[allocated_bs]
            
            # 绘制用户点
            plt.scatter(user_pos[0], user_pos[1], 
                       c=bs_colors[allocated_bs], 
                       marker=user_markers[user_type],
                       s=user_sizes[user_type],
                       alpha=0.7, 
                       edgecolors='black',
                       linewidth=0.5,
                       zorder=3)
            
            # 绘制连线（带箭头）
            plt.annotate('', xy=bs_pos, xytext=user_pos,
                        arrowprops=dict(arrowstyle='->', 
                                      color=bs_colors[allocated_bs],
                                      alpha=0.6,
                                      linewidth=1,
                                      shrinkA=5, shrinkB=5),
                        zorder=1)
            
            # 添加用户标签
            plt.annotate(user, user_pos, xytext=(2, 2), 
                        textcoords='offset points', fontsize=8)

# 创建图例
legend_elements = []
# 基站图例
for bs_name, color in bs_colors.items():
    legend_elements.append(plt.Line2D([0], [0], marker='s', color='w', 
                                    markerfacecolor=color, markersize=5, 
                                    markeredgecolor='black', label=f'{bs_name}基站'))

# 用户类型图例
for user_type, marker in user_markers.items():
    legend_elements.append(plt.Line2D([0], [0], marker=marker, color='w', 
                                    markerfacecolor='gray', markersize=8, 
                                    markeredgecolor='black', label=f'{user_type}用户'))

plt.legend(handles=legend_elements, loc='upper right', fontsize=10, framealpha=0.9)

# 设置图形属性
plt.grid(True, alpha=0.3)
plt.xlabel('X坐标', fontsize=12)
plt.ylabel('Y坐标', fontsize=12)
plt.title('第一次决策窗口用户与基站分配关系图', fontsize=20, fontweight='bold', pad=10)

# 设置坐标轴范围
all_x = [pos[0] for pos in user_positions.values()] + [pos[0] for pos in bs_positions.values()]
all_y = [pos[1] for pos in user_positions.values()] + [pos[1] for pos in bs_positions.values()]
plt.xlim(min(all_x) - 50, max(all_x) + 50)
plt.ylim(min(all_y) - 50, max(all_y) + 50)

# 调整布局并保存
plt.tight_layout()
plt.savefig('user_bs_allocation_window0.png', dpi=300, bbox_inches='tight')
plt.savefig('user_bs_allocation_window0.pdf', format='pdf', bbox_inches='tight')
print("图片已保存为 user_bs_allocation_window0.png")
print("图片已保存为 user_bs_allocation_window0.pdf")

# 输出统计信息
print("第0个窗口用户分配统计：")
allocation_stats = {}
for bs in bs_colors.keys():
    allocation_stats[bs] = sum(1 for allocated_bs in user_bs_allocation.values() if allocated_bs == bs)

for bs, count in allocation_stats.items():
    print(f"{bs}: {count}个用户")
    
print(f"\n总用户数: {len(user_bs_allocation)}")
