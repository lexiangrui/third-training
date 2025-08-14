#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 基站坐标 - 问题四的异构网络
bs_positions = {
    'MBS_1': (0, 0),        # 宏基站
    'SBS_1': (0, 500),      # 微基站1
    'SBS_2': (-433.0127, -250), # 微基站2
    'SBS_3': (433.0127, -250)   # 微基站3
}

# 基站颜色
bs_colors = {
    'MBS_1': 'red',
    'SBS_1': 'blue', 
    'SBS_2': 'green',
    'SBS_3': 'orange'
}

# 第0个时刻用户坐标（从taskflow_用户位置.csv第一行数据提取）
user_positions = {
    'U1': (599.2650255, -402.8801163),
    'U2': (-515.0375859, -565.0239406),
    'U3': (-403.9704089, -122.5778481),
    'U4': (431.0012707, 657.1032969),
    'U5': (-779.3680745, 71.78877133),
    'U6': (-265.6328344, 142.2265352),
    'U7': (689.0929412, -179.3026883),
    'U8': (88.21855281, -122.8038017),
    'U9': (32.03904523, -658.3337893),
    'U10': (-280.3576055, -415.2047582),
    'e1': (658.8096535, 133.5655329),
    'e2': (140.8728867, 98.50893562),
    'e3': (-272.8789199, 608.4558131),
    'e4': (-137.6224923, 55.75231957),
    'e5': (196.1279227, -671.9030457),
    'e6': (-527.4675443, 187.6982008),
    'e7': (19.83680664, -673.4884098),
    'e8': (-370.3823901, -545.7228436),
    'e9': (-275.9440639, 2.834116196),
    'e10': (-401.34141, -238.256001),
    'e11': (-22.20457007, 693.0496375),
    'e12': (517.806987, -423.4349872),
    'e13': (381.1552963, 452.714467),
    'e14': (219.0676482, -341.7714108),
    'e15': (29.37065327, 721.3002017),
    'e16': (155.8449682, 446.8902241),
    'e17': (-619.091576, 104.8965177),
    'e18': (-627.0444384, -372.236237),
    'e19': (-171.0719432, 746.8087167),
    'e20': (-507.6372971, 474.021435),
    'm1': (207.7941701, 73.26590059),
    'm2': (646.3297831, -284.4737417),
    'm3': (-592.2377614, 115.3274163),
    'm4': (243.4899468, 395.5635363),
    'm5': (-439.1397469, -79.58221056),
    'm6': (-50.53223938, 618.6388068),
    'm7': (-7.714158309, -664.1064541),
    'm8': (30.5064091, 229.5971586),
    'm9': (143.1998943, -277.5168044),
    'm10': (704.0516735, 376.6117461),
    'm11': (253.8117186, -61.95069202),
    'm12': (289.0084625, -642.1978992),
    'm13': (-187.8796203, 136.8958945),
    'm14': (-650.1493557, 298.9117323),
    'm15': (-29.54883038, 339.8668374),
    'm16': (201.0501528, -216.0152039),
    'm17': (363.7227268, 468.63205),
    'm18': (-373.9201271, 508.2319189),
    'm19': (450.428716, 232.9627341),
    'm20': (113.339281, 257.0580734),
    'm21': (491.8235848, 158.6225366),
    'm22': (-776.3215412, 44.61189538),
    'm23': (376.211071, -273.1386825),
    'm24': (50.40315556, -261.9711303),
    'm25': (-323.7687613, 223.2844248),
    'm26': (271.57325, -103.4961648),
    'm27': (564.4415818, 222.5752737),
    'm28': (205.6453536, -428.6394763),
    'm29': (80.86152265, 144.9010625),
    'm30': (-410.0380143, -547.8791861),
    'm31': (-169.7831705, 566.7964753),
    'm32': (-134.5396628, -320.6705372),
    'm33': (-342.0502736, -344.7206115),
    'm34': (205.7988375, -97.81405755),
    'm35': (-513.4465995, 218.9071672),
    'm36': (-442.1591378, -23.66103299),
    'm37': (201.0806462, -694.8734803),
    'm38': (185.7442308, -455.8606242),
    'm39': (439.4053182, -177.1699375),
    'm40': (-426.1874152, -412.8746833)
}

# 第0个窗口用户-基站分配关系（问题四）
user_bs_allocation = {
    'U1': 'SBS_3', 'U2': 'MBS_1', 'U3': 'MBS_1', 'U4': 'MBS_1', 'U5': 'MBS_1', 'U6': 'SBS_2',
    'U7': 'MBS_1', 'U8': 'MBS_1', 'U9': 'SBS_3', 'U10': 'SBS_2',
    'e1': 'MBS_1', 'e2': 'MBS_1', 'e3': 'SBS_1', 'e4': 'SBS_2', 'e5': 'MBS_1', 'e6': 'MBS_1',
    'e7': 'MBS_1', 'e8': 'MBS_1', 'e9': 'MBS_1', 'e10': 'SBS_2', 'e11': 'SBS_1', 'e12': 'MBS_1',
    'e13': 'SBS_1', 'e14': 'SBS_3', 'e15': 'MBS_1', 'e16': 'SBS_1', 'e17': 'MBS_1', 'e18': 'SBS_2',
    'e19': 'MBS_1', 'e20': 'SBS_1',
    'm1': 'SBS_3', 'm2': 'SBS_3', 'm3': 'MBS_1', 'm4': 'SBS_1', 'm5': 'SBS_2', 'm6': 'SBS_1',
    'm7': 'SBS_2', 'm8': 'SBS_1', 'm9': 'SBS_3', 'm10': 'MBS_1', 'm11': 'SBS_3', 'm12': 'MBS_1',
    'm13': 'SBS_1', 'm14': 'SBS_2', 'm15': 'MBS_1', 'm16': 'SBS_3', 'm17': 'MBS_1', 'm18': 'MBS_1',
    'm19': 'MBS_1', 'm20': 'MBS_1', 'm21': 'MBS_1', 'm22': 'SBS_2', 'm23': 'MBS_1', 'm24': 'SBS_3',
    'm25': 'MBS_1', 'm26': 'MBS_1', 'm27': 'MBS_1', 'm28': 'SBS_3', 'm29': 'MBS_1', 'm30': 'MBS_1',
    'm31': 'SBS_1', 'm32': 'MBS_1', 'm33': 'SBS_2', 'm34': 'SBS_3', 'm35': 'SBS_2', 'm36': 'SBS_2',
    'm37': 'MBS_1', 'm38': 'SBS_3', 'm39': 'SBS_3', 'm40': 'SBS_2'
}

# 创建图形
plt.figure(figsize=(16, 12))
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']  # 支持中文显示
plt.rcParams['axes.unicode_minus'] = False
# 设置matplotlib后端，避免显示问题
import matplotlib
matplotlib.use('Agg')

# 绘制基站
for bs_name, pos in bs_positions.items():
    if bs_name.startswith('MBS'):
        # 宏基站用更大的方形
        plt.scatter(pos[0], pos[1], c=bs_colors[bs_name], s=300, marker='s', 
                   edgecolors='black', linewidth=2, label=f'{bs_name}(宏基站)', zorder=5)
    else:
        # 微基站用小一些的方形
        plt.scatter(pos[0], pos[1], c=bs_colors[bs_name], s=200, marker='s', 
                   edgecolors='black', linewidth=1.5, label=f'{bs_name}(微基站)', zorder=5)
    plt.annotate(bs_name, (pos[0], pos[1]), xytext=(8, 8), 
                textcoords='offset points', fontsize=12, fontweight='bold')

# 绘制用户并连线到对应基站
user_types = {
    'URLLC': ['U1', 'U2', 'U3', 'U4', 'U5', 'U6', 'U7', 'U8', 'U9', 'U10'],
    'eMBB': [f'e{i}' for i in range(1, 21)],
    'mMTC': [f'm{i}' for i in range(1, 41)]
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
                                      shrinkA=5, shrinkB=8),
                        zorder=1)
            
            # 添加用户标签
            plt.annotate(user, user_pos, xytext=(2, 2), 
                        textcoords='offset points', fontsize=8)

# 创建图例
legend_elements = []
# 基站图例
for bs_name, color in bs_colors.items():
    if bs_name.startswith('MBS'):
        legend_elements.append(plt.Line2D([0], [0], marker='s', color='w', 
                                        markerfacecolor=color, markersize=8, 
                                        markeredgecolor='black', label=f'{bs_name}(宏基站)'))
    else:
        legend_elements.append(plt.Line2D([0], [0], marker='s', color='w', 
                                        markerfacecolor=color, markersize=6, 
                                        markeredgecolor='black', label=f'{bs_name}(微基站)'))

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
plt.title('异构网络第一次决策窗口用户与基站分配关系图', fontsize=20, fontweight='bold', pad=15)

# 设置坐标轴范围
all_x = [pos[0] for pos in user_positions.values()] + [pos[0] for pos in bs_positions.values()]
all_y = [pos[1] for pos in user_positions.values()] + [pos[1] for pos in bs_positions.values()]
plt.xlim(min(all_x) - 50, max(all_x) + 50)
plt.ylim(min(all_y) - 50, max(all_y) + 50)

# 调整布局并保存
plt.tight_layout()
plt.savefig('q4_user_bs_allocation_window0.png', dpi=300, bbox_inches='tight')
plt.savefig('q4_user_bs_allocation_window0.pdf', format='pdf', bbox_inches='tight')
print("问题四图片已保存为 q4_user_bs_allocation_window0.png")
print("问题四图片已保存为 q4_user_bs_allocation_window0.pdf")

# 输出统计信息
print("问题四第0个窗口用户分配统计：")
allocation_stats = {}
for bs in bs_colors.keys():
    allocation_stats[bs] = sum(1 for allocated_bs in user_bs_allocation.values() if allocated_bs == bs)

for bs, count in allocation_stats.items():
    bs_type = "宏基站" if bs.startswith('MBS') else "微基站"
    print(f"{bs}({bs_type}): {count}个用户")
    
print(f"\n总用户数: {len(user_bs_allocation)}")
