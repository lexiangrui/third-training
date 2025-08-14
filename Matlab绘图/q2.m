% 可视化问题二每个决策时刻三类切片的总QoS随时间变化
filename = '../report/code/q2_enum_results_all.csv';
data = readtable(filename);

% 获取窗口编号
windows = unique(data.window);
num_windows = length(windows);

% 初始化每类切片的最优QoS
qos_U = zeros(num_windows,1);
qos_E = zeros(num_windows,1);
qos_M = zeros(num_windows,1);

for i = 1:num_windows
    % 取当前窗口所有枚举方案
    idx = data.window == windows(i);
    sub = data(idx,:);
    % 找到obj最大值的方案
    [~,best_idx] = max(sub.obj);
    qos_U(i) = sub.sum_URLLC(best_idx);
    qos_E(i) = sub.sum_eMBB(best_idx);
    qos_M(i) = sub.sum_mMTC(best_idx);
end

% 双y轴可视化：左轴为QoS，右轴为RB分配
% 分组柱状图：每个决策时刻三类切片的RB分配，柱顶标注总QoS
rb_U = zeros(num_windows,1);
rb_E = zeros(num_windows,1);
rb_M = zeros(num_windows,1);
qos_total = zeros(num_windows,1);
for i = 1:num_windows
    idx = data.window == windows(i);
    sub = data(idx,:);
    [~,best_idx] = max(sub.obj);
    rb_U(i) = sub.R_U(best_idx);
    rb_E(i) = sub.R_E(best_idx);
    rb_M(i) = sub.R_M(best_idx);
    qos_total(i) = sub.obj(best_idx);
end

figure('Color','w','Position',[100 100 1000 420]);
bar_data = [rb_U, rb_E, rb_M];

yyaxis left;
b = bar(windows*100, bar_data, 'grouped');
b(1).FaceColor = [0.85 0.33 0.10]; % URLLC 红
b(2).FaceColor = [0 0.45 0.74];    % eMBB 蓝
b(3).FaceColor = [0.47 0.67 0.19]; % mMTC 绿
ylabel('分配的RB数量');
max_bar = max(bar_data(:));
ylim([0, max_bar*1.15]);
grid on; box on;

yyaxis right;
ph = plot(windows*100, qos_total, '-d', 'LineWidth',2, 'Color',[0.2 0.2 0.2], 'DisplayName','总QoS');
ymax = ylim;
offset = 0.03 * (ymax(2) - ymax(1));
for i = 1:num_windows
    x = windows(i)*100;
    y = qos_total(i);
    text(x, y+offset, sprintf('%.2f', y), 'HorizontalAlignment','center', 'VerticalAlignment','bottom', 'FontSize',10, 'Color',[0.2 0.2 0.2], 'FontWeight','bold');
end
ylabel('总QoS');

xlabel('决策时刻 (ms)');
title('各决策时刻三类切片的资源分配与总QoS');
legend({'URLLC','eMBB','mMTC','总QoS'},'Location','northeast');