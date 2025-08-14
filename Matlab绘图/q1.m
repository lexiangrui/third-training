% 参数
csvPath = 'D:/BISAI/8.11/third-training2/third-training/report/code/q1_enum_results.csv';
topN = 20;  % 显示前N个最优方案，可根据需要调整

% 读数
opts = detectImportOptions(csvPath, 'Encoding','UTF-8');
T = readtable(csvPath, opts);

% 排序，截取前N
T = sortrows(T, 'obj', 'descend');
topN = min(topN, height(T));
Ts = T(1:topN, :);

% 构造标签 (R_U,R_E,R_M)
labels = "(" + string(Ts.R_U) + "," + string(Ts.R_E) + "," + string(Ts.R_M) + ")";

%% 图1：目标函数柱状图
figure('Color','w','Position',[100 100 1200 420]);
b1 = bar(Ts.obj, 'FaceColor',[0.2 0.45 0.8]);
grid on; box on;
xticks(1:topN); xticklabels(labels); xtickangle(45);
ylabel('目标值 Q');
xlabel('切片分配 (R_U, R_e, R_m)');
title('问题一：各枚举方案的目标函数（前N个）');

% 可选：在柱顶标注数值（N较小时开启）
if topN <= 25
    text(1:topN, Ts.obj, compose('%.3f', Ts.obj), ...
        'HorizontalAlignment','center','VerticalAlignment','bottom','FontSize',9);
end

% 保存
out1 = fullfile(fileparts(csvPath), 'q1_enum_obj_topN.png');
try
    exportgraphics(gcf, out1, 'Resolution',300);
catch
    print(gcf, out1, '-dpng', '-r300');
end

%% 图2：组成项堆叠柱状图（URLLC/eMBB/mMTC）
figure('Color','w','Position',[100 550 1200 420]);
Y = [Ts.sum_URLLC, Ts.sum_eMBB, Ts.sum_mMTC];
b2 = bar(Y, 'stacked'); grid on; box on;
% 配色（可选）
co = [0.30 0.70 0.95; 0.35 0.80 0.45; 0.95 0.65 0.35];
for i=1:min(3, numel(b2))
    b2(i).FaceColor = co(i,:);
end
xticks(1:topN); xticklabels(labels); xtickangle(45);
ylabel('子项得分');
xlabel('切片分配 (R_U, R_e, R_m)');
legend({'∑URLLC','∑eMBB','∑mMTC'}, 'Location','northwest');
title('问题一：各枚举方案的组成项堆叠（前N个）');

% 保存
out2 = fullfile(fileparts(csvPath), 'q1_enum_stacked_topN.png');
try
    exportgraphics(gcf, out2, 'Resolution',300);
catch
    print(gcf, out2, '-dpng', '-r300');
end