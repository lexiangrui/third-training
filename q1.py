import pandas as pd
import numpy as np
from typing import List, Tuple
import itertools
import warnings
warnings.filterwarnings('ignore')

# 系统参数
class SystemParams:
    # 资源块参数
    TOTAL_RBS = 50  # 总资源块数
    RB_BANDWIDTH = 360e3  # 单个资源块带宽 360kHz
    RB_TIME = 1e-3  # 资源块时间跨度 1ms
    
    # 用户类型资源块需求
    URLLC_RB = 10
    EMBB_RB = 5
    MMTC_RB = 2
    
    # SLA参数
    URLLC_RATE_SLA = 10e6  # 10Mbps
    EMBB_RATE_SLA = 50e6   # 50Mbps
    MMTC_RATE_SLA = 1e6    # 1Mbps
    
    URLLC_DELAY_SLA = 5e-3   # 5ms
    EMBB_DELAY_SLA = 100e-3  # 100ms
    MMTC_DELAY_SLA = 500e-3  # 500ms
    
    # 惩罚系数
    URLLC_PENALTY = 5
    EMBB_PENALTY = 3
    MMTC_PENALTY = 1
    
    # URLLC效益折扣系数
    URLLC_ALPHA = 0.95
    
    # 功率参数
    TX_POWER_DBM = 30  # 发射功率 30dBm
    NOISE_DENSITY = -174  # 热噪声密度 dBm/Hz
    NOISE_FIGURE = 7  # 噪声系数 dB

class DataLoader:
    """加载问题一的数据"""
    def __init__(self, data_path):
        self.data_path = data_path
        self.load_data()
    
    def load_data(self):
        # 加载任务流数据
        self.task_data = pd.read_csv(f"{self.data_path}/q1_任务流.csv")
        # 加载大规模衰减数据
        self.large_scale_fading = pd.read_csv(f"{self.data_path}/q1_大规模衰减.csv")
        # 加载小规模瑞丽衰减数据
        self.small_scale_fading = pd.read_csv(f"{self.data_path}/q1_小规模瑞丽衰减.csv")
        
        # 提取用户列表
        self.urllc_users = ['U1', 'U2']
        self.embb_users = ['e1', 'e2', 'e3', 'e4']
        self.mmtc_users = ['m1', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7', 'm8', 'm9', 'm10']
        
        # 获取时刻0的数据
        self.tasks_t0 = self.task_data.iloc[0]
        self.large_fading_t0 = self.large_scale_fading.iloc[0]
        self.small_fading_t0 = self.small_scale_fading.iloc[0]

class NetworkSimulator:
    """网络仿真器"""
    def __init__(self, data_loader):
        self.data = data_loader
        self.params = SystemParams()
    
    def calculate_sinr(self, user, num_rbs):
        """计算信干噪比（根据修订后的附录）"""
        # 获取信道参数
        large_fading_db = self.data.large_fading_t0[user]
        small_fading = self.data.small_fading_t0[user]
        
        # 计算接收功率 (W) - 根据附录公式
        # p_r(W) = 10^((P_dBm - φ - 30)/10) * |h|^2
        rx_power_w = 10 ** ((self.params.TX_POWER_DBM - large_fading_db - 30) / 10) * small_fading
        
        # 计算噪声功率 (W) - 使用总带宽 i*b
        total_bandwidth = num_rbs * self.params.RB_BANDWIDTH  # i*b
        noise_power_dbm = self.params.NOISE_DENSITY + 10 * np.log10(total_bandwidth) + self.params.NOISE_FIGURE
        noise_power_w = 10 ** ((noise_power_dbm - 30) / 10)
        
        # 计算SINR（无干扰情况）
        sinr = rx_power_w / noise_power_w
        return sinr
    
    def calculate_rate(self, user, num_rbs):
        """计算传输速率（根据修订后的附录）"""
        if num_rbs == 0:
            return 0
        
        sinr = self.calculate_sinr(user, num_rbs)
        # r = i*b * log(1+γ)
        total_bandwidth = num_rbs * self.params.RB_BANDWIDTH
        rate = total_bandwidth * np.log2(1 + sinr)
        return rate
    
    def calculate_transmission_time(self, user, num_rbs):
        """计算传输时间"""
        task_size = self.data.tasks_t0[user] * 1e6  # Mbit转bit
        rate = self.calculate_rate(user, num_rbs)
        if rate > 0:
            return task_size / rate
        else:
            return float('inf')
    
    def evaluate_urllc_qos(self, user, num_rbs):
        """评估URLLC用户服务质量（根据修订后的附录）"""
        if num_rbs < self.params.URLLC_RB:
            return -self.params.URLLC_PENALTY  # 资源不足，任务丢失
        
        trans_time = self.calculate_transmission_time(user, self.params.URLLC_RB)
        total_delay = trans_time  # 假设无排队延迟
        
        if total_delay <= self.params.URLLC_DELAY_SLA:
            # y^URLLC = α^L (L以ms为单位)
            delay_ms = total_delay * 1000
            return self.params.URLLC_ALPHA ** delay_ms
        else:
            return -self.params.URLLC_PENALTY
    
    def evaluate_embb_qos(self, user, num_rbs):
        """评估eMBB用户服务质量（根据修订后的附录三段式）"""
        if num_rbs < self.params.EMBB_RB:
            return -self.params.EMBB_PENALTY  # 资源不足，任务丢失
        
        rate = self.calculate_rate(user, self.params.EMBB_RB)
        trans_time = self.calculate_transmission_time(user, self.params.EMBB_RB)
        total_delay = trans_time  # 假设无排队延迟
        
        # 三段式QoS函数
        if total_delay > self.params.EMBB_DELAY_SLA:
            return -self.params.EMBB_PENALTY  # 超时惩罚
        elif rate >= self.params.EMBB_RATE_SLA:
            return 1.0  # 达标
        else:
            return rate / self.params.EMBB_RATE_SLA  # 未达标按比例
    
    def evaluate_mmtc_qos_individual(self, user, num_rbs):
        """评估单个mMTC用户是否成功接入"""
        if num_rbs < self.params.MMTC_RB:
            return 0  # 未分配资源，未接入
        
        trans_time = self.calculate_transmission_time(user, self.params.MMTC_RB)
        total_delay = trans_time  # 假设无排队延迟
        
        if total_delay <= self.params.MMTC_DELAY_SLA:
            return 1  # 成功接入
        else:
            return 0  # 超时，接入失败
    
    def evaluate_total_qos(self, rb_allocation):
        """评估总体服务质量"""
        urllc_rbs, embb_rbs, mmtc_rbs = rb_allocation
        total_qos = 0
        
        # URLLC用户 - 按优先级分配
        remaining_urllc_rbs = urllc_rbs
        for user in self.data.urllc_users:
            if remaining_urllc_rbs >= self.params.URLLC_RB:
                qos = self.evaluate_urllc_qos(user, self.params.URLLC_RB)
                remaining_urllc_rbs -= self.params.URLLC_RB
            else:
                qos = -self.params.URLLC_PENALTY
            total_qos += qos
        
        # eMBB用户 - 按优先级分配
        remaining_embb_rbs = embb_rbs
        for user in self.data.embb_users:
            if remaining_embb_rbs >= self.params.EMBB_RB:
                qos = self.evaluate_embb_qos(user, self.params.EMBB_RB)
                remaining_embb_rbs -= self.params.EMBB_RB
            else:
                qos = -self.params.EMBB_PENALTY
            total_qos += qos
        
        # mMTC用户 - 计算接入比例（根据修订后的附录）
        mmtc_need_count = len(self.data.mmtc_users)  # 需要接入的用户数
        mmtc_served_count = 0  # 成功接入的用户数
        mmtc_has_timeout = False  # 是否有超时
        
        remaining_mmtc_rbs = mmtc_rbs
        for user in self.data.mmtc_users:
            if remaining_mmtc_rbs >= self.params.MMTC_RB:
                success = self.evaluate_mmtc_qos_individual(user, self.params.MMTC_RB)
                if success == 1:
                    mmtc_served_count += 1
                else:
                    # 分配了资源但超时
                    mmtc_has_timeout = True
                remaining_mmtc_rbs -= self.params.MMTC_RB
        
        # mMTC切片级QoS：接入比例或惩罚
        if mmtc_has_timeout:
            mmtc_qos = -self.params.MMTC_PENALTY * mmtc_need_count  # 有超时，给予惩罚
        else:
            # 接入比例
            if mmtc_need_count > 0:
                mmtc_qos = mmtc_served_count / mmtc_need_count
            else:
                mmtc_qos = 0
        
        total_qos += mmtc_qos
        
        return total_qos

class EnumerationOptimizer:
    """枚举法优化器"""
    def __init__(self, simulator):
        self.simulator = simulator
        self.params = SystemParams()
    
    def generate_feasible_allocations(self):
        """生成所有可行的资源分配方案"""
        n_urllc = len(self.simulator.data.urllc_users)
        n_embb = len(self.simulator.data.embb_users)
        n_mmtc = len(self.simulator.data.mmtc_users)
        
        # 计算各类用户的最大需求
        max_urllc = n_urllc * self.params.URLLC_RB  # 20
        max_embb = n_embb * self.params.EMBB_RB     # 20
        max_mmtc = n_mmtc * self.params.MMTC_RB     # 20
        
        feasible_allocations = []
        
        # 枚举所有可能的分配
        # URLLC: 0, 10, 20 (步长10)
        for urllc_rbs in range(0, min(max_urllc + 1, self.params.TOTAL_RBS + 1), self.params.URLLC_RB):
            # eMBB: 0, 5, 10, 15, 20 (步长5)
            for embb_rbs in range(0, min(max_embb + 1, self.params.TOTAL_RBS + 1), self.params.EMBB_RB):
                # mMTC: 0, 2, 4, ..., 20 (步长2)
                for mmtc_rbs in range(0, min(max_mmtc + 1, self.params.TOTAL_RBS + 1), self.params.MMTC_RB):
                    # 检查总资源约束
                    if urllc_rbs + embb_rbs + mmtc_rbs <= self.params.TOTAL_RBS:
                        feasible_allocations.append([urllc_rbs, embb_rbs, mmtc_rbs])
        
        return feasible_allocations
    
    def optimize(self):
        """枚举法求解最优资源分配"""
        print("\n=== 使用枚举法求解资源分配问题 ===")
        print(f"总资源块数: {self.params.TOTAL_RBS}")
        print(f"URLLC用户: {len(self.simulator.data.urllc_users)}个，每个需要{self.params.URLLC_RB}个RB")
        print(f"eMBB用户: {len(self.simulator.data.embb_users)}个，每个需要{self.params.EMBB_RB}个RB")
        print(f"mMTC用户: {len(self.simulator.data.mmtc_users)}个，每个需要{self.params.MMTC_RB}个RB")
        
        # 生成所有可行分配
        feasible_allocations = self.generate_feasible_allocations()
        print(f"\n生成了 {len(feasible_allocations)} 个可行分配方案")
        
        # 评估每个分配方案
        best_allocation = None
        best_qos = -float('inf')
        
        print("\n开始评估所有方案...")
        for i, allocation in enumerate(feasible_allocations):
            qos = self.simulator.evaluate_total_qos(allocation)
            
            if qos > best_qos:
                best_qos = qos
                best_allocation = allocation
                print(f"  找到更优解: {allocation} -> QoS = {qos:.4f}")
        
        return best_allocation, best_qos

def main():
    """主函数"""
    # 数据路径（使用相对路径）
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "题目/附件/附件1")
    
    # 如果相对路径不存在，尝试绝对路径
    if not os.path.exists(data_path):
        data_path = "/Users/lexiangrui/Desktop/训三/题目/附件/附件1"
    
    print(f"数据路径: {data_path}")
    
    # 加载数据
    data_loader = DataLoader(data_path)
    print("\n数据加载完成")
    print(f"URLLC用户: {data_loader.urllc_users}")
    print(f"eMBB用户: {data_loader.embb_users}")
    print(f"mMTC用户: {data_loader.mmtc_users}")
    
    # 创建网络仿真器
    simulator = NetworkSimulator(data_loader)
    
    # 创建优化器并求解
    optimizer = EnumerationOptimizer(simulator)
    best_allocation, best_qos = optimizer.optimize()
    
    # 输出最终结果
    print("\n" + "="*60)
    print("最优资源分配方案:")
    print("="*60)
    print(f"URLLC: {best_allocation[0]} 个资源块")
    print(f"eMBB:  {best_allocation[1]} 个资源块")
    print(f"mMTC:  {best_allocation[2]} 个资源块")
    print(f"总使用: {sum(best_allocation)} 个资源块")
    print(f"最大QoS: {best_qos:.4f}")
    
    # 详细分析
    print("\n" + "="*60)
    print("详细分析:")
    print("="*60)
    
    urllc_rbs, embb_rbs, mmtc_rbs = best_allocation
    
    # URLLC分析
    n_urllc_served = min(urllc_rbs // simulator.params.URLLC_RB, len(data_loader.urllc_users))
    print(f"\nURLLC切片:")
    print(f"  分配资源块: {urllc_rbs}")
    print(f"  服务用户数: {n_urllc_served}/{len(data_loader.urllc_users)}")
    
    # eMBB分析
    n_embb_served = min(embb_rbs // simulator.params.EMBB_RB, len(data_loader.embb_users))
    print(f"\neMBB切片:")
    print(f"  分配资源块: {embb_rbs}")
    print(f"  服务用户数: {n_embb_served}/{len(data_loader.embb_users)}")
    
    # mMTC分析
    n_mmtc_served = min(mmtc_rbs // simulator.params.MMTC_RB, len(data_loader.mmtc_users))
    print(f"\nmMTC切片:")
    print(f"  分配资源块: {mmtc_rbs}")
    print(f"  可服务用户数: {n_mmtc_served}/{len(data_loader.mmtc_users)}")
    
    # 验证一些关键分配方案的QoS
    print("\n" + "="*60)
    print("其他典型分配方案的QoS对比:")
    print("="*60)
    
    test_allocations = [
        ([20, 10, 20], "URLLC优先"),
        ([20, 20, 10], "URLLC+eMBB优先"),
        ([10, 20, 20], "eMBB+mMTC优先"),
        ([20, 15, 15], "平衡方案1"),
        ([10, 25, 15], "eMBB最大化"),
    ]
    
    for allocation, description in test_allocations:
        if sum(allocation) <= simulator.params.TOTAL_RBS:
            qos = simulator.evaluate_total_qos(allocation)
            print(f"{description:15} {allocation} -> QoS = {qos:.4f}")

if __name__ == "__main__":
    main()
