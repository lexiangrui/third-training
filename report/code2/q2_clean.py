#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""问题二：单微基站MPC滚动窗口求解器"""

import csv
import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Deque
from collections import deque

# 系统常量
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
ATTACH_DIR = os.path.join(ROOT_DIR, "题目", "附件", "附件2")

P_TX_DBM = 30.0
B_HZ = 360_000.0
NF_DB = 7.0
V_U, V_E, V_M = 10, 5, 2
ALPHA, M_U, M_E, M_M = 0.95, 5.0, 3.0, 1.0
SLA_L_U_MS, SLA_L_E_MS, SLA_L_M_MS = 5.0, 100.0, 500.0
SLA_R_E_MBPS = 50.0
WINDOW_MS, TOTAL_MS = 100, 1000

@dataclass
class Chunk:
    arrival_ms: int
    size_mbit: float
    remain_mbit: float
    start_ms: int | None = None
    finish_ms: int | None = None

@dataclass
class UserState:
    name: str
    category: str
    queue: Deque[Chunk] = field(default_factory=deque)

    def has_backlog(self) -> bool:
        return len(self.queue) > 0 and self.queue[0].remain_mbit > 1e-15

@dataclass
class Env:
    time_list: List[float]
    arrivals_mbit: Dict[str, List[float]]
    pl_db: Dict[str, List[float]]
    ray_raw: Dict[str, List[float]]
    small_scale_is_db: bool

    def get_phi_db(self, name: str, t_ms: int) -> float:
        arr = self.pl_db.get(name)
        if arr is None:
            return 100.0
        t_ms = max(0, min(t_ms, len(arr) - 1))
        return float(arr[t_ms])

    def get_h_pow(self, name: str, t_ms: int) -> float:
        arr = self.ray_raw.get(name)
        if arr is None or len(arr) == 0:
            return 1.0
        t_ms = max(0, min(t_ms, len(arr) - 1))
        val = float(arr[t_ms])
        if self.small_scale_is_db:
            return 10 ** (val / 10.0)
        return val * val if val >= 0 else 1e-6

def load_time_series_csv(path: str) -> Tuple[List[float], Dict[str, List[float]]]:
    """读取时间序列CSV"""
    time_list = []
    series = {}
    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            t_str = row.get("Time", row.get("time", row.get("TIME")))
            if t_str is None:
                continue
            try:
                t_val = float(t_str)
            except ValueError:
                continue
            time_list.append(t_val)
            for key, val in row.items():
                k = key.strip()
                if k.lower() == "time":
                    continue
                if val is None or val == "":
                    series.setdefault(k, []).append(0.0)
                    continue
                try:
                    v = float(val)
                except ValueError:
                    v = 0.0
                series.setdefault(k, []).append(v)
    # 对齐长度
    max_len = len(time_list)
    for k, arr in series.items():
        if len(arr) < max_len:
            arr.extend([0.0] * (max_len - len(arr)))
    return time_list, series

def build_env() -> Tuple[Env, List[str], List[str], List[str]]:
    """构建环境和用户列表"""
    time_list, arrivals = load_time_series_csv(os.path.join(ATTACH_DIR, "q2_用户任务流.csv"))
    _, pl = load_time_series_csv(os.path.join(ATTACH_DIR, "q2_大规模衰减.csv"))
    _, ray = load_time_series_csv(os.path.join(ATTACH_DIR, "q2_小规模瑞丽衰减.csv"))
    
    # 检测小规模衰减是否为dB
    small_is_db = any(v < 0 for arr in ray.values() for v in arr[:200])
    
    def sort_key(name: str) -> Tuple[int, int]:
        prefix_rank = 0 if name.startswith("U") else (1 if name.startswith("e") else 2)
        num = int(name[1:]) if name[1:].isdigit() else 0
        return (prefix_rank, num)

    all_names = list(arrivals.keys())
    U_names = sorted([n for n in all_names if n.startswith("U")], key=sort_key)
    E_names = sorted([n for n in all_names if n.startswith("e")], key=sort_key)
    M_names = sorted([n for n in all_names if n.startswith("m")], key=sort_key)

    env = Env(time_list=time_list, arrivals_mbit=arrivals, pl_db=pl, ray_raw=ray, small_scale_is_db=small_is_db)
    return env, U_names, E_names, M_names

def user_rate_bps(env: Env, name: str, t_ms: int, num_rbs: int) -> float:
    """计算用户传输速率"""
    if num_rbs <= 0:
        return 0.0
    phi_db = env.get_phi_db(name, t_ms)
    h_pow = env.get_h_pow(name, t_ms)
    p_rx_mw = 10 ** ((P_TX_DBM - phi_db) / 10.0) * h_pow
    n0_mw = 10 ** ((-174.0 + 10.0 * math.log10(num_rbs * B_HZ) + NF_DB) / 10.0)
    snr = p_rx_mw / max(n0_mw, 1e-30)
    return num_rbs * B_HZ * math.log(1.0 + snr, 2)

def simulate_window(env: Env, users: Dict[str, UserState], U_order: List[str], E_order: List[str], M_order: List[str], t0: int, nU: int, nE: int, nM: int):
    """仿真100ms窗口"""
    from copy import deepcopy
    st = {name: UserState(name=u.name, category=u.category, queue=deque(deepcopy(list(u.queue)))) for name, u in users.items()}
    
    capU = nU // V_U if V_U > 0 else 0
    capE = nE // V_E if V_E > 0 else 0
    capM = nM // V_M if V_M > 0 else 0
    
    activeU, activeE, activeM = [], [], []
    m_had_arrival = set()
    m_success_users = set()
    
    sum_U = sum_E = sum_m_score = 0.0
    
    t1 = min(t0 + WINDOW_MS, TOTAL_MS)
    for t in range(t0, t1):
        # 1) 到达处理
        for name, arr_series in env.arrivals_mbit.items():
            if t < len(arr_series):
                vol = arr_series[t]
                if vol > 0.0:
                    st[name].queue.append(Chunk(arrival_ms=t, size_mbit=vol, remain_mbit=vol))
                    if name.startswith("m") and t0 <= t < t1:
                        m_had_arrival.add(name)
        
        # 2) 填充并发槽位
        def fill_active(order: List[str], active: List[str], cap: int):
            active[:] = [nm for nm in active if st[nm].has_backlog()]
            for nm in order:
                if len(active) >= cap or nm in active:
                    continue
                if st[nm].has_backlog():
                    active.append(nm)
        
        fill_active(U_order, activeU, capU)
        fill_active(E_order, activeE, capE)
        fill_active(M_order, activeM, capM)
        
        # 3) 服务处理
        def serve_one(name: str, per_user_rbs: int):
            if not st[name].has_backlog():
                return
            head = st[name].queue[0]
            if head.start_ms is None:
                head.start_ms = t
            r_bps = user_rate_bps(env, name, t, per_user_rbs)
            served_mbit = (r_bps * 0.001) / 1e6
            head.remain_mbit -= served_mbit
            if head.remain_mbit <= 1e-12:
                head.remain_mbit = 0.0
                head.finish_ms = t + 1
        
        for nm in activeU:
            serve_one(nm, V_U)
        for nm in activeE:
            serve_one(nm, V_E)
        for nm in activeM:
            serve_one(nm, V_M)
        
        # 4) QoS统计
        def collect_finished(order: List[str], slice_type: str):
            nonlocal sum_U, sum_E, m_success_users
            for nm in order:
                while st[nm].queue and st[nm].queue[0].finish_ms == t + 1:
                    ch = st[nm].queue.popleft()
                    L_ms = (ch.finish_ms - ch.arrival_ms)
                    if slice_type == 'U':
                        sum_U += ALPHA ** L_ms if L_ms <= SLA_L_U_MS else -M_U
                    elif slice_type == 'E':
                        r_mbps = (ch.size_mbit * 1000.0) / L_ms
                        if L_ms <= SLA_L_E_MS and r_mbps >= SLA_R_E_MBPS:
                            sum_E += 1.0
                        elif L_ms <= SLA_L_E_MS:
                            sum_E += max(0.0, r_mbps / SLA_R_E_MBPS)
                        else:
                            sum_E += -M_E
                    else:  # mMTC
                        if t0 <= ch.arrival_ms < t1 and L_ms <= SLA_L_M_MS:
                            m_success_users.add(nm)
        
        collect_finished(U_order, 'U')
        collect_finished(E_order, 'E')
        collect_finished(M_order, 'M')
    
    # 5) mMTC最终计分
    ratio = len(m_success_users) / len(m_had_arrival) if m_had_arrival else 0.0
    for u in m_had_arrival:
        sum_m_score += ratio if u in m_success_users else -M_M
    
    obj = sum_U + sum_E + sum_m_score
    return st, type('SimResult', (), {'sum_U': sum_U, 'sum_E': sum_E, 'sum_m_score': sum_m_score, 'obj': obj})()

def enumerate_splits() -> List[Tuple[int, int, int]]:
    """枚举所有合法RB分配"""
    splits = []
    for nU in range(0, 51, V_U):
        for nE in range(0, 51 - nU, V_E):
            nM = 50 - nU - nE
            if nM >= 0 and nM % V_M == 0:
                splits.append((nU, nE, nM))
    return splits

def main():
    env, U_names, E_names, M_names = build_env()
    users = {}
    for nm in U_names:
        users[nm] = UserState(name=nm, category='U')
    for nm in E_names:
        users[nm] = UserState(name=nm, category='E')
    for nm in M_names:
        users[nm] = UserState(name=nm, category='M')
    
    splits = enumerate_splits()
    total_obj = 0.0
    
    # 逐窗口MPC决策
    for w in range(0, TOTAL_MS, WINDOW_MS):
        best_split, best_state_after, best_res = (0, 0, 50), None, None
        best_score = -1e18
        for (nU, nE, nM) in splits:
            new_state, res = simulate_window(env, users, U_names, E_names, M_names, w, nU, nE, nM)
            if res.obj > best_score:
                best_split, best_state_after, best_res = (nU, nE, nM), new_state, res
                best_score = res.obj
        users = best_state_after
        total_obj += best_res.obj
        
        # 简化输出：仅在结尾汇总
    
    print(f"累计目标函数: {total_obj:.4f}")

if __name__ == "__main__":
    main()
