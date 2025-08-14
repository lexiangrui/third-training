#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""问题四：异构网络GA-MPC求解器（MBS+SBS，接入受限）"""

import csv
import math
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
from collections import deque
from copy import deepcopy

# 路径和常量配置
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
ATTACH4_DIR = os.path.join(ROOT_DIR, "题目", "附件", "附件4")

B_HZ = 360_000.0
NF_DB = 7.0
RB_PER_SLICE = {"U": 10, "E": 5, "M": 2}
SLICE_LIST = ["U", "E", "M"]
ALPHA, M_U, M_E, M_M = 0.95, 5.0, 3.0, 1.0
SLA_L_MS = {"U": 5.0, "E": 100.0, "M": 500.0}
SLA_R_E_MBPS = 50.0
WINDOW_MS, TOTAL_MS = 100, 1000

# 基站配置
BS_LIST = ["MBS_1", "SBS_1", "SBS_2", "SBS_3"]
SBS_ONLY = [bs for bs in BS_LIST if bs.startswith("SBS_")]
RB_TOTAL = {"MBS_1": 100, "SBS_1": 50, "SBS_2": 50, "SBS_3": 50}
P_MIN_DBM = {"MBS_1": 10.0, "SBS_1": 10.0, "SBS_2": 10.0, "SBS_3": 10.0}
P_MAX_DBM = {"MBS_1": 40.0, "SBS_1": 30.0, "SBS_2": 30.0, "SBS_3": 30.0}

# 基站坐标
BS_COORD = {
    "MBS_1": (0.0, 0.0),
    "SBS_1": (0.0, 500.0),
    "SBS_2": (-433.0127, -250.0),
    "SBS_3": (433.0127, -250.0),
}

# GA参数（适度精简）
POP_SIZE, MAX_GEN = 30, 200
TOURN_K, CROSS_RATE, MUTATE_RATE, ELITE_NUM = 3, 0.8, 0.3, 5

@dataclass
class Env:
    time_list: List[float]
    arrivals: Dict[str, List[float]]
    phi: Dict[str, Dict[str, List[float]]]
    h_abs: Dict[str, Dict[str, List[float]]]
    pos_x: Dict[str, List[float]]
    pos_y: Dict[str, List[float]]

    def phi_db(self, bs: str, user: str, t: int) -> float:
        arr = self.phi.get(bs, {}).get(user)
        if arr is None or len(arr) == 0:
            return 120.0
        idx = min(max(t, 0), len(arr) - 1)
        return float(arr[idx])

    def h_pow(self, bs: str, user: str, t: int) -> float:
        arr = self.h_abs.get(bs, {}).get(user)
        if arr is None or len(arr) == 0:
            return 1.0
        idx = min(max(t, 0), len(arr) - 1)
        val = float(arr[idx])
        return val * val if val >= 0 else 0.0

    def user_xy(self, user: str, t: int) -> Tuple[float, float]:
        xs = self.pos_x.get(f"{user}_X") or self.pos_x.get(f"{user}X") or self.pos_x.get(user)
        ys = self.pos_y.get(f"{user}_Y") or self.pos_y.get(f"{user}Y") or self.pos_y.get(user)
        if not xs or not ys:
            return 0.0, 0.0
        idx = min(max(t, 0), min(len(xs), len(ys)) - 1)
        return float(xs[idx]), float(ys[idx])

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
    queue: deque[Chunk]

    def has_backlog(self) -> bool:
        return len(self.queue) > 0 and self.queue[0].remain_mbit > 1e-12

def load_time_series_csv(path: str) -> Tuple[List[float], Dict[str, List[float]]]:
    """读取时间序列CSV"""
    time_list = []
    series = {}
    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            t_val_str = row.get("Time", row.get("time", row.get("TIME")))
            if t_val_str is None:
                continue
            try:
                t_val = float(t_val_str)
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
    max_len = len(time_list)
    for arr in series.values():
        if len(arr) < max_len:
            arr.extend([0.0] * (max_len - len(arr)))
    return time_list, series

def build_env() -> Tuple[Env, List[str]]:
    """构建环境"""
    time_list, arrivals = load_time_series_csv(os.path.join(ATTACH4_DIR, "taskflow_用户任务流.csv"))
    
    # 信道数据
    phi = {}
    h_abs = {}
    CSV_PL = {
        "MBS_1": os.path.join(ATTACH4_DIR, "MBS_1_大规模衰减.csv"),
        "SBS_1": os.path.join(ATTACH4_DIR, "SBS_1_大规模衰减.csv"),
        "SBS_2": os.path.join(ATTACH4_DIR, "SBS_2_大规模衰减.csv"),
        "SBS_3": os.path.join(ATTACH4_DIR, "SBS_3_大规模衰减.csv"),
    }
    CSV_RAY = {
        "MBS_1": os.path.join(ATTACH4_DIR, "MBS_1_小规模瑞丽衰减.csv"),
        "SBS_1": os.path.join(ATTACH4_DIR, "SBS_1_小规模瑞丽衰减.csv"),
        "SBS_2": os.path.join(ATTACH4_DIR, "SBS_2_小规模瑞丽衰减.csv"),
        "SBS_3": os.path.join(ATTACH4_DIR, "SBS_3_小规模瑞丽衰减.csv"),
    }
    
    for bs in BS_LIST:
        _, phi_bs = load_time_series_csv(CSV_PL[bs])
        _, ray_bs = load_time_series_csv(CSV_RAY[bs])
        phi[bs] = phi_bs
        h_abs[bs] = ray_bs
    
    # 位置数据
    _, pos_series = load_time_series_csv(os.path.join(ATTACH4_DIR, "taskflow_用户位置.csv"))
    pos_x = {}
    pos_y = {}
    for key, arr in pos_series.items():
        if key.endswith("_X"):
            pos_x[key] = arr
        elif key.endswith("_Y"):
            pos_y[key] = arr
    
    all_users = sorted(arrivals.keys(), key=lambda x: (x[0], int(x[1:]) if x[1:].isdigit() else 0))
    return Env(time_list=time_list, arrivals=arrivals, phi=phi, h_abs=h_abs, pos_x=pos_x, pos_y=pos_y), all_users

def user_category(name: str) -> str:
    """确定用户类别"""
    if name.startswith("U"):
        return "U"
    if name.startswith("e"):
        return "E"
    return "M"

def user_rate_bps(env: Env, bs: str, name: str, cat: str, power_alloc: Dict[str, Dict[str, float]], active_sbs_same_slice: List[str], t_ms: int) -> float:
    """计算下行速率，MBS无干扰，SBS受其他SBS干扰"""
    p_tx_mw = 10 ** (power_alloc[bs][cat] / 10.0)
    phi_db = env.phi_db(bs, name, t_ms)
    h_pow = env.h_pow(bs, name, t_ms)
    recv_mw = p_tx_mw * 10 ** (-phi_db / 10.0) * h_pow
    
    interf_mw = 0.0
    if bs in SBS_ONLY:
        for b2 in active_sbs_same_slice:
            if b2 == bs:
                continue
            p_int_mw = 10 ** (power_alloc[b2][cat] / 10.0)
            phi_db_i = env.phi_db(b2, name, t_ms)
            h_pow_i = env.h_pow(b2, name, t_ms)
            interf_mw += p_int_mw * 10 ** (-phi_db_i / 10.0) * h_pow_i
    
    n0_mw = 10 ** ((-174.0 + 10.0 * math.log10(RB_PER_SLICE[cat] * B_HZ) + NF_DB) / 10.0)
    sinr = recv_mw / (interf_mw + n0_mw + 1e-30)
    return RB_PER_SLICE[cat] * B_HZ * math.log2(1.0 + sinr)

def simulate_window(env: Env, init_states: Dict[str, UserState], mapping: Dict[str, str], rb_alloc: Dict[str, Dict[str, int]], power_alloc: Dict[str, Dict[str, float]], t0: int, copy_state: bool = True):
    """异构网络窗口仿真"""
    states = {}
    if copy_state:
        for name, st in init_states.items():
            states[name] = UserState(name=st.name, category=st.category, queue=deque(deepcopy(list(st.queue))))
    else:
        states = init_states
    
    cap = {bs: {s: rb_alloc[bs][s] // RB_PER_SLICE[s] for s in SLICE_LIST} for bs in BS_LIST}
    active = {bs: {s: [] for s in SLICE_LIST} for bs in BS_LIST}
    
    def sort_key(u: str):
        return (u[0], int(u[1:]) if u[1:].isdigit() else 0)
    
    order = {bs: {s: [] for s in SLICE_LIST} for bs in BS_LIST}
    for nm, bs in mapping.items():
        cat = user_category(nm)
        order[bs][cat].append(nm)
    for bs in BS_LIST:
        for s in SLICE_LIST:
            order[bs][s].sort(key=sort_key)
    
    sum_U = sum_E = sum_M = 0.0
    had_m_users = set()
    success_m_users = set()
    
    t1 = min(t0 + WINDOW_MS, TOTAL_MS)
    for t in range(t0, t1):
        # 到达
        for nm in states.keys():
            arr_series = env.arrivals.get(nm, [])
            if t < len(arr_series):
                vol = arr_series[t]
                if vol > 0.0:
                    states[nm].queue.append(Chunk(arrival_ms=t, size_mbit=vol, remain_mbit=vol))
                    if states[nm].category == 'M':
                        had_m_users.add(nm)
        
        # 填充并发槽位
        for bs in BS_LIST:
            for s in SLICE_LIST:
                active[bs][s] = [u for u in active[bs][s] if states[u].has_backlog()]
                while len(active[bs][s]) < cap[bs][s]:
                    for cand in order[bs][s]:
                        if cand in active[bs][s]:
                            continue
                        if states[cand].has_backlog():
                            active[bs][s].append(cand)
                            break
                    else:
                        break
        
        # 服务
        for bs in BS_LIST:
            for s in SLICE_LIST:
                if len(active[bs][s]) == 0:
                    continue
                active_sbs_same_slice = [b for b in SBS_ONLY if len(active[b][s]) > 0]
                for u in list(active[bs][s]):
                    rate_bps = user_rate_bps(env, bs, u, s, power_alloc, active_sbs_same_slice, t)
                    served_mbit = (rate_bps * 0.001) / 1e6
                    head = states[u].queue[0]
                    if head.start_ms is None:
                        head.start_ms = t
                    head.remain_mbit -= served_mbit
                    if head.remain_mbit <= 1e-12:
                        head.remain_mbit = 0.0
                        head.finish_ms = t + 1
                        states[u].queue.popleft()
                        # QoS统计
                        L_ms = (head.finish_ms - head.arrival_ms)
                        if s == 'U':
                            sum_U += ALPHA ** L_ms if L_ms <= SLA_L_MS['U'] else -M_U
                        elif s == 'E':
                            r_mbps = (head.size_mbit * 1000.0) / max(L_ms, 1e-12)
                            if L_ms <= SLA_L_MS['E'] and r_mbps >= SLA_R_E_MBPS:
                                sum_E += 1.0
                            elif L_ms <= SLA_L_MS['E']:
                                sum_E += max(0.0, r_mbps / SLA_R_E_MBPS)
                            else:
                                sum_E += -M_E
                        else:
                            if L_ms <= SLA_L_MS['M']:
                                success_m_users.add(u)
    
    # mMTC评分
    if len(had_m_users) > 0:
        ratio = len(success_m_users) / len(had_m_users)
        for u in had_m_users:
            sum_M += ratio if u in success_m_users else -M_M
    
    obj = sum_U + sum_E + sum_M
    result = type('SimResult', (), {'sum_U': sum_U, 'sum_E': sum_E, 'sum_M': sum_M, 'obj': obj})()
    return states, result

def nearest_sbs_per_user(env: Env, users: List[str], t_ms: int) -> Dict[str, str]:
    """计算每个用户的最近SBS"""
    out = {}
    for u in users:
        x, y = env.user_xy(u, t_ms)
        best_bs = None
        best_d2 = 1e99
        for sbs in SBS_ONLY:
            bx, by = BS_COORD[sbs]
            d2 = (x - bx) ** 2 + (y - by) ** 2
            if d2 < best_d2:
                best_d2 = d2
                best_bs = sbs
        out[u] = best_bs or "SBS_1"
    return out

# 遗传算法实现
def random_individual(num_users: int):
    """生成随机个体：用户选择(0=MBS,1=nearest SBS) + RB分配 + 功率"""
    user_choice = np.random.randint(0, 2, size=num_users, dtype=np.int8)
    rb_counts = np.zeros(2 * len(BS_LIST), dtype=np.int16)
    idx = 0
    for bs in BS_LIST:
        total = RB_TOTAL[bs]
        u_units_max = total // RB_PER_SLICE['U']
        u_units = np.random.randint(0, u_units_max + 1)
        u_rb = int(u_units * RB_PER_SLICE['U'])
        e_units_max = (total - u_rb) // RB_PER_SLICE['E']
        e_units = np.random.randint(0, e_units_max + 1)
        e_rb = int(e_units * RB_PER_SLICE['E'])
        rb_counts[idx] = u_rb
        rb_counts[idx + 1] = e_rb
        idx += 2
    power = np.random.uniform(20.0, 30.0, size=3 * len(BS_LIST)).astype(np.float32)
    return [user_choice, rb_counts, power]

def decode_rb(rb_counts: np.ndarray) -> Dict[str, Dict[str, int]]:
    """解码RB分配"""
    out = {}
    idx = 0
    for bs in BS_LIST:
        total = RB_TOTAL[bs]
        u_req = int(rb_counts[idx])
        e_req = int(rb_counts[idx + 1])
        idx += 2
        u_rb = max(0, min(total, (u_req // RB_PER_SLICE['U']) * RB_PER_SLICE['U']))
        e_rb = max(0, min(total - u_rb, (e_req // RB_PER_SLICE['E']) * RB_PER_SLICE['E']))
        if u_rb + e_rb > total:
            e_rb = max(0, (total - u_rb) // RB_PER_SLICE['E'] * RB_PER_SLICE['E'])
        m_rb = total - (u_rb + e_rb)
        if m_rb % RB_PER_SLICE['M'] != 0:
            if e_rb >= RB_PER_SLICE['E']:
                e_rb -= RB_PER_SLICE['E']
            elif e_rb + RB_PER_SLICE['E'] <= total - u_rb:
                e_rb += RB_PER_SLICE['E']
            elif u_rb >= RB_PER_SLICE['U']:
                u_rb -= RB_PER_SLICE['U']
            m_rb = total - (u_rb + e_rb)
            if m_rb % RB_PER_SLICE['M'] != 0 and u_rb >= RB_PER_SLICE['U']:
                u_rb -= RB_PER_SLICE['U']
                m_rb = total - (u_rb + e_rb)
        out[bs] = {"U": u_rb, "E": e_rb, "M": m_rb}
    return out

def decode_power(power_arr: np.ndarray) -> Dict[str, Dict[str, float]]:
    """解码功率分配"""
    out = {}
    idx = 0
    for bs in BS_LIST:
        sub = {}
        for s in SLICE_LIST:
            p = float(power_arr[idx])
            p = max(P_MIN_DBM[bs], min(P_MAX_DBM[bs], p))
            sub[s] = p
            idx += 1
        out[bs] = sub
    return out

def evaluate_individual(indiv, env: Env, users: List[str], user_states: Dict[str, UserState], t0: int, nearest_sbs_map: Dict[str, str]) -> float:
    """评价个体适应度"""
    user_choice, rb_counts, power_arr = indiv
    rb_alloc = decode_rb(rb_counts)
    power_alloc = decode_power(power_arr)
    mapping = {users[i]: ("MBS_1" if int(user_choice[i]) == 0 else nearest_sbs_map[users[i]]) for i in range(len(users))}
    _, res = simulate_window(env, user_states, mapping, rb_alloc, power_alloc, t0, copy_state=True)
    return res.obj

def tournament_select(pop_scores: List[float]) -> int:
    """锦标赛选择"""
    cand = random.sample(range(len(pop_scores)), TOURN_K)
    cand.sort(key=lambda idx: pop_scores[idx], reverse=True)
    return cand[0]

def crossover(parent1, parent2):
    """交叉操作"""
    if random.random() > CROSS_RATE:
        return parent1, parent2
    u1, rb1, p1 = parent1
    u2, rb2, p2 = parent2
    num_users = len(u1)
    if num_users >= 2:
        pt = random.randint(1, num_users - 1)
        child_u1 = np.concatenate([u1[:pt], u2[pt:]]).astype(np.int8)
        child_u2 = np.concatenate([u2[:pt], u1[pt:]]).astype(np.int8)
    else:
        child_u1 = u1.copy()
        child_u2 = u2.copy()
    child_rb1 = ((rb1.astype(np.float32) + rb2) / 2.0).astype(np.int16)
    child_rb2 = child_rb1.copy()
    child_p1 = (p1 + p2) / 2.0
    child_p2 = child_p1.copy()
    return [child_u1, child_rb1, child_p1], [child_u2, child_rb2, child_p2]

def mutate(indiv):
    """变异操作"""
    user_choice, rb_counts, power_arr = indiv
    num_users = len(user_choice)
    # 接入变异
    for i in range(num_users):
        if random.random() < MUTATE_RATE:
            user_choice[i] = 1 - int(user_choice[i])
    # RB变异
    for b in range(len(BS_LIST)):
        iu = 2 * b
        ie = iu + 1
        total = RB_TOTAL[BS_LIST[b]]
        if random.random() < MUTATE_RATE:
            u_units_max = total // RB_PER_SLICE['U']
            u_units = np.random.randint(0, u_units_max + 1)
            rb_counts[iu] = int(u_units * RB_PER_SLICE['U'])
        if random.random() < MUTATE_RATE:
            u_rb = int(rb_counts[iu])
            e_units_max = (total - u_rb) // RB_PER_SLICE['E']
            e_units = np.random.randint(0, e_units_max + 1)
            rb_counts[ie] = int(e_units * RB_PER_SLICE['E'])
    # 功率高斯扰动
    for i in range(len(power_arr)):
        if random.random() < MUTATE_RATE:
            power_arr[i] += np.random.normal(0.0, 1.0)

def main():
    """附录精简：第四问省略与第二问相同的MPC滚动仿真，仅保留关键函数/接口。"""
    print("[附录] 问题四：已省略MPC滚动仿真与完整GA循环，仅保留核心组件与接口。")

if __name__ == "__main__":
    main()
