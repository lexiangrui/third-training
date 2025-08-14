#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""问题五：能耗优化GA-MPC求解器（二阶段：功率优先+RB枚举）"""

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
BS_LIST = ["MBS_1", "SBS_1", "SBS_2", "SBS_3"]
SBS_ONLY = [bs for bs in BS_LIST if bs.startswith("SBS_")]
RB_TOTAL = {"MBS_1": 100, "SBS_1": 50, "SBS_2": 50, "SBS_3": 50}
P_MIN_DBM = {"MBS_1": 10.0, "SBS_1": 10.0, "SBS_2": 10.0, "SBS_3": 10.0}
P_MAX_DBM = {"MBS_1": 40.0, "SBS_1": 30.0, "SBS_2": 30.0, "SBS_3": 30.0}

ALPHA, M_U, M_E, M_M = 0.95, 5.0, 3.0, 1.0
SLA_L_MS = {"U": 5.0, "E": 100.0, "M": 500.0}
SLA_R_E_MBPS = 50.0
WINDOW_MS, TOTAL_MS = 100, 1000

# 能耗模型参数
P_STATIC_W = 28.0
DELTA_W_PER_RB = 0.75
ETA = 0.35

# GA参数（阶段一功率优化，适度精简）
POP_SIZE, MAX_GEN = 16, 120
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
    phi = {}
    h_abs = {}
    
    CSV_PL = {
        "MBS_1": os.path.join(ATTACH4_DIR, "MBS_1_大规模衰减.csv"),
        "SBS_1": os.path.join(ATTACH4_DIR, "SBS_1_大规模衰减.csv"),
        "SBS_2": os.path.join(ATTACH4_DIR, "SBS_2_大规模衰减.csv"),
        "SBS_3": os.path.join(ATTACH4_DIR, "SBS_3_小规模瑞丽衰减.csv"),
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
    
    _, pos_series = load_time_series_csv(os.path.join(ATTACH4_DIR, "taskflow_用户位置.csv"))
    pos_x = {}
    pos_y = {}
    for key, arr in pos_series.items():
        if key.endswith("_X"):
            pos_x[key] = arr
        elif key.endswith("_Y"):
            pos_y[key] = arr
    
    users = sorted(arrivals.keys(), key=lambda x: (x[0], int(x[1:]) if x[1:].isdigit() else 0))
    return Env(time_list=time_list, arrivals=arrivals, phi=phi, h_abs=h_abs, pos_x=pos_x, pos_y=pos_y), users

def user_category(name: str) -> str:
    """确定用户类别"""
    if name.startswith("U"):
        return "U"
    if name.startswith("e"):
        return "E"
    return "M"

def user_rate_bps(env: Env, bs: str, name: str, cat: str, power_alloc: Dict[str, Dict[str, float]], active_sbs_same_slice: List[str], t_ms: int) -> float:
    """计算下行速率"""
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
    """带能耗计算的窗口仿真"""
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
    
    sum_U = sum_E = sum_M = energy_J = 0.0
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
        
        # 能耗累计
        p_total_w_all_bs = 0.0
        for bs in BS_LIST:
            n_active_rb = 0
            p_tx_w_bs = 0.0
            for s in SLICE_LIST:
                n_users_act = len(active[bs][s])
                n_active_rb += n_users_act * RB_PER_SLICE[s]
                p_tx_w_bs += n_users_act * (10 ** (power_alloc[bs][s] / 10.0) / 1000.0)  # mW to W
            p_bs = P_STATIC_W + DELTA_W_PER_RB * n_active_rb + (1.0 / ETA) * p_tx_w_bs
            p_total_w_all_bs += p_bs
        energy_J += p_total_w_all_bs * 0.001  # 1 ms
    
    # mMTC评分
    if len(had_m_users) > 0:
        ratio = len(success_m_users) / len(had_m_users)
        for u in had_m_users:
            sum_M += ratio if u in success_m_users else -M_M
    
    obj = sum_U + sum_E + sum_M
    result = type('SimResult', (), {'sum_U': sum_U, 'sum_E': sum_E, 'sum_M': sum_M, 'obj': obj, 'energy_J': energy_J})()
    return states, result

def nearest_sbs_per_user(env: Env, users: List[str], t_ms: int) -> Dict[str, str]:
    """计算最近SBS"""
    BS_COORD = {
        "SBS_1": (0.0, 500.0),
        "SBS_2": (-433.0127, -250.0),
        "SBS_3": (433.0127, -250.0),
    }
    out = {}
    for u in users:
        x, y = env.user_xy(u, t_ms)
        best_bs = None
        best_d2 = 1e99
        for sbs in ["SBS_1", "SBS_2", "SBS_3"]:
            bx, by = BS_COORD[sbs]
            d2 = (x - bx) ** 2 + (y - by) ** 2
            if d2 < best_d2:
                best_d2 = d2
                best_bs = sbs
        out[u] = best_bs or "SBS_1"
    return out

# 阶段一：功率优化GA
def random_individual(num_users: int):
    """阶段一个体：仅编码功率"""
    power = np.random.uniform(20.0, 30.0, size=3 * len(BS_LIST)).astype(np.float32)
    return [power]

def make_equal_rb_alloc() -> Dict[str, Dict[str, int]]:
    """生成均衡RB分配"""
    out = {}
    for bs in BS_LIST:
        remain = RB_TOTAL[bs]
        x = {"U": 0, "E": 0, "M": 0}
        order = ["U", "E", "M"]
        idx = 0
        while remain >= 2:
            s = order[idx % 3]
            need = RB_PER_SLICE[s]
            if remain >= need:
                x[s] += need
                remain -= need
            idx += 1
        out[bs] = x
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

def evaluate_power_individual(indiv, env: Env, users: List[str], user_states: Dict[str, UserState], t0: int, nearest_map: Dict[str, str]) -> Tuple[float, float, float]:
    """阶段一评价：以能耗最小为目标"""
    power_arr = indiv[0]
    power_alloc = decode_power(power_arr)
    rb_alloc = make_equal_rb_alloc()
    mapping = {u: ("MBS_1" if nearest_map.get(u) is None else nearest_map[u]) for u in users}
    _, res = simulate_window(env, user_states, mapping, rb_alloc, power_alloc, t0, copy_state=True)
    fitness = -res.energy_J  # 最小化能耗
    return fitness, res.obj, res.energy_J

def tournament_select(pop_scores: List[float]) -> int:
    cand = random.sample(range(len(pop_scores)), TOURN_K)
    cand.sort(key=lambda idx: pop_scores[idx], reverse=True)
    return cand[0]

def crossover(parent1, parent2):
    if random.random() > CROSS_RATE:
        return parent1, parent2
    p1 = parent1[0]
    p2 = parent2[0]
    child_p = (p1 + p2) / 2.0
    return [child_p.copy()], [child_p.copy()]

def mutate(indiv):
    power_arr = indiv[0]
    for i in range(len(power_arr)):
        if random.random() < MUTATE_RATE:
            power_arr[i] += np.random.normal(0.0, 1.0)

# 阶段二：RB枚举优化
def gen_splits_for_total(total: int) -> List[Tuple[int, int, int]]:
    """生成合法的(U,E,M) RB分配"""
    splits = []
    for nU in range(0, total + 1, RB_PER_SLICE['U']):
        for nE in range(0, total - nU + 1, RB_PER_SLICE['E']):
            nM = total - nU - nE
            if nM >= 0 and nM % RB_PER_SLICE['M'] == 0:
                splits.append((nU, nE, nM))
    return splits

def coordinate_enumeration_rb(env: Env, users: List[str], user_states: Dict[str, UserState], mapping: Dict[str, str], power_alloc: Dict[str, Dict[str, float]], t0: int, rounds: int = 2) -> Dict[str, Dict[str, int]]:
    """坐标枚举RB分配"""
    rb_alloc = make_equal_rb_alloc()
    bs_splits = {bs: gen_splits_for_total(RB_TOTAL[bs]) for bs in BS_LIST}
    
    for _ in range(rounds):
        for bs in BS_LIST:
            best_obj = -1e30
            best_split = rb_alloc[bs]['U'], rb_alloc[bs]['E'], rb_alloc[bs]['M']
            for (nU, nE, nM) in bs_splits[bs]:
                rb_alloc[bs]['U'] = nU
                rb_alloc[bs]['E'] = nE
                rb_alloc[bs]['M'] = nM
                _, res = simulate_window(env, user_states, mapping, rb_alloc, power_alloc, t0, copy_state=True)
                if res.obj > best_obj:
                    best_obj = res.obj
                    best_split = (nU, nE, nM)
            rb_alloc[bs]['U'], rb_alloc[bs]['E'], rb_alloc[bs]['M'] = best_split
    return rb_alloc

def main():
    """附录精简：第五问省略与第二问相同的MPC/仿真循环与二阶段细节，仅保留核心函数与接口。"""
    print("[附录] 问题五：已省略MPC滚动仿真与二阶段完整流程，仅保留能耗模型/评价与接口。")

if __name__ == "__main__":
    main()
