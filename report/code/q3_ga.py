#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
问题三（多基站同频干扰 + RB 切片 + 功率控制）——启发式求解器
==============================================================
采用“滚动 MPC + 单窗口遗传算法”策略。
每 100 ms 为一个决策窗口。外层逐窗口滚动，内层通过混合编码 GA 同时搜索：
  • 用户接入关联    a_{n,k}  (整型 0/1/2)
  • RB 切片分配    x_{n,s}  (整数，满足约束)
  • 切片功率水平    p_{n,s}  (连续 10–30 dBm)

本实现聚焦算法框架与可运行代码：
  ‑ 数据装载兼容附件 3 提供的 CSV；
  ‑ 物理层速率计算含同频干扰；
  ‑ QoS 评估使用附录中 3 类切片公式（排队 / 串并行调度在此版本中采用简化近似——假定一个窗口里所有到达量一次性完成，后续可在 TODO 标记处替换为精细仿真）。

依赖：numpy  (>=1.20)

运行示例：
$ python q3_ga.py
"""

from __future__ import annotations

import csv
import math
import os
import random
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import datetime

# =================== 路径配置 ===================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
ATTACH3_DIR = os.path.join(ROOT_DIR, "题目", "附件", "附件3")

CSV_TASKFLOW = os.path.join(ATTACH3_DIR, "taskflow_用户任务流.csv")
CSV_PL: Dict[str, str] = {
    "BS1": os.path.join(ATTACH3_DIR, "BS1_大规模衰减.csv"),
    "BS2": os.path.join(ATTACH3_DIR, "BS2_大规模衰减.csv"),
    "BS3": os.path.join(ATTACH3_DIR, "BS3_大规模衰减.csv"),
}
CSV_RAY: Dict[str, str] = {
    "BS1": os.path.join(ATTACH3_DIR, "BS1_小规模瑞丽衰减.csv"),
    "BS2": os.path.join(ATTACH3_DIR, "BS2_小规模瑞丽衰减.csv"),
    "BS3": os.path.join(ATTACH3_DIR, "BS3_小规模瑞丽衰减.csv"),
}


# =================== 系统常量 ===================
B_HZ = 360_000.0  # 单 RB 带宽 (Hz)
NF_DB = 7.0       # 噪声系数 (dB)

RB_PER_SLICE = {"U": 10, "E": 5, "M": 2}
CATEGORY_PREFIX = {"U": "U", "E": "e", "M": "m"}

ALPHA = 0.95
M_U = 5.0
M_E = 3.0
M_M = 1.0

SLA_L = {"U": 5.0, "E": 100.0, "M": 500.0}  # ms
SLA_R_E_MBPS = 50.0

# RL power bounds (dBm)
POWER_MIN = 10.0
POWER_MAX = 30.0

WINDOW_MS = 100
TOTAL_MS = 1000

# =================== 工具函数 ===================

def dbm_to_mw(dbm: float) -> float:
    return 10 ** (dbm / 10.0)


def mw_to_dbm(mw: float) -> float:
    if mw <= 0:
        return -1e9
    return 10 * math.log10(mw)


def noise_power_mw(num_rbs: int) -> float:
    """N0(dBm) = -174 + 10log10(i*b) + NF => mW"""
    if num_rbs <= 0:
        return 0.0
    n0_dbm = -174.0 + 10.0 * math.log10(num_rbs * B_HZ) + NF_DB
    return dbm_to_mw(n0_dbm)


# =================== 数据加载 ===================


def load_time_series_csv(path: str) -> Tuple[List[float], Dict[str, List[float]]]:
    """读取逐时刻 CSV。返回 (time_list, name -> series)。"""
    time_list: List[float] = []
    series: Dict[str, List[float]] = {}
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
                if key is None:
                    continue
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
    # 填齐列
    max_len = len(time_list)
    for arr in series.values():
        if len(arr) < max_len:
            arr.extend([0.0] * (max_len - len(arr)))
    return time_list, series


@dataclass
class Env:
    time_list: List[float]
    arrivals: Dict[str, List[float]]          # 用户 -> arrival (Mbit)
    phi: Dict[str, Dict[str, List[float]]]    # bs -> user -> phi_dB
    h_abs: Dict[str, Dict[str, List[float]]]  # bs -> user -> |h|

    def phi_db(self, bs: str, user: str, t: int) -> float:
        arr = self.phi.get(bs, {}).get(user)
        if arr is None or len(arr) == 0:
            return 120.0  # 极大衰减
        idx = min(max(t, 0), len(arr) - 1)
        return float(arr[idx])

    def h_pow(self, bs: str, user: str, t: int) -> float:
        arr = self.h_abs.get(bs, {}).get(user)
        if arr is None or len(arr) == 0:
            return 1.0
        idx = min(max(t, 0), len(arr) - 1)
        val = float(arr[idx])
        return val * val if val >= 0 else 0.0


def build_env() -> Tuple[Env, List[str]]:
    time_list, arrivals = load_time_series_csv(CSV_TASKFLOW)

    phi: Dict[str, Dict[str, List[float]]] = {}
    h_abs: Dict[str, Dict[str, List[float]]] = {}

    for bs in ["BS1", "BS2", "BS3"]:
        _, phi_bs = load_time_series_csv(CSV_PL[bs])
        _, ray_bs = load_time_series_csv(CSV_RAY[bs])
        phi[bs] = phi_bs
        h_abs[bs] = ray_bs

    # 确定所有用户列表（按编号顺序）
    all_users = sorted(arrivals.keys(), key=lambda x: (x[0], int(x[1:]) if x[1:].isdigit() else 0))

    return Env(time_list=time_list, arrivals=arrivals, phi=phi, h_abs=h_abs), all_users


# =================== QoS 计算 ===================

# ---- 基础分类与 QoS 函数 ----

def user_category(name: str) -> str:
    if name.startswith("U"):
        return "U"
    if name.startswith("e"):
        return "E"
    return "M"


def urllc_qos(L_ms: float) -> float:
    if L_ms <= SLA_L["U"]:
        return ALPHA ** L_ms
    return -M_U


def embb_qos(L_ms: float, r_mbps: float) -> float:
    if L_ms <= SLA_L["E"] and r_mbps >= SLA_R_E_MBPS:
        return 1.0
    if L_ms <= SLA_L["E"] and r_mbps < SLA_R_E_MBPS:
        return max(0.0, r_mbps / SLA_R_E_MBPS)
    return -M_E


def mmtc_qos(ratio: float, success: bool) -> float:
    return ratio if success else -M_M

# ---------------- 任务队列数据结构 ----------------
from collections import deque
from copy import deepcopy

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
    category: str  # 'U' | 'E' | 'M'
    queue: deque[Chunk]

    def has_backlog(self) -> bool:
        return len(self.queue) > 0 and self.queue[0].remain_mbit > 1e-12

    def total_backlog(self) -> float:
        return sum(ch.remain_mbit for ch in self.queue)

@dataclass
class SimResult:
    sum_U: float = 0.0
    sum_E: float = 0.0
    sum_M: float = 0.0
    obj: float = 0.0

# ---------------- 模拟器 ----------------

def user_rate_bps(env: Env, bs: str, name: str, cat: str, power_alloc: Dict[str, Dict[str, float]],
                  active_bs_same_slice: List[str], t_ms: int) -> float:
    """计算下行速率 (bps) , 考虑同 slice 干扰"""
    p_tx_mw = dbm_to_mw(power_alloc[bs][cat])
    phi_db = env.phi_db(bs, name, t_ms)
    h_pow = env.h_pow(bs, name, t_ms)
    recv_mw = p_tx_mw * 10 ** (-phi_db / 10.0) * h_pow

    interf_mw = 0.0
    for b2 in active_bs_same_slice:
        if b2 == bs:
            continue
        p_int_mw = dbm_to_mw(power_alloc[b2][cat])
        phi_db_i = env.phi_db(b2, name, t_ms)
        h_pow_i = env.h_pow(b2, name, t_ms)
        interf_mw += p_int_mw * 10 ** (-phi_db_i / 10.0) * h_pow_i

    n0_mw = noise_power_mw(RB_PER_SLICE[cat])
    sinr = recv_mw / (interf_mw + n0_mw + 1e-30)
    return RB_PER_SLICE[cat] * B_HZ * math.log2(1.0 + sinr)


def simulate_window_multibs(env: Env,
                            init_states: Dict[str, UserState],
                            mapping: Dict[str, str],
                            rb_alloc: Dict[str, Dict[str, int]],
                            power_alloc: Dict[str, Dict[str, float]],
                            t0: int,
                            copy_state: bool = True) -> Tuple[Dict[str, UserState], SimResult]:
    """多基站 100-ms 精细仿真。若 copy_state=True 则不影响原状态。"""
    states: Dict[str, UserState] = {}
    if copy_state:
        for name, st in init_states.items():
            states[name] = UserState(name=st.name, category=st.category, queue=deque(deepcopy(list(st.queue))))
    else:
        states = init_states  # 就地修改

    # 并发容量
    cap = {bs: {s: rb_alloc[bs][s] // RB_PER_SLICE[s] for s in SLICE_LIST} for bs in BS_LIST}
    active: Dict[str, Dict[str, List[str]]] = {bs: {s: [] for s in SLICE_LIST} for bs in BS_LIST}

    # 用户顺序（编号靠前优先） per bs-per cat
    def sort_key(u: str):
        return (u[0], int(u[1:]) if u[1:].isdigit() else 0)

    order: Dict[str, Dict[str, List[str]]] = {bs: {s: [] for s in SLICE_LIST} for bs in BS_LIST}
    for nm, bs in mapping.items():
        cat = user_category(nm)
        order[bs][cat].append(nm)
    for bs in BS_LIST:
        for s in SLICE_LIST:
            order[bs][s].sort(key=sort_key)

    res = SimResult()
    had_m_users: set[str] = set()
    success_m_users: set[str] = set()

    t1 = min(t0 + WINDOW_MS, TOTAL_MS)
    for t in range(t0, t1):
        # === 到达 ===
        for nm in states.keys():
            arr_series = env.arrivals.get(nm, [])
            if t < len(arr_series):
                vol = arr_series[t]
                if vol > 0.0:
                    states[nm].queue.append(Chunk(arrival_ms=t, size_mbit=vol, remain_mbit=vol))
                    if states[nm].category == 'M':
                        had_m_users.add(nm)

        # === 填充并发槽位 ===
        for bs in BS_LIST:
            for s in SLICE_LIST:
                # 移除已空用户
                active[bs][s] = [u for u in active[bs][s] if states[u].has_backlog()]
                while len(active[bs][s]) < cap[bs][s]:
                    # 在 order 队列中找下一个有 backlog 的
                    for cand in order[bs][s]:
                        if cand in active[bs][s]:
                            continue
                        if states[cand].has_backlog():
                            active[bs][s].append(cand)
                            break
                    else:
                        break  # 没有可加入者

        # === 服务 ===
        for bs in BS_LIST:
            for s in SLICE_LIST:
                if len(active[bs][s]) == 0:
                    continue
                # 列出此 slice 当前活动的 BS（自身 + 其他）
                active_bs_same_slice = [b for b in BS_LIST if len(active[b][s]) > 0]
                for u in list(active[bs][s]):
                    rate_bps = user_rate_bps(env, bs, u, s, power_alloc, active_bs_same_slice, t)
                    served_mbit = (rate_bps * 0.001) / 1e6  # 1 ms
                    head = states[u].queue[0]
                    if head.start_ms is None:
                        head.start_ms = t
                    head.remain_mbit -= served_mbit
                    if head.remain_mbit <= 1e-12:
                        head.remain_mbit = 0.0
                        head.finish_ms = t + 1
                        states[u].queue.popleft()
                        # QoS 统计
                        L_ms = (head.finish_ms - head.arrival_ms)
                        if s == 'U':
                            res.sum_U += urllc_qos(L_ms)
                        elif s == 'E':
                            r_mbps = (head.size_mbit * 1000.0) / max(L_ms, 1e-12)
                            res.sum_E += embb_qos(L_ms, r_mbps)
                        else:
                            if L_ms <= SLA_L['M']:
                                success_m_users.add(u)
                        # 当前用户可能还有 backlog，新块下 ms 再排队

    # mMTC 评分
    if len(had_m_users) > 0:
        ratio = len(success_m_users) / len(had_m_users)
        for u in had_m_users:
            res.sum_M += mmtc_qos(ratio, u in success_m_users)

    res.obj = res.sum_U + res.sum_E + res.sum_M
    return states, res

# =================== 遗传算法实现 ===================
POP_SIZE = 40
MAX_GEN = 200
TOURN_K = 3
CROSS_RATE = 0.8
MUTATE_RATE = 0.3
ELITE_NUM = 5

BS_LIST = ["BS1", "BS2", "BS3"]
SLICE_LIST = ["U", "E", "M"]

# --- 编码长度 ---
#   seg1: user->BS  (0,1,2) 长度 = K
#   seg2: RB 比例编码 2 ints / BS  => 6 ints  (0-100)
#   seg3: power 9 floats

def random_individual(num_users: int):
    """Return tuple (user_bs_array, rb_ratio_array, power_array)"""
    user_bs = np.random.randint(0, 3, size=num_users, dtype=np.int8)
    rb_ratio = np.random.randint(0, 101, size=6, dtype=np.int16)
    power = np.random.uniform(18.0, 26.0, size=9).astype(np.float32)
    return [user_bs, rb_ratio, power]


def decode_rb(rb_ratio: np.ndarray) -> Dict[str, Dict[str, int]]:
    """根据 2*3 比例整数 (0-100) 生成 x_{n,s} (RB 数量)。"""
    out: Dict[str, Dict[str, int]] = {}
    idx = 0
    for bs in BS_LIST:
        r1 = int(rb_ratio[idx])     # for U
        r2 = int(rb_ratio[idx + 1]) # for E
        idx += 2
        r1 = max(0, min(100, r1))
        r2 = max(0, min(100 - r1, r2))
        r3 = 100 - r1 - r2
        prop = {"U": r1, "E": r2, "M": r3}
        # 先按比例取整到合法最小粒度
        x = {}
        remain = 50
        for s in SLICE_LIST:
            base = (prop[s] * 50) // 100
            # 向下取到合法粒度
            base = (base // RB_PER_SLICE[s]) * RB_PER_SLICE[s]
            x[s] = base
            remain -= base
        # 把余量 2 RB 一份地给边际收益最大切片(这里简单给 URLLC)…可改进
        s_order = ["U", "E", "M"]
        idx_s = 0
        while remain >= 2:
            s_cur = s_order[idx_s % 3]
            incr = RB_PER_SLICE[s_cur]
            if remain >= incr:
                x[s_cur] += incr
                remain -= incr
            idx_s += 1
        out[bs] = x
    return out


def decode_power(power_arr: np.ndarray) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    idx = 0
    for bs in BS_LIST:
        sub = {}
        for s in SLICE_LIST:
            p = float(power_arr[idx])
            p = max(POWER_MIN, min(POWER_MAX, p))
            sub[s] = p
            idx += 1
        out[bs] = sub
    return out


# ---- 修改 evaluate_individual ----

def evaluate_individual(indiv, env: Env, users: List[str], user_states: Dict[str, UserState], t0: int) -> float:
    user_bs, rb_ratio, power_arr = indiv
    rb_alloc = decode_rb(rb_ratio)
    power_alloc = decode_power(power_arr)
    # 构建映射
    mapping = {users[i]: BS_LIST[int(user_bs[i])] for i in range(len(users))}
    # 仿真
    _, res = simulate_window_multibs(env, user_states, mapping, rb_alloc, power_alloc, t0, copy_state=True)
    return res.obj


# --------------- GA 主循环 ---------------

def tournament_select(pop_scores: List[float]) -> int:
    cand = random.sample(range(len(pop_scores)), TOURN_K)
    cand.sort(key=lambda idx: pop_scores[idx], reverse=True)
    return cand[0]


def crossover(parent1, parent2):
    if random.random() > CROSS_RATE:
        return parent1, parent2
    # 对用户段做单点交叉，其余段算术平均
    u1, rb1, p1 = parent1
    u2, rb2, p2 = parent2
    num_users = len(u1)
    pt = random.randint(1, num_users - 1)
    child_u1 = np.concatenate([u1[:pt], u2[pt:]]).astype(np.int8)
    child_u2 = np.concatenate([u2[:pt], u1[pt:]]).astype(np.int8)
    child_rb1 = ((rb1.astype(np.float32) + rb2) / 2.0).astype(np.int16)
    child_rb2 = child_rb1.copy()
    child_p1 = (p1 + p2) / 2.0
    child_p2 = child_p1.copy()
    return [child_u1, child_rb1, child_p1], [child_u2, child_rb2, child_p2]


def mutate(indiv):
    user_bs, rb_ratio, power_arr = indiv
    num_users = len(user_bs)
    # 用户 BS 变异
    for i in range(num_users):
        if random.random() < MUTATE_RATE:
            user_bs[i] = np.random.randint(0, 3)
    # RB 配额变异
    for i in range(len(rb_ratio)):
        if random.random() < MUTATE_RATE:
            rb_ratio[i] = np.random.randint(0, 101)
    # 功率 Gaussian 噪声
    for i in range(len(power_arr)):
        if random.random() < MUTATE_RATE:
            power_arr[i] += np.random.normal(0.0, 1.0)
            power_arr[i] = max(POWER_MIN, min(POWER_MAX, power_arr[i]))


# =================== 主函数 ===================

class Tee:
    """Duplicate stdout to console and logfile"""
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data)
    def flush(self):
        for s in self.streams:
            s.flush()

def main():
    env, users = build_env()
    # --- duplicate stdout to log file ---
    log_path = os.path.join(SCRIPT_DIR, "q3_run_log.txt")
    log_file = open(log_path, "w", encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, log_file)
    print("=== Problem 3 GA-MPC Run Log ===", datetime.datetime.now())

    num_users = len(users)

    # 初始化用户永久编号映射 => 方便个体编码
    print(f"加载完成：用户数={num_users}, 每窗口 GA 个体={POP_SIZE}×{MAX_GEN} 评估 ≈ {POP_SIZE*MAX_GEN}")

    # 记录跨窗口总 QoS
    total_qos = 0.0
    summary_rows: List[List[str]] = []  # will accumulate per-window csv rows

    for t0 in range(0, TOTAL_MS, WINDOW_MS):
        print(f"\n== 窗口 {t0}~{t0+WINDOW_MS} ms ==")
        # 种群初始化
        population = [random_individual(num_users) for _ in range(POP_SIZE)]
        best_score = -1e30
        best_indiv = None

        # 初始化用户状态（空队列）
        user_states: Dict[str, UserState] = {nm: UserState(name=nm, category=user_category(nm), queue=deque()) for nm in users}

        for gen in range(MAX_GEN):
            scores = [evaluate_individual(ind, env, users, user_states, t0) for ind in population]
            # 更新最优
            gen_best_idx = int(np.argmax(scores))
            if scores[gen_best_idx] > best_score:
                best_score = scores[gen_best_idx]
                best_indiv = [arr.copy() for arr in population[gen_best_idx]]
            # 打印进度
            if (gen + 1) % 20 == 0 or gen == 0:
                print(f"  [Gen {gen+1:3d}] best={max(scores):.4f}, avg={np.mean(scores):.4f}")
            # -------------- 产生下一代 --------------
            new_pop = []
            # 精英保留
            elite_idx = list(np.argsort(scores))[::-1][:ELITE_NUM]
            for idx in elite_idx:
                # 深拷贝
                indiv_cp = [arr.copy() for arr in population[idx]]
                new_pop.append(indiv_cp)
            # 其余通过交叉+变异生成
            while len(new_pop) < POP_SIZE:
                p1 = population[tournament_select(scores)]
                p2 = population[tournament_select(scores)]
                c1, c2 = crossover([arr.copy() for arr in p1], [arr.copy() for arr in p2])
                mutate(c1)
                mutate(c2)
                new_pop.append(c1)
                if len(new_pop) < POP_SIZE:
                    new_pop.append(c2)
            population = new_pop

        # --- 将最优方案作用于真实队列，得到窗口末状态 ---
        best_rb = decode_rb(best_indiv[1])
        best_power = decode_power(best_indiv[2])
        best_mapping = {users[i]: BS_LIST[int(best_indiv[0][i])] for i in range(len(users))}
        user_states, final_res = simulate_window_multibs(env, user_states, best_mapping, best_rb, best_power, t0, copy_state=False)
        print(f"窗口 {t0} 选择方案目标={final_res.obj:.4f} (与 GA 估计 {best_score:.4f})")
        total_qos += final_res.obj

        # 详细打印最优方案
        print("  最优RB分配/功率：")
        for bs in BS_LIST:
            ru, re, rm = best_rb[bs]["U"], best_rb[bs]["E"], best_rb[bs]["M"]
            pu, pe, pm = best_power[bs]["U"], best_power[bs]["E"], best_power[bs]["M"]
            print(f"    {bs}: RB(U,E,M)=({ru},{re},{rm}), P(U,E,M)=({pu:.1f},{pe:.1f},{pm:.1f}) dBm")

        # 收集 csv 行
        row: List[str] = []
        win_idx = t0 // WINDOW_MS
        row.append(str(win_idx))
        for bs in BS_LIST:
            row.extend([str(best_rb[bs][s]) for s in SLICE_LIST])
            row.extend([f"{best_power[bs][s]:.2f}" for s in SLICE_LIST])
        row.extend([f"{final_res.sum_U:.6f}", f"{final_res.sum_E:.6f}", f"{final_res.sum_M:.6f}", f"{final_res.obj:.6f}"])
        summary_rows.append(row)

    # 写入 CSV 文件
    out_csv = os.path.join(SCRIPT_DIR, "q3_window_results.csv")
    header = ["window"]
    for bs in BS_LIST:
        header += [f"RB_{bs}_{s}" for s in SLICE_LIST]
        header += [f"P_{bs}_{s}" for s in SLICE_LIST]
    header += ["sum_U", "sum_E", "sum_M", "objective"]
    try:
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(summary_rows)
        print(f"已保存窗口结果 -> {out_csv}")
    except Exception as e:
        print(f"写CSV失败: {e}")

    print(f"\n== 10 窗口累计目标 ≈ {total_qos:.4f}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted by user.")
        sys.exit(0)
