#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
问题四（异构网络：MBS+SBS，多站切片 + 接入模式 + 功率控制）— 启发式求解器
=================================================================
策略：滚动 MPC（100 ms 窗口） + 混合编码遗传算法（GA）

相较问题三，本实现新增/变化点：
  - 基站集合包含 1 个 MBS（100 RB，功率 10–40 dBm）与 3 个 SBS（各 50 RB，功率 10–30 dBm）
  - MBS 与 SBS 频谱不重叠 ⇒ 跨层无干扰；SBS 之间同频复用 ⇒ 仅 SBS 之间互扰
  - 接入选择受限：每个用户在每个决策窗口仅可接入 {MBS, 最近的SBS}
  - 其余：切片（URLLC/eMBB/mMTC）RB 粒度、速率/时延 SLA 与 QoS 评估沿用前文

数据：兼容“附件4”下的 CSV：
  - taskflow_用户任务流.csv（逐毫秒到达量，Mbit）
  - taskflow_用户位置.csv（逐毫秒坐标，用于确定最近SBS）
  - MBS_1/SBS_1/2/3 的大规模衰减与小规模瑞利衰减 CSV

运行：
  $ python q4_ga.py

输出：
  - report/code/q4_run_log.txt 日志
  - report/code/q4_window_results.csv 每窗口的 RB 与功率方案及得分
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
from collections import deque
from copy import deepcopy


# =================== 路径配置 ===================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
ATTACH4_DIR = os.path.join(ROOT_DIR, "题目", "附件", "附件4")

CSV_TASKFLOW = os.path.join(ATTACH4_DIR, "taskflow_用户任务流.csv")
CSV_POS = os.path.join(ATTACH4_DIR, "taskflow_用户位置.csv")

CSV_PL: Dict[str, str] = {
    "MBS_1": os.path.join(ATTACH4_DIR, "MBS_1_大规模衰减.csv"),
    "SBS_1": os.path.join(ATTACH4_DIR, "SBS_1_大规模衰减.csv"),
    "SBS_2": os.path.join(ATTACH4_DIR, "SBS_2_大规模衰减.csv"),
    "SBS_3": os.path.join(ATTACH4_DIR, "SBS_3_大规模衰减.csv"),
}
CSV_RAY: Dict[str, str] = {
    "MBS_1": os.path.join(ATTACH4_DIR, "MBS_1_小规模瑞丽衰减.csv"),
    "SBS_1": os.path.join(ATTACH4_DIR, "SBS_1_小规模瑞丽衰减.csv"),
    "SBS_2": os.path.join(ATTACH4_DIR, "SBS_2_小规模瑞丽衰减.csv"),
    "SBS_3": os.path.join(ATTACH4_DIR, "SBS_3_小规模瑞丽衰减.csv"),
}


# =================== 系统常量 ===================
B_HZ = 360_000.0  # 单 RB 带宽 (Hz)
NF_DB = 7.0       # 噪声系数 (dB)

RB_PER_SLICE = {"U": 10, "E": 5, "M": 2}
SLICE_LIST = ["U", "E", "M"]

ALPHA = 0.95
M_U = 5.0
M_E = 3.0
M_M = 1.0

SLA_L_MS = {"U": 5.0, "E": 100.0, "M": 500.0}  # ms
SLA_R_E_MBPS = 50.0

WINDOW_MS = 100
TOTAL_MS = 1000

# BS 集合与 RB/功率边界
BS_LIST = ["MBS_1", "SBS_1", "SBS_2", "SBS_3"]
SBS_ONLY = [bs for bs in BS_LIST if bs.startswith("SBS_")]
RB_TOTAL: Dict[str, int] = {"MBS_1": 100, "SBS_1": 50, "SBS_2": 50, "SBS_3": 50}
P_MIN_DBM: Dict[str, float] = {"MBS_1": 10.0, "SBS_1": 10.0, "SBS_2": 10.0, "SBS_3": 10.0}
P_MAX_DBM: Dict[str, float] = {"MBS_1": 40.0, "SBS_1": 30.0, "SBS_2": 30.0, "SBS_3": 30.0}

# 基站几何位置（来自附件4 readme）
BS_COORD = {
    "MBS_1": (0.0, 0.0),
    "SBS_1": (0.0, 500.0),
    "SBS_2": (-433.0127, -250.0),
    "SBS_3": (433.0127, -250.0),
}


# =================== 工具函数 ===================


def dbm_to_mw(dbm: float) -> float:
    return 10 ** (dbm / 10.0)


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
    pos_x: Dict[str, List[float]]             # user -> x(t)
    pos_y: Dict[str, List[float]]             # user -> y(t)

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

    def user_xy(self, user: str, t: int) -> Tuple[float, float]:
        xs = self.pos_x.get(f"{user}_X") or self.pos_x.get(f"{user}X") or self.pos_x.get(user)
        ys = self.pos_y.get(f"{user}_Y") or self.pos_y.get(f"{user}Y") or self.pos_y.get(user)
        if not xs or not ys:
            return 0.0, 0.0
        idx = min(max(t, 0), min(len(xs), len(ys)) - 1)
        return float(xs[idx]), float(ys[idx])


def build_env() -> Tuple[Env, List[str]]:
    # 到达
    time_list, arrivals = load_time_series_csv(CSV_TASKFLOW)

    # 信道
    phi: Dict[str, Dict[str, List[float]]] = {}
    h_abs: Dict[str, Dict[str, List[float]]] = {}
    for bs in BS_LIST:
        _, phi_bs = load_time_series_csv(CSV_PL[bs])
        _, ray_bs = load_time_series_csv(CSV_RAY[bs])
        phi[bs] = phi_bs
        h_abs[bs] = ray_bs

    # 位置
    _, pos_series = load_time_series_csv(CSV_POS)
    pos_x: Dict[str, List[float]] = {}
    pos_y: Dict[str, List[float]] = {}
    for key, arr in pos_series.items():
        if key.endswith("_X"):
            pos_x[key] = arr
        elif key.endswith("_Y"):
            pos_y[key] = arr

    # 用户集合（按编号/前缀排序）
    all_users = sorted(arrivals.keys(), key=lambda x: (x[0], int(x[1:]) if x[1:].isdigit() else 0))

    return Env(time_list=time_list, arrivals=arrivals, phi=phi, h_abs=h_abs, pos_x=pos_x, pos_y=pos_y), all_users


# =================== QoS 计算与用户结构 ===================


def user_category(name: str) -> str:
    if name.startswith("U"):
        return "U"
    if name.startswith("e"):
        return "E"
    return "M"


def urllc_qos(L_ms: float) -> float:
    if L_ms <= SLA_L_MS["U"]:
        return ALPHA ** L_ms
    return -M_U


def embb_qos(L_ms: float, r_mbps: float) -> float:
    if L_ms <= SLA_L_MS["E"] and r_mbps >= SLA_R_E_MBPS:
        return 1.0
    if L_ms <= SLA_L_MS["E"] and r_mbps < SLA_R_E_MBPS:
        return max(0.0, r_mbps / SLA_R_E_MBPS)
    return -M_E


def mmtc_qos(ratio: float, success: bool) -> float:
    return ratio if success else -M_M


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


@dataclass
class SimResult:
    sum_U: float = 0.0
    sum_E: float = 0.0
    sum_M: float = 0.0
    obj: float = 0.0


# =================== 速率与窗口仿真 ===================


def user_rate_bps(env: Env, bs: str, name: str, cat: str,
                  power_alloc: Dict[str, Dict[str, float]],
                  active_sbs_same_slice: List[str], t_ms: int) -> float:
    """计算下行速率 (bps)。MBS 无干扰；SBS 受其他 SBS 干扰。"""
    p_tx_mw = dbm_to_mw(power_alloc[bs][cat])
    phi_db = env.phi_db(bs, name, t_ms)
    h_pow = env.h_pow(bs, name, t_ms)
    recv_mw = p_tx_mw * 10 ** (-phi_db / 10.0) * h_pow

    interf_mw = 0.0
    if bs in SBS_ONLY:
        for b2 in active_sbs_same_slice:
            if b2 == bs:
                continue
            p_int_mw = dbm_to_mw(power_alloc[b2][cat])
            phi_db_i = env.phi_db(b2, name, t_ms)
            h_pow_i = env.h_pow(b2, name, t_ms)
            interf_mw += p_int_mw * 10 ** (-phi_db_i / 10.0) * h_pow_i

    n0_mw = noise_power_mw(RB_PER_SLICE[cat])
    sinr = recv_mw / (interf_mw + n0_mw + 1e-30)
    return RB_PER_SLICE[cat] * B_HZ * math.log2(1.0 + sinr)


def simulate_window(env: Env,
                    init_states: Dict[str, UserState],
                    mapping: Dict[str, str],
                    rb_alloc: Dict[str, Dict[str, int]],
                    power_alloc: Dict[str, Dict[str, float]],
                    t0: int,
                    copy_state: bool = True) -> Tuple[Dict[str, UserState], SimResult]:
    """100-ms 精细仿真。若 copy_state=True 则不影响原状态。"""
    states: Dict[str, UserState] = {}
    if copy_state:
        for name, st in init_states.items():
            states[name] = UserState(name=st.name, category=st.category, queue=deque(deepcopy(list(st.queue))))
    else:
        states = init_states

    # 并发容量
    cap = {bs: {s: rb_alloc[bs][s] // RB_PER_SLICE[s] for s in SLICE_LIST} for bs in BS_LIST}
    active: Dict[str, Dict[str, List[str]]] = {bs: {s: [] for s in SLICE_LIST} for bs in BS_LIST}

    # 用户顺序（编号优先）
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
                # 移除已空用户
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
                # 本 slice 当前活动的 SBS 列表（用于干扰）
                active_sbs_same_slice = [b for b in SBS_ONLY if len(active[b][s]) > 0]
                for u in list(active[bs][s]):
                    rate_bps = user_rate_bps(env, bs, u, s, power_alloc, active_sbs_same_slice, t)
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
                            if L_ms <= SLA_L_MS['M']:
                                success_m_users.add(u)

    # mMTC 评分
    if len(had_m_users) > 0:
        ratio = len(success_m_users) / len(had_m_users)
        for u in had_m_users:
            res.sum_M += mmtc_qos(ratio, u in success_m_users)

    res.obj = res.sum_U + res.sum_E + res.sum_M
    return states, res


# =================== 最近 SBS 计算 ===================


def nearest_sbs_per_user(env: Env, users: List[str], t_ms: int) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for u in users:
        x, y = env.user_xy(u, t_ms)
        # 在三个 SBS 中选最近
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


# =================== 遗传算法 ===================

POP_SIZE = 50
MAX_GEN = 500
TOURN_K = 3
CROSS_RATE = 0.8
MUTATE_RATE = 0.3
ELITE_NUM = 5


def random_individual(num_users: int):
    """编码：(user_choice, rb_counts, power)
    - user_choice: 长度 K，0=接 MBS_1，1=接最近 SBS
    - rb_counts: 每个基站 2 整数（U_RB,E_RB），M_RB=总RB-(U+E)
      对 MBS_1 总RB=100；对每个 SBS 总RB=50
    - power: 每个基站每切片 1 浮点，共 3*|BS|
    """
    user_choice = np.random.randint(0, 2, size=num_users, dtype=np.int8)
    rb_counts = np.zeros(2 * len(BS_LIST), dtype=np.int16)
    idx = 0
    for bs in BS_LIST:
        total = RB_TOTAL[bs]
        # U: 粒度10
        u_units_max = total // RB_PER_SLICE['U']
        u_units = np.random.randint(0, u_units_max + 1)
        u_rb = int(u_units * RB_PER_SLICE['U'])
        # E: 粒度5，受剩余约束
        e_units_max = (total - u_rb) // RB_PER_SLICE['E']
        e_units = np.random.randint(0, e_units_max + 1)
        e_rb = int(e_units * RB_PER_SLICE['E'])
        rb_counts[idx] = u_rb
        rb_counts[idx + 1] = e_rb
        idx += 2
    power = np.random.uniform(20.0, 30.0, size=3 * len(BS_LIST)).astype(np.float32)
    return [user_choice, rb_counts, power]


def decode_rb(rb_counts: np.ndarray) -> Dict[str, Dict[str, int]]:
    """根据每个 BS 的 (U_RB,E_RB) 计算 x_{n,s}，M_RB=总RB-(U+E)，并对齐粒度。
    """
    out: Dict[str, Dict[str, int]] = {}
    idx = 0
    for bs in BS_LIST:
        total = RB_TOTAL[bs]
        u_req = int(rb_counts[idx])
        e_req = int(rb_counts[idx + 1])
        idx += 2
        # 对齐粒度并裁剪
        u_rb = max(0, min(total, (u_req // RB_PER_SLICE['U']) * RB_PER_SLICE['U']))
        e_rb = max(0, min(total - u_rb, (e_req // RB_PER_SLICE['E']) * RB_PER_SLICE['E']))
        if u_rb + e_rb > total:
            e_rb = max(0, (total - u_rb) // RB_PER_SLICE['E'] * RB_PER_SLICE['E'])
        m_rb = total - (u_rb + e_rb)
        # 保障 m 粒度=2
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
    out: Dict[str, Dict[str, float]] = {}
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


def evaluate_individual(indiv, env: Env, users: List[str], user_states: Dict[str, UserState],
                        t0: int, nearest_sbs_map: Dict[str, str]) -> float:
    user_choice, rb_counts, power_arr = indiv
    rb_alloc = decode_rb(rb_counts)
    power_alloc = decode_power(power_arr)
    # 构建映射：0 -> MBS_1，1 -> 最近SBS（基于窗口起点）
    mapping = {users[i]: ("MBS_1" if int(user_choice[i]) == 0 else nearest_sbs_map[users[i]]) for i in range(len(users))}
    # 仿真
    _, res = simulate_window(env, user_states, mapping, rb_alloc, power_alloc, t0, copy_state=True)
    return res.obj


def tournament_select(pop_scores: List[float]) -> int:
    cand = random.sample(range(len(pop_scores)), TOURN_K)
    cand.sort(key=lambda idx: pop_scores[idx], reverse=True)
    return cand[0]


def crossover(parent1, parent2):
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
    # RB 段取算术平均后四舍五入到最近合法粒度（但最终 decode 会再校正）
    child_rb1 = ((rb1.astype(np.float32) + rb2) / 2.0).astype(np.int16)
    child_rb2 = child_rb1.copy()
    child_p1 = (p1 + p2) / 2.0
    child_p2 = child_p1.copy()
    return [child_u1, child_rb1, child_p1], [child_u2, child_rb2, child_p2]


def mutate(indiv):
    user_choice, rb_counts, power_arr = indiv
    num_users = len(user_choice)
    # 接入变异
    for i in range(num_users):
        if random.random() < MUTATE_RATE:
            user_choice[i] = 1 - int(user_choice[i])
    # RB 变异：按 BS 调整 U/E 绝对 RB
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
            # 解码时会截断至合法范围


# =================== 主程序 ===================


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
    log_path = os.path.join(SCRIPT_DIR, "q4_run_log.txt")
    log_file = open(log_path, "w", encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, log_file)
    print("=== Problem 4 GA-MPC Run Log ===", datetime.datetime.now())

    num_users = len(users)
    print(f"加载完成：用户数={num_users}, BS={len(BS_LIST)} (MBS+SBS)，每窗口评估 ≈ {POP_SIZE*MAX_GEN}")

    total_qos = 0.0
    summary_rows: List[List[str]] = []
    mapping_rows: List[List[str]] = []  # 将记录每个窗口的 user->BS 分配

    # 跨窗口的真实队列状态（保留）
    user_states: Dict[str, UserState] = {nm: UserState(name=nm, category=user_category(nm), queue=deque()) for nm in users}

    for t0 in range(0, TOTAL_MS, WINDOW_MS):
        print(f"\n== 窗口 {t0}~{t0+WINDOW_MS} ms ==")

        # 计算窗口起点的最近SBS
        nearest_map = nearest_sbs_per_user(env, users, t0)

        # 初始化种群
        population = [random_individual(num_users) for _ in range(POP_SIZE)]
        best_score = -1e30
        best_indiv = None

        for gen in range(MAX_GEN):
            scores = [evaluate_individual(ind, env, users, user_states, t0, nearest_map) for ind in population]
            # 更新最优
            gen_best_idx = int(np.argmax(scores))
            if scores[gen_best_idx] > best_score:
                best_score = float(scores[gen_best_idx])
                best_indiv = [arr.copy() for arr in population[gen_best_idx]]
            # 打印进度
            if (gen + 1) % 20 == 0 or gen == 0:
                print(f"  [Gen {gen+1:3d}] best={max(scores):.4f}, avg={np.mean(scores):.4f}")
            # 产生下一代
            new_pop = []
            elite_idx = list(np.argsort(scores))[::-1][:ELITE_NUM]
            for idx in elite_idx:
                new_pop.append([arr.copy() for arr in population[idx]])
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

        # --- 应用最优方案到真实队列 ---
        best_rb = decode_rb(best_indiv[1])
        best_power = decode_power(best_indiv[2])
        best_mapping = {users[i]: ("MBS_1" if int(best_indiv[0][i]) == 0 else nearest_map[users[i]]) for i in range(len(users))}
        user_states, final_res = simulate_window(env, user_states, best_mapping, best_rb, best_power, t0, copy_state=False)
        print(f"窗口 {t0} 选择方案目标={final_res.obj:.4f} (GA 估计 {best_score:.4f})")
        total_qos += final_res.obj

        # 打印方案
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

        # 追加用户->基站 映射行
        for u in users:
            mapping_rows.append([str(win_idx), u, best_mapping[u]])

    # 写入 CSV 文件
    out_csv = os.path.join(SCRIPT_DIR, "q4_window_results.csv")
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

    # 写入每用户-基站映射 CSV 文件
    out_map_csv = os.path.join(SCRIPT_DIR, "q4_user_bs_mapping.csv")
    try:
        with open(out_map_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["window", "user", "bs"])
            writer.writerows(mapping_rows)
        print(f"已保存用户-基站映射 -> {out_map_csv}")
    except Exception as e:
        print(f"写用户映射CSV失败: {e}")

    print(f"\n== 10 窗口累计目标 ≈ {total_qos:.4f}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted by user.")
        sys.exit(0)


