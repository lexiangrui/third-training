#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
问题五（在第四问基础上加入能耗优化）— GA-MPC 求解器
=====================================================
目标：在第四问同样的异构网络与 QoS 评估框架下，综合能耗模型
  P = P_static + δ·N_active + (1/η)·P_transmit
进行优化，使“能耗最低的同时能够达到最大的用户服务质量”。

实现：
  - 仍使用滚动 MPC（100 ms 窗口）+ 混合编码 GA（接入 + 切片RB + 功率）
  - 窗口内 1 ms 精细仿真，累积 QoS 与能耗（焦耳）
  - 适应度使用大权重组合：fitness = QoS*Wq - Energy(J)
    通过较大 Wq 逼近“QoS 优先，其次能耗”的字典序目标

数据与路径：复用“附件4”的 CSV（任务到达、位置、MBS/SBS 衰减）
输出：
  - report/code/q5_run_log.txt
  - report/code/q5_window_results.csv（含每窗口 QoS/能耗与方案）
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


# =================== 路径/常量 ===================
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

B_HZ = 360_000.0
NF_DB = 7.0

RB_PER_SLICE = {"U": 10, "E": 5, "M": 2}
SLICE_LIST = ["U", "E", "M"]
BS_LIST = ["MBS_1", "SBS_1", "SBS_2", "SBS_3"]
SBS_ONLY = [bs for bs in BS_LIST if bs.startswith("SBS_")]
RB_TOTAL: Dict[str, int] = {"MBS_1": 100, "SBS_1": 50, "SBS_2": 50, "SBS_3": 50}
P_MIN_DBM: Dict[str, float] = {"MBS_1": 10.0, "SBS_1": 10.0, "SBS_2": 10.0, "SBS_3": 10.0}
P_MAX_DBM: Dict[str, float] = {"MBS_1": 40.0, "SBS_1": 30.0, "SBS_2": 30.0, "SBS_3": 30.0}

ALPHA = 0.95
M_U = 5.0
M_E = 3.0
M_M = 1.0

SLA_L_MS = {"U": 5.0, "E": 100.0, "M": 500.0}
SLA_R_E_MBPS = 50.0

WINDOW_MS = 100
TOTAL_MS = 1000

# 能耗模型参数（附录 第6节）
P_STATIC_W = 28.0
DELTA_W_PER_RB = 0.75
ETA = 0.35

# 二阶段：阶段一仅能耗最小（固定近邻接入与均衡RB评估）；阶段二在固定功率下最大化 QoS


# =================== 工具函数 ===================
def dbm_to_mw(dbm: float) -> float:
    return 10 ** (dbm / 10.0)


def dbm_to_watt(dbm: float) -> float:
    return dbm_to_mw(dbm) / 1000.0


def noise_power_mw(num_rbs: int) -> float:
    if num_rbs <= 0:
        return 0.0
    n0_dbm = -174.0 + 10.0 * math.log10(num_rbs * B_HZ) + NF_DB
    return dbm_to_mw(n0_dbm)


def load_time_series_csv(path: str) -> Tuple[List[float], Dict[str, List[float]]]:
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
    # 填齐
    max_len = len(time_list)
    for arr in series.values():
        if len(arr) < max_len:
            arr.extend([0.0] * (max_len - len(arr)))
    return time_list, series


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


def build_env() -> Tuple[Env, List[str]]:
    time_list, arrivals = load_time_series_csv(CSV_TASKFLOW)
    phi: Dict[str, Dict[str, List[float]]] = {}
    h_abs: Dict[str, Dict[str, List[float]]] = {}
    for bs in BS_LIST:
        _, phi_bs = load_time_series_csv(CSV_PL[bs])
        _, ray_bs = load_time_series_csv(CSV_RAY[bs])
        phi[bs] = phi_bs
        h_abs[bs] = ray_bs
    _, pos_series = load_time_series_csv(CSV_POS)
    pos_x: Dict[str, List[float]] = {}
    pos_y: Dict[str, List[float]] = {}
    for key, arr in pos_series.items():
        if key.endswith("_X"):
            pos_x[key] = arr
        elif key.endswith("_Y"):
            pos_y[key] = arr
    users = sorted(arrivals.keys(), key=lambda x: (x[0], int(x[1:]) if x[1:].isdigit() else 0))
    return Env(time_list=time_list, arrivals=arrivals, phi=phi, h_abs=h_abs, pos_x=pos_x, pos_y=pos_y), users


# =================== QoS 与状态结构 ===================
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
    category: str
    queue: deque[Chunk]

    def has_backlog(self) -> bool:
        return len(self.queue) > 0 and self.queue[0].remain_mbit > 1e-12


@dataclass
class SimResult:
    sum_U: float = 0.0
    sum_E: float = 0.0
    sum_M: float = 0.0
    obj: float = 0.0
    energy_J: float = 0.0


# =================== 物理层与仿真 ===================
def user_rate_bps(env: Env, bs: str, name: str, cat: str,
                  power_alloc: Dict[str, Dict[str, float]],
                  active_sbs_same_slice: List[str], t_ms: int) -> float:
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
    states: Dict[str, UserState] = {}
    if copy_state:
        for name, st in init_states.items():
            states[name] = UserState(name=st.name, category=st.category, queue=deque(deepcopy(list(st.queue))))
    else:
        states = init_states

    cap = {bs: {s: rb_alloc[bs][s] // RB_PER_SLICE[s] for s in SLICE_LIST} for bs in BS_LIST}
    active: Dict[str, Dict[str, List[str]]] = {bs: {s: [] for s in SLICE_LIST} for bs in BS_LIST}

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
                            res.sum_U += urllc_qos(L_ms)
                        elif s == 'E':
                            r_mbps = (head.size_mbit * 1000.0) / max(L_ms, 1e-12)
                            res.sum_E += embb_qos(L_ms, r_mbps)
                        else:
                            if L_ms <= SLA_L_MS['M']:
                                success_m_users.add(u)

        # 能耗累计（每 ms 计算一次）
        p_total_w_all_bs = 0.0
        n_active_rb_all_bs = 0
        for bs in BS_LIST:
            n_active_rb = 0
            p_tx_w_bs = 0.0
            for s in SLICE_LIST:
                n_users_act = len(active[bs][s])
                n_active_rb += n_users_act * RB_PER_SLICE[s]
                p_tx_w_bs += n_users_act * dbm_to_watt(power_alloc[bs][s])
            p_bs = P_STATIC_W + DELTA_W_PER_RB * n_active_rb + (1.0 / ETA) * p_tx_w_bs
            p_total_w_all_bs += p_bs
            n_active_rb_all_bs += n_active_rb
        res.energy_J += p_total_w_all_bs * 0.001  # 1 ms

    if len(had_m_users) > 0:
        ratio = len(success_m_users) / len(had_m_users)
        for u in had_m_users:
            res.sum_M += mmtc_qos(ratio, u in success_m_users)

    res.obj = res.sum_U + res.sum_E + res.sum_M
    return states, res


# =================== “最近SBS”与编码 ===================
def nearest_sbs_per_user(env: Env, users: List[str], t_ms: int) -> Dict[str, str]:
    # SBS 坐标来源见 q4
    BS_COORD = {
        "SBS_1": (0.0, 500.0),
        "SBS_2": (-433.0127, -250.0),
        "SBS_3": (433.0127, -250.0),
    }
    out: Dict[str, str] = {}
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


# GA 参数（阶段一：功率最小化）
POP_SIZE = 24
MAX_GEN = 80
TOURN_K = 3
CROSS_RATE = 0.8
MUTATE_RATE = 0.3
ELITE_NUM = 5


def random_individual(num_users: int):
    """阶段一个体：仅编码功率数组 power_arr（长度=3*|BS|）。"""
    power = np.random.uniform(20.0, 30.0, size=3 * len(BS_LIST)).astype(np.float32)
    return [power]


def make_equal_rb_alloc() -> Dict[str, Dict[str, int]]:
    """生成均衡 RB 分配（轮转填充，保证全部 RB 用完且满足粒度）。"""
    out: Dict[str, Dict[str, int]] = {}
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


def evaluate_power_individual(indiv, env: Env, users: List[str], user_states: Dict[str, UserState],
                              t0: int, nearest_map: Dict[str, str]) -> Tuple[float, float, float]:
    """阶段一评价：固定近邻接入 + 均衡RB；以能耗最小为目标（适应度= -Energy）。"""
    power_arr = indiv[0]
    power_alloc = decode_power(power_arr)
    rb_alloc = make_equal_rb_alloc()
    mapping = {u: ("MBS_1" if nearest_map.get(u) is None else nearest_map[u]) for u in users}
    _, res = simulate_window(env, user_states, mapping, rb_alloc, power_alloc, t0, copy_state=True)
    fitness = -res.energy_J
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


# =================== 阶段二：固定功率下的 RB 枚举（含 MBS） ===================
def gen_splits_for_total(total: int) -> List[Tuple[int, int, int]]:
    splits: List[Tuple[int, int, int]] = []
    for nU in range(0, total + 1, RB_PER_SLICE['U']):
        for nE in range(0, total - nU + 1, RB_PER_SLICE['E']):
            nM = total - nU - nE
            if nM < 0:
                continue
            if nM % RB_PER_SLICE['M'] != 0:
                continue
            splits.append((nU, nE, nM))
    return splits


def coordinate_enumeration_rb(env: Env,
                              users: List[str],
                              user_states: Dict[str, UserState],
                              mapping: Dict[str, str],
                              power_alloc: Dict[str, Dict[str, float]],
                              t0: int,
                              rounds: int = 2) -> Dict[str, Dict[str, int]]:
    rb_alloc = make_equal_rb_alloc()
    # 预生成每个 BS 的候选 split 列表
    bs_splits: Dict[str, List[Tuple[int, int, int]]] = {bs: gen_splits_for_total(RB_TOTAL[bs]) for bs in BS_LIST}
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


# =================== 主程序 ===================
class Tee:
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
    log_path = os.path.join(SCRIPT_DIR, "q5_run_log.txt")
    log_file = open(log_path, "w", encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, log_file)
    print("=== Problem 5 GA-MPC (QoS then Energy) Run Log ===", datetime.datetime.now())

    num_users = len(users)
    print(f"加载完成：用户数={num_users}, BS={len(BS_LIST)}，每窗口评估≈{POP_SIZE*MAX_GEN}")

    total_qos = 0.0
    total_energy_J = 0.0
    summary_rows: List[List[str]] = []

    user_states: Dict[str, UserState] = {nm: UserState(name=nm, category=user_category(nm), queue=deque()) for nm in users}

    for t0 in range(0, TOTAL_MS, WINDOW_MS):
        print(f"\n== 窗口 {t0}~{t0+WINDOW_MS} ms ==")
        nearest_map = nearest_sbs_per_user(env, users, t0)

        # ------- 阶段一：GA 仅优化功率以最小化能耗 -------
        population = [random_individual(num_users) for _ in range(POP_SIZE)]
        best_fit = -1e99
        best_energy = 1e99
        best_power_indiv = None

        for gen in range(MAX_GEN):
            evals = [evaluate_power_individual(ind, env, users, user_states, t0, nearest_map) for ind in population]
            scores = [e[0] for e in evals]
            gen_best_idx = int(np.argmax(scores))
            fit, qos_est, eng = evals[gen_best_idx]
            if fit > best_fit:
                best_fit = float(fit)
                best_energy = float(eng)
                best_power_indiv = [arr.copy() for arr in population[gen_best_idx]]
            if (gen + 1) % 20 == 0 or gen == 0:
                print(f"  [P-Gen {gen+1:3d}] estE={eng:.3f}J, bestE={best_energy:.3f}J, fit={fit:.3e}")
            # 下一代
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

        assert best_power_indiv is not None
        best_power = decode_power(best_power_indiv[0])

        # ------- 阶段二：固定功率，坐标枚举每个基站的切片RB，最大化 QoS -------
        mapping = {u: ("MBS_1" if nearest_map.get(u) is None else nearest_map[u]) for u in users}
        best_rb = coordinate_enumeration_rb(env, users, user_states, mapping, best_power, t0, rounds=2)

        # 应用到真实队列
        user_states, final_res = simulate_window(env, user_states, mapping, best_rb, best_power, t0, copy_state=False)
        print(f"窗口 {t0}: QoS={final_res.obj:.4f}, Energy={final_res.energy_J:.3f} J (after enumeration)")
        total_qos += final_res.obj
        total_energy_J += final_res.energy_J

        # 打印方案
        print("  方案（功率先行、枚举RB）：")
        for bs in BS_LIST:
            ru, re, rm = best_rb[bs]['U'], best_rb[bs]['E'], best_rb[bs]['M']
            pu, pe, pm = best_power[bs]['U'], best_power[bs]['E'], best_power[bs]['M']
            print(f"    {bs}: RB(U,E,M)=({ru},{re},{rm}), P(U,E,M)=({pu:.1f},{pe:.1f},{pm:.1f}) dBm")

        # 汇总 CSV 行
        row: List[str] = []
        win_idx = t0 // WINDOW_MS
        row.append(str(win_idx))
        for bs in BS_LIST:
            row.extend([str(best_rb[bs][s]) for s in SLICE_LIST])
            row.extend([f"{best_power[bs][s]:.2f}" for s in SLICE_LIST])
        row.extend([
            f"{final_res.sum_U:.6f}", f"{final_res.sum_E:.6f}", f"{final_res.sum_M:.6f}", f"{final_res.obj:.6f}",
            f"{final_res.energy_J:.6f}", f"{total_energy_J:.6f}",
        ])
        summary_rows.append(row)

    # 输出 CSV
    out_csv = os.path.join(SCRIPT_DIR, "q5_window_results.csv")
    header = ["window"]
    for bs in BS_LIST:
        header += [f"RB_{bs}_{s}" for s in SLICE_LIST]
        header += [f"P_{bs}_{s}" for s in SLICE_LIST]
    header += ["sum_U", "sum_E", "sum_M", "objective", "energy_J", "cum_energy_J"]
    try:
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(summary_rows)
        print(f"已保存窗口结果 -> {out_csv}")
    except Exception as e:
        print(f"写CSV失败: {e}")

    print(f"\n== 总QoS ≈ {total_qos:.4f}, 总能耗 ≈ {total_energy_J:.3f} J ==")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted by user.")
        sys.exit(0)


