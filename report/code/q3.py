#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
问题三：三个微基站、同频复用引起小区间干扰，且允许每站对三类切片分别做功率控制 p_{n,s}∈[10,30] dBm。
系统在 1000 ms 内每 100 ms 决策一次：
- 每个基站的三切片 RB 划分 (x_{n,U}, x_{n,e}, x_{n,m})，满足 50、粒度倍数；
- 每个基站的三切片功率 (p_{n,U}, p_{n,e}, p_{n,m})。

采用滚动窗口（MPC-1步）+ 坐标下降（per-BS）启发式：
- 先以“就近基站（最小路径损耗）”确定窗口内关联；
- 初始功率设定（URLLC 高、eMBB 中、mMTC 低）；
- 初始每站 RB 划分用简单权重分配得到；
- 然后对每个基站枚举其 66 种合法 RB 划分（其余两站固定），用1ms仿真评估，选择提升最大者；
- 选做：对每个基站在小网格上枚举功率（4档）做一次“最佳响应”；
- 得到该窗口的 (x,p) 决策，推进到下窗口。

输出：逐窗口打印三站的 (R_U,R_e,R_m) 与 (p_U,p_e,p_m) 以及窗口 QoS；
最终汇总总 QoS，并导出 q3_window_results.csv。
"""

from __future__ import annotations

import csv
import math
import os
import copy
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Deque
from collections import deque, defaultdict


# 路径配置
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
ATTACH_DIR = os.path.join(ROOT_DIR, "题目", "附件", "附件3")

CSV_TASK = os.path.join(ATTACH_DIR, "taskflow_用户任务流.csv")
CSV_PL = {
    1: os.path.join(ATTACH_DIR, "BS1_大规模衰减.csv"),
    2: os.path.join(ATTACH_DIR, "BS2_大规模衰减.csv"),
    3: os.path.join(ATTACH_DIR, "BS3_大规模衰减.csv"),
}
CSV_RAY = {
    1: os.path.join(ATTACH_DIR, "BS1_小规模瑞丽衰减.csv"),
    2: os.path.join(ATTACH_DIR, "BS2_小规模瑞丽衰减.csv"),
    3: os.path.join(ATTACH_DIR, "BS3_小规模瑞丽衰减.csv"),
}


# 物理与系统常量
B_HZ: float = 360_000.0  # 360 kHz
NF_DB: float = 7.0

V_U: int = 10
V_E: int = 5
V_M: int = 2
RB_TOTAL: int = 50

ALPHA: float = 0.95
M_U: float = 5.0
M_E: float = 3.0
M_M: float = 1.0
SLA_L_U_MS: float = 5.0
SLA_L_E_MS: float = 100.0
SLA_L_M_MS: float = 500.0
SLA_R_E_MBPS: float = 50.0

WINDOW_MS: int = 100
TOTAL_MS: int = 1000


def dbm_to_mw(dbm: float) -> float:
    return 10 ** (dbm / 10.0)


def noise_power_mw(num_rbs: int, b_hz: float = B_HZ, nf_db: float = NF_DB) -> float:
    """N0(dBm) = -174 + 10log10(i*b) + NF → mW。"""
    if num_rbs <= 0:
        return 1e-30
    n0_dbm = -174.0 + 10.0 * math.log10(max(1.0, num_rbs) * b_hz) + nf_db
    return dbm_to_mw(n0_dbm)


def log2_safe(x: float) -> float:
    if x <= 0:
        return 0.0
    return math.log(x, 2)


# ===================== 数据加载 =====================

def load_time_series_csv(path: str) -> Tuple[List[float], Dict[str, List[float]]]:
    """
    读取逐时刻 CSV：返回 (time_list, name->series)。若列含非数值，忽略。
    """
    time_list: List[float] = []
    series: Dict[str, List[float]] = {}
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
    max_len = len(time_list)
    for k, arr in series.items():
        if len(arr) < max_len:
            arr.extend([0.0] * (max_len - len(arr)))
    return time_list, series


def detect_small_scale_is_db(ray_series: Dict[str, List[float]]) -> bool:
    # 任意列出现负值，则视为 dB（功率）
    for arr in ray_series.values():
        for v in arr[:200]:
            if v < 0:
                return True
    return False


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
    queue: Deque[Chunk] = field(default_factory=deque)

    def total_backlog_mbit(self) -> float:
        return sum(ch.remain_mbit for ch in self.queue)

    def has_backlog(self) -> bool:
        return len(self.queue) > 0 and self.queue[0].remain_mbit > 1e-15


@dataclass
class Env3:
    time_list: List[float]
    arrivals_mbit: Dict[str, List[float]]
    pl_db: Dict[int, Dict[str, List[float]]]
    ray_raw: Dict[int, Dict[str, List[float]]]
    ray_is_db: Dict[int, bool]

    def get_phi_db(self, n: int, name: str, t_ms: int) -> float:
        arr = self.pl_db.get(n, {}).get(name)
        if arr is None:
            return 100.0
        if t_ms < 0:
            t_ms = 0
        if t_ms >= len(arr):
            t_ms = len(arr) - 1
        return float(arr[t_ms])

    def get_h_pow(self, n: int, name: str, t_ms: int) -> float:
        arr = self.ray_raw.get(n, {}).get(name)
        if arr is None or len(arr) == 0:
            return 1.0
        if t_ms < 0:
            t_ms = 0
        if t_ms >= len(arr):
            t_ms = len(arr) - 1
        val = float(arr[t_ms])
        if self.ray_is_db.get(n, False):
            # 视为功率(dB)
            return 10 ** (val / 10.0)
        # 视为幅度 |h|
        if val < 0:
            return 1e-6
        return val * val


def build_env3() -> Tuple[Env3, List[str], List[str], List[str]]:
    time_list, arrivals = load_time_series_csv(CSV_TASK)

    pl_db: Dict[int, Dict[str, List[float]]] = {}
    ray_raw: Dict[int, Dict[str, List[float]]] = {}
    ray_is_db: Dict[int, bool] = {}

    for n in [1, 2, 3]:
        _, pln = load_time_series_csv(CSV_PL[n])
        _, rayn = load_time_series_csv(CSV_RAY[n])
        pl_db[n] = pln
        ray_raw[n] = rayn
        ray_is_db[n] = detect_small_scale_is_db(rayn)

    # 分类并确定编号顺序
    def sort_key(name: str) -> Tuple[int, int]:
        prefix_rank = 0
        if name.startswith("U"):
            prefix_rank = 0
        elif name.startswith("e"):
            prefix_rank = 1
        elif name.startswith("m"):
            prefix_rank = 2
        num = 0
        for i in range(len(name)):
            if name[i].isdigit():
                num = int(name[i:])
                break
        return (prefix_rank, num)

    all_names = [k for k in arrivals.keys()]
    U_names = sorted([n for n in all_names if n.startswith("U")], key=sort_key)
    E_names = sorted([n for n in all_names if n.startswith("e")], key=sort_key)
    M_names = sorted([n for n in all_names if n.startswith("m")], key=sort_key)

    env = Env3(
        time_list=time_list,
        arrivals_mbit=arrivals,
        pl_db=pl_db,
        ray_raw=ray_raw,
        ray_is_db=ray_is_db,
    )
    return env, U_names, E_names, M_names


# ===================== QoS 计算 =====================

def urllc_qos_from_L(L_ms: float) -> float:
    if L_ms <= SLA_L_U_MS:
        return ALPHA ** L_ms
    return -M_U


def embb_qos_from_chunk(size_mbit: float, L_ms: float) -> float:
    if L_ms <= 0:
        return 0.0
    r_mbps = (size_mbit * 1000.0) / L_ms
    if L_ms <= SLA_L_E_MS and r_mbps >= SLA_R_E_MBPS:
        return 1.0
    if L_ms <= SLA_L_E_MS and r_mbps < SLA_R_E_MBPS:
        return max(0.0, r_mbps / SLA_R_E_MBPS)
    return -M_E


def mmtc_success_from_L(L_ms: float) -> bool:
    return L_ms <= SLA_L_M_MS


# ===================== 关联、RB切片、功率工具 =====================

def associate_by_min_pathloss(env: Env3, t0: int, names: List[str]) -> Dict[str, int]:
    assoc: Dict[str, int] = {}
    for nm in names:
        best_n = 1
        best_phi = float("inf")
        for n in [1, 2, 3]:
            phi = env.get_phi_db(n, nm, t0)
            if phi < best_phi:
                best_phi = phi
                best_n = n
        assoc[nm] = best_n
    return assoc


def enumerate_splits_single_bs() -> List[Tuple[int, int, int]]:
    splits: List[Tuple[int, int, int]] = []
    for nU in range(0, RB_TOTAL + 1, V_U):
        for nE in range(0, RB_TOTAL - nU + 1, V_E):
            nM = RB_TOTAL - nU - nE
            if nM < 0:
                continue
            if nM % V_M != 0:
                continue
            splits.append((nU, nE, nM))
    return splits


def initial_power_grid() -> List[float]:
    # 小网格功率候选（dBm）
    return [14.0, 20.0, 24.0, 28.0]


def initial_power_profile() -> Dict[str, float]:
    # 初值：URLLC 高、eMBB 中、mMTC 低
    return {"U": 28.0, "e": 22.0, "m": 14.0}


def greedy_initial_split(demand_weights: Dict[str, float]) -> Tuple[int, int, int]:
    """根据需求粗略贪心分配 RB，单位为 RB，满足粒度倍数。"""
    # 以“以用户为单位”的块大小：10/5/2 RB
    blocks = {"U": 0, "e": 0, "m": 0}
    rb_used = 0
    # 估计单位块效益=权重/占用RB
    score_per_block = {
        "U": demand_weights.get("U", 0.0) / max(V_U, 1),
        "e": demand_weights.get("e", 0.0) / max(V_E, 1),
        "m": demand_weights.get("m", 0.0) / max(V_M, 1),
    }
    # 先给每类至少一个块（若空间允许）
    order = sorted([("U", score_per_block["U"]), ("e", score_per_block["e"]), ("m", score_per_block["m"])], key=lambda x: -x[1])
    for s, _ in order:
        need = {"U": V_U, "e": V_E, "m": V_M}[s]
        if rb_used + need <= RB_TOTAL:
            blocks[s] += need
            rb_used += need
    # 其余 RB 按单位效益贪心填满
    while rb_used + min(V_M, V_E, V_U) <= RB_TOTAL:
        s_best = max(["U", "e", "m"], key=lambda ss: score_per_block[ss])
        need = {"U": V_U, "e": V_E, "m": V_M}[s_best]
        if rb_used + need > RB_TOTAL:
            # 换次优
            candidates = [ss for ss in ["U", "e", "m"] if rb_used + {"U": V_U, "e": V_E, "m": V_M}[ss] <= RB_TOTAL]
            if not candidates:
                break
            s_best = max(candidates, key=lambda ss: score_per_block[ss])
            need = {"U": V_U, "e": V_E, "m": V_M}[s_best]
        blocks[s_best] += need
        rb_used += need
        if rb_used >= RB_TOTAL:
            break
    # 可能存在剩余不足以形成完整块的 RB，忽略（不分配）
    nU, nE, nM = blocks["U"], blocks["e"], blocks["m"]
    # 若因初始至少一块导致超过 50，则回退
    while nU + nE + nM > RB_TOTAL:
        # 去掉单位效益最小的一个块
        s_worst = min(["U", "e", "m"], key=lambda ss: score_per_block[ss] if blocks[ss] > 0 else float("inf"))
        if blocks[s_worst] <= 0:
            break
        blocks[s_worst] -= {"U": V_U, "e": V_E, "m": V_M}[s_worst]
        nU, nE, nM = blocks["U"], blocks["e"], blocks["m"]
    return nU, nE, nM


# ===================== 仿真（3站带干扰） =====================

@dataclass
class SimResult3:
    sum_U: float = 0.0
    sum_E: float = 0.0
    sum_m_score: float = 0.0
    obj: float = 0.0
    # 调试：mMTC 成功比
    m_ratio_num: int = 0
    m_ratio_den: int = 0


def slice_intervals_from_split(split: Tuple[int, int, int]) -> Dict[str, Tuple[int, int]]:
    """给定 (nU,nE,nM) 返回切片在 [0,49] 上的连续区间（闭区间）。顺序：U,e,m。"""
    nU, nE, nM = split
    cur = 0
    res: Dict[str, Tuple[int, int]] = {}
    if nU > 0:
        res["U"] = (cur, cur + nU - 1)
        cur += nU
    else:
        res["U"] = (-1, -2)  # 空区间
    if nE > 0:
        res["e"] = (cur, cur + nE - 1)
        cur += nE
    else:
        res["e"] = (-1, -2)
    if nM > 0:
        res["m"] = (cur, cur + nM - 1)
        cur += nM
    else:
        res["m"] = (-1, -2)
    return res


def rb_block_for_slot(interval: Tuple[int, int], per_user_rbs: int, slot_index: int) -> List[int]:
    """返回该切片中第 slot_index 个并发用户的 i_s 个 RB 索引。"""
    l, r = interval
    if l > r or per_user_rbs <= 0:
        return []
    start = l + slot_index * per_user_rbs
    end = start + per_user_rbs - 1
    if end > r:
        return []
    return list(range(start, end + 1))


def simulate_window_3bs(env: Env3,
                        users: Dict[str, UserState],
                        U_all: List[str], E_all: List[str], M_all: List[str],
                        assoc: Dict[str, int],
                        t0: int,
                        splits: Dict[int, Tuple[int, int, int]],
                        powers: Dict[int, Dict[str, float]]) -> Tuple[Dict[str, UserState], SimResult3]:
    """仿真 [t0, t0+WINDOW_MS) 窗口，三站分配与功率固定。返回窗口末 users 状态副本与 QoS。"""
    st: Dict[str, UserState] = {name: UserState(name=u.name, category=u.category, queue=deque(copy.deepcopy(list(u.queue)))) for name, u in users.items()}

    # 并发容量
    caps: Dict[int, Dict[str, int]] = {}
    for n in [1, 2, 3]:
        nU, nE, nM = splits[n]
        caps[n] = {
            "U": (nU // V_U) if V_U > 0 else 0,
            "e": (nE // V_E) if V_E > 0 else 0,
            "m": (nM // V_M) if V_M > 0 else 0,
        }

    intervals: Dict[int, Dict[str, Tuple[int, int]]] = {n: slice_intervals_from_split(splits[n]) for n in [1, 2, 3]}

    # 活动列表 per 基站 per 切片
    active: Dict[int, Dict[str, List[str]]] = {n: {"U": [], "e": [], "m": []} for n in [1, 2, 3]}

    # 编号顺序（每站内按关联过滤）
    U_order_by_bs: Dict[int, List[str]] = {n: [nm for nm in U_all if assoc.get(nm, 0) == n] for n in [1, 2, 3]}
    E_order_by_bs: Dict[int, List[str]] = {n: [nm for nm in E_all if assoc.get(nm, 0) == n] for n in [1, 2, 3]}
    M_order_by_bs: Dict[int, List[str]] = {n: [nm for nm in M_all if assoc.get(nm, 0) == n] for n in [1, 2, 3]}

    res = SimResult3()
    t1 = min(t0 + WINDOW_MS, TOTAL_MS)

    # mMTC 窗口级统计
    m_had_arrival: set[str] = set()
    m_success_users: set[str] = set()

    for t in range(t0, t1):
        # 1) 到达
        for name, arr_series in env.arrivals_mbit.items():
            if t < len(arr_series):
                vol = arr_series[t]
                if vol > 0.0:
                    st[name].queue.append(Chunk(arrival_ms=t, size_mbit=vol, remain_mbit=vol))
                    if name.startswith("m"):
                        m_had_arrival.add(name)

        # 2) 填充并发槽位（编号靠前优先，站内）
        def fill_active_for_bs(n: int):
            for s, order in [("U", U_order_by_bs[n]), ("e", E_order_by_bs[n]), ("m", M_order_by_bs[n])]:
                cap = caps[n][s]
                lst = active[n][s]
                # 移除无 backlog 用户
                lst[:] = [nm for nm in lst if st[nm].has_backlog()]
                # 补位
                for nm in order:
                    if len(lst) >= cap:
                        break
                    if nm in lst:
                        continue
                    if st[nm].has_backlog():
                        lst.append(nm)

        for n in [1, 2, 3]:
            fill_active_for_bs(n)

        # 3) 构建每站每切片当前占用的 RB 索引集合（按槽位分块）
        occ_by_bs_slice_slot: Dict[int, Dict[str, List[List[int]]]] = {n: {"U": [], "e": [], "m": []} for n in [1, 2, 3]}
        for n in [1, 2, 3]:
            for s in ["U", "e", "m"]:
                per_rbs = {"U": V_U, "e": V_E, "m": V_M}[s]
                for j in range(len(active[n][s])):
                    block = rb_block_for_slot(intervals[n][s], per_rbs, j)
                    occ_by_bs_slice_slot[n][s].append(block)

        # 4) 服务（同一 ms）：对每个占用者计算含干扰速率并扣减队头块
        n0_cache = {V_U: noise_power_mw(V_U), V_E: noise_power_mw(V_E), V_M: noise_power_mw(V_M)}

        def serve_user(n: int, s: str, user_index_in_active: int):
            nm = active[n][s][user_index_in_active]
            if not st[nm].has_backlog():
                return
            head = st[nm].queue[0]
            if head.start_ms is None:
                head.start_ms = t
            # 本站接收功率
            phi_db = env.get_phi_db(n, nm, t)
            h_pow = env.get_h_pow(n, nm, t)
            p_dbm = powers[n][s]
            p_rx_mw = 10 ** ((p_dbm - phi_db) / 10.0) * h_pow

            # 干扰功率：来自其他两站在与本用户 RB 块重叠的部分
            target_block = occ_by_bs_slice_slot[n][s][user_index_in_active]
            target_len = len(target_block)
            I_total = 0.0
            if target_len > 0:
                target_set = set(target_block)
                for u in [1, 2, 3]:
                    if u == n:
                        continue
                    # 来自站 u 的不同切片，按各自功率叠加
                    for s2 in ["U", "e", "m"]:
                        blocks_u_s2 = occ_by_bs_slice_slot[u][s2]
                        if not blocks_u_s2:
                            continue
                        # 合并该切片在站 u 的所有占用块
                        occ_union: set[int] = set()
                        for bl in blocks_u_s2:
                            occ_union.update(bl)
                        if not occ_union:
                            continue
                        overlap_count = len(target_set & occ_union)
                        if overlap_count <= 0:
                            continue
                        frac = overlap_count / float(target_len)
                        phi_u_db = env.get_phi_db(u, nm, t)
                        h_u_pow = env.get_h_pow(u, nm, t)
                        p_u_dbm = powers[u][s2]
                        p_u_rx_mw = 10 ** ((p_u_dbm - phi_u_db) / 10.0) * h_u_pow
                        I_total += frac * p_u_rx_mw

            num_rbs = {"U": V_U, "e": V_E, "m": V_M}[s]
            n0_mw = n0_cache[num_rbs]
            sinr = p_rx_mw / max(I_total + n0_mw, 1e-30)
            r_bps = num_rbs * B_HZ * log2_safe(1.0 + sinr)
            served_mbit = (r_bps * 0.001) / 1e6  # 1 ms
            if served_mbit < 0.0:
                served_mbit = 0.0
            head.remain_mbit -= served_mbit
            if head.remain_mbit <= 1e-12:
                head.remain_mbit = 0.0
                head.finish_ms = t + 1

        # 执行服务
        for n in [1, 2, 3]:
            for s in ["U", "e", "m"]:
                for idx in range(len(active[n][s])):
                    serve_user(n, s, idx)

        # 5) 统计本 ms 完成的头块并打分
        for n in [1, 2, 3]:
            for s in ["U", "e", "m"]:
                order = {"U": U_order_by_bs[n], "e": E_order_by_bs[n], "m": M_order_by_bs[n]}[s]
                for nm in order:
                    while st[nm].queue and st[nm].queue[0].finish_ms is not None and st[nm].queue[0].finish_ms == t + 1:
                        ch = st[nm].queue.popleft()
                        L_ms = (ch.finish_ms - ch.arrival_ms)
                        if s == 'U':
                            res.sum_U += urllc_qos_from_L(L_ms)
                        elif s == 'e':
                            res.sum_E += embb_qos_from_chunk(ch.size_mbit, L_ms)
                        else:
                            if t0 <= ch.arrival_ms < t1 and mmtc_success_from_L(L_ms):
                                m_success_users.add(nm)

    # 窗口末 mMTC 逐用户口径
    res.m_ratio_den = len(m_had_arrival)
    res.m_ratio_num = len([u for u in m_had_arrival if u in m_success_users])
    ratio = (res.m_ratio_num / res.m_ratio_den) if res.m_ratio_den > 0 else 0.0
    sum_m = 0.0
    for u in m_had_arrival:
        if u in m_success_users:
            sum_m += ratio
        else:
            sum_m += -M_M
    res.sum_m_score = sum_m
    res.obj = res.sum_U + res.sum_E + res.sum_m_score

    return st, res


# ===================== 主流程（MPC+坐标下降） =====================

def main() -> None:
    env, U_names, E_names, M_names = build_env3()

    # 初始化用户状态
    users: Dict[str, UserState] = {}
    for nm in U_names:
        users[nm] = UserState(name=nm, category='U')
    for nm in E_names:
        users[nm] = UserState(name=nm, category='E')
    for nm in M_names:
        users[nm] = UserState(name=nm, category='M')

    # 枚举表（用于 per-BS 局部搜索）
    splits_all = enumerate_splits_single_bs()
    p_grid = initial_power_grid()

    # 记录
    window_rows: List[List[str]] = []

    for w in range(0, TOTAL_MS, WINDOW_MS):
        idx = w // WINDOW_MS
        # 1) 关联（按窗口起点最小路径损耗，就近）
        assoc = associate_by_min_pathloss(env, w, U_names + E_names + M_names)

        # 2) 初始功率（每站相同初值）
        powers: Dict[int, Dict[str, float]] = {n: initial_power_profile() for n in [1, 2, 3]}

        # 3) 初始 RB 划分（每站单独按需求权重粗分）
        # 需求权重：窗口内到达+积压（Mbit）粗略估计
        demand_by_bs: Dict[int, Dict[str, float]] = {n: {"U": 0.0, "e": 0.0, "m": 0.0} for n in [1, 2, 3]}
        t1 = min(w + WINDOW_MS, TOTAL_MS)
        for nm, st_u in users.items():
            cat = st_u.category
            n = assoc.get(nm, 0)
            if n not in [1, 2, 3]:
                continue
            demand_by_bs[n]["U" if cat == 'U' else ('e' if cat == 'E' else 'm')] += st_u.total_backlog_mbit()
        # 加上窗口内预计到达量
        for nm, arr in env.arrivals_mbit.items():
            n = assoc.get(nm, 0)
            if n not in [1, 2, 3]:
                continue
            add = sum(arr[w:t1]) if w < len(arr) else 0.0
            cat = users[nm].category
            demand_by_bs[n]["U" if cat == 'U' else ('e' if cat == 'E' else 'm')] += add

        splits: Dict[int, Tuple[int, int, int]] = {}
        for n in [1, 2, 3]:
            splits[n] = greedy_initial_split(demand_by_bs[n])
            # 保底：若全为 0，给 mMTC 25*2 的上限内一块
            if sum(splits[n]) == 0:
                splits[n] = (0, 0, min(RB_TOTAL, V_M))

        # 4) 坐标下降：依次优化每个基站的 RB 划分（其余两站固定）
        improved = True
        best_state_after = None
        best_res = None
        # 先评估初值
        cur_state, cur_res = simulate_window_3bs(env, users, U_names, E_names, M_names, assoc, w, splits, powers)
        best_score = cur_res.obj
        best_state_after = cur_state
        best_res = cur_res
        for cd_round in range(2):  # 2 轮即可
            changed = False
            for n in [1, 2, 3]:
                local_best = best_score
                local_best_split = splits[n]
                local_best_state = best_state_after
                local_best_res = best_res
                for sp in splits_all:
                    tmp_splits = dict(splits)
                    tmp_splits[n] = sp
                    new_state, res = simulate_window_3bs(env, users, U_names, E_names, M_names, assoc, w, tmp_splits, powers)
                    if res.obj > local_best:
                        local_best = res.obj
                        local_best_split = sp
                        local_best_state = new_state
                        local_best_res = res
                if local_best > best_score + 1e-9:
                    splits[n] = local_best_split
                    best_score = local_best
                    best_state_after = local_best_state
                    best_res = local_best_res
                    changed = True
            if not changed:
                break

        # 5) 可选：功率最佳响应（在小网格上）
        for n in [1, 2, 3]:
            local_best = best_score
            local_best_p = powers[n].copy()
            local_best_state = best_state_after
            local_best_res = best_res
            for pU in p_grid:
                for pE in p_grid:
                    for pM in p_grid:
                        tmp_powers = {nn: powers[nn].copy() for nn in [1, 2, 3]}
                        tmp_powers[n] = {"U": pU, "e": pE, "m": pM}
                        new_state, res = simulate_window_3bs(env, users, U_names, E_names, M_names, assoc, w, splits, tmp_powers)
                        if res.obj > local_best:
                            local_best = res.obj
                            local_best_p = tmp_powers[n].copy()
                            local_best_state = new_state
                            local_best_res = res
            if local_best > best_score + 1e-9:
                powers[n] = local_best_p
                best_score = local_best
                best_state_after = local_best_state
                best_res = local_best_res

        # 6) 采用最终方案推进到窗口末状态
        assert best_state_after is not None and best_res is not None
        users = best_state_after

        # 打印与记录
        sp1, sp2, sp3 = splits[1], splits[2], splits[3]
        p1, p2, p3 = powers[1], powers[2], powers[3]
        print(f"[决策{idx:02d}] 窗口 {w}~{w+WINDOW_MS} ms")
        print(f"  BS1: (R_U,R_e,R_m)={sp1}, (p_U,p_e,p_m)=({p1['U']:.1f},{p1['e']:.1f},{p1['m']:.1f})")
        print(f"  BS2: (R_U,R_e,R_m)={sp2}, (p_U,p_e,p_m)=({p2['U']:.1f},{p2['e']:.1f},{p2['m']:.1f})")
        print(f"  BS3: (R_U,R_e,R_m)={sp3}, (p_U,p_e,p_m)=({p3['U']:.1f},{p3['e']:.1f},{p3['m']:.1f})")
        print(f"  窗口QoS: URLLC={best_res.sum_U:.4f}, eMBB={best_res.sum_E:.4f}, mMTC累计={best_res.sum_m_score:.4f}, 目标={best_res.obj:.4f}")

        window_rows.append([
            str(idx),
            str(sp1[0]), str(sp1[1]), str(sp1[2]), f"{p1['U']:.2f}", f"{p1['e']:.2f}", f"{p1['m']:.2f}",
            str(sp2[0]), str(sp2[1]), str(sp2[2]), f"{p2['U']:.2f}", f"{p2['e']:.2f}", f"{p2['m']:.2f}",
            str(sp3[0]), str(sp3[1]), str(sp3[2]), f"{p3['U']:.2f}", f"{p3['e']:.2f}", f"{p3['m']:.2f}",
            f"{best_res.sum_U:.6f}", f"{best_res.sum_E:.6f}", f"{best_res.sum_m_score:.6f}", f"{best_res.obj:.6f}",
        ])

        # 导出本窗口的接入关联（a_{n,k}）
        csv_assoc = os.path.join(SCRIPT_DIR, f"q3_assoc_win_{idx:02d}.csv")
        try:
            with open(csv_assoc, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["user", "bs"])
                for nm in (U_names + E_names + M_names):
                    writer.writerow([nm, assoc.get(nm, 0)])
            print(f"  已导出接入关联 -> {csv_assoc}")
        except Exception as e:
            print(f"  导出接入关联失败: {e}")

    # 汇总
    total_U = sum(float(r[-4]) for r in window_rows) if window_rows else 0.0
    total_E = sum(float(r[-3]) for r in window_rows) if window_rows else 0.0
    total_m = sum(float(r[-2]) for r in window_rows) if window_rows else 0.0
    total_obj = sum(float(r[-1]) for r in window_rows) if window_rows else 0.0

    print("\n== 总结 ==")
    print(f"累计 URLLC={total_U:.4f}, eMBB={total_E:.4f}, mMTC累计={total_m:.4f}, 目标累计={total_obj:.4f}")

    # 导出逐窗口结果
    csv_out = os.path.join(SCRIPT_DIR, "q3_window_results.csv")
    try:
        with open(csv_out, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "window",
                "BS1_RU", "BS1_Re", "BS1_Rm", "BS1_pU", "BS1_pe", "BS1_pm",
                "BS2_RU", "BS2_Re", "BS2_Rm", "BS2_pU", "BS2_pe", "BS2_pm",
                "BS3_RU", "BS3_Re", "BS3_Rm", "BS3_pU", "BS3_pe", "BS3_pm",
                "sum_URLLC", "sum_eMBB", "sum_mMTC", "obj",
            ])
            for row in window_rows:
                writer.writerow(row)
        print(f"结果已导出 -> {csv_out}")
    except Exception as e:
        print(f"导出 CSV 失败: {e}")


if __name__ == "__main__":
    main()


