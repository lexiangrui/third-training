#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
问题二：单微基站、无同频干扰、功率 30 dBm。系统在 1000 ms 内每 100 ms 决策一次三类切片的 RB 分配
(n_U, n_e, n_m)，并在每个决策窗口内进行“编号靠前优先”的串并行调度，评估总体服务质量。

主要特性：
- 数据来自 附件2：
  - 任务流：../../题目/附件/附件2/q2_用户任务流.csv （单位：Mbit）逐 ms 到达
  - 大规模衰减：../../题目/附件/附件2/q2_大规模衰减.csv （单位：dB）逐 ms 变化
  - 小规模瑞丽衰减：../../题目/附件/附件2/q2_小规模瑞丽衰减.csv（dB 或幅度，自动识别）逐 ms 变化
- 物理层：p_tx=30 dBm，b=360 kHz，NF=7 dB；N0(dBm)=-174+10log10(i*b)+NF。
- 切片并发占用：v_U=10, v_e=5, v_m=2（RB/用户）。
- 决策：每个 100ms 窗口内 RB 恒定；窗口内允许按编号优先排队，完成即释放，下一名接续。
- QoS：
  - URLLC：y=alpha^L(ms)，若 L>5ms 给 -M_U
  - eMBB：若 L<=100ms 且 r_avg>=50Mbps → 1；若 L<=100ms 且 r_avg<50 → r_avg/50；否则 -M_E
  - mMTC：与 q1.py 一致的“逐用户加和”口径（在每个窗口内）：
      记本窗有到达的 m 用户集合为 H，成功集合 S（本窗最早到达任务在本窗内完成且 L<=500ms），
      ratio = |S|/|H|（|H|=0 时取 0），则对每个 k∈H，若 k∈S，y_k^m=ratio；否则 y_k^m=-M_M；本窗 mMTC 得分为 ∑_{k∈H} y_k^m。

求解策略（MPC-1步）：
- 在每个决策起点，枚举所有合法 (n_U,n_e,n_m)（避免 RB 碎片化），在 100ms 窗内仿真队列与调度，计算该窗 QoS 得分，选择得分最高的方案执行。
"""

from __future__ import annotations

import csv
import math
import os
import copy
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Deque
from collections import deque


# 路径配置
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
ATTACH_DIR = os.path.join(ROOT_DIR, "题目", "附件", "附件2")

CSV_TASK = os.path.join(ATTACH_DIR, "q2_用户任务流.csv")
CSV_PL = os.path.join(ATTACH_DIR, "q2_大规模衰减.csv")
CSV_RAY = os.path.join(ATTACH_DIR, "q2_小规模瑞丽衰减.csv")


# 物理与系统常量
P_TX_DBM: float = 30.0
B_HZ: float = 360_000.0  # 360 kHz
NF_DB: float = 7.0

V_U: int = 10
V_E: int = 5
V_M: int = 2

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
    # 使用 utf-8-sig 以移除可能存在的 BOM，避免首列名出现 "\ufeffTime"
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
                    # 填 0 以便对齐
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
        if t_ms < 0:
            t_ms = 0
        if t_ms >= len(arr):
            t_ms = len(arr) - 1
        return float(arr[t_ms])

    def get_h_pow(self, name: str, t_ms: int) -> float:
        arr = self.ray_raw.get(name)
        if arr is None or len(arr) == 0:
            return 1.0
        if t_ms < 0:
            t_ms = 0
        if t_ms >= len(arr):
            t_ms = len(arr) - 1
        val = float(arr[t_ms])
        if self.small_scale_is_db:
            # 视为功率(dB)
            return 10 ** (val / 10.0)
        # 视为幅度 |h|
        if val < 0:
            # 防御：若为负仍返回极小正值
            return 1e-6
        return val * val


def detect_small_scale_is_db(ray_series: Dict[str, List[float]]) -> bool:
    # 任意列出现负值，则视为 dB（功率）
    for arr in ray_series.values():
        for v in arr[:200]:  # 采样检查
            if v < 0:
                return True
    return False


def build_env() -> Tuple[Env, List[str], List[str], List[str]]:
    time_list, arrivals = load_time_series_csv(CSV_TASK)
    _, pl = load_time_series_csv(CSV_PL)
    _, ray = load_time_series_csv(CSV_RAY)

    small_is_db = detect_small_scale_is_db(ray)

    # 分类并确定编号顺序
    def sort_key(name: str) -> Tuple[int, int]:
        # 确保 U1,U2,..., e1,e2,..., m1,m2,... 按数字升序
        prefix_rank = 0
        if name.startswith("U"):
            prefix_rank = 0
        elif name.startswith("e"):
            prefix_rank = 1
        elif name.startswith("m"):
            prefix_rank = 2
        # 提取数字后缀
        num = 0
        i = 0
        for i in range(len(name)):
            if name[i].isdigit():
                num = int(name[i:])
                break
        return (prefix_rank, num)

    all_names = [k for k in arrivals.keys()]
    U_names = sorted([n for n in all_names if n.startswith("U")], key=sort_key)
    E_names = sorted([n for n in all_names if n.startswith("e")], key=sort_key)
    M_names = sorted([n for n in all_names if n.startswith("m")], key=sort_key)

    env = Env(
        time_list=time_list,
        arrivals_mbit=arrivals,
        pl_db=pl,
        ray_raw=ray,
        small_scale_is_db=small_is_db,
    )
    return env, U_names, E_names, M_names


# ===================== 速率 / QoS 计算 =====================

def user_rate_bps(env: Env, name: str, t_ms: int, num_rbs: int) -> float:
    if num_rbs <= 0:
        return 0.0
    phi_db = env.get_phi_db(name, t_ms)
    h_pow = env.get_h_pow(name, t_ms)
    p_rx_mw = 10 ** ((P_TX_DBM - phi_db) / 10.0) * h_pow
    n0_mw = noise_power_mw(num_rbs)
    snr = p_rx_mw / max(n0_mw, 1e-30)
    return num_rbs * B_HZ * log2_safe(1.0 + snr)


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


# ===================== 仿真核心 =====================

@dataclass
class SimResult:
    sum_U: float = 0.0
    sum_E: float = 0.0
    sum_m_score: float = 0.0  # mMTC逐用户求和口径
    obj: float = 0.0
    # 调试用字段
    m_ratio_num: int = 0
    m_ratio_den: int = 0


def simulate_window(env: Env,
                    users: Dict[str, UserState],
                    U_order: List[str],
                    E_order: List[str],
                    M_order: List[str],
                    t0: int,
                    nU: int,
                    nE: int,
                    nM: int) -> Tuple[Dict[str, UserState], SimResult]:
    """
    从时间 t0 开始，仿真 [t0, t0+WINDOW_MS) 的窗口，给定切片 RB 配置。
    返回复制后的新状态（窗口末的 users）与本窗 QoS 结果。
    - 无抢占：窗口内某用户占到并发槽位后，持续占用至其队列清空；队列空则立即释放，该 ms 内不复用。
    - 每 ms 先到达再服务。
    - URLLC/eMBB：仅统计在本窗内完成的任务块的得分；mMTC：统计本窗内“有到达的用户中、完成其本窗最早到达任务且 L<=SLA 的用户占比”。
    """
    # 深拷贝 users 状态，以免影响外层
    st: Dict[str, UserState] = {name: UserState(name=u.name, category=u.category, queue=deque(copy.deepcopy(list(u.queue)))) for name, u in users.items()}

    # 并发容量
    capU = nU // V_U if V_U > 0 else 0
    capE = nE // V_E if V_E > 0 else 0
    capM = nM // V_M if V_M > 0 else 0

    # 活动用户集合（当前占用并发槽位）
    activeU: List[str] = []
    activeE: List[str] = []
    activeM: List[str] = []

    # 本窗内 mMTC 统计集合
    m_had_arrival: set[str] = set()
    m_success_users: set[str] = set()

    res = SimResult()

    t1 = min(t0 + WINDOW_MS, TOTAL_MS)
    for t in range(t0, t1):
        # 1) 到达：逐用户把该 ms 到达的任务压入队列
        for name, arr_series in env.arrivals_mbit.items():
            if t < len(arr_series):
                vol = arr_series[t]
                if vol > 0.0:
                    st[name].queue.append(Chunk(arrival_ms=t, size_mbit=vol, remain_mbit=vol))
                    if name.startswith("m") and t0 <= t < t1:
                        m_had_arrival.add(name)

        # 2) 填充并发槽位（编号靠前优先）
        def fill_active(order: List[str], active: List[str], cap: int):
            # 移除已空队列的
            active[:] = [nm for nm in active if st[nm].has_backlog()]
            # 补位
            for nm in order:
                if len(active) >= cap:
                    break
                if nm in active:
                    continue
                if st[nm].has_backlog():
                    active.append(nm)

        fill_active(U_order, activeU, capU)
        fill_active(E_order, activeE, capE)
        fill_active(M_order, activeM, capM)

        # 3) 服务：对每个占用者按当 ms 信道计算速率，并扣减头块
        def serve_one(name: str, per_user_rbs: int):
            if not st[name].has_backlog():
                return
            head = st[name].queue[0]
            if head.start_ms is None:
                head.start_ms = t
            r_bps = user_rate_bps(env, name, t, per_user_rbs)
            served_mbit = (r_bps * 0.001) / 1e6  # 1 ms
            if served_mbit <= 0.0:
                served_mbit = 0.0
            head.remain_mbit -= served_mbit
            if head.remain_mbit <= 1e-12:
                head.remain_mbit = 0.0
                head.finish_ms = t + 1  # 在该 ms 末完成

        for nm in activeU:
            serve_one(nm, V_U)
        for nm in activeE:
            serve_one(nm, V_E)
        for nm in activeM:
            serve_one(nm, V_M)

        # 4) 统计：仅统计本窗内完成的头块（可能一个用户在本窗内完成多个块）
        def collect_finished(order: List[str], per_user_rbs: int, sum_acc: str):
            nonlocal res
            for nm in order:
                # 循环弹出本 ms 完成的多个块
                while st[nm].queue and st[nm].queue[0].finish_ms is not None and st[nm].queue[0].finish_ms == t + 1:
                    ch = st[nm].queue.popleft()
                    L_ms = (ch.finish_ms - ch.arrival_ms)
                    if st[nm].category == 'U':
                        res.sum_U += urllc_qos_from_L(L_ms)
                    elif st[nm].category == 'E':
                        res.sum_E += embb_qos_from_chunk(ch.size_mbit, L_ms)
                    else:  # mMTC：记录用户是否在本窗成功（最早到达块在本窗内完成且 L<=SLA）
                        # 本窗成功判定：该块到达于本窗内，且 L<=SLA
                        if t0 <= ch.arrival_ms < t1 and mmtc_success_from_L(L_ms):
                            m_success_users.add(nm)
                    # 若该用户队列仍有块，且仍在 active 集合中，下一个块将于下一 ms 开始服务

        collect_finished(U_order, V_U, 'sum_U')
        collect_finished(E_order, V_E, 'sum_E')
        collect_finished(M_order, V_M, 'sum_m')

    # 5) 窗口末计算 mMTC 逐用户求和口径
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


def enumerate_splits() -> List[Tuple[int, int, int]]:
    splits: List[Tuple[int, int, int]] = []
    for nU in range(0, 51, V_U):
        for nE in range(0, 51 - nU, V_E):
            nM = 50 - nU - nE
            if nM < 0:
                continue
            if nM % V_M != 0:
                continue
            splits.append((nU, nE, nM))
    return splits


def main() -> None:
    env, U_names, E_names, M_names = build_env()

    # 初始化用户状态
    users: Dict[str, UserState] = {}
    for nm in U_names:
        users[nm] = UserState(name=nm, category='U')
    for nm in E_names:
        users[nm] = UserState(name=nm, category='E')
    for nm in M_names:
        users[nm] = UserState(name=nm, category='M')

    splits = enumerate_splits()
    enum_rows_all: List[Tuple[int, int, int, int, float, float, float, float]] = []  # (win, nU,nE,nM,sumU,sumE,sumM,obj)

    decisions: List[Tuple[int, int, int]] = []
    window_results: List[SimResult] = []

    # 逐窗口决策
    for w in range(0, TOTAL_MS, WINDOW_MS):
        best_score = -1e18
        best_split = (0, 0, 50)
        best_state_after: Dict[str, UserState] | None = None
        best_res: SimResult | None = None

        enum_rows_window: List[Tuple[int, int, int, float, float, float, float]] = []
        for (nU, nE, nM) in splits:
            new_state, res = simulate_window(env, users, U_names, E_names, M_names, w, nU, nE, nM)
            enum_rows_window.append((nU, nE, nM, res.sum_U, res.sum_E, res.sum_m_score, res.obj))
            enum_rows_all.append((w // WINDOW_MS, nU, nE, nM, res.sum_U, res.sum_E, res.sum_m_score, res.obj))
            if res.obj > best_score:
                best_score = res.obj
                best_split = (nU, nE, nM)
                best_state_after = new_state
                best_res = res

        assert best_state_after is not None and best_res is not None
        users = best_state_after
        decisions.append(best_split)
        window_results.append(best_res)

        # 输出该窗口结果
        idx = w // WINDOW_MS
        print(f"[决策{idx:02d}] 窗口 {w}~{w+WINDOW_MS} ms: 选择 (R_U,R_e,R_m)={best_split}, "
              f"URLLC={best_res.sum_U:.4f}, eMBB={best_res.sum_E:.4f}, mMTC累计={best_res.sum_m_score:.4f}, 目标={best_res.obj:.4f}")

        # 并列最优方案输出
        tol = 1e-9
        best_list = [row for row in enum_rows_window if abs(row[6] - best_res.obj) <= tol]
        if len(best_list) > 1:
            print(f"  并列最优方案（{len(best_list)}个）：")
            for (nU, nE, nM, sU, sE, sM, obj) in best_list:
                print(f"    (R_U={nU}, R_e={nE}, R_m={nM}), ∑U={sU:.4f}, ∑e={sE:.4f}, ∑m={sM:.4f}, Q={obj:.4f}")
        elif len(best_list) == 1:
            # 明确打印唯一最优
            (nU, nE, nM, sU, sE, sM, obj) = best_list[0]
            print(f"  唯一最优： (R_U={nU}, R_e={nE}, R_m={nM})")

        # 导出本窗口全部枚举结果为 CSV
        csv_win = os.path.join(SCRIPT_DIR, f"q2_enum_results_win_{idx:02d}.csv")
        try:
            with open(csv_win, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["R_U", "R_E", "R_M", "sum_URLLC", "sum_eMBB", "sum_mMTC", "obj"])
                for (nU, nE, nM, sU, sE, sM, obj) in enum_rows_window:
                    writer.writerow([nU, nE, nM, f"{sU:.6f}", f"{sE:.6f}", f"{sM:.6f}", f"{obj:.6f}"])
            print(f"  已导出窗口枚举结果 -> {csv_win}")
        except Exception as e:
            print(f"  导出窗口CSV失败: {e}")

    # 汇总
    total_U = sum(r.sum_U for r in window_results)
    total_E = sum(r.sum_E for r in window_results)
    total_m = sum(r.sum_m_score for r in window_results)
    total_obj = sum(r.obj for r in window_results)

    print("\n== 总结 ==")
    print("10次决策的切片RB分配序列：")
    for i, sp in enumerate(decisions):
        print(f"  t={i*WINDOW_MS:3d}ms: R_U={sp[0]:2d}, R_e={sp[1]:2d}, R_m={sp[2]:2d}")
    print(f"累计 URLLC={total_U:.4f}, eMBB={total_E:.4f}, mMTC累计={total_m:.4f}, 目标累计={total_obj:.4f}")

    # 导出所有窗口的枚举结果（汇总）
    csv_all = os.path.join(SCRIPT_DIR, "q2_enum_results_all.csv")
    try:
        with open(csv_all, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["window", "R_U", "R_E", "R_M", "sum_URLLC", "sum_eMBB", "sum_mMTC", "obj"])
            for (win, nU, nE, nM, sU, sE, sM, obj) in enum_rows_all:
                writer.writerow([win, nU, nE, nM, f"{sU:.6f}", f"{sE:.6f}", f"{sM:.6f}", f"{obj:.6f}"])
        print(f"已导出全部窗口的枚举结果 -> {csv_all}")
    except Exception as e:
        print(f"导出汇总CSV失败: {e}")


if __name__ == "__main__":
    main()


