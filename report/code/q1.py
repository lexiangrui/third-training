#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
问题一：单小区、无干扰、功率30 dBm。采用“切片RB枚举 + 同周期内可排队的串并行调度”以最大化总体服务质量（URLLC+eMBB+mMTC）。

数据来源（相对路径，基于本脚本所在目录）：
- 任务量：../../题目/附件/附件1/q1_任务流.csv （单位：Mbit）
- 大规模衰减：../../题目/附件/附件1/q1_大规模衰减.csv （单位：dB）
- 小规模瑞利：../../题目/附件/附件1/q1_小规模瑞丽衰减.csv （幅度 |h|）

假设与参数：
- p_tx = 30 dBm，b = 360 kHz，NF = 7 dB。
- v_U=10, v_E=5, v_M=2（每类用户并发占用RB数）。
- 决策周期 T_window = 100 ms；同一周期内允许在各切片内按“编号靠前优先”进行排队服务（先占先得，完成即释放RB，后继用户接续占用）。
- SLA：URLLC: L<=5 ms；eMBB: L<=100 ms 且 r>=50 Mbps；mMTC: L<=500 ms。
- QoS：
  URLLC: y = alpha^L(ms) if L<=5ms else -M_U；alpha=0.95, M_U=5
  eMBB:  y = 1 if (L<=100ms & r>=50Mbps)；y = r/50 if (L<=100ms & r<50Mbps)；else -M_E, M_E=3
  mMTC:  y = (#接入且满足L<=500ms)/(#任务存在的mMTC总数)

实现要点：
- 枚举(R_U, R_E, R_M)且满足R_U+R_E+R_M=50，且R_U%10==0, R_E%5==0, R_M%2==0以避免RB浪费。
- 对每类用户以并发容量 cap_s=R_s/v_s 进行最短完工时间的无抢占调度，计算等待 Q 与总时延 L=Q+T，继而按切片QoS定义计分并汇总。
"""

from __future__ import annotations

import csv
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable


"""基于脚本位置的相对路径，确保可移植性。"""
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
ATTACH_DIR = os.path.join(ROOT_DIR, "题目", "附件", "附件1")

CSV_TASK = os.path.join(ATTACH_DIR, "q1_任务流.csv")
CSV_PL = os.path.join(ATTACH_DIR, "q1_大规模衰减.csv")
CSV_RAY = os.path.join(ATTACH_DIR, "q1_小规模瑞丽衰减.csv")


# 系统常量
P_TX_DBM: float = 30.0
B_HZ: float = 360_000.0  # 360 kHz
NF_DB: float = 7.0
T_WINDOW_MS: float = 100.0  # 决策周期（可排队服务窗口）

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


@dataclass
class User:
    name: str
    category: str  # 'U' | 'E' | 'M'
    data_mbit: float
    pl_db: float  # 路径损耗 φ (dB)
    h_abs: float  # 瑞利幅度 |h|

    @property
    def h_pow(self) -> float:
        return self.h_abs * self.h_abs


def read_single_row_csv(path: str) -> Dict[str, float]:
    """读取仅包含一行数据的CSV：返回列名到数值的映射（忽略'Time'列）。"""
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        row = next(reader)
        result: Dict[str, float] = {}
        for k, v in row.items():
            if k is None:
                continue
            if k.strip().lower() == "time":
                continue
            if v is None or v == "":
                continue
            try:
                result[k.strip()] = float(v)
            except ValueError:
                # 忽略非数值
                pass
        return result


def build_users() -> List[User]:
    task = read_single_row_csv(CSV_TASK)
    pl = read_single_row_csv(CSV_PL)
    ray = read_single_row_csv(CSV_RAY)

    users: List[User] = []
    for name in task.keys():
        data_mbit = task.get(name, 0.0)
        pl_db = pl.get(name, None)
        h_abs = ray.get(name, None)
        if pl_db is None or h_abs is None:
            # 数据缺失则跳过
            continue
        if name.startswith("U") or name.startswith("u"):
            cat = "U"
        elif name.startswith("e") or name.startswith("E"):
            cat = "E"
        elif name.startswith("m") or name.startswith("M"):
            cat = "M"
        else:
            # 未知命名，跳过
            continue
        users.append(User(name=name, category=cat, data_mbit=data_mbit, pl_db=pl_db, h_abs=h_abs))
    return users


def dbm_to_mw(dbm: float) -> float:
    return 10 ** (dbm / 10.0)


def noise_power_mw(num_rbs: int, b_hz: float = B_HZ, nf_db: float = NF_DB) -> float:
    """按附录：N0(dBm) = -174 + 10log10(i*b) + NF，返回mW。"""
    if num_rbs <= 0:
        # 不占用RB则无意义，返回极小正数避免除零
        return 1e-30
    n0_dbm = -174.0 + 10.0 * math.log10(num_rbs * b_hz) + nf_db
    return dbm_to_mw(n0_dbm)


def prx_mw(p_tx_dbm: float, pl_db: float, h_abs: float) -> float:
    """接收功率：p_rx = 10^((p_tx - φ)/10) * |h|^2 (mW)。"""
    return 10 ** ((p_tx_dbm - pl_db) / 10.0) * (h_abs * h_abs)


def user_rate_bps(user: User, num_rbs: int) -> float:
    """香农速率：r = i*b*log2(1+SNR)。单位：bit/s"""
    p_rx = prx_mw(P_TX_DBM, user.pl_db, user.h_abs)
    n0 = noise_power_mw(num_rbs)
    snr = p_rx / n0
    return num_rbs * B_HZ * math.log2(1.0 + snr)


def delay_ms(user: User, num_rbs: int) -> float:
    r_bps = user_rate_bps(user, num_rbs)
    if r_bps <= 0.0:
        return float("inf")
    t_s = (user.data_mbit * 1e6) / r_bps
    return t_s * 1e3


def urllc_qos_from_L(L_ms: float) -> float:
    if L_ms <= SLA_L_U_MS:
        return ALPHA ** L_ms
    return -M_U


def embb_qos_from_L_and_r(L_ms: float, r_bps: float) -> float:
    r_mbps = r_bps / 1e6
    if L_ms <= SLA_L_E_MS and r_mbps >= SLA_R_E_MBPS:
        return 1.0
    if L_ms <= SLA_L_E_MS and r_mbps < SLA_R_E_MBPS:
        return max(0.0, r_mbps / SLA_R_E_MBPS)
    return -M_E


def mm_tc_success_from_L(L_ms: float) -> bool:
    return L_ms <= SLA_L_M_MS


def schedule_slice(users: List[User], num_rbs_slice: int, per_user_rbs: int) -> Dict[str, Dict[str, float]]:
    """
    在给定切片RB与每用户固定RB占用下，对该切片内所有用户进行同周期(100ms)内的串并行调度。
    - 并发容量 cap = floor(num_rbs_slice / per_user_rbs)。
    - 调度顺序：输入列表顺序（与数据列顺序/编号先后对齐）。
    - 计算每个用户的开始时间、完成时间、等待 Q 与总时延 L=Q+T。
    返回：name -> { 'Q_ms', 'T_ms', 'L_ms', 'r_bps' }。
    """
    import heapq

    results: Dict[str, Dict[str, float]] = {}

    # 并发容量
    cap = num_rbs_slice // per_user_rbs if per_user_rbs > 0 else 0
    if cap <= 0:
        # 无法服务：用无穷大时延表示
        for u in users:
            r_bps = user_rate_bps(u, per_user_rbs) if per_user_rbs > 0 else 0.0
            results[u.name] = {
                "Q_ms": float("inf"),
                "T_ms": delay_ms(u, per_user_rbs) if per_user_rbs > 0 else float("inf"),
                "L_ms": float("inf"),
                "r_bps": r_bps,
            }
        return results

    # 小根堆：存放正在服务的会话的完成时刻
    active_heap: List[Tuple[float, int]] = []  # (finish_time_ms, counter)
    counter = 0

    # 先填满并发槽位
    idx = 0
    n = len(users)
    while idx < min(cap, n):
        u = users[idx]
        T_ms = delay_ms(u, per_user_rbs)
        Q_ms = 0.0
        L_ms = Q_ms + T_ms
        r_bps = user_rate_bps(u, per_user_rbs)
        results[u.name] = {"Q_ms": Q_ms, "T_ms": T_ms, "L_ms": L_ms, "r_bps": r_bps}
        heapq.heappush(active_heap, (L_ms, counter))
        counter += 1
        idx += 1

    # 其余用户按完成最早者释放后接续
    while idx < n:
        u = users[idx]
        # 取出最早释放时间
        earliest_finish, _ = heapq.heappop(active_heap)

        T_ms = delay_ms(u, per_user_rbs)
        start_ms = earliest_finish
        Q_ms = start_ms
        L_ms = Q_ms + T_ms
        r_bps = user_rate_bps(u, per_user_rbs)
        results[u.name] = {"Q_ms": Q_ms, "T_ms": T_ms, "L_ms": L_ms, "r_bps": r_bps}

        heapq.heappush(active_heap, (L_ms, counter))
        counter += 1
        idx += 1

    return results


def choose_best_subset(values: List[Tuple[str, float]], k: int) -> List[str]:
    """从(name,score)中选择最多k个且score>0的项，按score降序。返回被选name列表。"""
    filtered = [it for it in values if it[1] > 0]
    filtered.sort(key=lambda x: x[1], reverse=True)
    return [name for name, _ in filtered[:k]]


def enumerate_solution(users: List[User]):
    # 按类别拆分
    U = [u for u in users if u.category == "U"]
    E = [u for u in users if u.category == "E"]
    M = [u for u in users if u.category == "M"]

    best = {
        "obj": -1e18,
        "R": (0, 0, 0),
        "sel_U": [],
        "sel_E": [],
        "sel_M": [],
        "details": {},
    }

    for R_U in range(0, 51, V_U):  # 0,10,20,30,40,50
        for R_E in range(0, 51 - R_U, V_E):  # 0,5,10,...
            R_M = 50 - R_U - R_E
            if R_M < 0:
                continue
            if R_M % V_M != 0:
                continue  # 避免RB浪费

            # 在各切片内进行同周期调度，得到每个用户的 L 与 r
            U_sched = schedule_slice(U, R_U, V_U)
            E_sched = schedule_slice(E, R_E, V_E)
            M_sched = schedule_slice(M, R_M, V_M)

            # 计算 QoS
            sum_U = 0.0
            for u in U:
                L_ms = U_sched[u.name]["L_ms"]
                sum_U += urllc_qos_from_L(L_ms)

            sum_E = 0.0
            for e in E:
                L_ms = E_sched[e.name]["L_ms"]
                r_bps = E_sched[e.name]["r_bps"]
                sum_E += embb_qos_from_L_and_r(L_ms, r_bps)

            denom_M = sum(1 for m in M if m.data_mbit > 0.0)
            num_M = 0
            for m in M:
                L_ms = M_sched[m.name]["L_ms"]
                if mm_tc_success_from_L(L_ms):
                    num_M += 1
            y_M = (num_M / denom_M) if denom_M > 0 else 0.0

            obj = sum_U + sum_E + y_M

            if obj > best["obj"]:
                # 保存详情
                details = {}
                for u in U:
                    L_ms = U_sched[u.name]["L_ms"]
                    Q_ms = U_sched[u.name]["Q_ms"]
                    r_mbps = U_sched[u.name]["r_bps"] / 1e6
                    details[u.name] = {
                        "selected": True,
                        "Q_ms": Q_ms,
                        "L_ms": L_ms,
                        "r_Mbps": r_mbps,
                        "qos": urllc_qos_from_L(L_ms),
                    }
                for e in E:
                    L_ms = E_sched[e.name]["L_ms"]
                    Q_ms = E_sched[e.name]["Q_ms"]
                    r_mbps = E_sched[e.name]["r_bps"] / 1e6
                    details[e.name] = {
                        "selected": True,
                        "Q_ms": Q_ms,
                        "L_ms": L_ms,
                        "r_Mbps": r_mbps,
                        "qos": embb_qos_from_L_and_r(L_ms, E_sched[e.name]["r_bps"]),
                    }
                for m in M:
                    L_ms = M_sched[m.name]["L_ms"]
                    Q_ms = M_sched[m.name]["Q_ms"]
                    r_mbps = M_sched[m.name]["r_bps"] / 1e6
                    details[m.name] = {
                        "selected": True,
                        "Q_ms": Q_ms,
                        "L_ms": L_ms,
                        "r_Mbps": r_mbps,
                        "success": mm_tc_success_from_L(L_ms),
                    }

                best.update(
                    {
                        "obj": obj,
                        "R": (R_U, R_E, R_M),
                        "sel_U": [u.name for u in U],
                        "sel_E": [e.name for e in E],
                        "sel_M": [m.name for m in M if mm_tc_success_from_L(M_sched[m.name]["L_ms"])],
                        "y_M": y_M,
                        "sum_U": sum_U,
                        "sum_E": sum_E,
                        "details": details,
                    }
                )

    return best


def main() -> None:
    users = build_users()
    result = enumerate_solution(users)

    R_U, R_E, R_M = result["R"]

    print("== 最优结果（枚举法）==")
    print(f"切片RB分配: R_U={R_U}, R_E={R_E}, R_M={R_M} (总计=50)")
    print(f"URLLC接入: {result['sel_U']}")
    print(f"eMBB接入:  {result['sel_E']}")
    print(f"mMTC接入:  {result['sel_M']}")
    print(f"mMTC比例:  y_M={result['y_M']:.4f}")
    print(f"URLLC QoS合计: {result['sum_U']:.4f}")
    print(f"eMBB  QoS合计: {result['sum_E']:.4f}")
    print(f"目标函数: {result['obj']:.4f}")
    print()

    print("-- 详细指标 --")
    for name in sorted(result["details"].keys()):
        info = result["details"][name]
        if name.startswith("U") or name.startswith("u"):
            print(
                f"{name:>4s}: sel={info['selected']}, L(ms)={info['L_ms']:.3f}, r(Mbps)={info['r_Mbps']:.3f}, qos={info['qos']:.4f}"
            )
        elif name.startswith("e") or name.startswith("E"):
            print(
                f"{name:>4s}: sel={info['selected']}, L(ms)={info['L_ms']:.3f}, r(Mbps)={info['r_Mbps']:.3f}, qos={info['qos']:.4f}"
            )
        else:
            print(
                f"{name:>4s}: sel={info['selected']}, L(ms)={info['L_ms']:.3f}, r(Mbps)={info['r_Mbps']:.3f}, success={info['success']}"
            )


if __name__ == "__main__":
    main()


