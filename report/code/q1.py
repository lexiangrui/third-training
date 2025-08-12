#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
问题一：单小区、无干扰、功率30 dBm，使用枚举法进行切片RB分配与并发接入选择，
最大化总体服务质量（URLLC+eMBB+mMTC）。

数据来源（相对路径，基于本脚本所在目录）：
- 任务量：../../题目/附件/附件1/q1_任务流.csv （单位：Mbit）
- 大规模衰减：../../题目/附件/附件1/q1_大规模衰减.csv （单位：dB）
- 小规模瑞利：../../题目/附件/附件1/q1_小规模瑞丽衰减.csv （幅度 |h|）

假设与参数：
- p_tx = 30 dBm，b = 360 kHz，NF = 7 dB。
- v_U=10, v_E=5, v_M=2（每类用户并发占用RB数）。
- SLA：URLLC: L<=5 ms；eMBB: L<=100 ms 且 r>=50 Mbps；mMTC: L<=500 ms。
- QoS：
  URLLC: y = alpha^L(ms) if L<=5ms else -M_U；alpha=0.95, M_U=5
  eMBB:  y = 1 if (L<=100ms & r>=50Mbps)；y = r/50 if (L<=100ms & r<50Mbps)；else -M_E, M_E=3
  mMTC:  y = (#接入且满足L<=500ms)/(#任务存在的mMTC总数)；若选择集合中任何一个超过500ms，可不选择该用户避免惩罚。

实现要点：
- 枚举(R_U, R_E, R_M)且满足R_U+R_E+R_M=50，且R_U%10==0, R_E%5==0, R_M%2==0以避免RB浪费。
- 对每类在容量上限内选择子集，使得对应QoS贡献最大（URLLC/eMBB按个体QoS排序取前若干个正收益者；mMTC取满足SLA的前若干个）。
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


def urllc_qos(user: User, num_rbs: int) -> float:
    L_ms = delay_ms(user, num_rbs)
    if L_ms <= SLA_L_U_MS:
        return ALPHA ** L_ms
    return -M_U


def embb_qos(user: User, num_rbs: int) -> float:
    L_ms = delay_ms(user, num_rbs)
    r_bps = user_rate_bps(user, num_rbs)
    r_mbps = r_bps / 1e6
    if L_ms <= SLA_L_E_MS and r_mbps >= SLA_R_E_MBPS:
        return 1.0
    if L_ms <= SLA_L_E_MS and r_mbps < SLA_R_E_MBPS:
        return max(0.0, r_mbps / SLA_R_E_MBPS)
    return -M_E


def mm_tc_success(user: User, num_rbs: int) -> bool:
    L_ms = delay_ms(user, num_rbs)
    return L_ms <= SLA_L_M_MS


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

            cap_U = min(len(U), R_U // V_U)
            cap_E = min(len(E), R_E // V_E)
            cap_M = min(len(M), R_M // V_M)

            # 逐个用户计算候选QoS
            U_scores: List[Tuple[str, float]] = [(u.name, urllc_qos(u, V_U)) for u in U]
            E_scores: List[Tuple[str, float]] = [(e.name, embb_qos(e, V_E)) for e in E]

            # mMTC：只统计满足SLA的用户用于接入
            M_success: List[Tuple[str, bool]] = [(m.name, mm_tc_success(m, V_M)) for m in M]

            sel_U = choose_best_subset(U_scores, cap_U)
            sel_E = choose_best_subset(E_scores, cap_E)
            # mMTC 选择：优先选择满足成功条件者
            m_ok = [name for name, ok in M_success if ok]
            sel_M = m_ok[:cap_M]

            # 计算目标：sum(URLLC y)+sum(eMBB y)+ y^M
            sum_U = sum(score for name, score in U_scores if name in sel_U)
            sum_E = sum(score for name, score in E_scores if name in sel_E)

            denom_M = sum(1 for m in M if m.data_mbit > 0.0)
            num_M = len(sel_M)
            y_M = (num_M / denom_M) if denom_M > 0 else 0.0

            obj = sum_U + sum_E + y_M

            if obj > best["obj"]:
                # 保存详情
                details = {}
                for u in U:
                    L_ms = delay_ms(u, V_U)
                    r_mbps = user_rate_bps(u, V_U) / 1e6
                    details[u.name] = {
                        "selected": u.name in sel_U,
                        "L_ms": L_ms,
                        "r_Mbps": r_mbps,
                        "qos": urllc_qos(u, V_U),
                    }
                for e in E:
                    L_ms = delay_ms(e, V_E)
                    r_mbps = user_rate_bps(e, V_E) / 1e6
                    details[e.name] = {
                        "selected": e.name in sel_E,
                        "L_ms": L_ms,
                        "r_Mbps": r_mbps,
                        "qos": embb_qos(e, V_E),
                    }
                for m in M:
                    L_ms = delay_ms(m, V_M)
                    r_mbps = user_rate_bps(m, V_M) / 1e6
                    details[m.name] = {
                        "selected": m.name in sel_M,
                        "L_ms": L_ms,
                        "r_Mbps": r_mbps,
                        "success": mm_tc_success(m, V_M),
                    }

                best.update(
                    {
                        "obj": obj,
                        "R": (R_U, R_E, R_M),
                        "sel_U": sel_U,
                        "sel_E": sel_E,
                        "sel_M": sel_M,
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


