#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import csv
import math
import os
from dataclasses import dataclass
from typing import Dict, List

# 系统常量
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
ATTACH_DIR = os.path.join(ROOT_DIR, "题目", "附件", "附件1")

CSV_TASK = os.path.join(ATTACH_DIR, "q1_任务流.csv")
CSV_PL = os.path.join(ATTACH_DIR, "q1_大规模衰减.csv")
CSV_RAY = os.path.join(ATTACH_DIR, "q1_小规模瑞丽衰减.csv")

P_TX_DBM = 20.0
B_HZ = 360_000.0
NF_DB = 7.0
V_U, V_E, V_M = 10, 5, 2
ALPHA, M_U, M_E, M_M = 0.95, 5.0, 3.0, 1.0
SLA_L_U_MS, SLA_L_E_MS, SLA_L_M_MS = 5.0, 100.0, 500.0
SLA_R_E_MBPS = 50.0

@dataclass
class User:
    name: str
    category: str
    data_mbit: float
    pl_db: float
    h_abs: float

def load_csv_users() -> List[User]:
    """加载CSV并构建用户最小所需信息"""
    with open(CSV_TASK, "r", encoding="utf-8") as f:
        task_reader = csv.DictReader(f)
        tasks = {row["用户"]: float(row["任务数据量(Mbit)"]) for row in task_reader}

    phi_db: Dict[str, float] = {}
    with open(CSV_PL, "r", encoding="utf-8") as f:
        pl_reader = csv.DictReader(f)
        for row in pl_reader:
            phi_db[row["用户"]] = float(row["衰减(dB)"])

    h_raw: Dict[str, float] = {}
    with open(CSV_RAY, "r", encoding="utf-8") as f:
        ray_reader = csv.DictReader(f)
        for row in ray_reader:
            h_raw[row["用户"]] = float(row["衰减(dB)"])

    users: List[User] = []
    for name, data_mbit in tasks.items():
        category = "U" if name.startswith("U") else ("E" if name.startswith("e") else "M")
        pl = phi_db.get(name, 100.0)
        h_val = h_raw.get(name, 0.0)
        # CSV中瑞丽一般以dB给出（负值），取功率/幅度绝对值的近似
        h_abs = math.sqrt(10 ** (h_val / 10.0)) if h_val <= 0 else h_val
        users.append(User(name, category, data_mbit, pl, h_abs))
    return users

def dbm_to_mw(dbm: float) -> float:
    return 10 ** (dbm / 10.0)

def noise_power_mw(num_rbs: int) -> float:
    if num_rbs <= 0:
        return 1e-30
    n0_dbm = -174.0 + 10.0 * math.log10(num_rbs * B_HZ) + NF_DB
    return dbm_to_mw(n0_dbm)

def user_rate_bps(user: User, num_rbs: int) -> float:
    p_rx = 10 ** ((P_TX_DBM - user.pl_db) / 10.0) * (user.h_abs * user.h_abs)
    n0 = noise_power_mw(num_rbs)
    snr = p_rx / n0
    return num_rbs * B_HZ * math.log2(1.0 + snr)

def delay_ms(user: User, num_rbs: int) -> float:
    r_bps = user_rate_bps(user, num_rbs)
    if r_bps <= 0.0:
        return float("inf")
    return (user.data_mbit * 1e6) / r_bps * 1e3

def urllc_qos_from_L(L_ms: float) -> float:
    return ALPHA ** L_ms if L_ms <= SLA_L_U_MS else -M_U

def embb_qos_from_L_and_r(L_ms: float, r_bps: float) -> float:
    r_mbps = r_bps / 1e6
    if L_ms <= SLA_L_E_MS and r_mbps >= SLA_R_E_MBPS:
        return 1.0
    elif L_ms <= SLA_L_E_MS and r_mbps < SLA_R_E_MBPS:
        return max(0.0, r_mbps / SLA_R_E_MBPS)
    return -M_E

def schedule_slice(users: List[User], num_rbs_slice: int, per_user_rbs: int) -> Dict[str, Dict[str, float]]:
    """切片内串并行调度"""
    import heapq
    results = {}
    cap = num_rbs_slice // per_user_rbs if per_user_rbs > 0 else 0
    if cap <= 0:
        for u in users:
            r_bps = user_rate_bps(u, per_user_rbs) if per_user_rbs > 0 else 0.0
            results[u.name] = {"Q_ms": float("inf"), "T_ms": delay_ms(u, per_user_rbs) if per_user_rbs > 0 else float("inf"), "L_ms": float("inf"), "r_bps": r_bps}
        return results
    
    active_heap = []
    counter = 0
    idx = 0
    n = len(users)
    
    # 填满初始并发槽位
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

def enumerate_solution(users: List[User]) -> Dict[str, object]:
    """核心：枚举RB切片分配并计算目标值，返回最优方案"""
    def sort_key(u: User):
        pr = 0 if u.category == "U" else (1 if u.category == "E" else 2)
        num = int(u.name[1:]) if u.name[1:].isdigit() else 0
        return (pr, num)

    U = sorted([u for u in users if u.category == "U"], key=sort_key)
    E = sorted([u for u in users if u.category == "E"], key=sort_key)
    M = sorted([u for u in users if u.category == "M"], key=sort_key)

    best = {"obj": -1e18, "R": (0, 0, 0)}

    for R_U in range(0, 51, V_U):
        for R_E in range(0, 51 - R_U, V_E):
            R_M = 50 - R_U - R_E
            if R_M < 0 or R_M % V_M != 0:
                continue

            U_sched = schedule_slice(U, R_U, V_U)
            E_sched = schedule_slice(E, R_E, V_E)
            M_sched = schedule_slice(M, R_M, V_M)

            sum_U = sum(urllc_qos_from_L(U_sched[u.name]["L_ms"]) for u in U)
            sum_E = sum(embb_qos_from_L_and_r(E_sched[e.name]["L_ms"], E_sched[e.name]["r_bps"]) for e in E)

            denom_M = sum(1 for m in M if m.data_mbit > 0.0)
            num_success = sum(1 for m in M if m.data_mbit > 0.0 and M_sched[m.name]["L_ms"] <= SLA_L_M_MS)
            ratio = (num_success / denom_M) if denom_M > 0 else 0.0
            sum_M = sum((ratio if M_sched[m.name]["L_ms"] <= SLA_L_M_MS else -M_M) for m in M if m.data_mbit > 0.0)

            obj = sum_U + sum_E + sum_M
            if obj > best["obj"]:
                best["obj"] = obj
                best["R"] = (R_U, R_E, R_M)
    return best

def main():
    users = load_csv_users()
    result = enumerate_solution(users)
    R_U, R_E, R_M = result["R"]
    print(f"最优RB分配: R_U={R_U}, R_E={R_E}, R_M={R_M}")
    print(f"目标函数: {result['obj']:.4f}")

if __name__ == "__main__":
    main()
