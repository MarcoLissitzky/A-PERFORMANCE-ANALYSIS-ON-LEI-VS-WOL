# ============================================================================
# leicester_extensions.py
# 扩展模块：上场时间归一化 · 吸收态马尔科夫链 · xG期望威胁 · 复合分析
# ============================================================================
# 依赖：pip install mplsoccer networkx pandas numpy matplotlib scipy
# 用法：在原有 PassingNetwork 和 MarkovChain 之后 import 本文件即可

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.lines import Line2D
from mplsoccer import Pitch
from scipy.linalg import inv as scipy_inv
import warnings
warnings.filterwarnings('ignore')

# 导入基类
from Graph import PassingNetwork, MarkovChain

import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei',
                                           'Noto Sans CJK SC', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False   # 解决负号显示问题


# ============================================================================
# PART 1 — 上场时间推断 + Per-90 归一化
# ============================================================================

class PlayingTimeAnalyzer:
    """
    从原始事件 CSV 推断每位球员的上场时间，
    并将传球数据归一化到 Per-90 分钟。

    设计思路（为什么不写进 PassingNetwork？）：
        单一职责原则 —— 上场时间分析是独立的功能模块，
        通过「组合」而非「修改」来扩展系统。

    用法:
        pta = PlayingTimeAnalyzer()
        pta.analyze(raw_csv="raw_events.csv", team="Leicester")
        normalized_df = pta.normalize_passes(pass_df)
    """

    def __init__(self):
        self.minutes_played = {}   # {player_name: minutes}
        self.sub_events     = []   # 换人事件列表
        self.match_length   = 90   # 比赛总时长（不含补时默认90）
        self.team           = None

    def analyze(self, raw_csv: str, team: str = "Leicester",
                match_length: int = 90):
        """
        从原始事件中推断每位球员的上场时间。

        逻辑:
        ┌─────────────────────────────────────────────────┐
        │  1. 找所有换人事件 (SubstitutionOn/Off)         │
        │  2. 对于被换下的球员：上场时间 = 换下分钟       │
        │  3. 对于被换上的球员：上场时间 = 终场 - 换上分钟│
        │  4. 对于全场球员：上场时间 = match_length        │
        │  5. 回退策略：用该球员的首末事件分钟差估算       │
        └─────────────────────────────────────────────────┘
        """
        self.team = team
        self.match_length = match_length

        print("=" * 65)
        print(f"  上场时间分析  ——  {team}")
        print("=" * 65)

        df = pd.read_csv(raw_csv)

        # 只看目标球队
        df_team = df[df['team'].str.contains(team, case=False, na=False)].copy()
        df_team['minute'] = pd.to_numeric(df_team['minute'], errors='coerce')
        df_team.dropna(subset=['minute'], inplace=True)

        # ---- 1. 提取换人事件 ----
        # WhoScored 的 type 列：'SubstitutionOn', 'SubstitutionOff'
        sub_on  = df_team[df_team['type'].str.contains(
            'SubstitutionOn|Sub.*On', case=False, na=False)]
        sub_off = df_team[df_team['type'].str.contains(
            'SubstitutionOff|Sub.*Off', case=False, na=False)]

        # 被换上的球员 → 从换上分钟踢到终场
        players_subbed_on = {}
        for _, row in sub_on.iterrows():
            name = row['player']
            minute = row['minute']
            players_subbed_on[name] = minute

        # 被换下的球员 → 从开场踢到换下分钟
        players_subbed_off = {}
        for _, row in sub_off.iterrows():
            name = row['player']
            minute = row['minute']
            players_subbed_off[name] = minute

        # ---- 2. 获取所有出场球员 ----
        all_players = df_team['player'].dropna().unique()

        # ---- 3. 计算上场时间 ----
        for player in all_players:
            if player in players_subbed_off:
                # 首发但被换下
                mins = players_subbed_off[player]
            elif player in players_subbed_on:
                # 替补上场
                mins = match_length - players_subbed_on[player]
            else:
                # 全场出场 —— 默认 match_length
                # 但用事件范围做 sanity check
                p_events = df_team[df_team['player'] == player]
                first_min = p_events['minute'].min()
                last_min  = p_events['minute'].max()

                if first_min > 60:
                    # 很可能是替补（事件中没有明确换人记录的兜底）
                    mins = match_length - first_min
                elif last_min < 50 and match_length > 50:
                    # 可能上半场就被换下
                    mins = last_min
                else:
                    mins = match_length

            # 最少给 1 分钟，防止除零
            self.minutes_played[player] = max(mins, 1)

        # ---- 4. 打印结果 ----
        print(f"\n  {'球员':<22} {'上场时间':>8} {'状态':>10}")
        print("  " + "-" * 44)

        for p in sorted(self.minutes_played,
                         key=self.minutes_played.get, reverse=True):
            mins = self.minutes_played[p]
            if p in players_subbed_off:
                status = f"换下@{players_subbed_off[p]}'"
            elif p in players_subbed_on:
                status = f"换上@{players_subbed_on[p]}'"
            else:
                status = "全场"
            print(f"  {p:<22} {mins:>6.0f} min  {status:>10}")

        print(f"\n  共 {len(self.minutes_played)} 名球员出场。\n")
        return self.minutes_played

    def normalize_passes(self, pass_df: pd.DataFrame,
                          target_minutes: float = 90.0):
        """
        将传球数据归一化到 Per-90 分钟。

        原理:
            如果球员只踢了 60 分钟传了 30 脚，
            Per-90 = 30 × (90/60) = 45 脚。

        实现方式:
            不改变原始 DataFrame 的行数（每行仍是一次传球），
            而是给每行一个「权重」列，用于后续构建加权图。

        Parameters
        ----------
        pass_df       : 原始成功传球 DataFrame（来自 PassingNetwork.pass_df）
        target_minutes: 归一化目标分钟数，默认 90

        Returns
        -------
        DataFrame : 添加了 'pass_weight' 列的新 DataFrame
        """
        print("=" * 65)
        print(f"  Per-{target_minutes:.0f} 归一化传球数据")
        print("=" * 65)

        df = pass_df.copy()

        # 给每个传球人算归一化系数
        def get_weight(player_name):
            mins = self.minutes_played.get(player_name, target_minutes)
            return target_minutes / mins

        df['passer_weight']   = df['player'].map(get_weight)
        df['receiver_weight'] = df['receiver'].map(get_weight)
        # 综合权重：取传球人和接球人的几何平均
        df['pass_weight'] = np.sqrt(df['passer_weight'] * df['receiver_weight'])

        # 展示归一化系数
        print(f"\n  {'球员':<22} {'原始分钟':>8} {'归一化系数':>10}")
        print("  " + "-" * 44)
        for p in sorted(self.minutes_played,
                         key=self.minutes_played.get, reverse=True):
            mins = self.minutes_played[p]
            w = target_minutes / mins
            marker = " ◀ 放大" if w > 1.3 else ""
            print(f"  {p:<22} {mins:>6.0f} min  ×{w:>7.2f}{marker}")

        print(f"\n  ✓ 已添加 'pass_weight' 列。\n")
        return df

    def build_normalized_pair_counts(self, pass_df_weighted: pd.DataFrame):
        """
        用归一化权重重新计算球员对之间的传球次数。

        Returns
        -------
        DataFrame : columns = ['player', 'receiver', 'count', 'count_raw']
        """
        df = pass_df_weighted.copy()

        # 原始计数
        raw = (df.groupby(['player', 'receiver'])
                 .size()
                 .reset_index(name='count_raw'))

        # 加权计数（Per-90）
        weighted = (df.groupby(['player', 'receiver'])['pass_weight']
                      .sum()
                      .reset_index(name='count'))

        merged = raw.merge(weighted, on=['player', 'receiver'])
        merged['count'] = merged['count'].round(1)

        return merged


# ============================================================================
# PART 2 — 简易 xG 模型（基于射门位置）
# ============================================================================

class SimpleXGModel:
    """
    基于射门位置的简易 xG 估算器。

    没有 StatsBomb 那样的精确 xG？没关系。
    用射门距离和角度拟合一个 Logistic 函数，
    足以支撑我们的期望威胁模型。

    数学原理:
        xG = 1 / (1 + exp(-(β₀ + β₁·distance + β₂·angle)))

        distance = 射门点到球门中心的距离
        angle    = 射门点看球门两柱的张角

    系数来源: 经典 xG 模型的近似值（Caley 2015 风格）
    """

    # Opta 坐标系：球场 100×100，球门中心 (100, 50)
    GOAL_X      = 100.0
    GOAL_Y      = 50.0
    # 球门宽度 7.32m，球场宽度 ≈ 68m → Opta 单位
    GOAL_WIDTH  = 7.32 / 68.0 * 100.0   # ≈ 10.76
    POST_LEFT   = GOAL_Y - GOAL_WIDTH / 2  # ≈ 44.6
    POST_RIGHT  = GOAL_Y + GOAL_WIDTH / 2  # ≈ 55.4

    # Logistic 回归系数（近似值）
    BETA_0 =  1.10   # 截距
    BETA_1 = -0.09   # 距离系数（越远 xG 越低）
    BETA_2 =  0.80   # 角度系数（角度越大 xG 越高）

    @classmethod
    def calculate_xg(cls, x: float, y: float) -> float:
        """
        计算单次射门的 xG。

        Parameters
        ----------
        x, y : 射门位置（Opta 坐标 0-100）

        Returns
        -------
        float : xG ∈ [0, 1]
        """
        # 到球门中心的距离（Opta 单位）
        dx = cls.GOAL_X - x
        dy = cls.GOAL_Y - y
        dist = np.sqrt(dx**2 + dy**2)

        # 射门角度（弧度）—— 射门点看两根门柱的张角
        # 用向量夹角公式
        vec_left  = np.array([cls.GOAL_X - x, cls.POST_LEFT - y])
        vec_right = np.array([cls.GOAL_X - x, cls.POST_RIGHT - y])

        cos_angle = (np.dot(vec_left, vec_right) /
                     (np.linalg.norm(vec_left) * np.linalg.norm(vec_right)
                      + 1e-10))
        cos_angle = np.clip(cos_angle, -1, 1)
        angle = np.arccos(cos_angle)   # 弧度

        # Logistic 函数
        z = cls.BETA_0 + cls.BETA_1 * dist + cls.BETA_2 * angle
        xg = 1.0 / (1.0 + np.exp(-z))

        # 常识修正：距离 > 40 Opta 单位（约 27 米）的远射 xG 压低
        if dist > 40:
            xg *= 0.5
        # 极近距离（门前 < 8 Opta 单位）提升
        if dist < 8:
            xg = min(xg * 1.3, 0.95)

        return round(xg, 4)

    @classmethod
    def batch_calculate(cls, shot_df: pd.DataFrame,
                         x_col: str = 'x', y_col: str = 'y') -> pd.Series:
        """批量计算 xG。"""
        return shot_df.apply(
            lambda row: cls.calculate_xg(row[x_col], row[y_col]), axis=1)


# ============================================================================
# PART 3 — 吸收态马尔科夫链 + 期望威胁模型（双吸收态：射门 + 丢球）
# ============================================================================

class AbsorbingMarkovChain:
    """
    将传球网络建模为带「射门」和「丢球」双吸收态的马尔科夫链。

    ┌──────────────────────────────────────────────────────────────┐
    │  为什么必须加丢球吸收态？                                    │
    │                                                              │
    │  只有射门一个出口时：                                        │
    │    Q 的每一行几乎和为 1（只有射门球员的行略小于 1）          │
    │    → N = (I-Q)⁻¹ 的所有元素都很大且接近                     │
    │    → xT = N @ R 对所有球员几乎一样 → 无异质性               │
    │                                                              │
    │  加入丢球后：                                                │
    │    每个球员的出球 = 传球 + 射门 + 丢球                       │
    │    → Q 的行和显著小于 1（概率被两个吸收态分走）              │
    │    → N 的元素出现分化                                        │
    │    → 经常丢球的球员 → 球到他那里容易"死掉" → 低 xT         │
    │    → 高效传球+射门的球员 → 球到他那里能产出 → 高 xT         │
    │                                                              │
    │  矩阵结构（canonical form）:                                 │
    │                                                              │
    │             球员₁ ··· 球员ₙ │ Shot  Turnover                 │
    │    球员₁  [                 │                 ]               │
    │      :    [      Q          │   R_s    R_t    ]               │
    │    球员ₙ  [                 │                 ]               │
    │    ────────┼────────────────┼─────────────────                │
    │    Shot    [      0         │    1       0    ]               │
    │    Turnov  [      0         │    0       1    ]               │
    │                                                              │
    │    Q   = n×n  球员间转移概率                                 │
    │    R_s = n×1  射门概率（或 xG 加权）                         │
    │    R_t = n×1  丢球概率                                       │
    │    行约束: Q[i,:].sum() + R_s[i] + R_t[i] = 1               │
    │                                                              │
    │    N  = (I - Q)⁻¹                                            │
    │    xT = N @ R_s   ← 期望进球贡献（丢球贡献 = 0）            │
    │    pT = N @ R_t   ← 期望丢球概率（诊断用）                  │
    │                                                              │
    └──────────────────────────────────────────────────────────────┘
    """

    # 射门事件类型（来自你的数据集）
    SHOT_TYPES = ['Goal', 'SavedShot', 'ShotOnPost', 'MissedShots']

    # 丢球事件类型（来自你的数据集）
    TURNOVER_TYPES = [
        'Dispossessed',   # 被断球
        'Error',          # 失误
        'BlockedPass',    # 传球被封堵
        'OffsidePass',    # 越位传球
    ]

    def __init__(self, raw_csv: str, pass_df: pd.DataFrame,
                 team: str = "Leicester",
                 pair_counts: pd.DataFrame = None,
                 minutes_played: dict = None,
                 use_xg_weighting: bool = True):

        self.team = team
        self.use_xg = use_xg_weighting
        self.minutes_played = minutes_played or {}

        print("=" * 65)
        print("  吸收态马尔科夫链 + 期望威胁模型（双吸收态）")
        print("=" * 65)

        # ---- 3.1 提取射门数据 ----
        self.shot_df = self._extract_shots(raw_csv, team)

        # ---- 3.2 提取丢球数据 ----
        self.turnover_df = self._extract_turnovers(raw_csv, team)

        # ---- 3.3 计算每位球员的射门/丢球概率 & xG ----
        self.shot_stats = self._compute_action_stats(pass_df)

        # ---- 3.4 构建双吸收态转移矩阵 ----
        self._build_absorbing_matrix(pass_df, pair_counts)

        # ---- 3.5 高斯消元（求逆）得到基础矩阵 N ----
        self._compute_fundamental_matrix()

        # ---- 3.6 计算期望威胁 xT + 期望丢球概率 pT ----
        self._compute_expected_threat()

        # ---- 3.7 打印结果 ----
        self._print_results()

    # ----------------------------------------------------------------
    def _extract_shots(self, raw_csv: str, team: str) -> pd.DataFrame:
        """从原始事件中提取射门数据。"""
        print("\n  ▸ 提取射门事件...")

        df = pd.read_csv(raw_csv)
        df_team = df[df['team'].str.contains(team, case=False, na=False)].copy()

        mask = df_team['type'].isin(self.SHOT_TYPES)
        shots = df_team[mask].copy()

        # 清洗坐标
        for c in ['x', 'y']:
            shots[c] = pd.to_numeric(shots[c], errors='coerce')
        shots.dropna(subset=['x', 'y'], inplace=True)

        # 计算 xG
        if len(shots) > 0:
            shots['xG'] = SimpleXGModel.batch_calculate(shots)
            shots['is_goal'] = shots['type'].str.contains(
                'Goal', case=False, na=False).astype(int)
        else:
            shots['xG'] = []
            shots['is_goal'] = []

        print(f"    射门总数: {len(shots)}")
        if len(shots) > 0:
            print(f"    总 xG   : {shots['xG'].sum():.3f}")
            print(f"    进球数  : {shots['is_goal'].sum()}")

            per_player = shots.groupby('player').agg(
                shots=('xG', 'count'),
                total_xG=('xG', 'sum'),
                goals=('is_goal', 'sum')
            ).sort_values('total_xG', ascending=False)

            print(f"\n    {'球员':<22} {'射门':>4} {'xG':>7} {'进球':>4}")
            print("    " + "-" * 40)
            for p, row in per_player.iterrows():
                print(f"    {p:<22} {row['shots']:>4} "
                      f"{row['total_xG']:>7.3f} {row['goals']:>4}")

        return shots

    # ----------------------------------------------------------------
    def _extract_turnovers(self, raw_csv: str, team: str) -> pd.DataFrame:
        """
        从原始事件中提取丢球数据。

        丢球 = 控球回合终止（非射门方式），包括：
            · Dispossessed  — 被对方断球
            · Error         — 失误送球
            · BlockedPass   — 传球被封堵
            · OffsidePass   — 越位传球
        """
        print("\n  ▸ 提取丢球事件...")

        df = pd.read_csv(raw_csv)
        df_team = df[df['team'].str.contains(team, case=False, na=False)].copy()

        mask = df_team['type'].isin(self.TURNOVER_TYPES)
        turnovers = df_team[mask].copy()

        print(f"    丢球总数: {len(turnovers)}")

        if len(turnovers) > 0:
            # 按类型统计
            type_counts = turnovers['type'].value_counts()
            print(f"\n    丢球类型分布:")
            for t, cnt in type_counts.items():
                print(f"      {t:<20} {cnt:>4}")

            # 按球员统计
            per_player = turnovers.groupby('player').size().sort_values(
                ascending=False).reset_index(name='turnovers')

            print(f"\n    {'球员':<22} {'丢球次数':>8}")
            print("    " + "-" * 32)
            for _, row in per_player.iterrows():
                print(f"    {row['player']:<22} {row['turnovers']:>8}")

        return turnovers

    # ----------------------------------------------------------------
    def _compute_action_stats(self, pass_df: pd.DataFrame) -> dict:
        """
        计算每位球员的 射门概率 + 丢球概率。

        总动作 = 传球(作为传球人) + 射门 + 丢球

        Returns
        -------
        dict : {player: {'passes_out', 'shots', 'total_xG',
                          'turnovers', 'total_actions',
                          'shot_prob', 'turnover_prob',
                          'xg_weighted_prob'}}
        """
        print("\n  ▸ 计算射门/丢球概率向量...")

        # 传球次数（作为传球人）
        passer_counts = pass_df['player'].value_counts().to_dict()

        # 射门
        shot_counts = {}
        shot_xg = {}
        if len(self.shot_df) > 0:
            for player, group in self.shot_df.groupby('player'):
                shot_counts[player] = len(group)
                shot_xg[player] = group['xG'].sum()

        # 丢球
        turnover_counts = {}
        if len(self.turnover_df) > 0:
            for player, group in self.turnover_df.groupby('player'):
                turnover_counts[player] = len(group)

        # 合并
        all_players = (set(passer_counts.keys()) |
                       set(shot_counts.keys()) |
                       set(turnover_counts.keys()))
        stats = {}

        for p in all_players:
            passes = passer_counts.get(p, 0)
            shots = shot_counts.get(p, 0)
            xg_sum = shot_xg.get(p, 0.0)
            to_cnt = turnover_counts.get(p, 0)
            total = passes + shots + to_cnt

            if total > 0:
                shot_prob = shots / total
                turnover_prob = to_cnt / total
                xg_weighted = xg_sum / total
            else:
                shot_prob = 0.0
                turnover_prob = 0.0
                xg_weighted = 0.0

            stats[p] = {
                'passes_out':       passes,
                'shots':            shots,
                'total_xG':         xg_sum,
                'turnovers':        to_cnt,
                'total_actions':    total,
                'shot_prob':        shot_prob,
                'turnover_prob':    turnover_prob,
                'xg_weighted_prob': xg_weighted,
            }

        # 打印汇总
        print(f"\n    {'球员':<22} {'传球':>5} {'射门':>4} {'丢球':>4} "
              f"{'P(射门)':>8} {'P(丢球)':>8}")
        print("    " + "-" * 56)
        for p in sorted(stats, key=lambda x: stats[x]['total_actions'],
                        reverse=True):
            s = stats[p]
            print(f"    {p:<22} {s['passes_out']:>5} {s['shots']:>4} "
                  f"{s['turnovers']:>4} {s['shot_prob']:>8.3f} "
                  f"{s['turnover_prob']:>8.3f}")

        return stats

    # ----------------------------------------------------------------
    def _build_absorbing_matrix(self, pass_df: pd.DataFrame,
                                 pair_counts: pd.DataFrame = None):
        """
        构建双吸收态转移矩阵。

        每行归一化:  传球概率 + P(射门) + P(丢球) = 1
        """
        print("\n  ▸ 构建双吸收态转移矩阵...")

        if pair_counts is None:
            pair_counts = (pass_df.groupby(['player', 'receiver'])
                           .size()
                           .reset_index(name='count'))

        # 确定球员列表
        all_in_passes = set(pair_counts['player']) | set(pair_counts['receiver'])
        all_in_stats = set(self.shot_stats.keys())
        self.players = sorted(all_in_passes | all_in_stats)
        self.n = len(self.players)
        self.player_to_idx = {p: i for i, p in enumerate(self.players)}

        print(f"    瞬态状态数: {self.n} (球员)")
        print(f"    吸收态: 2 (射门 + 丢球)")

        # ---- 传球计数矩阵 ----
        count_matrix = np.zeros((self.n, self.n))
        for _, row in pair_counts.iterrows():
            i = self.player_to_idx.get(row['player'])
            j = self.player_to_idx.get(row['receiver'])
            if i is not None and j is not None:
                count_matrix[i, j] = row['count']

        # ---- 射门 & 丢球计数向量 ----
        shot_counts_vec = np.zeros(self.n)
        xg_vec = np.zeros(self.n)
        turnover_counts_vec = np.zeros(self.n)

        for p, stats in self.shot_stats.items():
            idx = self.player_to_idx.get(p)
            if idx is not None:
                shot_counts_vec[idx] = stats['shots']
                xg_vec[idx] = stats['total_xG']
                turnover_counts_vec[idx] = stats['turnovers']

        # ---- 行归一化 ----
        # 总出球 = 传队友 + 射门 + 丢球
        row_totals = (count_matrix.sum(axis=1) +
                      shot_counts_vec +
                      turnover_counts_vec)
        row_totals[row_totals == 0] = 1  # 防除零

        # Q: 球员间转移概率
        self.Q = count_matrix / row_totals[:, np.newaxis]

        # R_shot: 射门吸收概率
        if self.use_xg:
            self.R_shot = xg_vec / row_totals
            print("    射门概率向量: xG 加权 ✓")
        else:
            self.R_shot = shot_counts_vec / row_totals
            print("    射门概率向量: 纯频率")

        # R_turnover: 丢球吸收概率
        self.R_turnover = turnover_counts_vec / row_totals

        # 向后兼容：self.R 指向 R_shot
        self.R = self.R_shot

        # 存储原始数据
        self.count_matrix = count_matrix
        self.shot_counts_vec = shot_counts_vec
        self.xg_vec = xg_vec
        self.turnover_counts_vec = turnover_counts_vec

        # ---- 校验 ----
        if not self.use_xg:
            row_check = self.Q.sum(axis=1) + self.R_shot + self.R_turnover
            deviation = np.abs(row_check - 1.0).max()
            print(f"    行和校验（最大偏差）: {deviation:.6f}")
        else:
            # xG 模式下 R_shot 用 xG 替代了频率，行和不严格为 1
            row_pass_turn = self.Q.sum(axis=1) + self.R_turnover
            remaining = 1.0 - row_pass_turn
            print(f"    Q行和 + R_turnover 范围: "
                  f"[{row_pass_turn.min():.4f}, {row_pass_turn.max():.4f}]")
            print(f"    留给射门的概率空间: "
                  f"[{remaining.min():.4f}, {remaining.max():.4f}]")

        # ---- 打印行和分解（核心诊断） ----
        print(f"\n    行和分解（每个球员的概率去向）:")
        print(f"    {'球员':<22} {'P(传球)':>8} {'P(射门)':>8} "
              f"{'P(丢球)':>8} {'合计':>6}")
        print("    " + "-" * 56)
        for i in range(self.n):
            p_pass = self.Q[i, :].sum()
            p_shot = self.R_shot[i]
            p_turn = self.R_turnover[i]
            total = p_pass + p_shot + p_turn
            flag = ""
            if p_turn > 0.3:
                flag = " ← 高丢球风险"
            elif p_shot > 0.1:
                flag = " ← 高射门率"
            print(f"    {self.players[i]:<22} {p_pass:>8.3f} {p_shot:>8.4f} "
                  f"{p_turn:>8.3f} {total:>6.3f}{flag}")

    # ----------------------------------------------------------------
    def _compute_fundamental_matrix(self):
        """
        高斯消元求基础矩阵 N = (I - Q)⁻¹

        现在 Q 的行和显著 < 1（概率被射门+丢球分走），
        所以 (I-Q) 有更好的条件数，N 的元素也更有区分度。
        """
        print("\n  ▸ 高斯消元求基础矩阵 N = (I - Q)⁻¹ ...")

        I = np.eye(self.n)
        I_minus_Q = I - self.Q

        det = np.linalg.det(I_minus_Q)
        cond = np.linalg.cond(I_minus_Q)
        print(f"    det(I - Q) = {det:.6f}", end="")

        if abs(det) < 1e-12:
            print(" ← 接近奇异！使用伪逆。")
            self.N = np.linalg.pinv(I_minus_Q)
        else:
            print(" ← 可逆 ✓")
            self.N = scipy_inv(I_minus_Q)

        print(f"    条件数 cond(I-Q) = {cond:.2f}")
        print(f"    基础矩阵 N: {self.n}×{self.n}")
        print(f"    N 元素范围: [{self.N.min():.4f}, {self.N.max():.4f}]")
        print(f"    N 对角线范围: [{self.N.diagonal().min():.4f}, "
              f"{self.N.diagonal().max():.4f}]")

        # ---- 期望吸收步数: t = N @ 1 ----
        # 这是到「任一吸收态」的期望步数（射门或丢球）
        self.expected_steps_to_absorption = self.N @ np.ones(self.n)

        # 向后兼容
        self.expected_steps_to_shot = self.expected_steps_to_absorption

        print(f"\n    期望吸收步数（球从该球员出发到被吸收的平均传球脚数）:")
        print(f"    {'球员':<22} {'E[步数]':>8}")
        print("    " + "-" * 32)
        order = np.argsort(self.expected_steps_to_absorption)
        for idx in order:
            p = self.players[idx]
            steps = self.expected_steps_to_absorption[idx]
            print(f"    {p:<22} {steps:>8.2f}")

    # ----------------------------------------------------------------
    def _compute_expected_threat(self):
        """
        期望威胁与期望丢球概率:

            xT = N @ R_shot      → 球从 i 出发的期望进球贡献
            pT = N @ R_turnover  → 球从 i 出发的丢球概率

        xT + pT ≈ 1（在非 xG 模式下严格等于 1）

        异质性来源:
            · 丢球多的球员 → pT 高 → xT 低
            · 射门多/效率高的球员 → R_shot 大 → xT 高
            · 连接射手的球员 → N 行传导到射手 → xT 间接高
        """
        print("\n  ▸ 计算期望威胁 xT = N @ R_shot ...")
        print("  ▸ 计算丢球概率 pT = N @ R_turnover ...")

        self.xT = self.N @ self.R_shot
        self.pT = self.N @ self.R_turnover

        # 净效率：xT / (xT + pT)，衡量"球到这个人脚下有多大概率变成进球而非丢球"
        self.efficiency = np.where(
            (self.xT + self.pT) > 1e-10,
            self.xT / (self.xT + self.pT),
            0.0
        )

        print(f"\n    {'球员':<22} {'xT':>10} {'pT':>10} "
              f"{'效率':>8} {'解读'}")
        print("    " + "-" * 70)
        order = np.argsort(-self.xT)
        max_xt = self.xT.max() if self.xT.max() > 0 else 1

        for idx in order:
            p = self.players[idx]
            xt = self.xT[idx]
            pt = self.pT[idx]
            eff = self.efficiency[idx]
            bar = "█" * int(xt / max_xt * 15)
            bar_t = "░" * int(pt * 10)
            print(f"    {p:<22} {xt:>10.4f} {pt:>10.4f} "
                  f"{eff:>7.1%}  {bar}{bar_t}")

    # ----------------------------------------------------------------
    def _print_results(self):
        """汇总打印。"""
        print(f"\n{'=' * 65}")
        print(f"  期望威胁模型汇总（双吸收态）")
        print(f"{'=' * 65}")

        total_xt = self.xT.sum()
        avg_eff = self.efficiency.mean()
        max_pt_idx = np.argmax(self.pT)
        max_xt_idx = np.argmax(self.xT)

        print(f"\n  模式: {'xG 加权' if self.use_xg else '纯频率'}吸收态马尔科夫链")
        print(f"  吸收态: 射门 + 丢球（双吸收态）")
        print(f"\n  球队 xT 总和:    {total_xt:.4f}")
        print(f"  平均进攻效率:    {avg_eff:.1%}")
        print(f"  最高 xT 球员:    {self.players[max_xt_idx]} "
              f"({self.xT[max_xt_idx]:.4f})")
        print(f"  最高丢球率球员:  {self.players[max_pt_idx]} "
              f"({self.pT[max_pt_idx]:.4f})")

        # 异质性诊断
        xt_cv = np.std(self.xT) / (np.mean(self.xT) + 1e-10)
        print(f"\n  xT 变异系数 (CV): {xt_cv:.3f}", end="")
        if xt_cv > 0.5:
            print("  ← 高异质性 ✓ 球员间差异显著")
        elif xt_cv > 0.2:
            print("  ← 中等异质性")
        else:
            print("  ← 低异质性 ⚠ 球员间差异不大")
        print()

    # ----------------------------------------------------------------
    # 公开接口
    # ----------------------------------------------------------------
    def get_xT_dict(self) -> dict:
        """返回 {player: xT} 字典。"""
        return {self.players[i]: self.xT[i] for i in range(self.n)}

    def get_pT_dict(self) -> dict:
        """返回 {player: pT (丢球概率)} 字典。"""
        return {self.players[i]: self.pT[i] for i in range(self.n)}

    def get_efficiency_dict(self) -> dict:
        """返回 {player: efficiency} 字典。"""
        return {self.players[i]: self.efficiency[i] for i in range(self.n)}

    def get_fundamental_matrix(self) -> pd.DataFrame:
        """返回基础矩阵 N 作为 DataFrame。"""
        short = [p.split()[-1] for p in self.players]
        return pd.DataFrame(self.N, index=short, columns=short)

    def expected_steps_between(self, from_player: str, to_player: str):
        """
        利用基础矩阵计算任意两人之间的期望传球步数。
        公式: E[从 i 到 j 的步数] ≈ N[i,j] / N[j,j]
        """
        i = self._fuzzy_match(from_player)
        j = self._fuzzy_match(to_player)
        if i is None or j is None:
            return np.inf
        if self.N[j, j] > 1e-10:
            return self.N[i, j] / self.N[j, j]
        return np.inf

    def _fuzzy_match(self, name: str):
        """模糊匹配球员名。"""
        matches = [i for i, p in enumerate(self.players)
                   if name.lower() in p.lower()]
        if matches:
            return matches[0]
        print(f"    [!] 找不到包含 '{name}' 的球员")
        return None

    # ----------------------------------------------------------------
    # 可视化
    # ----------------------------------------------------------------
    def plot_expected_threat(self, positions: dict = None,
                              save_path: str = None):
        """
        在球场上绘制期望威胁图 + xT/pT 对比柱状图。
        """
        fig = plt.figure(figsize=(22, 10), facecolor='#0E1117')
        BG = '#0E1117'

        # ========================
        # 左图: 球场上的 xT 气泡图
        # ========================
        ax1 = fig.add_axes([0.02, 0.05, 0.48, 0.85])
        pitch = Pitch(pitch_type='opta', pitch_color=BG,
                      line_color='#2a3a4a', linewidth=1)
        pitch.draw(ax=ax1)
        ax1.set_title('Expected Threat (xT) on Pitch\n'
                       'large=xT · color=efficiency',
                       color='white', fontsize=14, fontweight='bold', pad=10)

        if positions is None:
            print("  [!] 需要传入 positions 字典（来自 PassingNetwork.positions）")
            print("      使用默认均匀分布。")
            angles_pos = np.linspace(0, 2 * np.pi, self.n, endpoint=False)
            positions = {p: (50 + 30 * np.cos(a), 50 + 30 * np.sin(a))
                         for p, a in zip(self.players, angles_pos)}

        xt_dict = self.get_xT_dict()
        eff_dict = self.get_efficiency_dict()
        max_xt = max(xt_dict.values()) if xt_dict else 1

        cmap_fill = LinearSegmentedColormap.from_list(
            'xt_cmap', ['#1a1a2e', '#00C8FF', '#FFD700', '#FF4444'])
        norm_fill = Normalize(vmin=0, vmax=max_xt)

        cmap_edge = LinearSegmentedColormap.from_list(
            'eff_cmap', ['#FF4444', '#FFD700', '#00FF88'])
        norm_edge = Normalize(vmin=0, vmax=1)

        for player, (px, py) in positions.items():
            if player not in xt_dict:
                continue
            xt = xt_dict[player]
            eff = eff_dict.get(player, 0.5)
            size = 300 + (xt / max_xt) * 2500
            fill_color = cmap_fill(norm_fill(xt))
            edge_color = cmap_edge(norm_edge(eff))

            ax1.scatter(px, py, s=size, c=[fill_color],
                        edgecolors=[edge_color],
                        linewidths=3, zorder=5, alpha=0.9)

            short = player.split()[-1]
            txt = ax1.text(px, py + 0.5, short, color='white',
                           fontsize=8, ha='center', va='bottom',
                           fontweight='bold', zorder=6)
            txt.set_path_effects([
                path_effects.Stroke(linewidth=2.5, foreground='black'),
                path_effects.Normal()])

            val_txt = ax1.text(px, py - 4, f'xT={xt:.3f}', color='#FFD700',
                               fontsize=7, ha='center', va='top', zorder=6)
            val_txt.set_path_effects([
                path_effects.Stroke(linewidth=2, foreground='black'),
                path_effects.Normal()])

            eff_txt = ax1.text(px, py - 7, f'eff={eff:.0%}',
                               color=edge_color,
                               fontsize=6.5, ha='center', va='top', zorder=6)
            eff_txt.set_path_effects([
                path_effects.Stroke(linewidth=1.5, foreground='black'),
                path_effects.Normal()])

        sm = plt.cm.ScalarMappable(cmap=cmap_fill, norm=norm_fill)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax1, fraction=0.03, pad=0.02,
                             location='bottom', aspect=40)
        cbar.set_label('Expected Threat (xT)', color='white', fontsize=10)
        cbar.ax.tick_params(colors='white', labelsize=8)

        # ========================
        # 右图: xT vs pT 水平堆叠柱状图
        # ========================
        ax2 = fig.add_axes([0.55, 0.05, 0.42, 0.85])
        ax2.set_facecolor(BG)

        # 按 xT 降序排列
        order = np.argsort(-self.xT)
        names = [self.players[i].split()[-1] for i in order]
        xt_vals = [self.xT[i] for i in order]
        pt_vals = [self.pT[i] for i in order]
        eff_vals = [self.efficiency[i] for i in order]

        y_pos = np.arange(len(names))

        # 堆叠柱状图：xT（金色）+ pT（红色）
        bars_xt = ax2.barh(y_pos, xt_vals, height=0.6,
                            color='#FFD700', alpha=0.9, label='xT (expected Threat)')
        bars_pt = ax2.barh(y_pos, pt_vals, height=0.6,
                            left=xt_vals, color='#FF4444', alpha=0.7,
                            label='pT (turnover probability)')

        # 在柱子右边标注效率
        for i, (xt, pt, eff) in enumerate(zip(xt_vals, pt_vals, eff_vals)):
            ax2.text(xt + pt + 0.005, i,
                     f'{eff:.0%}',
                     color='#00FF88', fontsize=8,
                     ha='left', va='center', fontweight='bold')

        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(names, color='white', fontsize=10)
        ax2.invert_yaxis()

        ax2.set_xlabel('Probability (xT + pT ≈ 1)', color='white', fontsize=11)
        ax2.set_title('Expected Threat vs Turnover Probability\n'
                       'AttackEfficiency xT/(xT+pT)',
                       color='white', fontsize=14, fontweight='bold', pad=15)

        ax2.tick_params(colors='white')
        for spine in ax2.spines.values():
            spine.set_color('#2a3a4a')

        lines = [Line2D([0], [0], color='#FFD700', lw=8, alpha=0.9),
                 Line2D([0], [0], color='#FF4444', lw=8, alpha=0.7),
                 Line2D([0], [0], color='#00FF88', lw=0,
                        marker='$eff$', markersize=12)]
        ax2.legend(lines, ['xT (Expected Threat)', 'pT (Turnover Prob)',
                           'Efficiency'],
                   loc='lower right', facecolor=BG, edgecolor='#2a3a4a',
                   labelcolor='white', fontsize=9)

        if save_path:
            plt.savefig(save_path, dpi=150, facecolor=BG, bbox_inches='tight')
            print(f"\n  ✓ 已保存至 {save_path}")
        plt.show()



# ============================================================================
# PART 4 — 复合分析：度中心性 × 期望威胁 × 战术角色画像
# ============================================================================

class TacticalCompositeAnalyzer:
    """
    将图论指标（度中心性、中介中心性、聚集系数）与期望威胁模型复合，
    生成「战术角色画像」。

    核心输出：
        1. 四象限散点图（度中心性 vs xT）
        2. 雷达图（多维度球员画像）
        3. 战术报告（文字）

    四象限解读:
    ┌──────────────────────────────────────────────────┐
    │              高 xT                               │
    │                │                                 │
    │   「高效射手」   │   「核心创造者」                │
    │   低中心性       │   高中心性                     │
    │   高威胁         │   高威胁                       │
    │   → 不常拿球     │   → 频繁拿球                  │
    │     但一拿就致命  │     且每次都能推进威胁         │
    │                │                                 │
    │   ─────────────┼─────────── 高度中心性           │
    │                │                                 │
    │   「边缘球员」   │   「空转枢纽」                  │
    │   低中心性       │   高中心性                     │
    │   低威胁         │   低威胁                       │
    │   → 基本隐身     │   → 横传回传多                 │
    │                │     但对进攻无实质帮助           │
    │              低 xT                               │
    └──────────────────────────────────────────────────┘

    用法:
        comp = TacticalCompositeAnalyzer(net, absorbing_mc)
        comp.plot_quadrant()
        comp.plot_radar(["Vardy", "Ndidi", "Buonanotte"])
        comp.print_tactical_report()
    """

    # 战术角色标签定义
    ROLE_LABELS = {
        'core_creator':    '核心创造者',
        'efficient_finisher': '高效射手',
        'idle_hub':        '空转枢纽',
        'peripheral':      '边缘球员',
    }

    def __init__(self, passing_network, absorbing_mc: AbsorbingMarkovChain):
        """
        Parameters
        ----------
        passing_network : PassingNetwork 实例（已调用 build_graph）
        absorbing_mc    : AbsorbingMarkovChain 实例
        """
        self.net = passing_network
        self.amc = absorbing_mc

        # 汇总全部指标
        self.metrics = self._collect_metrics()

    # ----------------------------------------------------------------
    # 内部：收集指标
    # ----------------------------------------------------------------
    def _collect_metrics(self) -> pd.DataFrame:
        """
        从 PassingNetwork 和 AbsorbingMarkovChain 中提取所有指标，
        合并为一张宽表。

        列:
            player, dc_all, dc_in, dc_out, bc, cc,
            xT, steps_to_shot, shots, total_xG, touches,
            role
        """
        rows = []

        # ---- 从 PassingNetwork 获取图论指标 ----
        dc_all = self.net.degree_centrality("all")
        dc_in  = self.net.degree_centrality("in")
        dc_out = self.net.degree_centrality("out")
        bc     = self.net.betweenness_centrality(weighted=True)
        _, cc  = self.net.clustering_coefficient()

        # ---- 从 AbsorbingMarkovChain 获取威胁指标 ----
        xt_dict    = self.amc.get_xT_dict()
        steps_dict = {self.amc.players[i]: self.amc.expected_steps_to_shot[i]
                      for i in range(self.amc.n)}
        shot_stats = self.amc.shot_stats

        # ---- 合并（以 PassingNetwork 的节点为准） ----
        all_players = set(dc_all.keys()) | set(xt_dict.keys())

        for p in all_players:
            row = {
                'player':         p,
                'short_name':     p.split()[-1] if ' ' in p else p,
                'dc_all':         dc_all.get(p, 0.0),
                'dc_in':          dc_in.get(p, 0.0),
                'dc_out':         dc_out.get(p, 0.0),
                'bc':             bc.get(p, 0.0),
                'cc':             cc.get(p, 0.0),
                'xT':             xt_dict.get(p, 0.0),
                'steps_to_shot':  steps_dict.get(p, 99.0),
                'shots':          shot_stats.get(p, {}).get('shots', 0),
                'total_xG':       shot_stats.get(p, {}).get('total_xG', 0.0),
                'touches':        self.net.touch_counts.get(p, 0),
            }
            rows.append(row)

        df = pd.DataFrame(rows)

        # ---- 归一化列（用于雷达图 0~1 范围） ----
        for col in ['dc_all', 'dc_in', 'dc_out', 'bc', 'cc',
                     'xT', 'steps_to_shot', 'touches']:
            col_max = df[col].max()
            if col_max > 0:
                df[f'{col}_norm'] = df[col] / col_max
            else:
                df[f'{col}_norm'] = 0.0

        # steps_to_shot 越小越好，所以反转归一化
        if df['steps_to_shot'].max() > 0:
            df['steps_to_shot_norm'] = 1.0 - df['steps_to_shot_norm']

        # ---- 判定战术角色 ----
        dc_median = df['dc_all'].median()
        xt_median = df['xT'].median()

        def classify(row):
            high_dc = row['dc_all'] >= dc_median
            high_xt = row['xT'] >= xt_median
            if high_dc and high_xt:
                return 'core_creator'
            elif not high_dc and high_xt:
                return 'efficient_finisher'
            elif high_dc and not high_xt:
                return 'idle_hub'
            else:
                return 'peripheral'

        df['role'] = df.apply(classify, axis=1)
        df['role_cn'] = df['role'].map(self.ROLE_LABELS)

        # 按 xT 降序
        df.sort_values('xT', ascending=False, inplace=True)
        df.reset_index(drop=True, inplace=True)

        return df

    # ----------------------------------------------------------------
    # 公开接口：获取指标表
    # ----------------------------------------------------------------
    def get_metrics_table(self) -> pd.DataFrame:
        """返回完整指标 DataFrame。"""
        return self.metrics.copy()

    # ----------------------------------------------------------------
    # 可视化 1：四象限散点图
    # ----------------------------------------------------------------
    def plot_quadrant(self, x_metric: str = 'dc_all', y_metric: str = 'xT',
                       size_metric: str = 'touches',
                       color_metric: str = 'bc',
                       save_path: str = None):
        """
        四象限散点图：任意两个指标作为 X/Y 轴。

        默认:
            X = 度中心性（球员在传球网络中的连接度）
            Y = 期望威胁 xT（球从该球员出发的进球贡献）
            气泡大小 = 触球次数
            气泡颜色 = 中介中心性

        Parameters
        ----------
        x_metric     : 'dc_all', 'dc_in', 'dc_out', 'bc', 'cc'
        y_metric     : 'xT', 'steps_to_shot', 'total_xG'
        size_metric  : 控制气泡大小的列名
        color_metric : 控制气泡颜色的列名
        save_path    : 保存路径
        """
        BG = '#0E1117'
        df = self.metrics.copy()

        fig, ax = plt.subplots(figsize=(14, 10), facecolor=BG)
        ax.set_facecolor(BG)

        # ---- 坐标值 ----
        x_vals = df[x_metric].values
        y_vals = df[y_metric].values

        # ---- 气泡大小 ----
        s_vals = df[size_metric].values.astype(float)
        s_max  = s_vals.max() if s_vals.max() > 0 else 1
        sizes  = 200 + (s_vals / s_max) * 2000

        # ---- 气泡颜色 ----
        c_vals = df[color_metric].values.astype(float)
        cmap   = LinearSegmentedColormap.from_list(
            'quad_cmap', ['#1a1a2e', '#00C8FF', '#FFD700', '#FF4444'])
        norm   = Normalize(vmin=c_vals.min(), vmax=max(c_vals.max(), 0.01))

        # ---- 绘制象限线 ----
        x_med = np.median(x_vals)
        y_med = np.median(y_vals)
        ax.axvline(x=x_med, color='#444444', linestyle='--',
                    linewidth=1.5, zorder=1)
        ax.axhline(y=y_med, color='#444444', linestyle='--',
                    linewidth=1.5, zorder=1)

        # 象限标签
        quad_style = dict(fontsize=11, alpha=0.3, fontweight='bold',
                           ha='center', va='center')
        x_range = x_vals.max() - x_vals.min()
        y_range = y_vals.max() - y_vals.min()

        ax.text(x_med + x_range * 0.25, y_med + y_range * 0.35,
                'Core Creator', color='#FFD700', **quad_style)
        ax.text(x_med - x_range * 0.25, y_med + y_range * 0.35,
                'High Efficient Finisher', color='#00FF88', **quad_style)
        ax.text(x_med + x_range * 0.25, y_med - y_range * 0.35,
                'Idle Hub', color='#FF6B6B', **quad_style)
        ax.text(x_med - x_range * 0.25, y_med - y_range * 0.35,
                'Peripheral', color='#888888', **quad_style)

        # ---- 绘制散点 ----
        role_markers = {
            'core_creator':      'o',
            'efficient_finisher': '^',
            'idle_hub':          's',
            'peripheral':        'D',
        }

        for _, row in df.iterrows():
            idx = row.name
            marker = role_markers.get(row['role'], 'o')
            sc = ax.scatter(
                x_vals[idx], y_vals[idx],
                s=sizes[idx], c=[cmap(norm(c_vals[idx]))],
                marker=marker, edgecolors='white', linewidths=1.5,
                zorder=5, alpha=0.9
            )

            # 球员名标注
            txt = ax.annotate(
                row['short_name'],
                (x_vals[idx], y_vals[idx]),
                xytext=(8, 8), textcoords='offset points',
                color='white', fontsize=9, fontweight='bold', zorder=6
            )
            txt.set_path_effects([
                path_effects.Stroke(linewidth=2.5, foreground='black'),
                path_effects.Normal()])

        # ---- 标签 ----
        label_map = {
            'dc_all':         'Degree Centrality (总度中心性)',
            'dc_in':          'In-Degree Centrality (入度中心性)',
            'dc_out':         'Out-Degree Centrality (出度中心性)',
            'bc':             'Betweenness Centrality (中介中心性)',
            'cc':             'Clustering Coefficient (聚集系数)',
            'xT':             'Expected Threat (期望威胁)',
            'steps_to_shot':  'Expected Steps to Shot (期望射门步数)',
            'total_xG':       'Total xG',
            'touches':        'Touches (触球次数)',
        }

        ax.set_xlabel(label_map.get(x_metric, x_metric),
                       color='white', fontsize=12)
        ax.set_ylabel(label_map.get(y_metric, y_metric),
                       color='white', fontsize=12)
        ax.set_title(
            f"{self.net.team} — Tactical Role Quadrant\n"
            f"气泡大小 = {label_map.get(size_metric, size_metric)} · "
            f"气泡颜色 = {label_map.get(color_metric, color_metric)}",
            color='white', fontsize=15, fontweight='bold', pad=15)

        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_color('#2a3a4a')

        # colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_label(label_map.get(color_metric, color_metric),
                        color='white', fontsize=10)
        cbar.ax.tick_params(colors='white', labelsize=8)

        # 图例：marker 形状 = 角色
        legend_handles = [
            Line2D([0], [0], marker='o', color=BG, markerfacecolor='#FFD700',
                   markersize=10, label='Core creator'),
            Line2D([0], [0], marker='^', color=BG, markerfacecolor='#00FF88',
                   markersize=10, label='High efficient finisher'),
            Line2D([0], [0], marker='s', color=BG, markerfacecolor='#FF6B6B',
                   markersize=10, label='Idle hub'),
            Line2D([0], [0], marker='D', color=BG, markerfacecolor='#888888',
                   markersize=10, label='Peripheral'),
        ]
        ax.legend(handles=legend_handles, loc='upper left',
                  facecolor=BG, edgecolor='#2a3a4a',
                  labelcolor='white', fontsize=10)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, facecolor=BG, bbox_inches='tight')
            print(f"  ✓ 四象限图已保存至 {save_path}")
        plt.show()

    # ----------------------------------------------------------------
    # 可视化 2：雷达图（多球员对比）
    # ----------------------------------------------------------------
    def plot_radar(self, player_names: list = None,
                    dimensions: list = None,
                    save_path: str = None):
        """
        多维度雷达图：对比任意球员的指标画像。

        Parameters
        ----------
        player_names : 球员名列表（支持模糊匹配），默认取 xT 前 5
        dimensions   : 雷达维度列名列表，默认:
                       ['dc_all', 'dc_in', 'dc_out', 'bc', 'cc', 'xT']
        save_path    : 保存路径
        """
        BG = '#0E1117'
        df = self.metrics.copy()

        # 默认维度
        if dimensions is None:
            dimensions = ['dc_all_norm', 'dc_in_norm', 'dc_out_norm',
                          'bc_norm', 'cc_norm', 'xT_norm',
                          'steps_to_shot_norm']

        dim_labels = {
            'dc_all_norm':         'All Degree\nCentrality',
            'dc_in_norm':          'In-Degree\n(Receptions)',
            'dc_out_norm':         'Out-Degree\n(Passes)',
            'bc_norm':             'Betweenness\nCentrality',
            'cc_norm':             'Clustering\nCoefficient',
            'xT_norm':             'Expected\nThreat',
            'steps_to_shot_norm':  'Expected\nSteps to Shot',
            'touches_norm':        'Touches\n(Touch Count)',
        }

        # 默认球员：xT 前 5
        if player_names is None:
            player_names = df.nlargest(5, 'xT')['player'].tolist()
        else:
            # 模糊匹配
            matched = []
            for name in player_names:
                found = df[df['player'].str.contains(
                    name, case=False, na=False)]
                if len(found) > 0:
                    matched.append(found.iloc[0]['player'])
                else:
                    print(f"  [!] 找不到包含 '{name}' 的球员，跳过。")
            player_names = matched

        if len(player_names) == 0:
            print("  [!] 没有有效球员，无法绘制雷达图。")
            return

        # ---- 雷达图数据准备 ----
        n_dims  = len(dimensions)
        angles  = np.linspace(0, 2 * np.pi, n_dims, endpoint=False).tolist()
        angles += angles[:1]  # 闭合

        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'polar': True},
                                facecolor=BG)
        ax.set_facecolor(BG)

        # 颜色循环
        colors = ['#00C8FF', '#FFD700', '#FF6B6B', '#00FF88',
                  '#FF66FF', '#FF8800', '#88FF00']

        for idx, player in enumerate(player_names):
            row = df[df['player'] == player]
            if row.empty:
                continue
            row = row.iloc[0]

            values = [row.get(d, 0.0) for d in dimensions]
            values += values[:1]  # 闭合

            color = colors[idx % len(colors)]
            ax.plot(angles, values, 'o-', linewidth=2.5,
                    color=color, label=row['short_name'], markersize=6)
            ax.fill(angles, values, alpha=0.12, color=color)

        # ---- 标签 ----
        labels = [dim_labels.get(d, d) for d in dimensions]
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, color='white', fontsize=10,
                            fontweight='bold')

        # 径向刻度
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'],
                            color='#666666', fontsize=7)
        ax.set_ylim(0, 1.05)

        # 网格颜色
        ax.xaxis.grid(True, color='#2a3a4a', linewidth=0.8)
        ax.yaxis.grid(True, color='#2a3a4a', linewidth=0.8)
        ax.spines['polar'].set_color('#2a3a4a')

        ax.set_title(f"{self.net.team} — 球员多维画像",
                      color='white', fontsize=16, fontweight='bold',
                      pad=25)
        ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1.15),
                  facecolor=BG, edgecolor='#2a3a4a',
                  labelcolor='white', fontsize=11)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, facecolor=BG, bbox_inches='tight')
            print(f"  ✓ 雷达图已保存至 {save_path}")
        plt.show()

    # ----------------------------------------------------------------
    # 可视化 3：综合大看板
    # ----------------------------------------------------------------
    def plot_composite_dashboard(self, save_path: str = None):
        """
        四合一综合看板:
            [左上] 四象限散点图
            [右上] 入度 vs 出度 → xT 颜色
            [左下] 雷达图（前5球员）
            [右下] 指标相关性热力图
        """
        BG = '#0E1117'
        df = self.metrics.copy()

        fig = plt.figure(figsize=(24, 20), facecolor=BG)
        fig.suptitle(
            f"{self.net.team} — Tactical Composite Analysis",
            color='white', fontsize=22, fontweight='bold', y=0.98)

        # =====================
        # 左上：四象限散点
        # =====================
        ax1 = fig.add_axes([0.03, 0.52, 0.45, 0.42])
        ax1.set_facecolor(BG)

        x_med = df['dc_all'].median()
        y_med = df['xT'].median()
        ax1.axvline(x=x_med, color='#444444', ls='--', lw=1.2)
        ax1.axhline(y=y_med, color='#444444', ls='--', lw=1.2)

        role_colors = {
            'core_creator':      '#FFD700',
            'efficient_finisher': '#00FF88',
            'idle_hub':          '#FF6B6B',
            'peripheral':        '#888888',
        }

        for _, row in df.iterrows():
            c = role_colors.get(row['role'], 'white')
            size = 150 + row['touches'] * 8
            ax1.scatter(row['dc_all'], row['xT'], s=size,
                        c=c, edgecolors='white', lw=1.2, zorder=5, alpha=0.9)
            txt = ax1.annotate(
                row['short_name'], (row['dc_all'], row['xT']),
                xytext=(6, 6), textcoords='offset points',
                color='white', fontsize=8, fontweight='bold', zorder=6)
            txt.set_path_effects([
                path_effects.Stroke(linewidth=2, foreground='black'),
                path_effects.Normal()])

        ax1.set_xlabel('Degree Centrality', color='white', fontsize=11)
        ax1.set_ylabel('Expected Threat (xT)', color='white', fontsize=11)
        ax1.set_title('Degree Centrality vs Expected Threat (xT)',
                       color='#00C8FF', fontsize=13, fontweight='bold', pad=8)
        ax1.tick_params(colors='white')
        for sp in ax1.spines.values():
            sp.set_color('#2a3a4a')

        # 图例
        from matplotlib.patches import Patch
        legend_patches = [Patch(fc=v, ec='white', label=self.ROLE_LABELS[k])
                          for k, v in role_colors.items()]
        ax1.legend(handles=legend_patches, loc='upper left',
                   facecolor=BG, edgecolor='#2a3a4a',
                   labelcolor='white', fontsize=9)

        # =====================
        # 右上：入度 vs 出度，颜色=xT
        # =====================
        ax2 = fig.add_axes([0.55, 0.52, 0.42, 0.42])
        ax2.set_facecolor(BG)

        cmap_xt = LinearSegmentedColormap.from_list(
            'xt', ['#1a1a2e', '#00C8FF', '#FFD700', '#FF4444'])
        norm_xt = Normalize(vmin=0, vmax=max(df['xT'].max(), 0.001))

        for _, row in df.iterrows():
            ax2.scatter(row['dc_out'], row['dc_in'],
                        s=200 + row['touches'] * 6,
                        c=[cmap_xt(norm_xt(row['xT']))],
                        edgecolors='white', lw=1.2, zorder=5, alpha=0.9)
            txt = ax2.annotate(
                row['short_name'], (row['dc_out'], row['dc_in']),
                xytext=(6, 6), textcoords='offset points',
                color='white', fontsize=8, fontweight='bold', zorder=6)
            txt.set_path_effects([
                path_effects.Stroke(linewidth=2, foreground='black'),
                path_effects.Normal()])

        # 对角线 y=x（入度=出度的平衡线）
        lim_max = max(df['dc_out'].max(), df['dc_in'].max()) * 1.1
        ax2.plot([0, lim_max], [0, lim_max], '--', color='#444444', lw=1)
        ax2.text(lim_max * 0.7, lim_max * 0.55, 'In-Degree < Out-Degree\n(More Passes Out than In)',
                 color='#888888', fontsize=8, ha='center')
        ax2.text(lim_max * 0.3, lim_max * 0.75, 'In-Degree > Out-Degree\n(More Passes In than Out)',
                 color='#888888', fontsize=8, ha='center')

        ax2.set_xlabel('Out-Degree Centrality (出度)', color='white', fontsize=11)
        ax2.set_ylabel('In-Degree Centrality (入度)', color='white', fontsize=11)
        ax2.set_title('In-Degree vs Out-Degree · Color = xT',
                       color='#00C8FF', fontsize=13, fontweight='bold', pad=8)
        ax2.tick_params(colors='white')
        for sp in ax2.spines.values():
            sp.set_color('#2a3a4a')

        sm = plt.cm.ScalarMappable(cmap=cmap_xt, norm=norm_xt)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax2, fraction=0.03, pad=0.02)
        cbar.set_label('xT', color='white', fontsize=10)
        cbar.ax.tick_params(colors='white', labelsize=8)

        # =====================
        # 左下：雷达图（前5球员）
        # =====================
        ax3 = fig.add_axes([0.03, 0.05, 0.42, 0.40], polar=True)
        ax3.set_facecolor(BG)

        dims = ['dc_all_norm', 'dc_in_norm', 'dc_out_norm',
                'bc_norm', 'cc_norm', 'xT_norm', 'steps_to_shot_norm']
        dim_labels_short = ['All Degree', 'In-Degree', 'Out-Degree', 'Betweenness', 'Clustering', 'xT', 'Steps to Shot']

        n_dims = len(dims)
        angles = np.linspace(0, 2 * np.pi, n_dims, endpoint=False).tolist()
        angles += angles[:1]

        top5 = df.nlargest(5, 'xT')
        colors_radar = ['#00C8FF', '#FFD700', '#FF6B6B', '#00FF88', '#FF66FF']

        for idx, (_, row) in enumerate(top5.iterrows()):
            values = [row.get(d, 0) for d in dims]
            values += values[:1]
            c = colors_radar[idx % len(colors_radar)]
            ax3.plot(angles, values, 'o-', lw=2.2, color=c,
                     label=row['short_name'], markersize=5)
            ax3.fill(angles, values, alpha=0.08, color=c)

        ax3.set_xticks(angles[:-1])
        ax3.set_xticklabels(dim_labels_short, color='white', fontsize=9,
                             fontweight='bold')
        ax3.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax3.set_yticklabels(['', '0.5', '', '1.0'], color='#555', fontsize=7)
        ax3.set_ylim(0, 1.05)
        ax3.xaxis.grid(True, color='#2a3a4a', lw=0.8)
        ax3.yaxis.grid(True, color='#2a3a4a', lw=0.8)
        ax3.spines['polar'].set_color('#2a3a4a')
        ax3.set_title('xT Top 5 Players · Multi-Dimensional Profile',
                       color='#00C8FF', fontsize=13, fontweight='bold', pad=18)
        ax3.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1),
                   facecolor=BG, edgecolor='#2a3a4a',
                   labelcolor='white', fontsize=9)

        # =====================
        # 右下：指标相关性热力图 + 球员排名表
        # =====================
        ax4 = fig.add_axes([0.55, 0.05, 0.42, 0.40])
        ax4.set_facecolor(BG)

        # 选取核心指标做相关矩阵
        corr_cols = ['dc_all', 'dc_in', 'dc_out', 'bc', 'cc',
                     'xT', 'steps_to_shot', 'touches']
        corr_labels = ['All Degree', 'In-Degree', 'Out-Degree', 'Betweenness', 'Clustering',
                        'xT', 'Steps to Shot', 'Touches']

        corr_data = df[corr_cols].copy()
        # steps_to_shot 反转（越小越好 → 取负数再算相关）
        corr_data['steps_to_shot'] = -corr_data['steps_to_shot']
        corr_matrix = corr_data.corr()

        # 绘制热力图
        cmap_corr = LinearSegmentedColormap.from_list(
            'corr', ['#FF4444', '#1a1a2e', '#00FF88'])
        im = ax4.imshow(corr_matrix.values, cmap=cmap_corr,
                         vmin=-1, vmax=1, aspect='auto')

        ax4.set_xticks(range(len(corr_labels)))
        ax4.set_yticks(range(len(corr_labels)))
        ax4.set_xticklabels(corr_labels, color='white', fontsize=9,
                             rotation=45, ha='right')
        ax4.set_yticklabels(corr_labels, color='white', fontsize=9)

        # 在每个格子里写数字
        for i in range(len(corr_labels)):
            for j in range(len(corr_labels)):
                val = corr_matrix.values[i, j]
                text_color = 'white' if abs(val) > 0.5 else '#aaaaaa'
                ax4.text(j, i, f'{val:.2f}', ha='center', va='center',
                         color=text_color, fontsize=8, fontweight='bold')

        ax4.set_title('Correlation Matrix of Key Metrics\n(Steps to Shot reversed: smaller is better → larger is better)',
                       color='#00C8FF', fontsize=13, fontweight='bold', pad=10)

        cbar2 = fig.colorbar(im, ax=ax4, fraction=0.03, pad=0.02)
        cbar2.set_label('Pearson r', color='white', fontsize=10)
        cbar2.ax.tick_params(colors='white', labelsize=8)

        plt.subplots_adjust(hspace=0.3, wspace=0.3)
        if save_path:
            plt.savefig(save_path, dpi=150, facecolor=BG, bbox_inches='tight')
            print(f"  ✓ 综合看板已保存至 {save_path}")
        plt.show()

    # ----------------------------------------------------------------
    # 可视化 4：球场上叠加 xT 热力 + 传球网络
    # ----------------------------------------------------------------
    def plot_pitch_overlay(self, positions: dict = None,
                            pair_df: pd.DataFrame = None,
                            save_path: str = None):
        """
        在球场上同时绘制：
            · 传球网络（线条粗细 = 传球频率）
            · xT 气泡（大小 = xT 值）
            · 节点颜色 = 战术角色

        Parameters
        ----------
        positions : {player: (x, y)} 平均位置字典
        pair_df   : 传球对统计 DataFrame (需有 player, receiver, count 列)
        save_path : 保存路径
        """
        BG = '#0E1117'

        fig, ax = plt.subplots(figsize=(16, 11), facecolor=BG)
        pitch = Pitch(pitch_type='opta', pitch_color=BG,
                      line_color='#2a3a4a', linewidth=1)
        pitch.draw(ax=ax)

        if positions is None:
            print("  [!] 需要 positions 字典。使用 PassingNetwork.positions")
            try:
                positions = self.net.positions
            except AttributeError:
                print("  [!] 无法获取位置信息，跳过。")
                return

        df = self.metrics.copy()
        xt_dict = self.amc.get_xT_dict()
        max_xt = max(xt_dict.values()) if xt_dict else 1.0

        role_colors = {
            'core_creator':      '#FFD700',
            'efficient_finisher': '#00FF88',
            'idle_hub':          '#FF6B6B',
            'peripheral':        '#888888',
        }

        # ---- 画传球连线 ----
        if pair_df is not None:
            max_count = pair_df['count'].max() if len(pair_df) > 0 else 1
            for _, row in pair_df.iterrows():
                p1 = row['player']
                p2 = row['receiver']
                if p1 not in positions or p2 not in positions:
                    continue

                x1, y1 = positions[p1]
                x2, y2 = positions[p2]
                cnt     = row['count']

                # 粗细 & 透明度
                lw    = 0.5 + (cnt / max_count) * 5.0
                alpha = 0.15 + (cnt / max_count) * 0.5

                ax.annotate(
                    '', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(
                        arrowstyle='->', color='#00C8FF',
                        lw=lw, alpha=alpha,
                        connectionstyle='arc3,rad=0.08',
                        shrinkA=12, shrinkB=12
                    ), zorder=2
                )

        # ---- 画节点 ----
        for player, (px, py) in positions.items():
            row = df[df['player'] == player]
            if row.empty:
                continue
            row = row.iloc[0]

            xt   = xt_dict.get(player, 0.0)
            role = row['role']
            c    = role_colors.get(role, '#888888')

            # 节点大小 = xT
            size = 400 + (xt / max_xt) * 3000

            ax.scatter(px, py, s=size, c=c, edgecolors='white',
                       linewidths=2.5, zorder=5, alpha=0.9)

            # 球员名
            short = row['short_name']
            txt = ax.text(px, py + 0.5, short, color='white',
                          fontsize=9, ha='center', va='bottom',
                          fontweight='bold', zorder=6)
            txt.set_path_effects([
                path_effects.Stroke(linewidth=3, foreground='black'),
                path_effects.Normal()])

            # xT 数值
            val_txt = ax.text(px, py - 4.5, f'xT={xt:.3f}',
                              color='#FFD700', fontsize=7,
                              ha='center', va='top', zorder=6)
            val_txt.set_path_effects([
                path_effects.Stroke(linewidth=2, foreground='black'),
                path_effects.Normal()])

            # 角色标签
            role_txt = ax.text(px, py - 7.5, row['role_cn'],
                               color=c, fontsize=6.5,
                               ha='center', va='top', zorder=6,
                               fontstyle='italic', alpha=0.8)
            role_txt.set_path_effects([
                path_effects.Stroke(linewidth=1.5, foreground='black'),
                path_effects.Normal()])

        # 图例
        from matplotlib.patches import Patch
        legend_patches = [Patch(fc=v, ec='white', label=self.ROLE_LABELS[k])
                          for k, v in role_colors.items()]
        ax.legend(handles=legend_patches, loc='lower left',
                  facecolor=BG, edgecolor='#2a3a4a',
                  labelcolor='white', fontsize=10)

        ax.set_title(
            f"{self.net.team} — Passing Network × Expected Threat × Tactical Roles",
            color='white', fontsize=16, fontweight='bold', pad=12)

        if save_path:
            plt.savefig(save_path, dpi=150, facecolor=BG, bbox_inches='tight')
            print(f"  ✓ 球场叠加图已保存至 {save_path}")
        plt.show()

    # ----------------------------------------------------------------
    # 文字输出：战术分析报告
    # ----------------------------------------------------------------
    def print_tactical_report(self):
        """
        生成完整的文字版战术分析报告。

        涵盖:
            1. 球队整体战术风格
            2. 每位球员的角色分析
            3. 关键通道（期望步数最短的球员对）
            4. 薄弱环节
        """
        df = self.metrics.copy()

        print("\n" + "=" * 70)
        print(f"    {self.net.team} — 战术复合分析报告")
        print("=" * 70)

        # ===== 1. 整体画像 =====
        print("\n┌─────────────────────────────────────────────────┐")
        print("│  1. 球队整体战术画像                             │")
        print("└─────────────────────────────────────────────────┘")

        n_core    = len(df[df['role'] == 'core_creator'])
        n_finish  = len(df[df['role'] == 'efficient_finisher'])
        n_idle    = len(df[df['role'] == 'idle_hub'])
        n_periph  = len(df[df['role'] == 'peripheral'])

        print(f"\n  角色分布:")
        print(f"    核心创造者     : {n_core} 人")
        print(f"    高效射手       : {n_finish} 人")
        print(f"    空转枢纽       : {n_idle} 人")
        print(f"    边缘球员       : {n_periph} 人")

        # 集中度分析
        top3_xt = df.nlargest(3, 'xT')
        top3_xt_share = top3_xt['xT'].sum() / max(df['xT'].sum(), 1e-10)

        print(f"\n  威胁集中度:")
        print(f"    xT 前三球员占比: {top3_xt_share:.1%}")
        if top3_xt_share > 0.6:
            print(f"    → 进攻高度集中于少数球员，对方可针对性盯防")
        elif top3_xt_share > 0.4:
            print(f"    → 进攻较为均衡，有多个威胁点")
        else:
            print(f"    → 进攻非常分散，缺乏明确的主攻手")

        # 入度/出度不平衡分析
        df['dc_imbalance'] = df['dc_out'] - df['dc_in']
        big_out = df[df['dc_imbalance'] > 0.1]
        big_in  = df[df['dc_imbalance'] < -0.1]

        print(f"\n  传球方向性:")
        if len(big_out) > 0:
            names = ', '.join(big_out['short_name'].tolist())
            print(f"    出球>接球（组织点）: {names}")
        if len(big_in) > 0:
            names = ', '.join(big_in['short_name'].tolist())
            print(f"    接球>出球（终结点）: {names}")

        # ===== 2. 球员逐一分析 =====
        print("\n┌─────────────────────────────────────────────────┐")
        print("│  2. 球员角色详解                                 │")
        print("└─────────────────────────────────────────────────┘")

        for _, row in df.iterrows():
            print(f"\n  ▸ {row['player']}  [{row['role_cn']}]")
            print(f"    度中心性: {row['dc_all']:.3f} "
                  f"(入={row['dc_in']:.3f}, 出={row['dc_out']:.3f})")
            print(f"    中介中心性: {row['bc']:.3f}  "
                  f"聚集系数: {row['cc']:.3f}")
            print(f"    期望威胁 xT: {row['xT']:.4f}  "
                  f"射门: {row['shots']}次  xG: {row['total_xG']:.3f}")
            print(f"    触球: {row['touches']}次  "
                  f"射门期望步数: {row['steps_to_shot']:.1f}")

            # 个性化评语
            if row['role'] == 'core_creator':
                if row['bc'] > df['bc'].quantile(0.75):
                    print(f"    ★ 核心中转站——大量进攻经过此人，"
                          f"且最终能产生高威胁")
                else:
                    print(f"    ★ 高威胁组织者——虽非最忙的中转站，"
                          f"但每次触球都有价值")

            elif row['role'] == 'efficient_finisher':
                if row['shots'] > 0:
                    print(f"    ★ 冷血终结者——参与传球不多，"
                          f"但拿球就能制造射门威胁")
                else:
                    print(f"    ★ 隐蔽威胁点——不常出现在数据中，"
                          f"但位置/跑动创造了高 xT")

            elif row['role'] == 'idle_hub':
                if row['cc'] > df['cc'].quantile(0.75):
                    print(f"    ⚠ 短传循环者——高聚集系数 + 高度中心性 → "
                          f"倾向在小范围内倒脚")
                else:
                    print(f"    ⚠ 横传回传者——控球多但缺乏向前推进能力")

            elif row['role'] == 'peripheral':
                if row['touches'] < df['touches'].quantile(0.25):
                    print(f"    ▽ 极少参与进攻，可能是后卫或刚替补上场")
                else:
                    print(f"    ▽ 参与有限——需要更多跑位来融入进攻体系")

        # ===== 3. 关键通道 =====
        print("\n┌─────────────────────────────────────────────────┐")
        print("│  3. 关键进攻通道 (期望步数最短的连接)            │")
        print("└─────────────────────────────────────────────────┘")

        N = self.amc.N
        players = self.amc.players
        n = self.amc.n

        # 找所有有意义的 (i→j) 对的期望首达步数近似
        channels = []
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if N[j, j] > 1e-10:
                    steps_ij = N[i, j] / N[j, j]
                    # 只看有传球关系的
                    if self.amc.count_matrix[i, j] > 0:
                        channels.append({
                            'from':   players[i],
                            'to':     players[j],
                            'steps':  steps_ij,
                            'passes': self.amc.count_matrix[i, j],
                            'to_xT':  self.amc.xT[j],
                        })

        if channels:
            ch_df = pd.DataFrame(channels)
            # 按 to_xT * passes / steps 排序（综合评分）
            ch_df['score'] = ch_df['to_xT'] * ch_df['passes'] / ch_df['steps']
            ch_df.sort_values('score', ascending=False, inplace=True)

            print(f"\n  {'传球人':<18} {'→':>2} {'接球人':<18} "
                  f"{'传球数':>6} {'步数':>6} {'接球人xT':>8} {'评分':>7}")
            print("  " + "-" * 70)

            for _, row in ch_df.head(10).iterrows():
                f_short = row['from'].split()[-1]
                t_short = row['to'].split()[-1]
                print(f"  {f_short:<18} → {t_short:<18} "
                      f"{row['passes']:>6.0f} {row['steps']:>6.1f} "
                      f"{row['to_xT']:>8.4f} {row['score']:>7.3f}")

        # ===== 4. 薄弱环节 =====
        print("\n┌─────────────────────────────────────────────────┐")
        print("│  4. 潜在薄弱环节                                 │")
        print("└─────────────────────────────────────────────────┘")

        # 中介中心性极高但 xT 低 → 依赖但不产出
        bottlenecks = df[(df['bc'] > df['bc'].quantile(0.8)) &
                          (df['xT'] < df['xT'].median())]
        if len(bottlenecks) > 0:
            print(f"\n  ⚠ 瓶颈球员（高中介 + 低 xT）:")
            for _, row in bottlenecks.iterrows():
                print(f"    {row['short_name']}: "
                      f"BC={row['bc']:.3f}, xT={row['xT']:.4f}")
                print(f"    → 大量球权经过此人但未产生威胁，"
                      f"对方截断此人可瘫痪进攻")

        # 聚集系数过高的区域 → 可能存在过多无效倒脚
        high_cc = df[df['cc'] > df['cc'].quantile(0.8)]
        if len(high_cc) > 0:
            print(f"\n  ⚠ 高聚集区域（可能的无效倒脚区）:")
            for _, row in high_cc.iterrows():
                print(f"    {row['short_name']}: CC={row['cc']:.3f}")

        # xT 前 3 的球员如果期望步数也很大 → 难以有效送球
        danger_far = df.nlargest(3, 'xT')
        danger_far = danger_far[danger_far['steps_to_shot'] >
                                 df['steps_to_shot'].median()]
        if len(danger_far) > 0:
            print(f"\n  ⚠ 威胁点偏远（高 xT 但高步数）:")
            for _, row in danger_far.iterrows():
                print(f"    {row['short_name']}: "
                      f"xT={row['xT']:.4f}, 步数={row['steps_to_shot']:.1f}")
                print(f"    → 虽然此人拿球后威胁大，"
                      f"但球很难传到他脚下")

        print(f"\n{'=' * 70}")
        print(f"  报告结束")
        print(f"{'=' * 70}\n")

        return df


# ============================================================================
# PART 5 — 顶层编排器：一键串联全部流程
# ============================================================================

class FullPipelineOrchestrator:
    """
    顶层编排器，不修改任何已有类，
    通过组合调用串联完整分析流程。

    用法（最简）:
        orchestrator = FullPipelineOrchestrator(
            raw_csv     = "raw_events.csv",
            pass_csv    = "successful_passes.csv",
            team        = "Leicester",
            opponent    = "Tottenham"
        )
        orchestrator.run()

    流程:
        ┌──────────────────────────────────────┐
        │  1. 原始 PassingNetwork.build_graph  │  ← 不修改
        │  2. 原始 MarkovChain                 │  ← 不修改
        ├──────────────────────────────────────┤
        │  3. PlayingTimeAnalyzer              │  ← 新增
        │     → 推断上场时间                    │
        │     → Per-90 归一化传球               │
        ├──────────────────────────────────────┤
        │  4. AbsorbingMarkovChain             │  ← 新增
        │     → 提取射门 + 计算 xG             │
        │     → 构建吸收态矩阵                 │
        │     → 高斯消元求基础矩阵 N           │
        │     → 计算期望威胁 xT                │
        ├──────────────────────────────────────┤
        │  5. TacticalCompositeAnalyzer        │  ← 新增
        │     → 复合度中心性 × xT              │
        │     → 四象限图 + 雷达图 + 看板       │
        │     → 战术报告                       │
        └──────────────────────────────────────┘
    """

    def __init__(self, raw_csv: str, pass_csv: str,
                 team: str = "Leicester",
                 opponent: str = "Opponent",
                 match_length: int = 90,
                 use_xg_weighting: bool = True):

        self.raw_csv          = raw_csv
        self.pass_csv         = pass_csv
        self.team             = team
        self.opponent         = opponent
        self.match_length     = match_length
        self.use_xg_weighting = use_xg_weighting

        # 存储各模块实例
        self.net      = None   # PassingNetwork
        self.mc       = None   # MarkovChain (原始)
        self.pta      = None   # PlayingTimeAnalyzer
        self.amc      = None   # AbsorbingMarkovChain
        self.comp     = None   # TacticalCompositeAnalyzer

    def run(self, skip_original: bool = False,
            per90: bool = True,
            plot_all: bool = True,
            save_prefix: str = "leicester"):
        """
        一键运行全部分析。

        Parameters
        ----------
        skip_original : 是否跳过原始 PassingNetwork/MarkovChain 的可视化
        per90         : 是否进行 Per-90 归一化
        plot_all      : 是否绘制所有图表
        save_prefix   : 图片保存文件名前缀
        """
        print("\n" + "█" * 70)
        print("█" + " " * 68 + "█")
        print("█" + f"  {self.team} vs {self.opponent}".center(68) + "█")
        print("█" + "  完整战术分析流水线".center(64) + "█")
        print("█" + " " * 68 + "█")
        print("█" * 70 + "\n")

        # ============================================================
        # Step 1: 原始 PassingNetwork（调用已有代码，不修改）
        # ============================================================
        print("━" * 70)
        print("  STEP 1 / 5 — 构建传球网络（原始 PassingNetwork）")
        print("━" * 70)

        # 这里假设 PassingNetwork 类已经存在于同一命名空间
        # from leicester_network import PassingNetwork
        try:
            self.net = PassingNetwork(self.pass_csv, self.team)
            self.net.build_graph()
            if plot_all and not skip_original:
                self.net.plot_network(save_path=f"{save_prefix}_01_network.png")
                self.net.plot_centrality(save_path=f"{save_prefix}_02_centrality.png")
        except NameError:
            print("  [!] PassingNetwork 类未找到。")
            print("      请确保已 import 或在同文件中定义。")
            print("      尝试仅用本文件的功能继续...\n")
            return

        # ============================================================
        # Step 2: 原始 MarkovChain（调用已有代码，不修改）
        # ============================================================
        print("\n" + "━" * 70)
        print("  STEP 2 / 5 — 原始马尔可夫链（MarkovChain）")
        print("━" * 70)

        try:
            self.mc = MarkovChain(self.net)
            if plot_all and not skip_original:
                self.mc.plot_transition_matrix(
                    save_path=f"{save_prefix}_03_transition.png")
                self.mc.plot_steady_state(
                    save_path=f"{save_prefix}_04_steady.png")
        except NameError:
            print("  [!] MarkovChain 类未找到，跳过。\n")

        # ============================================================
        # Step 3: 上场时间分析 + Per-90 归一化
        # ============================================================
        print("\n" + "━" * 70)
        print("  STEP 3 / 5 — 上场时间 + Per-90 归一化")
        print("━" * 70)

        self.pta = PlayingTimeAnalyzer()
        self.pta.analyze(
            raw_csv=self.raw_csv,
            team=self.team,
            match_length=self.match_length
        )

        # 归一化传球数据
        if per90 and hasattr(self.net, 'pass_df'):
            pass_df_weighted = self.pta.normalize_passes(self.net.pass_df)
            pair_counts_90 = self.pta.build_normalized_pair_counts(
                pass_df_weighted)
        else:
            pass_df_weighted = None
            pair_counts_90   = None

        # ============================================================
        # Step 4: 吸收态马尔可夫链 + 期望威胁
        # ============================================================
        print("\n" + "━" * 70)
        print("  STEP 4 / 5 — 吸收态马尔可夫链 + xG 期望威胁")
        print("━" * 70)

        self.amc = AbsorbingMarkovChain(
            raw_csv          = self.raw_csv,
            pass_df          = self.net.pass_df,
            team             = self.team,
            pair_counts      = pair_counts_90,
            minutes_played   = self.pta.minutes_played,
            use_xg_weighting = self.use_xg_weighting,
        )

        if plot_all:
            positions = getattr(self.net, 'positions', None)
            self.amc.plot_expected_threat(
                positions=positions,
                save_path=f"{save_prefix}_05_xT.png"
            )

        # ============================================================
        # Step 5: 复合分析
        # ============================================================
        print("\n" + "━" * 70)
        print("  STEP 5 / 5 — 复合战术分析")
        print("━" * 70)

        self.comp = TacticalCompositeAnalyzer(self.net, self.amc)

        if plot_all:
            self.comp.plot_quadrant(
                save_path=f"{save_prefix}_06_quadrant.png")

            self.comp.plot_radar(
                save_path=f"{save_prefix}_07_radar.png")

            self.comp.plot_pitch_overlay(
                positions=getattr(self.net, 'positions', None),
                pair_df=pair_counts_90 if pair_counts_90 is not None
                        else self.net.pair_counts,
                save_path=f"{save_prefix}_08_overlay.png")

            self.comp.plot_composite_dashboard(
                save_path=f"{save_prefix}_09_dashboard.png")

        # 文字报告
        self.comp.print_tactical_report()

        # ============================================================
        # 完成
        # ============================================================
        print("\n" + "█" * 70)
        print("█" + " " * 68 + "█")
        print("█" + "  全部分析完成 ✓".center(64) + "█")
        print("█" + " " * 68 + "█")

        if plot_all:
            print("█" + f"  已生成图表:".center(64) + "█")
            for i, name in enumerate([
                "传球网络", "中心性", "转移矩阵", "稳态分布",
                "期望威胁", "四象限", "雷达图", "球场叠加", "综合看板"
            ], 1):
                fn = f"{save_prefix}_{i:02d}_*.png"
                print("█" + f"    {fn:40s} {name}".center(64) + "█")

        print("█" + " " * 68 + "█")
        print("█" * 70 + "\n")

        return self.comp.get_metrics_table()


# ============================================================================
# 入口：如果直接运行本文件
# ============================================================================

if __name__ == "__main__":

    import sys

    # 默认文件名（可通过命令行参数覆盖）
    RAW_CSV  = sys.argv[1] if len(sys.argv) > 1 else "raw_events.csv"
    PASS_CSV = sys.argv[2] if len(sys.argv) > 2 else "successful_passes.csv"
    TEAM     = sys.argv[3] if len(sys.argv) > 3 else "Leicester"
    OPPONENT = sys.argv[4] if len(sys.argv) > 4 else "Tottenham"

    print(f"\n  原始事件文件: {RAW_CSV}")
    print(f"  传球文件:     {PASS_CSV}")
    print(f"  球队:         {TEAM}")
    print(f"  对手:         {OPPONENT}\n")

    # ---- 需要先导入原始类 ----
    # 方式 1：如果原始代码在 leicester_network.py 中
    # from leicester_network import PassingNetwork, MarkovChain

    # 方式 2：如果在同一个 Jupyter Notebook 中，直接运行即可

    orchestrator = FullPipelineOrchestrator(
        raw_csv          = RAW_CSV,
        pass_csv         = PASS_CSV,
        team             = TEAM,
        opponent         = OPPONENT,
        match_length     = 90,
        use_xg_weighting = True,
    )

    results = orchestrator.run(
        skip_original = False,
        per90         = True,
        plot_all      = True,
        save_prefix   = "leicester_tactical"
    )

    # 打印最终指标表
    if results is not None:
        print("\n  ===== 最终指标表 =====\n")
        display_cols = ['short_name', 'role_cn', 'dc_all', 'dc_in', 'dc_out',
                        'bc', 'cc', 'xT', 'steps_to_shot', 'shots',
                        'total_xG', 'touches']
        print(results[display_cols].to_string(index=False))
