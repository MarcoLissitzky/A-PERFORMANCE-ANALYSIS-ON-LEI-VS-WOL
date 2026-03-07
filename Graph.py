# ============================================================================
# Leicester City 24/25 — Passing Network Graph Theory + Markov Chain
# ============================================================================
# pip install mplsoccer networkx pandas numpy matplotlib scipy

import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.lines import Line2D
from mplsoccer import Pitch
from scipy.linalg import eig
import warnings, ast, os
warnings.filterwarnings('ignore')

# 数据加载与清洗
# 

class PassingNetwork:
    """
    从 ScraperFC 输出的 CSV 构建传球网络，并提供完整的图论分析接口。
    用法:
        net = PassingNetwork()
        net.load_csv("leicester_wolves.csv", team="Leicester")
        net.build_graph(min_pass_count=3)
        net.print_metrics()
        net.plot_full_dashboard()
        mc = net.markov_chain()
    """

    # ---------- 初始化 ----------
    def __init__(self):
        #self.raw_df        = None   # 原始 CSV
        self.pass_df       = None   # 清洗后的传球数据
        self.team          = None
        self.G             = None   # nx.DiGraph — 有向传球图
        self.G_undirected  = None   # nx.Graph  — 用于聚集系数
        self.positions     = {}     # {player: (x, y)}  球员平均位置
        self.touch_counts  = {}     # {player: int}     触球次数
        self.pair_counts   = None   # DataFrame (passer, receiver, count)

    # STEP 1 : 加载 CSV + 提取成功传球
    # 
    def load_csv(self, filepath: str, team: str = "Leicester",
                 period: str = "all", max_minute: int = 999):
        """
        加载 ScraperFC 爬取的 CSV 。

        Parameters
        ----------
        filepath   : CSV 文件路径
        """
        #self.team = team
        print("=" * 65)
        print(f"  STEP 1 · 加载数据  ——  目标球队: {team}")
        print("=" * 65)
        df = pd.read_csv(filepath)
        print(f"  传球总数           : {len(df)}")
        """

        # ---- 1.2 过滤球队 ----
        mask_team = df['team'].str.contains(team, case=False, na=False)
        df = df[mask_team].copy()
        print(f"  {team} 事件数          : {len(df)}")

        # ---- 1.3 过滤半场 ----
        if period != "all":
            df = df[df['period'] == period].copy()
            print(f"  {period} 事件数       : {len(df)}")

        # ---- 1.4 过滤分钟 ----
        df = df[df['minute'] <= max_minute].copy()

        # ---- 1.5 只保留「传球类」事件且结果成功 ----
        # WhoScored 的 type 列：'Pass', 'KeyPass', 'Assist' 等均含 Pass
        # 也包括 'TakeOn', 'BallTouch' 等，但我们只要传球
        pass_types = ['Pass', 'KeyPass', 'Assist', 'CrossNotClaimed',
                      'BlockedPass', 'OffsidePass']
        # 先做宽松匹配：type 列包含 'Pass' 或在上述列表中
        mask_pass = (
            df['type'].str.contains('Pass', case=False, na=False) |
            df['type'].isin(pass_types)
        )
        df_pass = df[mask_pass].copy()
        print(f"  传球类事件（含失败）   : {len(df_pass)}")

        # 只保留成功传球
        mask_success = df_pass['outcome_type'].str.contains(
            'Successful|Success', case=False, na=False)
        df_pass = df_pass[mask_success].copy()
        print(f"  成功传球               : {len(df_pass)}")

        # ---- 1.6 坐标清洗 ----
        for c in ['x', 'y', 'end_x', 'end_y']:
            df_pass[c] = pd.to_numeric(df_pass[c], errors='coerce')
        df_pass.dropna(subset=['x', 'y', 'end_x', 'end_y'], inplace=True)
        print(f"  有效坐标传球           : {len(df_pass)}")

        # ---- 1.7 推断接球人 ----
        # WhoScored 数据没有直接的 receiver 列
        # 方法：同队下一条「触球」事件的球员即为接球人
        df_team_all = df[mask_team].sort_values(
            ['period', 'minute', 'second']).reset_index(drop=True)

        # 建立 related_player_id 映射（如果有值则优先使用）
        # 否则回退到「下一个触球球员」
        receiver_map = {}
        for idx, row in df_team_all.iterrows():
            if idx + 1 < len(df_team_all):
                next_row = df_team_all.iloc[idx + 1]
                # 同一半场且分钟差 <= 1
                if (row['period'] == next_row['period'] and
                        abs(next_row['minute'] - row['minute']) <= 1):
                    receiver_map[idx] = next_row['player']

        # 把 receiver 映射回 df_pass（按原始 index 对齐）
        df_pass = df_pass.copy()
        df_pass['receiver'] = df_pass.index.map(
            lambda i: receiver_map.get(i, np.nan))

        # 去掉自传自接
        df_pass = df_pass[df_pass['player'] != df_pass['receiver']].copy()
        df_pass.dropna(subset=['receiver'], inplace=True)
        print(f"  有接球人的成功传球     : {len(df_pass)}")
        """

        #self.raw_df  = df
        self.pass_df = df
        print()

    # STEP 2 : 构建有向图 + 无向图
    # 
    def build_graph(self, min_pass_count: int = 3):
        """
        从传球数据构建 NetworkX 图。

        Parameters
        ----------
        min_pass_count : 两人之间传球次数 >= 此值才画边
        """
        print("=" * 65)
        print(f"  STEP 2 · 构建图  ——  最小边权: {min_pass_count}")
        print("=" * 65)

        df = self.pass_df.copy()

        # 2.1 每对球员之间的传球次数
        pair = (df.groupby(['player', 'receiver'])
                  .size()
                  .reset_index(name='count'))
        self.pair_counts = pair
        print(f"  唯一传球对: {len(pair)}")

        # 2.2 球员平均位置
        passer_pos  = df.groupby('player')[['x', 'y']].mean()
        recv_pos    = df.groupby('receiver')[['end_x', 'end_y']].mean()
        recv_pos.columns = ['x', 'y']
        combined    = pd.concat([passer_pos, recv_pos])
        avg_pos     = combined.groupby(combined.index).mean()
        self.positions = {n: (r['x'], r['y']) for n, r in avg_pos.iterrows()}

        # ---- 2.3 触球次数 ----
        t1 = df['player'].value_counts()
        t2 = df['receiver'].value_counts()
        self.touch_counts = (t1.add(t2, fill_value=0)).astype(int).to_dict()

        # ---- 2.4 构建有向图 ----
        G = nx.DiGraph()
        for player, (px, py) in self.positions.items():
            G.add_node(player, x=px, y=py,
                       touches=self.touch_counts.get(player, 0))

        edges_total = 0
        edges_kept  = 0
        for _, row in pair.iterrows():
            edges_total += 1
            if row['count'] >= min_pass_count:
                G.add_edge(row['player'], row['receiver'],
                           weight=int(row['count']))
                edges_kept += 1

        self.G = G
        print(f"  节点数（球员）         : {G.number_of_nodes()}")
        print(f"  边数（>= {min_pass_count} 次传球）   : {edges_kept} / {edges_total}")

        # ---- 2.5 构建无向图（用于聚集系数） ----
        #     无向图边权 = 双方传球次数之和
        G_u = nx.Graph()
        for n, d in G.nodes(data=True):
            G_u.add_node(n, **d)
        for u, v, d in G.edges(data=True):
            if G_u.has_edge(u, v):
                G_u[u][v]['weight'] += d['weight']
            else:
                G_u.add_edge(u, v, weight=d['weight'])
        self.G_undirected = G_u
        print(f"  无向图边数             : {G_u.number_of_edges()}")
        print()

    # ================================================================
    # STEP 3 : 图论指标计算
    # ================================================================

    # ---------- 3A : 度中心性 ----------
    def degree_centrality(self, direction: str = "all"):
        """
        Parameters
        ----------
        direction : 'in' / 'out' / 'all'
            in  = 入度中心性（谁接球最多）
            out = 出度中心性（谁传球最多）
            all = 总度中心性（有向图转无向后）
        Returns : dict {player: centrality}
        """
        if direction == "in":
            return nx.in_degree_centrality(self.G)
        elif direction == "out":
            return nx.out_degree_centrality(self.G)
        else:
            return nx.degree_centrality(self.G_undirected)

    # ---------- 3B : 中介中心性 ----------
    def betweenness_centrality(self, weighted: bool = True):
        """
        中介中心性：一个节点在多少「最短路径」上充当桥梁。
        值越高 → 该球员是传球枢纽；如果中场值低 → 中场被跳过。

        Parameters
        ----------
        weighted : 是否用传球次数作为权重
                   注意：nx 中 weight 越大 = 越「容易」通过，
                   所以我们用 1/weight 作为距离。
        Returns : dict {player: centrality}
        """
        if weighted:
            # 给每条边添加 distance = 1 / weight
            G_copy = self.G.copy()
            for u, v, d in G_copy.edges(data=True):
                d['distance'] = 1.0 / d['weight']
            return nx.betweenness_centrality(G_copy, weight='distance',
                                              normalized=True)
        else:
            return nx.betweenness_centrality(self.G, weight=None,
                                              normalized=True)

    # ---------- 3C : 聚集系数 ----------
    def clustering_coefficient(self, player: str = None):
        """
        聚集系数：衡量一个球员的「传球伙伴」之间是否也互相传球。
        值高 → 局部三角配合多（tiki-taka）；值低 → 传球链条是线性的。

        Parameters
        ----------
        player : str or None
            None  → 返回全局平均聚集系数 + 每人的字典
            'xxx' → 返回该球员的局部聚集系数
        Returns : float  或  (float_global, dict_local)
        """
        # 对有向图计算
        local = nx.clustering(self.G_undirected, weight='weight')

        if player is not None:
            if player in local:
                return local[player]
            else:
                print(f"  [!] 球员 '{player}' 不在图中。")
                return 0.0

        global_cc = nx.average_clustering(self.G_undirected, weight='weight')
        return global_cc, local

    # ---------- 3D : 网络密度 ----------
    def density(self):
        """有向图密度：实际边数 / 最大可能边数。"""
        return nx.density(self.G)

    # ---------- 3E : 代数连通度 ----------
    def algebraic_connectivity(self):
        """
        拉普拉斯矩阵第二小特征值（Fiedler value）。
        值越小 → 网络越容易被「切成两半」→ 球队结构脆弱。
        仅对连通的无向图有意义。
        """
        if not nx.is_connected(self.G_undirected):
            print("  [!] 无向图不连通，代数连通度 = 0")
            components = list(nx.connected_components(self.G_undirected))
            print(f"      连通分量: {[list(c) for c in components]}")
            return 0.0
        return nx.algebraic_connectivity(self.G_undirected, weight='weight')

    # ================================================================
    # STEP 4 : 马尔科夫链
    # ================================================================
    def markov_chain(self):
        """
        将传球网络建模为离散时间马尔科夫链。

        转移概率: P(j | i) = passes(i→j) / Σ_k passes(i→k)

        Returns
        -------
        MarkovChain 对象，包含:
            .players          : 球员列表（状态空间）
            .transition_matrix: numpy 2D array
            .stationary_dist  : 稳态分布（哪个球员长期持球占比最高）
            .expected_steps(a,b) : 从 a 到 b 的期望传球次数
        """
        print("=" * 65)
        print("  STEP 4 · 构建马尔科夫链")
        print("=" * 65)

        return MarkovChain(self.pair_counts, self.G)

    # ================================================================
    # STEP 5 : 打印所有指标
    # ================================================================
    def print_metrics(self):
        """一键打印全部图论指标，格式化输出。"""
        print("=" * 65)
        print("  STEP 3 · 图论指标汇总")
        print("=" * 65)

        # ---- 密度 ----
        d = self.density()
        print(f"\n  ▸ 网络密度 (Density)              : {d:.4f}")
        if d < 0.3:
            print("    → 低密度：球队传球线路单一，缺乏多样化配合。")
        elif d < 0.5:
            print("    → 中等密度：传球线路有一定多样性。")
        else:
            print("    → 高密度：球队传球网络丰富。")

        # ---- 代数连通度 ----
        ac = self.algebraic_connectivity()
        print(f"\n  ▸ 代数连通度 (Algebraic Conn.)    : {ac:.4f}")
        if ac < 0.5:
            print("    → 极低：网络极易断裂，前后场脱节严重。")

        # ---- 度中心性 ----
        dc_all = self.degree_centrality("all")
        dc_in  = self.degree_centrality("in")
        dc_out = self.degree_centrality("out")

        print(f"\n  ▸ 度中心性 (Degree Centrality)")
        print(f"    {'球员':<18} {'总度':>6} {'入度':>6} {'出度':>6} {'触球':>5}")
        print("    " + "-" * 48)
        for p in sorted(dc_all, key=dc_all.get, reverse=True):
            print(f"    {p:<18} {dc_all[p]:6.3f} {dc_in.get(p,0):6.3f} "
                  f"{dc_out.get(p,0):6.3f} {self.touch_counts.get(p,0):5d}")

        # ---- 中介中心性 ----
        bc = self.betweenness_centrality(weighted=True)
        print(f"\n  ▸ 中介中心性 (Betweenness Centrality)")
        print(f"    {'球员':<18} {'BC':>8}")
        print("    " + "-" * 28)
        for p in sorted(bc, key=bc.get, reverse=True):
            marker = " ◀ 枢纽" if bc[p] > 0.15 else ""
            print(f"    {p:<18} {bc[p]:8.4f}{marker}")

        # ---- 聚集系数 ----
        global_cc, local_cc = self.clustering_coefficient()
        print(f"\n  ▸ 聚集系数 (Clustering Coefficient)")
        print(f"    全局平均                        : {global_cc:.4f}")
        if global_cc < 0.3:
            print("    → 低聚集：缺少三角短传配合，多为线性传递。")
        print(f"    {'球员':<18} {'CC':>8}")
        print("    " + "-" * 28)
        for p in sorted(local_cc, key=local_cc.get, reverse=True):
            print(f"    {p:<18} {local_cc[p]:8.4f}")

        print()

    # ================================================================
    # STEP 6 : 可视化大看板
    # ================================================================
    def plot_full_dashboard(self, save_path: str = r"C:\Users\86185\Desktop\LEI\leicester_dashboard.png", dpi: int = 150):
        """
        一张图包含四个子图:
            [左上]  传球网络 (Passing Network)
            [右上]  中介中心性热力图
            [左下]  度中心性柱状图
            [右下]  聚集系数柱状图
        """
        print("=" * 65)
        print("  STEP 5 · 绘制可视化面板")
        print("=" * 65)

        fig = plt.figure(figsize=(22, 18), facecolor='#0E1117')
        fig.suptitle(f"{self.team} — Passing Network & Graph Theory",
                     color='white', fontsize=22, fontweight='bold', y=0.97)

        # ---------- 颜色方案 ----------
        BG      = '#0E1117'
        ACCENT  = '#00C8FF'
        EDGE_C  = '#FFD700'
        TEXT_C  = '#FFFFFF'

        # ========================
        # 子图 1 : 传球网络
        # ========================
        ax1 = fig.add_axes([-0.04, 0.42, 0.45, 0.45])   # [left, bottom, w, h]
        pitch = Pitch(pitch_type='opta', pitch_color=BG,
                      line_color='#2a3a4a', linewidth=1)
        pitch.draw(ax=ax1)
        ax1.set_title("Passing Network", color=TEXT_C, fontsize=16,
                       fontweight='bold', pad=8)

        # 节点大小：触球次数
        max_touch = max(self.touch_counts.values()) if self.touch_counts else 1

        # 中介中心性用于节点颜色
        bc = self.betweenness_centrality(weighted=True)
        bc_vals = np.array([bc.get(p, 0) for p in self.positions])
        norm_bc = Normalize(vmin=0, vmax=max(bc_vals.max(), 0.01))
        cmap = LinearSegmentedColormap.from_list('bc_cmap',
                    ['#1a1a2e', '#00C8FF', '#FFD700', '#FF4444'])

        # 画边
        for u, v, d in self.G.edges(data=True):
            if u in self.positions and v in self.positions:
                x1, y1 = self.positions[u]
                x2, y2 = self.positions[v]
                w = d['weight']
                alpha = min(0.9, 0.2 + w / 20)
                lw = 0.5 + w * 0.35
                ax1.annotate("",
                    xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(
                        arrowstyle='->,head_length=0.4,head_width=0.25',
                        color=EDGE_C, alpha=alpha, lw=lw,
                        connectionstyle='arc3,rad=0.08'))

        # 画节点
        for player, (px, py) in self.positions.items():
            touches = self.touch_counts.get(player, 1)
            size = 200 + (touches / max_touch) * 1800
            color = cmap(norm_bc(bc.get(player, 0)))
            ax1.scatter(px, py, s=size, c=[color], edgecolors='white',
                        linewidths=1.5, zorder=5, alpha=0.95)

            # 球员名（取姓氏）
            short_name = player.split()[-1] if ' ' in player else player
            txt = ax1.text(px, py - 4.5, short_name, color=TEXT_C,
                           fontsize=7.5, ha='center', va='top',
                           fontweight='bold', zorder=6)
            txt.set_path_effects([
                path_effects.Stroke(linewidth=2, foreground='black'),
                path_effects.Normal()])

        # colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm_bc)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax1, fraction=0.03, pad=0.02,
                             location='bottom', aspect=40)
        cbar.set_label('Betweenness Centrality', color=TEXT_C, fontsize=9)
        cbar.ax.tick_params(colors=TEXT_C, labelsize=7)

        # ========================
        # 子图 2 : 中介中心性条形图
        # ========================
        ax2 = fig.add_axes([0.58, 0.55, 0.38, 0.35])
        ax2.set_facecolor(BG)

        sorted_bc = sorted(bc.items(), key=lambda x: x[1], reverse=True)
        names_bc  = [p.split()[-1] for p, _ in sorted_bc]
        vals_bc   = [v for _, v in sorted_bc]
        colors_bc = [cmap(norm_bc(v)) for v in vals_bc]

        bars = ax2.barh(range(len(names_bc)), vals_bc, color=colors_bc,
                         edgecolor='white', linewidth=0.5, height=0.7)
        ax2.set_yticks(range(len(names_bc)))
        ax2.set_yticklabels(names_bc, color=TEXT_C, fontsize=9)
        ax2.invert_yaxis()
        ax2.set_xlabel('Betweenness Centrality', color=TEXT_C, fontsize=10)
        ax2.set_title('Betweenness Centrality - Node Importance', color=ACCENT,
                       fontsize=13, fontweight='bold', pad=10)
        ax2.tick_params(colors=TEXT_C)
        for spine in ax2.spines.values():
            spine.set_color('#2a3a4a')

        # 标注数值
        for i, v in enumerate(vals_bc):
            ax2.text(v + 0.005, i, f'{v:.3f}', color=TEXT_C,
                     va='center', fontsize=8)

        # ========================
        # 子图 3 : 度中心性对比（入度 vs 出度）
        # ========================
        ax3 = fig.add_axes([0.02, 0.05, 0.44, 0.30])
        ax3.set_facecolor(BG)

        dc_in  = self.degree_centrality("in")
        dc_out = self.degree_centrality("out")
        all_players = sorted(self.positions.keys(),
                              key=lambda p: dc_in.get(p, 0) + dc_out.get(p, 0),
                              reverse=True)
        short_names = [p.split()[-1] for p in all_players]
        in_vals  = [dc_in.get(p, 0) for p in all_players]
        out_vals = [dc_out.get(p, 0) for p in all_players]

        x_idx = np.arange(len(all_players))
        bar_w = 0.35
        ax3.bar(x_idx - bar_w/2, in_vals,  bar_w, color='#00C8FF',
                label='In-Degree', edgecolor='white', linewidth=0.5)
        ax3.bar(x_idx + bar_w/2, out_vals, bar_w, color='#FF6B6B',
                label='Out-Degree', edgecolor='white', linewidth=0.5)

        ax3.set_xticks(x_idx)
        ax3.set_xticklabels(short_names, color=TEXT_C, fontsize=8,
                             rotation=45, ha='right')
        ax3.set_ylabel('Degree Centrality', color=TEXT_C, fontsize=10)
        ax3.set_title('Degree Centrality - In-Degree vs Out-Degree', color=ACCENT,
                       fontsize=13, fontweight='bold', pad=10)
        ax3.legend(facecolor=BG, edgecolor='#2a3a4a', labelcolor=TEXT_C,
                   fontsize=9)
        ax3.tick_params(colors=TEXT_C)
        for spine in ax3.spines.values():
            spine.set_color('#2a3a4a')

        # ========================
        # 子图 4 : 聚集系数
        # ========================
        ax4 = fig.add_axes([0.55, 0.05, 0.42, 0.30])
        ax4.set_facecolor(BG)

        global_cc, local_cc = self.clustering_coefficient()
        sorted_cc = sorted(local_cc.items(), key=lambda x: x[1], reverse=True)
        names_cc  = [p.split()[-1] for p, _ in sorted_cc]
        vals_cc   = [v for _, v in sorted_cc]

        # 渐变色
        cc_colors = plt.cm.YlOrRd(Normalize(0, max(max(vals_cc), 0.01))
                                    (np.array(vals_cc)))

        ax4.bar(range(len(names_cc)), vals_cc, color=cc_colors,
                edgecolor='white', linewidth=0.5)
        # 全局均值线
        ax4.axhline(y=global_cc, color=ACCENT, linestyle='--', linewidth=1.5,
                     label=f'Global Average = {global_cc:.3f}')

        ax4.set_xticks(range(len(names_cc)))
        ax4.set_xticklabels(names_cc, color=TEXT_C, fontsize=8,
                             rotation=45, ha='right')
        ax4.set_ylabel('Clustering Coefficient', color=TEXT_C, fontsize=10)
        ax4.set_title('Clustering Coefficient - Triangle Density', color=ACCENT,
                       fontsize=13, fontweight='bold', pad=10)
        ax4.legend(facecolor=BG, edgecolor='#2a3a4a', labelcolor=TEXT_C,
                   fontsize=9)
        ax4.tick_params(colors=TEXT_C)
        for spine in ax4.spines.values():
            spine.set_color('#2a3a4a')

        # 标注数值
        for i, v in enumerate(vals_cc):
            if v > 0:
                ax4.text(i, v + 0.01, f'{v:.2f}', color=TEXT_C,
                         ha='center', fontsize=7)

        plt.savefig(save_path or "leicester_network_dashboard.png",
                    dpi=dpi, facecolor=BG, bbox_inches='tight')
        print(f"  ✓ 已保存至 {save_path or 'leicester_network_dashboard.png'}\n")
        plt.show()


# ============================================================================
# 马尔科夫链类
# ============================================================================

class MarkovChain:
    """
    将传球网络建模为离散时间齐次马尔科夫链。

    状态空间 S = {球员1, 球员2, ..., 球员n}
    转移概率 P(i→j) = passes(i,j) / Σ_k passes(i,k)

    用途:
        - stationary_dist: 长期持球概率 → 哪个球员理论上应该拿球最多
        - absorbing analysis: 如果某球员被「移除」,链条断裂程度
        - expected_steps: 球从 A 到 B 平均需要几脚传球
    """

    def __init__(self, pair_counts: pd.DataFrame, G: nx.DiGraph):
        # 球员列表（仅保留有出度的）
        all_players = sorted(set(pair_counts['player']) |
                              set(pair_counts['receiver']))
        self.players = all_players
        self.n = len(all_players)
        self.player_to_idx = {p: i for i, p in enumerate(all_players)}

        # ---- 构建转移矩阵 ----
        # 注意：用全部传球（不只是 >= min_pass_count 的）
        T = np.zeros((self.n, self.n))
        for _, row in pair_counts.iterrows():
            i = self.player_to_idx.get(row['player'])
            j = self.player_to_idx.get(row['receiver'])
            if i is not None and j is not None:
                T[i, j] = row['count']

        # 行归一化 → 概率
        row_sums = T.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # 避免除零（吸收态）
        self.transition_matrix = T / row_sums

        print(f"  状态空间: {self.n} 个球员")
        print(f"  转移矩阵: {self.n}×{self.n}")

        # ---- 计算稳态分布 ----
        self._compute_stationary()

        # ---- 打印 ----
        self._print_summary()

    def _compute_stationary(self):
        """
        求解稳态分布 π，满足 πP = π, Σπ_i = 1。
        用左特征向量法。
        """
        P = self.transition_matrix.T  # 转置后求右特征向量 = 原矩阵左特征向量
        eigenvalues, eigenvectors = eig(P)

        # 找特征值最接近 1 的
        idx = np.argmin(np.abs(eigenvalues - 1.0))
        stationary = np.real(eigenvectors[:, idx])
        stationary = stationary / stationary.sum()  # 归一化
        self.stationary_dist = stationary

    def _print_summary(self):
        """打印稳态分布。"""
        print(f"\n  ▸ 稳态分布（长期持球概率）:")
        print(f"    {'球员':<18} {'π(i)':>8} {'百分比':>7}")
        print("    " + "-" * 35)

        order = np.argsort(-self.stationary_dist)
        for idx in order:
            p = self.players[idx]
            pi = self.stationary_dist[idx]
            pct = pi * 100
            print(f"    {p:<18} {pi:8.4f} {pct:6.1f}%")
        print()

    def expected_steps(self, from_player: str, to_player: str):
        """
        计算从 from_player 到 to_player 的平均首达时间。
        （球从 A 传到 B 平均需要几脚？）

        使用公式: h(i→j) = 1 + Σ_{k≠j} P(i,k) * h(k→j)
        转化为线性方程组求解。
        """
        j = self.player_to_idx.get(to_player)
        if j is None:
            print(f"  [!] 球员 '{to_player}' 不在马尔科夫链中。")
            return np.inf

        i = self.player_to_idx.get(from_player)
        if i is None:
            print(f"  [!] 球员 '{from_player}' 不在马尔科夫链中。")
            return np.inf

        # 构建方程组: (I - P_reduced) h = 1
        # 其中 P_reduced 是去掉目标行列后的子矩阵
        P = self.transition_matrix.copy()
        n = self.n

        # 去掉第 j 行和第 j 列
        mask = np.ones(n, dtype=bool)
        mask[j] = False
        P_sub = P[mask][:, mask]

        # (I - P_sub) h = 1
        A = np.eye(n - 1) - P_sub
        b = np.ones(n - 1)

        try:
            h = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return np.inf

        # 找到 from_player 在缩减后的索引
        reduced_idx = i if i < j else i - 1
        steps = h[reduced_idx]

        return steps

    def transition_heatmap(self, ax=None, save_path: str = None):
        """绘制转移概率矩阵热力图。"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 10), facecolor='#0E1117')

        short_names = [p.split()[-1] for p in self.players]

        im = ax.imshow(self.transition_matrix, cmap='YlOrRd',
                        interpolation='nearest', aspect='auto')
        ax.set_xticks(range(self.n))
        ax.set_yticks(range(self.n))
        ax.set_xticklabels(short_names, rotation=45, ha='right',
                            color='white', fontsize=8)
        ax.set_yticklabels(short_names, color='white', fontsize=8)
        ax.set_title('Markov Transition Matrix  P(i → j)',
                      color='#00C8FF', fontsize=14, fontweight='bold', pad=12)
        ax.set_xlabel('Receiver (j)', color='white', fontsize=11)
        ax.set_ylabel('Passer (i)', color='white', fontsize=11)
        ax.set_facecolor('#0E1117')

        # 在格子里标注概率
        for r in range(self.n):
            for c in range(self.n):
                val = self.transition_matrix[r, c]
                if val > 0.02:
                    color = 'white' if val > 0.3 else 'black'
                    ax.text(c, r, f'{val:.2f}', ha='center', va='center',
                            fontsize=6.5, color=color, fontweight='bold')

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                      label='Transition Probability')

        if save_path:
            plt.savefig(save_path, dpi=150, facecolor='#0E1117',
                        bbox_inches='tight')
            print(f"  ✓ 转移矩阵热力图已保存至 {save_path}")

        return ax


# ============================================================================
# MAIN — 运行入口
# ============================================================================

if __name__ == "__main__":

    # ---- 用户输入 ----
    print("\n" + "=" * 65)
    print("  Leicester City 24/25 — 传球网络图论分析")
    print("=" * 65)

    csv_path = input("\n  请输入 CSV 文件路径: ").strip()
    if not csv_path:
        csv_path = "leicester_wolves.csv"  # 默认

    team = input("  请输入目标球队名 (默认 Leicester): ").strip()
    if not team:
        team = "Leicester"

    min_pass = input("  最小传球次数阈值 (默认 3): ").strip()
    min_pass = int(min_pass) if min_pass else 3

    # ---- 执行分析 ----
    net = PassingNetwork()
    net.load_csv(csv_path, team=team)
    net.build_graph(min_pass_count=min_pass)
    net.print_metrics()
    net.plot_full_dashboard()

    # ---- 马尔科夫链 ----
    mc = net.markov_chain()

    # 转移矩阵热力图
    fig_mc, ax_mc = plt.subplots(figsize=(12, 10), facecolor='#0E1117')
    mc.transition_heatmap(ax=ax_mc, save_path=r"C:\Users\86185\Desktop\LEI\leicester_transition_heatmap.png")
    plt.show()

    # ---- 交互式查询 ----
    print("\n" + "=" * 65)
    print("  交互查询: 任意两人之间的期望传球步数")
    print("  输入格式:  球员A, 球员B    (输入 q 退出)")
    print("=" * 65)

    while True:
        query = input("\n  > ").strip()
        if query.lower() in ('q', 'quit', 'exit', ''):
            break
        try:
            a, b = [x.strip() for x in query.split(',')]
            # 模糊匹配球员名
            match_a = [p for p in mc.players if a.lower() in p.lower()]
            match_b = [p for p in mc.players if b.lower() in p.lower()]
            if not match_a:
                print(f"    [!] 找不到包含 '{a}' 的球员")
                continue
            if not match_b:
                print(f"    [!] 找不到包含 '{b}' 的球员")
                continue
            pa, pb = match_a[0], match_b[0]
            steps = mc.expected_steps(pa, pb)
            print(f"    {pa} → {pb} : 平均 {steps:.1f} 脚传球")
        except Exception as e:
            print(f"    [!] 输入格式错误: {e}")

    print("\n  分析完毕。\n")
