# Leicester City Passing Network Extractor
import soccerdata as sd
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

def get_passing_network_data(match_url):
    print("\n" + "="*50)
    print("EXTRACTING PASSING NETWORK DATA")
    print("="*50)

    # 1. 解析比赛 ID
    try:
        match_id = int(match_url.split('/Matches/')[1].split('/')[0])
    except Exception:
        print("网址格式不正确！")
        return

    print(f"[1/3] Fetching full chronological events for Match {match_id}...")
    ws = sd.WhoScored(leagues="ENG-Premier League", seasons="2425")
    try:
        events = ws.read_events(match_id=[match_id])
    except Exception as e:
        print(f"抓取失败: {e}")
        return

    # 重置索引，获取展平的二维表
    df = events.reset_index()

    raw_output = r"C:\Users\86185\Desktop\LEI\RAW_match_{match_id}_events.csv".format(match_id=match_id)
    df.to_csv(raw_output, index=False, encoding='utf-8-sig')
    print(f"\n已将没有任何操作的原始数据保存到指定路径：{raw_output}")

    print("[2/3] Performing chronological shift(-1) to find receivers...")
    # 按时间顺序执行shift-1，找到每个事件的下一个事件（即潜在的接球人）
    df['receiver_id'] = df['player_id'].shift(-1)
    df['receiver'] = df['player'].shift(-1)
    df['next_event_team'] = df['team'].shift(-1)

    full_output = r"C:\Users\86185\Desktop\LEI\FULL_match_{match_id}_events_with_receivers.csv".format(match_id=match_id)
    df.to_csv(full_output, index=False, encoding='utf-8-sig')
    print(f"已将包含接球人信息的完整事件数据保存到：{full_output}")

    print("[3/3] Filtering for successful passes by Leicester City...")
    # 确保兼容不同的参数名称
    team_col = 'team' if 'team' in df.columns else df.columns[df.columns.str.contains('team', case=False)][0]
    type_col = 'type' if 'type' in df.columns else df.columns[df.columns.str.contains('type', case=False)][0]
    outcome_col = 'outcomeType' if 'outcomeType' in df.columns else df.columns[df.columns.str.contains('outcome', case=False)][0]
    
    is_leicester = df[team_col].astype(str).str.contains('Leicester', case=False, na=False)
    is_pass_base = df[type_col].astype(str).str.contains('Pass', case=False, na=False)
    is_excluded = df[type_col].astype(str).str.contains('Blocked|Offside', case=False, na=False)
    is_pass = is_pass_base & ~is_excluded
    is_successful = df[outcome_col].astype(str).str.contains('Successful|Success|1', case=False, na=False)
    is_same_team_next = df['next_event_team'].astype(str).str.contains('Leicester', case=False, na=False) # 下一个事件必须是本方球员

    pass_df = df[is_leicester & is_pass & is_successful & is_same_team_next].copy()

    # 仅保留：时间、传球人、接球人、起点坐标、终点坐标
    final_cols = ['minute', 'player_id', 'player', 'receiver_id', 'receiver', 'x', 'y', 'endX', 'endY']
    
    # 防止有些列不存在导致报错
    existing_cols = [col for col in final_cols if col in pass_df.columns]
    graph_ready_df = pass_df[existing_cols]

    output_name = r"C:\Users\86185\Desktop\LEI\Leicester_Passing_Network_{match_id}.csv".format(match_id=match_id)
    graph_ready_df.to_csv(output_name, index=False, encoding='utf-8-sig')

    print("\n" + "="*50)
    print("GRAPH THEORY DATA READY")
    print("="*50)
    print(f"Total Completed Passes: {len(graph_ready_df)}")
    print(f"File Saved: {output_name}")
    print("\nSample Edge List:")
    if 'receiver' in graph_ready_df.columns:
        preview = graph_ready_df[['minute', 'player', 'receiver']].head(5)
        print(preview.to_string(index=False))

if __name__ == "__main__":
    # 莱斯特城 vs 狼队
    TARGET_URL = "https://www.whoscored.com/Matches/1821238/Live/England-Premier-League-2024-2025-Leicester-Wolverhampton-Wanderers"
    get_passing_network_data(TARGET_URL)