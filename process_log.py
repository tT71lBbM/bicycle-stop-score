"""
一時停止の遵守判定（log.csv → near_log.csv / route_log.csv / visit_log.csv）

設計
  1. 訪問の状態を保つ
     一時停止地点ごとに「訪問」の状態を持ち、その地点の半径の外に出るまで保持する。
     他の地点の状態が変わっても、この地点の状態はリセットしない。
  2. 方位は進入した瞬間だけで判定する
     地点の半径に入った最初の行の方位で、判定対象かどうかを決める。
     決めた結果は退出するまで変えない（半径内で曲がっても判定し直さない）。
  3. 向きが合わない地点は進入として扱わない
     進入時に方位が合わなかった地点は「対象外」として、退出まで停止判定もしない。
     対象外の地点が近くにあっても、対象の地点の判定には影響しない。
  4. 同じ地点に連続で進入しない
     ある地点から退出したあと、ほかの地点に進入する前に同じ地点へ入り直した場合は
     新しい進入とせず、前の訪問の続きとして扱う（GPSの揺れで半径を出入りしたときなど）。
     進入時の判定（対象／対象外）や停止の結果は前の訪問のものを引き継ぐ。

停止判定
  判定対象の訪問中に、IMUの静止（加速度・角速度がどちらも閾値未満）が
  STOP_DURATION 秒以上続けば「停止あり」。一度も続かずに退出したら「停止なし」。

使い方
  python process_log.py          … 未処理の走行IDだけ追加で処理する
  python process_log.py --all    … 全走行を処理し直す（判定ロジックを変えたとき）
"""
import json
import math
import os
import sys

import pandas as pd

# ===== 設定 =====
STOP_THRESHOLD_ACCEL = 0.8      # 加速度閾値
STOP_THRESHOLD_GYRO  = 12.0     # 角速度閾値 (°/s)
STOP_DURATION        = 2.0      # 停止継続時間 (秒)
GPS_RADIUS           = 10.0     # 近接半径 (m)
HEADING_TOLERANCE    = 45       # 方位許容角 (°)
GRID_SIZE            = 0.0005   # 地点検索用のグリッド (度) ≒ 50m。GPS_RADIUS より大きくすること

# ===== ファイルパス =====
LOG_FILE       = 'log.csv'
GEOJSON_FILE   = 'stop_edited.geojson'
NEAR_LOG_FILE  = 'near_log.csv'
ROUTE_LOG_FILE = 'route_log.csv'
VISIT_LOG_FILE = 'visit_log.csv'    # 対象外を含む全訪問の記録（検証用）

NEAR_COLUMNS  = ['走行ID', '時刻(JST)', '地点ID', '緯度', '経度', '距離(m)',
                 '速度(km/h)', '方位角(deg)', '停止',
                 '地点緯度', '地点経度']   # 地点緯度・経度 = GeoJSONの一時停止地点の座標（地図のピン用）
VISIT_COLUMNS = ['走行ID', '進入時刻', '退出時刻', '地点ID', '地点方位(deg)',
                 '進入方位(deg)', '方位差(deg)', '判定対象', '停止', '最小距離(m)', '最長静止(s)']
ROUTE_COLUMNS = ['走行ID', '時刻(JST)', '緯度', '経度', '速度(km/h)', '方位角(deg)']


# ===== 一時停止地点の読み込み =====
def parse_direction(props):
    """地点の向き（度）。数値で入っていなければ None"""
    for key in ('direction_deg', 'direction'):
        try:
            return float(props.get(key)) % 360
        except (TypeError, ValueError):
            continue
    return None


def load_stop_points(filepath):
    with open(filepath) as f:
        data = json.load(f)
    stops = []
    for i, feature in enumerate(data['features']):
        lon, lat = feature['geometry']['coordinates'][:2]
        stops.append({'id': i + 1, 'lat': lat, 'lon': lon,
                      'direction': parse_direction(feature['properties'])})
    return stops


def build_grid(stops):
    """近くの地点だけを調べるためのグリッド（14万地点を毎行すべて調べないため）"""
    grid = {}
    for s in stops:
        grid.setdefault(grid_key(s['lat'], s['lon']), []).append(s)
    return grid


def grid_key(lat, lon):
    return (math.floor(lat / GRID_SIZE), math.floor(lon / GRID_SIZE))


def stops_in_radius(lat, lon, grid):
    """半径 GPS_RADIUS 以内の地点を {地点ID: (地点, 距離)} で返す"""
    gy, gx = grid_key(lat, lon)
    found = {}
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            for s in grid.get((gy + dy, gx + dx), []):
                d = haversine(lat, lon, s['lat'], s['lon'])
                if d <= GPS_RADIUS:
                    found[s['id']] = (s, d)
    return found


# ===== ユーティリティ =====
def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def heading_diff(a, b):
    d = abs(a - b) % 360
    return 360 - d if d > 180 else d


def to_float(v):
    try:
        f = float(v)
        return None if math.isnan(f) else f
    except (TypeError, ValueError):
        return None


def fmt(v, nd=1):
    return '--' if v is None else f"{v:.{nd}f}"


# ===== 1走行の処理 =====
def process_ride(ride_id, group, grid):
    near_rows, visit_rows = [], []

    # 設計1：地点ごとの訪問状態。半径の外に出るまで保持する
    visits = {}
    # 設計4：退出したがまだ確定していない訪問。ほかの地点に進入した時点で確定する
    exited = {}
    still_start = None      # 走行全体で静止が続いている開始時刻（入り直し時の静止継続判定用）

    def finish_visit(v):
        """退出時の記録。判定対象なら near_log に1行出す"""
        s = v['stop']
        visit_rows.append({
            '走行ID': ride_id, '進入時刻': v['entry_time'], '退出時刻': v['last_time'],
            '地点ID': s['id'], '地点方位(deg)': fmt(s['direction'], 0),
            '進入方位(deg)': fmt(v['entry_heading']), '方位差(deg)': fmt(v['diff']),
            '判定対象': v['target'], '停止': v['stopped'],
            '最小距離(m)': fmt(v['min_dist']), '最長静止(s)': fmt(v['max_still']),
        })
        if not v['target']:
            return                              # 設計3：対象外は記録しない
        r, d = (v['stop_row'], v['stop_dist']) if v['stopped'] else (v['last_row'], None)
        near_rows.append({
            '走行ID': ride_id, '時刻(JST)': r['time'], '地点ID': s['id'],
            '緯度': f"{r['lat']:.6f}", '経度': f"{r['lon']:.6f}", '距離(m)': fmt(d),
            '速度(km/h)': fmt(r['speed'], 2), '方位角(deg)': fmt(r['heading'], 2),
            '停止': 'True' if v['stopped'] else 'False',
            '地点緯度': f"{s['lat']:.7f}", '地点経度': f"{s['lon']:.7f}",
        })

    for _, row in group.iterrows():
        r = {
            'time': row['時刻(JST)'], 't': pd.Timestamp(row['時刻(JST)']),
            'lat': to_float(row['緯度']), 'lon': to_float(row['経度']),
            'speed': to_float(row['速度(km/h)']), 'heading': to_float(row['方位角(deg)']),
            'accel': to_float(row['加速度']), 'gyro': to_float(row['角速度']),
        }
        if None in (r['lat'], r['lon'], r['accel'], r['gyro']):
            continue
        is_still = r['accel'] < STOP_THRESHOLD_ACCEL and r['gyro'] < STOP_THRESHOLD_GYRO
        if is_still:
            if still_start is None:
                still_start = r['t']
        else:
            still_start = None

        inside = stops_in_radius(r['lat'], r['lon'], grid)

        # 退出：前の行までは半径内にいて、この行で外に出た地点。すぐには確定せず保留する
        for sid in [sid for sid in visits if sid not in inside]:
            exited[sid] = visits.pop(sid)

        # 設計4：ほかの地点に進入したら、それより前に退出した訪問を確定する
        entering = [sid for sid in inside if sid not in visits]
        for sid in list(exited):
            if any(e != sid for e in entering):
                finish_visit(exited.pop(sid))

        for sid, (s, d) in inside.items():
            # 設計4：退出後、ほかの地点に入る前に同じ地点へ戻った → 前の訪問の続き
            if sid not in visits and sid in exited:
                v = exited.pop(sid)
                # 半径の外にいた間も静止が途切れていなければ、静止時間をそのまま数え続ける
                if not (v['still_start'] is not None and still_start is not None
                        and still_start <= v['still_start']):
                    v['still_start'] = None
                visits[sid] = v

            # 進入：この行で初めて半径に入った地点
            if sid not in visits:
                # 設計2：進入した瞬間の方位だけで判定対象かどうかを決める
                if s['direction'] is None or r['heading'] is None:
                    diff, target = None, True       # 判断材料がなければ対象に含める
                else:
                    diff = heading_diff(r['heading'], s['direction'])
                    target = diff <= HEADING_TOLERANCE
                visits[sid] = {
                    'stop': s, 'target': target, 'entry_time': r['time'],
                    'entry_heading': r['heading'], 'diff': diff,
                    'still_start': None, 'max_still': 0.0,
                    'stopped': False, 'stop_row': None, 'stop_dist': None,
                    'min_dist': d,
                }

            v = visits[sid]
            v['last_row'], v['last_time'] = r, r['time']
            v['min_dist'] = min(v['min_dist'], d)

            # 設計3：対象外の地点では停止判定をしない
            if not v['target']:
                continue

            # 停止判定：訪問中に静止が STOP_DURATION 秒続いたら停止あり
            if is_still:
                if v['still_start'] is None:
                    v['still_start'] = r['t']
                still = (r['t'] - v['still_start']).total_seconds()
                v['max_still'] = max(v['max_still'], still)
                if still >= STOP_DURATION and not v['stopped']:
                    v['stopped'], v['stop_row'], v['stop_dist'] = True, r, d
            else:
                v['still_start'] = None

    # 走行の終わりで、半径内に残っている訪問と保留中の訪問を確定する
    for v in list(exited.values()) + list(visits.values()):
        finish_visit(v)

    near_rows.sort(key=lambda x: x['時刻(JST)'])
    visit_rows.sort(key=lambda x: x['進入時刻'])
    return near_rows, visit_rows


# ===== メイン処理 =====
def read_csv_if_exists(path):
    if os.path.exists(path):
        try:
            return pd.read_csv(path, dtype=str)
        except Exception:
            pass
    return None


def append_and_write(path, new_rows, columns, old_df):
    df = pd.DataFrame(new_rows, columns=columns)
    if old_df is not None and len(old_df):
        df = pd.concat([old_df.reindex(columns=columns), df], ignore_index=True)
    df.to_csv(path, index=False)
    print(f"{path} を出力しました（{len(df)}行、うち新規{len(new_rows)}行）")


def main():
    reprocess_all = '--all' in sys.argv

    print("ログ読み込み中...")
    df = pd.read_csv(LOG_FILE)
    stops = load_stop_points(GEOJSON_FILE)
    grid = build_grid(stops)
    print(f"ログ行数: {len(df)}, 一時停止地点数: {len(stops)}")

    old = {path: (None if reprocess_all else read_csv_if_exists(path))
           for path in (NEAR_LOG_FILE, ROUTE_LOG_FILE, VISIT_LOG_FILE)}

    # 処理済みかどうかは route_log で判断する（一時停止地点を通らなかった走行も含めるため）
    done = set()
    if old[ROUTE_LOG_FILE] is not None:
        done = set(old[ROUTE_LOG_FILE]['走行ID'].astype(int))
    new_ids = [rid for rid in sorted(df['走行ID'].unique()) if int(rid) not in done]
    if not new_ids:
        print("新しい走行データはありません")
        return
    print(f"処理対象: 走行ID {[int(x) for x in new_ids]}")

    all_near, all_visits = [], []
    for ride_id, group in df[df['走行ID'].isin(new_ids)].groupby('走行ID'):
        near, visits = process_ride(int(ride_id), group, grid)
        all_near += near
        all_visits += visits
        n_stop = sum(r['停止'] == 'True' for r in near)
        n_skip = sum(not v['判定対象'] for v in visits)
        rate = n_stop / len(near) * 100 if near else 0
        print(f"  走行ID {int(ride_id)}: 遵守 {n_stop}/{len(near)} ({rate:.0f}%)  対象外 {n_skip}件")

    append_and_write(NEAR_LOG_FILE, all_near, NEAR_COLUMNS, old[NEAR_LOG_FILE])
    append_and_write(VISIT_LOG_FILE, all_visits, VISIT_COLUMNS, old[VISIT_LOG_FILE])
    route_rows = df[df['走行ID'].isin(new_ids)][ROUTE_COLUMNS].to_dict('records')
    append_and_write(ROUTE_LOG_FILE, route_rows, ROUTE_COLUMNS, old[ROUTE_LOG_FILE])


if __name__ == '__main__':
    main()
