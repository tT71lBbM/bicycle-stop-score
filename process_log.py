import pandas as pd
import math
import json

# ===== 設定 =====
STOP_THRESHOLD_ACCEL = 0.8      # 加速度閾値
STOP_THRESHOLD_GYRO  = 12.0     # 角速度閾値 (°/s)
STOP_DURATION        = 2.0      # 停止継続時間 (秒)
GPS_RADIUS           = 10.0     # 近接半径 (m)
HEADING_TOLERANCE    = 45       # 方位許容角 (°)

# ===== ファイルパス =====
LOG_FILE      = 'log.csv'
GEOJSON_FILE  = 'stop_edited.geojson'
NEAR_LOG_FILE  = 'near_log.csv'
ROUTE_LOG_FILE = 'route_log.csv'

# ===== GeoJSONから一時停止座標の読み込み =====
def load_stop_points(filepath):
    with open(filepath) as f:
        data = json.load(f)
    stops = []
    for i, feature in enumerate(data['features']):
        lon, lat = feature['geometry']['coordinates']
        direction = feature['properties'].get('direction', None)
        stops.append({'id': i + 1, 'lat': lat, 'lon': lon, 'direction': direction})
    return stops

# ===== ユーティリティ =====
def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

def check_heading(current_heading, direction_tag):
    if direction_tag is None or current_heading is None:
        return True
    if direction_tag == 'forward':
        target = 0.0
    elif direction_tag == 'backward':
        target = 180.0
    else:
        try:
            target = float(direction_tag)
        except:
            return True
    diff = abs(float(current_heading) - target) % 360
    if diff > 180:
        diff = 360 - diff
    return diff <= HEADING_TOLERANCE

def find_nearest_stop(lat, lon, heading, stop_points):
    best_dist = float('inf')
    best_stop = None
    best_heading_ok = False
    for stop in stop_points:
        dist = haversine(lat, lon, stop['lat'], stop['lon'])
        if dist <= GPS_RADIUS and dist < best_dist:
            best_dist = dist
            best_stop = stop
            best_heading_ok = check_heading(heading, stop['direction'])
    return best_stop, best_dist if best_stop else None, best_heading_ok

# ===== メイン処理 =====
print("ログ読み込み中...")
df = pd.read_csv(LOG_FILE)
stop_points = load_stop_points(GEOJSON_FILE)
print(f"ログ行数: {len(df)}, 一時停止地点数: {len(stop_points)}")

near_log_rows = []

for ride_id, group in df.groupby('走行ID'):
    group = group.reset_index(drop=True)
    print(f"\n走行ID {ride_id} を処理中（{len(group)}行）...")

    valid_entry_ids = set()
    prev_in_radius  = set()
    near_ids        = set()
    checked_ids     = set()
    stop_start_time = None
    near_start_time = None
    last_logged_id  = None
    stop_count      = 0
    total_near      = 0

    for i, row in group.iterrows():
        try:
            lat     = float(row['緯度'])
            lon     = float(row['経度'])
            speed   = float(row['速度(km/h)'])
            heading = row['方位角(deg)']
            accel   = float(row['加速度'])
            gyro    = float(row['角速度'])
            time_str = row['時刻(JST)']
        except:
            continue

        nearest_stop, dist, heading_ok = find_nearest_stop(lat, lon, heading, stop_points)
        stop_id = nearest_stop['id'] if nearest_stop else None
        in_radius = nearest_stop is not None

        # 侵入瞬間の方位チェック
        if in_radius and stop_id is not None:
            if stop_id not in prev_in_radius:
                if heading_ok:
                    valid_entry_ids.add(stop_id)
            is_near = stop_id in valid_entry_ids
        else:
            is_near = False

        # prev_in_radiusを更新
        if in_radius and stop_id is not None:
            prev_in_radius.add(stop_id)
        else:
            if stop_id in prev_in_radius:
                prev_in_radius.discard(stop_id)
                valid_entry_ids.discard(stop_id)

        is_stopped = (accel < STOP_THRESHOLD_ACCEL and gyro < STOP_THRESHOLD_GYRO)

        # 経過時間（近接開始からの秒数）
        if is_near:
            if near_start_time is None:
                near_start_time = pd.Timestamp(time_str)
            elapsed = (pd.Timestamp(time_str) - near_start_time).total_seconds()
        else:
            near_start_time = None
            elapsed = 0.0

        # 近接カウント
        if is_near and stop_id not in near_ids:
            near_ids.add(stop_id)
            total_near += 1

        # 停止判定
        if is_near and is_stopped:
            if stop_start_time is None:
                stop_start_time = pd.Timestamp(time_str)
            elapsed_stop = (pd.Timestamp(time_str) - stop_start_time).total_seconds()

            if elapsed_stop >= STOP_DURATION and stop_id not in checked_ids:
                stop_count += 1
                checked_ids.add(stop_id)

                if stop_id != last_logged_id:
                    near_log_rows.append({
                        '走行ID': ride_id,
                        '時刻(JST)': time_str,
                        '地点ID': stop_id,
                        '緯度': f"{lat:.6f}",
                        '経度': f"{lon:.6f}",
                        '距離(m)': f"{dist:.1f}",
                        '速度(km/h)': f"{speed:.2f}",
                        '方位角(deg)': heading,
                        '停止': 'True'
                    })
                    last_logged_id = stop_id
        else:
            if not is_near:
                if near_ids - checked_ids:
                    for mid in near_ids - checked_ids:
                        if mid != last_logged_id:
                            near_log_rows.append({
                                '走行ID': ride_id,
                                '時刻(JST)': time_str,
                                '地点ID': mid,
                                '緯度': f"{lat:.6f}",
                                '経度': f"{lon:.6f}",
                                '距離(m)': '--',
                                '速度(km/h)': f"{speed:.2f}",
                                '方位角(deg)': heading,
                                '停止': 'False'
                            })
                            last_logged_id = mid
                checked_ids     = set()
                near_ids        = set()
                stop_start_time = None
                last_logged_id  = None
                valid_entry_ids = set()
                prev_in_radius  = set()
            else:
                if stop_id not in checked_ids:
                    stop_start_time = None

    rate = stop_count / total_near * 100 if total_near > 0 else 0
    print(f"  結果: {stop_count}/{total_near} ({rate:.0f}%)")

# near_log.csv出力
near_df = pd.DataFrame(near_log_rows, columns=[
    '走行ID', '時刻(JST)', '地点ID', '緯度', '経度', '距離(m)', '速度(km/h)', '方位角(deg)', '停止'
])
near_df.to_csv(NEAR_LOG_FILE, index=False)
print(f"\nnear_log.csv を出力しました（{len(near_df)}行）")

# route_log.csv出力（全GPSログをそのまま出力）
route_df = df[['走行ID', '時刻(JST)', '緯度', '経度', '速度(km/h)', '方位角(deg)']].copy()
route_df.to_csv(ROUTE_LOG_FILE, index=False)
print(f"route_log.csv を出力しました（{len(route_df)}行）")
