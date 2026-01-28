import numpy as np
import pandas as pd
import glob
import os
from scipy.signal import welch, butter, filtfilt
from scipy.linalg import eigh
from scipy.stats import linregress
from scipy.ndimage import label

# ==========================================
# 1. 自動検出エンジン (S002以降用)
# ==========================================
def detect_timings_auto(df, fs=256):
    """
    S002以降のための自動検出ロジック
    ルール: 59秒以降のイベントを順に Jaw -> Eye -> Blink とする
    """
    raw_cols = [c for c in df.columns if 'RAW' in c and ('TP9' in c or 'AF7' in c or 'AF8' in c or 'TP10' in c)]
    if len(raw_cols) < 4:
        alt = ['TP9', 'AF7', 'AF8', 'TP10']
        raw_cols = []
        for t in alt:
             match = [c for c in df.columns if t in c and 'Delta' not in c]
             if match: raw_cols.append(match[0])
    if not raw_cols: return None

    # データ抽出 & 補間
    data_df = df[raw_cols].interpolate(method='linear', limit_direction='both').fillna(0)
    data = data_df.values
    
    # フィルタリング
    b, a = butter(4, 30 / (fs/2), btype='highpass')
    try:
        filt_data = filtfilt(b, a, data, axis=0)
    except: return None

    # パワー計算 (対数スケールでダイナミックレンジ圧縮)
    power = np.mean(filt_data**2, axis=1)
    power = np.maximum(power, 1e-10)
    log_power = np.log10(power)
    
    window = int(fs * 0.5)
    smooth_log_power = pd.Series(log_power).rolling(window=window, center=True).mean().fillna(np.min(log_power)).values
    
    # --- 59秒ルール ---
    scan_start_idx = int(fs * 59.0)
    
    # しきい値計算 (59秒以降のデータに基づく)
    if len(smooth_log_power) > scan_start_idx + (fs * 5):
        ref_data = smooth_log_power[scan_start_idx:]
    else:
        ref_data = smooth_log_power

    floor = np.percentile(ref_data, 5)
    ceiling = np.percentile(ref_data, 99)
    threshold = floor + (ceiling - floor) * 0.30 # 感度30%
    
    is_active = smooth_log_power > threshold
    if len(is_active) > scan_start_idx:
        is_active[:scan_start_idx] = False # 59秒以前は無視
    
    labeled_array, num_features = label(is_active)
    
    raw_events = []
    for i in range(1, num_features+1):
        idx = np.where(labeled_array == i)[0]
        if len(idx) > fs * 2.0:
            raw_events.append((idx[0], idx[-1]))
    
    merged_events = []
    if raw_events:
        curr_s, curr_e = raw_events[0]
        for next_s, next_e in raw_events[1:]:
            if (next_s - curr_e) < fs * 5.0: 
                curr_e = next_e
            else:
                merged_events.append((curr_s, curr_e))
                curr_s, curr_e = next_s, next_e
        merged_events.append((curr_s, curr_e))
    
    timings = {}
    
    # Rest (固定)
    rest_end = 55.0
    if len(df)/fs < 55.0: rest_end = (len(df)/fs) - 5.0
    timings['Rest'] = (10.0, rest_end)
    
    # Tasks
    task_order = ['Jaw', 'Eye', 'Blink']
    for i, task_name in enumerate(task_order):
        if i < len(merged_events):
            s, e = merged_events[i]
            duration = (e - s) / fs
            margin = min(1.0, duration * 0.1)
            timings[task_name] = ((s/fs)+margin, (e/fs)-margin)
        else:
            timings[task_name] = None
            
    return timings

# ==========================================
# 2. タイミング決定ロジック (ハイブリッド)
# ==========================================
def get_timings_hybrid(df, subject_id, fs=256):
    """
    S001なら固定値、それ以外なら自動検出を返す
    """
    if subject_id == "S001":
        # S001専用: グラフ検証済みの正解データ
        # Jaw: 69-79s, Eye: 87-93s, Blink: 100s+
        return {
            'Rest': (10.0, 50.0),
            'Jaw':  (71.0, 77.0), # 安定区間
            'Eye':  (88.0, 92.0),
            'Blink':(102.0, 108.0)
        }
    else:
        # S002以降: 自動検出
        return detect_timings_auto(df, fs)

# ==========================================
# 3. 共通解析クラス
# ==========================================
class RobustCCA:
    def __init__(self, n_components=4):
        self.n_components = n_components
        self.w_ = None 
    def fit(self, X):
        X = np.nan_to_num(X)
        X = X - np.mean(X, axis=1, keepdims=True)
        n_ch, n_times = X.shape
        if n_times < 2: return self
        X_t, X_t1 = X[:, :-1], X[:, 1:]
        try:
            R_xx = np.dot(X_t, X_t.T) / (n_times - 1)
            R_xy = np.dot(X_t, X_t1.T) / (n_times - 1)
            R_xy = (R_xy + R_xy.T) / 2
            eigvals, eigvecs = eigh(R_xy, R_xx)
            idx = np.argsort(eigvals)[::-1]
            self.w_ = eigvecs[:, idx[:self.n_components]]
        except: self.w_ = np.eye(n_ch)[:, :self.n_components]
        return self
    def transform(self, X):
        X = np.nan_to_num(X)
        X = X - np.mean(X, axis=1, keepdims=True)
        if self.w_ is None: return X
        return np.dot(self.w_.T, X)

def calc_simple_exponent(freqs, psd, f_range=[1, 50]):
    idx = np.logical_and(freqs >= f_range[0], freqs <= f_range[1])
    f_sel, p_sel = freqs[idx], psd[idx]
    valid = (f_sel > 0) & (p_sel > 0)
    f_sel, p_sel = f_sel[valid], p_sel[valid]
    if len(f_sel) < 2: return 0.0
    slope, _, _, _, _ = linregress(np.log10(f_sel), np.log10(p_sel))
    return -slope

def verify_segment(raw_df, start_sec, end_sec, fs=256):
    raw_cols = [c for c in raw_df.columns if 'RAW' in c and ('TP9' in c or 'AF7' in c or 'AF8' in c or 'TP10' in c)]
    if len(raw_cols) < 4:
        alt = ['TP9', 'AF7', 'AF8', 'TP10']
        raw_cols = []
        for t in alt:
            match = [c for c in raw_df.columns if t in c and 'Delta' not in c and 'Theta' not in c]
            if match: raw_cols.append(match[0])
    if len(raw_cols) < 4: return None
    start_idx = int(start_sec * fs)
    end_idx = int(end_sec * fs)
    if len(raw_df) < start_idx: return None
    if len(raw_df) < end_idx: end_idx = len(raw_df)
    data = raw_df[raw_cols].iloc[start_idx:end_idx].values.T
    data = np.nan_to_num(data)
    if data.shape[1] < fs: return None
    cca = RobustCCA(n_components=len(raw_cols))
    cca.fit(data)
    comps = cca.transform(data)
    freqs, psd = welch(comps, fs=fs, nperseg=min(fs*2, data.shape[1]))
    if len(freqs) == 0: return 0.0
    avg_psd = np.mean(psd, axis=0)
    exponent = calc_simple_exponent(freqs, avg_psd, f_range=[2, 45])
    return exponent

# ==========================================
# 4. メイン処理 (Production Mode)
# ==========================================
def batch_process_production(file_pattern="S*_Calibration.csv", fs=256):
    files = sorted(glob.glob(file_pattern))
    if not files:
        print(f"No files found matching: {file_pattern}")
        return

    print(f"Found {len(files)} files. Running PRODUCTION HYBRID ANALYSIS...\n")
    print(f"{'Subject':<8} | {'Rest':<5} | {'Jaw':<5} | {'Eye':<5} | {'Blink':<5} | {'Diff':<6} | {'Status'} | {'Timing Info (s)'}")
    print("-" * 115)

    for file_path in files:
        filename = os.path.basename(file_path)
        subj = filename.split('_')[0]

        try:
            df = pd.read_csv(file_path)
            
            # ★ハイブリッド判定呼び出し★
            timings = get_timings_hybrid(df, subj, fs)
            
            if timings is None:
                print(f"{subj:<8} | ERROR: Signal processing failed")
                continue
                
            results = {}
            time_str = ""
            for task in ['Rest', 'Jaw', 'Eye', 'Blink']:
                t_range = timings.get(task)
                if t_range:
                    exp = verify_segment(df, t_range[0], t_range[1], fs)
                    results[task] = exp
                    time_str += f"{task[0]}:{int(t_range[0])}-{int(t_range[1])} "
                else:
                    results[task] = None
                    time_str += f"{task[0]}:-- "

            exp_rest = results['Rest']
            exp_jaw = results['Jaw']
            
            if exp_rest is not None and exp_jaw is not None:
                diff = exp_rest - exp_jaw
                # 合格基準: 明確な差があること
                if diff > 0.3: status = "OK"
                elif diff > 0.1: status = "WEAK"
                else: status = "CHECK"
                diff_str = f"{diff:.2f}"
            else:
                diff_str = "N/A"
                status = "ERR"

            def fmt(v): return f"{v:.2f}" if v is not None else " - "
            print(f"{subj:<8} | {fmt(results['Rest']):<5} | {fmt(results['Jaw']):<5} | {fmt(results['Eye']):<5} | {fmt(results['Blink']):<5} | {diff_str:<6} | {status:<6} | {time_str}")
            
        except Exception as e:
            print(f"{subj:<8} | ERROR: {str(e)}")

    print("-" * 115)

# 実行
batch_process_production("S*_Calibration.csv")