import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
import warnings
from scipy.stats import entropy
from scipy.signal import butter, sosfiltfilt, welch, iirnotch
from sklearn.decomposition import FastICA

# =============================================================================
# SAIM Analysis Pipeline v1.0 (Official Release)
# =============================================================================
# Systemic Attractor Instability Metric (SAIM) - PNAS Submission Version
#
# Designed for: Raw EEG/Physiological Data (256 Hz, Constant Interval)
# Description:
#   This pipeline computes the Free Energy proxy (F) and associated metrics
#   from raw physiological signals. It utilizes Independent Component Analysis (ICA)
#   for artifact removal and Welch's method for spectral density estimation.
#
# Key Parameters:
#   - Sampling Rate: 256 Hz
#   - Lambda (Scaling Factor): 0.5 (Dynamic Range Equalizer)
#   - Window Size: 10.0 seconds
#
# Author: Takafumi Shiga
# Date: January 2026
# =============================================================================

warnings.filterwarnings('ignore')

class SAIMConfig:
    # --- Analysis Parameters ---
    WINDOW_SEC = 10
    STEP_SEC = 2
    FS = 256.0
    EPS = 1e-9

    # --- Frequency Bands ---
    BANDS = {
        'Delta': (1, 4),
        'Theta': (4, 8),
        'Alpha': (8, 13),
        'Beta':  (13, 30),
        'Gamma': (30, 45)
    }

    # --- Metrics to Export ---
    METRIC_COLS = [
        'PE', 'F', 'SII', 'I',
        'NCI', 'NCI_Vol', 'LZC', 'SOM', 'HEMO', 'AUT',
        'Gamma', 'Beta', 'Alpha', 'Theta', 'Delta',
        'HSI_Gamma', 'HSI_Beta', 'HSI_Alpha', 'HSI_Theta', 'HSI_Delta'
    ]

    # --- HEMO Configuration (Muse S Gen2 / Athena MS-03 Mapping) ---
    # 730nm (Left/Right, Inner/Outer): 1, 5, 6, 2
    # 850nm (Left/Right, Inner/Outer): 3, 7, 8, 4
    HEMO_SIGNAL_CHANNELS = [
        'Optics1', 'Optics2', 'Optics3', 'Optics4',
        'Optics5', 'Optics6', 'Optics7', 'Optics8'
    ]
    # Ambient (Env Light Correction): 11, 15, 16, 12
    HEMO_AMBIENT_CHANNELS = ['Optics11', 'Optics12', 'Optics15', 'Optics16']
    
    # --- Visualization Colors ---
    COLORS = {
        'PE': '#DC143C',   'F': '#8B0000',    'SII': '#800080',  'I': '#4682B4',
        'NCI': '#00BFFF',  'NCI_Vol': '#1E90FF', 'LZC': '#DAA520',
        'SOM': '#FFD700',  'HEMO': '#FF4500', 'AUT': '#2E8B57',
        'Gamma': '#FF1493', 'Beta': '#8A2BE2', 'Alpha': '#32CD32', 'Theta': '#FF8C00', 'Delta': '#708090',
        'HSI_Gamma': '#FF69B4', 'HSI_Beta': '#9370DB', 'HSI_Alpha': '#98FB98', 
        'HSI_Theta': '#FFA07A', 'HSI_Delta': '#B0C4DE'
    }

# --- Signal Processing Helper Functions ---
def filter_data(data, fs):
    # Bandpass 0.5-100Hz
    sos = butter(4, [0.5, 100.0], btype='band', fs=fs, output='sos')
    filt = sosfiltfilt(sos, data, axis=1)
    # Notch 50Hz & 60Hz
    for f in [50, 60]:
        b, a = iirnotch(f, 30, fs)
        from scipy.signal import filtfilt
        filt = filtfilt(b, a, filt, axis=1)
    return filt

def clean_with_ica(data):
    # FastICA for artifact removal
    try:
        ica = FastICA(n_components=4, random_state=42, max_iter=200)
        S = ica.fit_transform(data.T)
        A = ica.mixing_
        # Reject blink artifacts (High Variance)
        vars = np.var(S, axis=0)
        if np.max(vars) > 5 * np.median(vars):
            S[:, np.argmax(vars)] = 0
        return np.dot(S, A.T).T
    except:
        return data

def calc_entropy(s):
    try:
        counts, _ = np.histogram(s.dropna(), bins='fd')
        if np.sum(counts) == 0: return 0.0
        p = counts / np.sum(counts)
        p = p[p > 0]
        ent = -np.sum(p * np.log(p))
        
        # NORMALIZATION (Dynamic Range Equalization)
        n_bins = len(counts)
        if n_bins > 1:
            return ent / np.log(n_bins)
        else:
            return 0.0
    except: return np.nan

def calc_lzc(s):
    try:
        b = (s > s.median()).astype(int)
        seq = "".join(map(str, b)); n = len(seq)
        if n == 0: return np.nan
        i, k, l = 0, 1, 1; k_max = 1
        while i+k+l-1 < n:
            if seq[i+k : i+k+l] in seq[i : i+k+l-1]: l += 1
            else: k_max += 1; i += k; k = 1; l = 1
        return k_max / (n / np.log2(n))
    except: return np.nan

# --- Main Analysis Class ---
class SAIMAnalyzer:
    def __init__(self, file_path):
        self.file_path = file_path
        base = os.path.basename(file_path).replace('.csv', '')
        if "mindMonitor" in base:
            self.subject_id = "Subj"
            self.visit_id = base.split('--')[-1] if '--' in base else "Test"
        else:
            self.subject_id = "S00"; self.visit_id = "V1"
        
        self.prefix = f"{self.subject_id}_{self.visit_id}"
        self.results = []
        try: self.E_inv = np.linalg.pinv(SAIMConfig.E)
        except: self.E_inv = None

    def _process_window(self, df_win):
        metrics = {}
        
        # 1. EEG
        raw_cols = ['RAW_TP9', 'RAW_AF7', 'RAW_AF8', 'RAW_TP10']
        if all(c in df_win.columns for c in raw_cols):
            raw = df_win[raw_cols].values.T
            clean = clean_with_ica(filter_data(raw, SAIMConfig.FS))
            
            freqs, psd = welch(clean, fs=SAIMConfig.FS, nperseg=int(SAIMConfig.FS))
            
            for band, (l, h) in SAIMConfig.BANDS.items():
                idx = np.logical_and(freqs >= l, freqs <= h)
                pwr = np.sum(psd[:, idx], axis=1)
                metrics[band] = np.mean(pwr)
                P_L = np.mean(pwr[[0, 1]]); P_R = np.mean(pwr[[2, 3]])
                metrics[f'HSI_{band}'] = 1.0 - abs(P_L - P_R)/(P_L + P_R + SAIMConfig.EPS)
                
            metrics['PE'] = metrics['Gamma'] / metrics['Delta'] if metrics['Delta'] > 0 else 0
            metrics['NCI'] = np.mean([calc_entropy(df_win[c]) for c in raw_cols])
            metrics['LZC'] = np.mean([calc_lzc(df_win[c]) for c in raw_cols])

        # 2. Physiological
        acc = [c for c in df_win.columns if 'Accelerometer' in c]
        if acc: metrics['SOM'] = 1.0 / (1.0 + np.std(np.sqrt(np.sum(df_win[acc]**2, axis=1))))
        
        # --- HEMO (Hemodynamic Coupling) - UPDATED for Athena MS-03 ---
        sig_cols = [c for c in SAIMConfig.HEMO_SIGNAL_CHANNELS if c in df_win.columns]
        amb_cols = [c for c in SAIMConfig.HEMO_AMBIENT_CHANNELS if c in df_win.columns]
        
        if sig_cols:
            # Mean variance across signal channels (730nm + 850nm)
            sig_var = df_win[sig_cols].var().mean()
            # Mean variance across ambient channels (Noise)
            amb_var = df_win[amb_cols].var().mean() if amb_cols else 0.0
            
            # Normalized Stability Metric
            metrics['HEMO'] = 1.0 / (1.0 + (sig_var / (amb_var + SAIMConfig.EPS)))
        else:
            metrics['HEMO'] = np.nan
        
        # --- AUT ---
        if 'Heart_Rate' in df_win.columns: metrics['AUT'] = calc_entropy(df_win['Heart_Rate'])

        # 3. Systemic
        comps = [metrics.get(k, np.nan) for k in ['HSI_Gamma', 'SOM', 'HEMO', 'AUT']]
        metrics['I'] = np.nanmean(comps)
        if not np.isnan(metrics['I']) and 'PE' in metrics:
            metrics['F'] = (1 - metrics['I']) + 0.5 * metrics['PE']
            metrics['SII'] = 1.0 / (1.0 + np.exp(-5 * (metrics['I'] - metrics['PE'])))
            
        return metrics

    def analyze(self):
        print(f"Analyzing {self.file_path}...")
        try:
            df = pd.read_csv(self.file_path)
            if 'TimeStamp' in df.columns: df['TimeStamp'] = pd.to_datetime(df['TimeStamp'], errors='coerce')
            df = df.dropna(subset=['RAW_TP9']).sort_values('TimeStamp').reset_index(drop=True)
            
            if len(df) < SAIMConfig.FS * 5: 
                print("Error: Data too short (< 5 sec)")
                return False
            
            start_t = df['TimeStamp'].iloc[0]
            end_t = df['TimeStamp'].iloc[-1]
            curr_t = start_t
            
            while curr_t + pd.Timedelta(seconds=SAIMConfig.WINDOW_SEC) <= end_t:
                end_win = curr_t + pd.Timedelta(seconds=SAIMConfig.WINDOW_SEC)
                win = df[(df['TimeStamp'] >= curr_t) & (df['TimeStamp'] < end_win)]
                if len(win) > SAIMConfig.FS:
                    m = self._process_window(win)
                    m['Time'] = (curr_t - start_t).total_seconds()
                    m['Phase'] = "Observation"
                    m['Session'] = "Session1"
                    m['Condition'] = "Rest"
                    self.results.append(m)
                curr_t += pd.Timedelta(seconds=SAIMConfig.STEP_SEC)
            return True
        except Exception as e:
            print(f"Analysis Error: {e}")
            return False

    def save_outputs(self):
        if not self.results: return
        df_res = pd.DataFrame(self.results)
        
        if 'NCI' in df_res.columns:
            df_res['NCI_Vol'] = df_res['NCI'].rolling(5, center=True).std().fillna(0)
            
        # Save CSVs & Plots
        ts_name = f"{self.prefix}_TimeSeries.csv"
        df_res.to_csv(ts_name, index=False)
        self.plot_continuous(df_res)
        self.plot_omni(df_res)
        
        # Save Stats
        ov_name = f"{self.prefix}_Overall_Stats.csv"
        metrics = [c for c in SAIMConfig.METRIC_COLS if c in df_res.columns]
        df_res.groupby('Session')[metrics].mean().to_csv(ov_name)
        
        sub_name = f"{self.prefix}_SubPhase_Stats.csv"
        df_res.groupby(['Session', 'Condition'])[metrics].agg(['mean', 'std']).to_csv(sub_name)
        
        print(f"Generated output: {ts_name}")

    def plot_continuous(self, df):
        plt.figure(figsize=(12, 10))
        cols = ['PE', 'F', 'I', 'SII', 'NCI']
        for i, c in enumerate(cols):
            if c in df.columns:
                plt.subplot(len(cols), 1, i+1)
                plt.plot(df['Time'], df[c], color=SAIMConfig.COLORS.get(c, 'black'), linewidth=2)
                plt.ylabel(c); plt.grid(True, alpha=0.3)
        plt.xlabel('Time (s)')
        plt.suptitle(f"Continuous Dynamics ({self.prefix})")
        plt.tight_layout()
        out_name = f"{self.prefix}_Continuous_Dynamics.png"
        plt.savefig(out_name, dpi=300)
        plt.close()

    def plot_omni(self, df):
        fig, axes = plt.subplots(5, 4, figsize=(20, 20))
        axes = axes.flatten()
        plot_list = [
            'F', 'PE', 'SII', 'SOM',
            'I', 'LZC', 'NCI', 'HEMO',
            'Gamma', 'Beta', 'Alpha', 'Theta',
            'Delta', 'HSI_Gamma', 'HSI_Beta', 'HSI_Alpha',
            'HSI_Theta', 'HSI_Delta', 'AUT', 'NCI_Vol'
        ]
        stats = df.groupby('Condition')[plot_list].mean().reset_index()
        for i, metric in enumerate(plot_list):
            if i < len(axes) and metric in stats.columns:
                sns.barplot(data=stats, x='Condition', y=metric, ax=axes[i], 
                            color=SAIMConfig.COLORS.get(metric, 'grey'))
                axes[i].set_title(metric)
        plt.tight_layout()
        out_name = f"{self.prefix}_OmniPanel.png"
        plt.savefig(out_name, dpi=300)
        plt.close()

if __name__ == "__main__":
    files = glob.glob("*.csv")
    for f in files:
        if "Stats" in f or "TimeSeries" in f: continue
        analyzer = SAIMAnalyzer(f)
        if analyzer.analyze():
            analyzer.save_outputs()
