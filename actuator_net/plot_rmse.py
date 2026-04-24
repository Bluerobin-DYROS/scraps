"""
Plot predicted vs measured joint-space torque for Right Ankle Pitch.
Top: 20–25 s overview with circular zoom indicator
Bottom: zoomed peak region with connector lines
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch, Ellipse
import torch
from utils import JOINT_GROUPS, load_single_experiment

# ─── Configuration ───────────────────────────────────────────────────────────
EXPERIMENT_DIR = '/home/user/actuatornet/actuator_net/data/pkl'
MODEL_DIR      = '/home/user/actuatornet/actuator_net'
EVAL_PKL_NAME  = 'data_chirp_amplitude0.3_f00.1_f10.5_disturbance.pkl'
TORQUE_SCALE   = 0.01
TARGET_JOINT   = 'right_ankle_pitch'

T_START, T_END = 20.0, 25.0
PEAK_HALF = 0.25

FONT_TITLE  = 13
FONT_LABEL  = 11
FONT_TICK   = 10
FONT_LEGEND = 10
LINE_MEASURED = 1.6
LINE_PRED     = 1.2

# ─── Load data ───────────────────────────────────────────────────────────────
eval_pkl_path = os.path.join(EXPERIMENT_DIR, EVAL_PKL_NAME)
jpe, jv, te = load_single_experiment(eval_pkl_path, torque_scaling=TORQUE_SCALE)
N = jpe.shape[0]
t_axis = np.arange(N) * 0.001

ji = None
for indices, name in JOINT_GROUPS:
    if name == TARGET_JOINT:
        ji = indices[0]
        break
assert ji is not None

# ─── Inference ───────────────────────────────────────────────────────────────
model_path = os.path.join(MODEL_DIR, f"p73_lstm_{TARGET_JOINT}.pt")
model = torch.jit.load(model_path, map_location='cpu')
model.eval()

xs = torch.stack([jpe[:, ji], jv[:, ji]], dim=1).unsqueeze(1)
with torch.no_grad():
    y_pred_scaled, _ = model(xs)

y_pred_Nm = (y_pred_scaled[:, 0] / TORQUE_SCALE).numpy()
y_true_Nm = (te[:, ji] / TORQUE_SCALE).numpy()
t_np = t_axis

def time_slice(t, y_true, y_pred, t0, t1):
    mask = (t >= t0) & (t <= t1)
    return t[mask], y_true[mask], y_pred[mask]

t_ov, yt_ov, yp_ov = time_slice(t_np, y_true_Nm, y_pred_Nm, T_START, T_END)

peak_idx = np.argmax(np.abs(yt_ov))
peak_time = t_ov[peak_idx]
PEAK_START = peak_time - PEAK_HALF
PEAK_END   = peak_time + PEAK_HALF

print(f"Peak at {peak_time:.3f} s  (|τ| = {np.abs(yt_ov[peak_idx]):.2f} Nm)")

t_pk, yt_pk, yp_pk = time_slice(t_np, y_true_Nm, y_pred_Nm, PEAK_START, PEAK_END)

rmse = np.sqrt(np.mean((yp_ov - yt_ov) ** 2))
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif']  = ['Liberation Serif', 'DejaVu Serif', 'Times New Roman']
plt.rcParams['mathtext.fontset'] = 'stix'
# ─── Figure (tight_layout 먼저, ConnectionPatch 나중) ────────────────────────
fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(7.0, 4.5),
    gridspec_kw={'height_ratios': [1, 1], 'hspace': 0.65},
)

# ── Top: overview ──
ax1.plot(t_ov, yt_ov, label='Measured', color='#2ca02c', linewidth=LINE_MEASURED)
ax1.plot(t_ov, yp_ov, label='Predicted', color='#d62728',
         linewidth=LINE_PRED, linestyle='--')
ax1.set_title(
    f"Right Ankle Pitch  (RMSE = {rmse:.3f} Nm)",
    fontsize=FONT_TITLE, fontweight='bold',
)
ax1.set_ylabel('Torque [Nm]', fontsize=FONT_LABEL)
ax1.set_xlabel('Time [s]', fontsize=FONT_LABEL)
ax1.tick_params(labelsize=FONT_TICK)
ax1.legend(fontsize=FONT_LEGEND, loc='upper right')
ax1.grid(True, alpha=0.25)

# Ellipse
cx = peak_time
cy = yt_ov[peak_idx]
y_range = yt_ov.max() - yt_ov.min()
if y_range == 0:
    y_range = 1.0
rx_data = PEAK_HALF * 1.2
ry_data = y_range * 0.25

ellipse = Ellipse(
    (cx, cy), width=2*rx_data, height=2*ry_data,
    fill=False, edgecolor='black', linewidth=1.5, zorder=5
)
ax1.add_patch(ellipse)

# ── Bottom: zoom ──
ax2.plot(t_pk, yt_pk, label='Measured', color='#2ca02c', linewidth=LINE_MEASURED + 0.3)
ax2.plot(t_pk, yp_pk, label='Predicted', color='#d62728',
         linewidth=LINE_PRED + 0.3, linestyle='--')
ax2.set_title('Peak Detail', fontsize=FONT_TITLE, fontweight='bold')
ax2.set_ylabel('Torque [Nm]', fontsize=FONT_LABEL)
ax2.set_xlabel('Time [s]', fontsize=FONT_LABEL)
ax2.tick_params(labelsize=FONT_TICK)
ax2.legend(fontsize=FONT_LEGEND, loc='upper right')
ax2.grid(True, alpha=0.25)
ax2.set_xlim(PEAK_START, PEAK_END)

# ── 레이아웃 확정 먼저 ──
fig.subplots_adjust(hspace=0.55)
fig.canvas.draw()   # 모든 좌표 확정

# ── ConnectionPatch: 레이아웃 확정 후에 추가 ──
y2_top = ax2.get_ylim()[1]

con_left = ConnectionPatch(
    xyA=(cx - rx_data, cy - ry_data), coordsA=ax1.transData,
    xyB=(PEAK_START, y2_top),         coordsB=ax2.transData,
    color='black', linewidth=0.8, linestyle='--', alpha=0.6,
    clip_on=False,
)
con_right = ConnectionPatch(
    xyA=(cx + rx_data, cy - ry_data), coordsA=ax1.transData,
    xyB=(PEAK_END, y2_top),           coordsB=ax2.transData,
    color='black', linewidth=0.8, linestyle='--', alpha=0.6,
    clip_on=False,
)
fig.add_artist(con_left)
fig.add_artist(con_right)

# ── 저장 (tight_layout 호출 안 함 — 이미 확정됨) ──
out_path = os.path.join(MODEL_DIR, 'eval_right_ankle_pitch_zoom.png')
plt.savefig(out_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"Saved → {out_path}")
print(f"  RMSE = {rmse:.4f} Nm")

# ─── Peak torque 상세 출력 ───────────────────────────────────────────────────
peak_measured = yt_ov[peak_idx]
peak_predicted = yp_ov[peak_idx]
peak_error = peak_predicted - peak_measured

print(f"\n  Peak info ({T_START}–{T_END} s window):")
print(f"    Time         = {peak_time:.4f} s")
print(f"    Measured      = {peak_measured:.4f} Nm")
print(f"    Predicted     = {peak_predicted:.4f} Nm")
print(f"    Error         = {peak_error:.4f} Nm  ({abs(peak_error/peak_measured)*100:.2f}%)")
print(f"    |Peak|        = {abs(peak_measured):.4f} Nm")