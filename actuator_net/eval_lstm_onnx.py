"""Evaluate LSTM actuator networks (ONNX) on the held-out eval set.

Runs single-step sequential inference — h and c are carried between
timesteps — so results match the C++ runtime behaviour exactly.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import onnxruntime as ort
import datetime

from utils import JOINT_GROUPS, EVAL_PKL_NAME, load_single_experiment

EXPERIMENT_DIR = '/home/dyros/scraps/actuator_net/data/pkl'
ONNX_DIR       = '/home/dyros/scraps/actuator_net'
HIDDEN_SIZE    = 32
NUM_LAYERS     = 3

eval_pkl_path = os.path.join(EXPERIMENT_DIR, EVAL_PKL_NAME)
if not os.path.exists(eval_pkl_path):
    raise FileNotFoundError(f"Eval pkl not found: {eval_pkl_path}")

jpe, jv, te = load_single_experiment(eval_pkl_path, torque_scaling=0.01)
N = jpe.shape[0]
print(f"Eval set: {EVAL_PKL_NAME}  ({N} timesteps)\n")

n_joints = len(JOINT_GROUPS)
n_cols   = 2
n_rows   = (n_joints + 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 3.5 * n_rows))
axes = axes.flatten()

results = {}
for idx, (joint_indices, group_name) in enumerate(JOINT_GROUPS):
    onnx_path = os.path.join(ONNX_DIR, f"p73_lstm_{group_name}.onnx")
    ax = axes[idx]

    if not os.path.exists(onnx_path):
        ax.set_title(f"{group_name}\n(no model)", fontsize=8)
        ax.axis('off')
        print(f"[{group_name}] No ONNX model found — skipped: {onnx_path}")
        continue

    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads = 1
    sess = ort.InferenceSession(onnx_path,
                                sess_options=sess_opts,
                                providers=['CPUExecutionProvider'])

    ji = joint_indices[0]

    # Model was trained with h=0, c=0 reset for every independent sample
    # (shuffled DataLoader, seq_len=1, no state carried during training).
    # Must reset each step to match training behaviour.
    h_zero = np.zeros((NUM_LAYERS, 1, HIDDEN_SIZE), dtype=np.float32)
    c_zero = np.zeros((NUM_LAYERS, 1, HIDDEN_SIZE), dtype=np.float32)

    y_pred_scaled = np.zeros(N, dtype=np.float32)

    for t in range(N):
        x = np.array([[[float(jpe[t, ji]), float(jv[t, ji])]]], dtype=np.float32)  # (1,1,2)
        out, _, _ = sess.run(
            ['output', 'h_out', 'c_out'],
            {'input': x, 'h_in': h_zero, 'c_in': c_zero},
        )
        y_pred_scaled[t] = out[0, 0]  # model output in scaled units (Nm * 0.01)

    # Convert to Nm (same unscaling as C++: divide by 0.01)
    torque_scale = 0.01
    y_pred_Nm = y_pred_scaled / torque_scale
    y_true_Nm = te[:, ji].numpy() / torque_scale

    mse = float(np.mean((y_pred_Nm - y_true_Nm) ** 2))
    mae = float(np.mean(np.abs(y_pred_Nm - y_true_Nm)))
    print(f"[{group_name:22s}]  MSE={mse:.4f}  MAE={mae:.4f} Nm")
    results[group_name] = {"mse": mse, "mae": mae}

    t_axis = np.arange(N) * 0.001  # 1 kHz data
    ax.plot(t_axis, y_true_Nm, label="Measured",  color="green", linewidth=1.5)
    ax.plot(t_axis, y_pred_Nm, label="LSTM pred", color="red",   linewidth=0.8, linestyle="--")
    ax.set_title(f"{group_name}  MSE={mse:.3f}  MAE={mae:.3f} Nm", fontsize=8)
    ax.set_ylabel("Torque [Nm]", fontsize=7)
    ax.set_xlabel("Time [s]", fontsize=7)
    ax.legend(fontsize=6)
    ax.grid(True, alpha=0.3)

for i in range(n_joints, len(axes)):
    axes[i].axis('off')

fig.suptitle(f"LSTM ActuatorNet (ONNX, sequential) — {EVAL_PKL_NAME}", fontsize=11)
plt.tight_layout()

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
out_path  = os.path.join(os.path.dirname(__file__), f"eval_lstm_onnx_{timestamp}.png")
plt.savefig(out_path, dpi=120)
plt.close()
print(f"\nSaved: {out_path}")

print("\nSummary:")
for name, r in results.items():
    print(f"  {name:22s}  MSE={r['mse']:.4f}  MAE={r['mae']:.4f} Nm")
