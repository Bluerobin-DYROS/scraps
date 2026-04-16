import torch
import torch.nn as nn
import os

PT_DIR   = "/home/dyros/scraps/actuator_net"
ONNX_DIR = "/home/dyros/ros2_ws/src/p73_walker_controller/p73_lib/src/actuatornet_models"

JOINT_NAMES = [
    "left_hip_roll",   "left_hip_pitch",   "left_hip_yaw",
    "left_knee_pitch",
    "left_ankle_pitch","left_ankle_roll",
    "right_hip_roll",  "right_hip_pitch",  "right_hip_yaw",
    "right_knee_pitch",
    "right_ankle_pitch","right_ankle_roll",
]


class LSTMOnnxWrapper(nn.Module):
    """Flattens the nested (h, c) tuple so ONNX sees three flat outputs."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, h, c):
        out, (h_n, c_n) = self.model(x, (h, c))
        return out, h_n, c_n


for joint_name in JOINT_NAMES:
    pt_path   = os.path.join(PT_DIR,   f"p73_lstm_{joint_name}.pt")
    onnx_path = os.path.join(ONNX_DIR, f"p73_lstm_{joint_name}.onnx")

    if not os.path.exists(pt_path):
        print(f"Skipped (not found): {pt_path}")
        continue

    model = torch.jit.load(pt_path, map_location="cpu")
    model.eval()

    hidden_size = model.hidden_size
    num_layers  = model.num_layers

    wrapper = LSTMOnnxWrapper(model)
    wrapper.eval()

    # Dummy inputs matching real-time single-step inference:
    #   x     : (batch=1, seq_len=1, input_dim=2)  — [pos_error, velocity]
    #   h, c  : (num_layers, batch=1, hidden_size)
    x_dummy = torch.zeros(1, 1, 2)
    h_dummy = torch.zeros(num_layers, 1, hidden_size)
    c_dummy = torch.zeros(num_layers, 1, hidden_size)

    # Trace first: ONNX tracing cannot follow a TorchScript submodule that
    # wasn't registered inside the active trace.  torch.jit.trace executes
    # the wrapper eagerly and records all ops (including those inside the
    # scripted model), producing a graph ONNX can then serialise.
    wrapper_traced = torch.jit.trace(wrapper, (x_dummy, h_dummy, c_dummy))

    torch.onnx.export(
        wrapper_traced,
        (x_dummy, h_dummy, c_dummy),
        onnx_path,
        input_names=["input", "h_in", "c_in"],
        output_names=["output", "h_out", "c_out"],
        opset_version=11,
    )
    print(f"Converted: p73_lstm_{joint_name}.onnx  "
          f"(hidden={hidden_size}, layers={num_layers})")
