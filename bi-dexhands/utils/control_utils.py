""" The utility function for PID control
"""
import numpy as np
import torch
from utils.torch_jit_utils import *

# PID control for rotation
def pid_control_rotation(target_rot, current_rot, kp):
    quat_diff = quat_mul(current_rot, quat_conjugate(target_rot))
    quat_diff = torch.nn.functional.normalize(quat_diff, dim=1)
    # compute torque from quaternion difference
    angle = torch.acos(torch.clamp(quat_diff[:, 3:4], -1.0, 1.0))
    axis_norm = torch.sqrt(1.0 - quat_diff[:, 3:4] * quat_diff[:, 3:4])
    axis_x = quat_diff[:, 0:1] / axis_norm
    axis_y = quat_diff[:, 1:2] / axis_norm
    axis_z = quat_diff[:, 2:3] / axis_norm
    torque = torch.hstack([axis_x, axis_y, axis_z]) * angle

    return - kp * torque