""" Provide math utilities for isaacgym & torch. """
from isaacgym import gymapi
from utils.torch_jit_utils import *

# Isaacgym
def quaternion_mul(quat_1, quat_2):
    """Multiply two quaternions.

    Args:
        quat_1 (isaacgym.gymapi.Quat): a quaternion of shape (4,).
        quat_2 (isaacgym.gymapi.Quat): a quaternion of shape (4,).

    Returns:
        isaacgym.gymapi.Quat: quat_1 * quat_2.
    """
    # Parse
    w1 = quat_1.w
    x1 = quat_1.x
    y1 = quat_1.y
    z1 = quat_1.z
    w2 = quat_2.w
    x2 = quat_2.x
    y2 = quat_2.y
    z2 = quat_2.z

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return gymapi.Quat(x, y, z, w)


# Torch
@torch.jit.script
def relative_velocity(vel_pos, vel_rot, rot_ref):
    """Compute the relative pose of a body with respect to a reference frame.

    Args:
        vel_pos (torch.Tensor): linear velocity of the body.
        vel_rot (torch.Tensor): rotation velocity of the body.
        rot_ref (torch.Tensor): rotation of the reference frame.

    Returns:
        torch.Tensor: relative linear velocity of the body.
        torch.Tensor: relative rotation velocity of the body.
    """
    # Compute relative position
    vel_pos_rel = quat_rotate_inverse(rot_ref, vel_pos) 

    # Compute relative rotation
    vel_rot_rel = quat_mul(quat_conjugate(rot_ref), vel_rot)

    return vel_pos_rel, vel_rot_rel


@torch.jit.script
def relative_pose(pos, rot, pos_ref, rot_ref):
    """Compute the relative pose of a body with respect to a reference frame.

    Args:
        pos (torch.Tensor): position of the body.
        rot (torch.Tensor): rotation of the body.
        pos_ref (torch.Tensor): position of the reference frame.
        rot_ref (torch.Tensor): rotation of the reference frame.

    Returns:
        torch.Tensor: relative position of the body.
        torch.Tensor: relative rotation of the body.
    """
    # Compute relative position
    pos_rel = quat_rotate_inverse(rot_ref, pos - pos_ref) 

    # Compute relative rotation
    rot_rel = quat_mul(quat_conjugate(rot_ref), rot)

    return pos_rel, rot_rel