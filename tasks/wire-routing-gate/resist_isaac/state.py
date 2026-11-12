"""Read native state and initialize passive joints during a task reset."""

import math


def tensor(value):
    """Expose the installed Isaac backend's tensor without replacing simulator data."""
    return value.torch if hasattr(value, "torch") else value


def unit_quaternion(value):
    """Reject non-finite or non-unit native orientation values."""
    values = list(map(float, value))
    return (len(values) == 4 and all(math.isfinite(v) for v in values)
            and abs(sum(v*v for v in values) - 1.0) <= 0.001)


def pose_state(asset, body=None):
    """Read link-origin pose and COM velocity for environment zero, in world coordinates."""
    if body is None:
        pose = tensor(asset.data.root_link_pose_w)[0]
        velocity = tensor(asset.data.root_com_vel_w)[0]
    else:
        index = asset.body_names.index(body)
        pose = tensor(asset.data.body_link_pose_w)[0, index]
        velocity = tensor(asset.data.body_com_vel_w)[0, index]
    return pose[:3], pose[3:], velocity


def gripper_distance(base, position):
    """Minimum TCP distance; a geometric clearance proxy, not a contact verdict."""
    from yam_arena.runtime import tcp_world

    point = tuple(map(float, position))
    if not all(math.isfinite(value) for value in point):
        raise ValueError("Cannot measure gripper distance to non-finite state")
    return min(math.dist(tcp_world(base.scene[name]), point) for name in ("left_robot", "right_robot"))


def reset_joints(base, object_name, positions):
    """Initialize named passive joint coordinates and zero velocity before stepping."""
    import torch

    if not positions or not all(math.isfinite(value) for value in positions.values()):
        raise ValueError("Reset positions must be nonempty and finite")
    asset = base.scene[object_name]
    names = list(positions)
    ids = asset.find_joints(names, preserve_order=True)[0]
    if len(ids) != len(names):
        raise ValueError(f"Missing reset joint in {object_name}: {names}")
    current = tensor(asset.data.joint_pos)
    values = torch.tensor([list(positions.values())], dtype=current.dtype, device=current.device)
    values = values.expand(current.shape[0], -1).contiguous()
    asset.write_joint_state_to_sim_index(position=values, velocity=torch.zeros_like(values), joint_ids=ids)
