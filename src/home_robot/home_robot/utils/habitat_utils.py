import math

def quat_wxyz_to_yaw(rotation) -> float:
    """
    rotation: a quaternion object in (w, x, y, z) layout, e.g. numpy-quaternion / Habitat:
              rotation = quaternion(w, x, y, z)

    Returns:
        yaw in radians, normalized to (-pi, pi]
    """
    # numpy-quaternion uses attributes: rotation.w, rotation.x, rotation.y, rotation.z
    w = float(rotation.w)
    x = float(rotation.x)
    y = float(rotation.y)
    z = float(rotation.z)

    # General yaw extraction (Y-up convention)
    yaw = math.atan2(
        2.0 * (w * y + x * z),
        1.0 - 2.0 * (x * x + y * y),
    )

    # normalize to (-pi, pi]
    yaw = (yaw + math.pi) % (2.0 * math.pi) - math.pi
    return yaw

def rad_norm(yaw):
    yaw = (yaw + math.pi) % (2.0 * math.pi) - math.pi
    return yaw

# aa.base_rot = base
# self._sim.maybe_update_articulated_agent()
# obs = self._sim.get_sensor_observations()['head_rgb']
# imageio.imwrite('datadump/map_debug/rgb_debug.png', obs)