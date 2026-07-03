from typing import TYPE_CHECKING, Any, Dict, Optional
import numpy as np
from habitat_sim.utils.common import quat_to_magnum


def get_phase_id_from_info(data_info: Dict) -> int:
    if 'is_curr_skill_NAV_TO_OBJ' in data_info and data_info['is_curr_skill_NAV_TO_OBJ']:
        phase_id = 1
        return phase_id
    if 'is_curr_skill_PICK' in data_info and data_info['is_curr_skill_NAV_TO_OBJ']:
        phase_id = 2
        return phase_id
    if 'is_curr_skill_NAV_TO_REC' in data_info and data_info['is_curr_skill_NAV_TO_OBJ']:
        phase_id = 3
        return phase_id
    if 'is_curr_skill_PLACE' in data_info and data_info['is_curr_skill_NAV_TO_OBJ']:
        phase_id = 4
        return phase_id
    return 0



def get_success_np(data_info: Dict) -> np.ndarray:
    result = np.zeros(4)
    if 'ovmm_find_object_phase_success' in data_info:
        result[0] = data_info['ovmm_find_object_phase_success']
    if 'ovmm_pick_object_phase_success' in data_info:
        result[1] = data_info['ovmm_pick_object_phase_success']
    if 'ovmm_find_recep_phase_success' in data_info:
        result[2] = data_info['ovmm_find_recep_phase_success']
    if 'ovmm_place_object_phase_success' in data_info:
        result[3] = data_info['ovmm_place_object_phase_success']
    return result

def get_phase_distance(
        last_info: Dict,
        next_info: Dict) -> np.ndarray:
    distance = np.zeros((2,2))
    if last_info['is_curr_skill_NAV_TO_OBJ'] or last_info['is_curr_skill_PICK']:
        distance[0,0] = last_info['ovmm_dist_to_pick_goal']
        distance[0,1] = last_info['ovmm_rot_dist_to_pick_goal']
        distance[1,0] = next_info['ovmm_dist_to_pick_goal']
        distance[1,1] = next_info['ovmm_rot_dist_to_pick_goal']
    if last_info['is_curr_skill_NAV_TO_REC'] or last_info['is_curr_skill_PLACE']:
        distance[0,0] = last_info['ovmm_dist_to_place_goal']
        distance[0,1] = last_info['ovmm_rot_dist_to_place_goal']
        distance[1,0] = next_info['ovmm_dist_to_place_goal']
        distance[1,1] = next_info['ovmm_rot_dist_to_place_goal']

    return distance


def get_agent_yaw_from_state(state):
    # 1. 把 Habitat quaternion 转成 magnum.quaternion
    q_mag = quat_to_magnum(state.rotation)

    # 2. “前向基向量”在你坐标系里是 (0, 0, -1)
    base_forward = np.array([0.0, 0.0, -1.0])

    # 3. 旋转到世界坐标
    f = q_mag.transform_vector(base_forward)  # 得到当前前向向量 (fx, fy, fz)

    # 4. 在 x-z 平面上计算 yaw：前 = -z 时的常用形式
    fx, fz = float(f[0]), float(f[2])
    yaw = np.arctan2(fx, -fz)     # [-pi, pi]
    return yaw

def get_target_yaw(agent_pos, target_pos):
    ax, ay, az = agent_pos
    tx, ty, tz = target_pos

    dx = tx - ax
    dz = tz - az

    # 指向 target 的方向的 yaw
    yaw = np.arctan2(dx, -dz)     # 一样是前 = -z 的定义
    return yaw

def angle_diff_agent_to_target(state, target_pos):
    agent_pos = np.array(state.position, dtype=float)
    yaw_agent = get_agent_yaw_from_state(state)
    yaw_target = get_target_yaw(agent_pos, target_pos)

    diff = yaw_target - yaw_agent
    # 归一化到 [-pi, pi]
    diff = (diff + np.pi) % (2.0 * np.pi) - np.pi
    return diff  # 有符号弧度

