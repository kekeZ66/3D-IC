# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.nn import DataParallel

import home_robot.utils.pose as pu
from home_robot.core.abstract_agent import Agent
from home_robot.core.interfaces import DiscreteNavigationAction, Observations
from home_robot.mapping.instance import InstanceMemory
from home_robot.mapping.semantic.categorical_2d_semantic_map_state import (
    Categorical2DSemanticMapState,
)
from home_robot.mapping.semantic.constants import MapConstants as MC
from home_robot.navigation_planner.discrete_planner import DiscretePlanner

from .feature_map_agent_module import ObjectNavAgentModule

import os
from home_robot.mapping.semantic.constants import MapConstants as MC
from scipy.ndimage import label as nd_label

from scipy.ndimage import distance_transform_edt
import cv2
from PIL import Image
import torch.nn.functional as F

import skimage.morphology
from home_robot.utils.nav_utils import los_exists_visible_mask_numpy
from home_robot.utils.habitat_utils import quat_wxyz_to_yaw, rad_norm
import math
from home_robot.mapping.clip3d.voxel_map import VoxelClipMap

import imageio.v2 as imageio

import json
import os
from datetime import datetime
from typing import Optional, Any, Dict

def _append_jsonl(path: str, obj: Dict[str, Any]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


from habitat.utils.geometry_utils import (
    quaternion_from_coeff,
    quaternion_rotate_vector,
)

debug_frontier_map = False


class ObjectNavAgent(Agent):
    """Simple object nav agent based on a 2D semantic map"""

    # Flag for debugging data flow and task configuration
    verbose = True

    def __init__(
        self,
        config,
        device_id: int = 0,
        min_goal_distance_cm: float = 50.0,
        continuous_angle_tolerance: float = 30.0,
        get_timing: bool = False,
    ):
        self.config = config
        self.get_timing = get_timing
        self.max_steps = config.AGENT.max_steps
        self.num_environments = config.NUM_ENVIRONMENTS
        self.store_all_categories_in_map = getattr(
            config.AGENT, "store_all_categories", False
        )
        if config.AGENT.panorama_start:
            self.panorama_start_steps = int(360 / config.ENVIRONMENT.turn_angle)
        else:
            self.panorama_start_steps = 0

        self.instance_memory = None
        self.record_instance_ids = getattr(
            config.AGENT.SEMANTIC_MAP, "record_instance_ids", False
        )

        if self.record_instance_ids:
            self.instance_memory = InstanceMemory(
                self.num_environments,
                config.AGENT.SEMANTIC_MAP.du_scale,
                instance_association=getattr(
                    config.AGENT.SEMANTIC_MAP, "instance_association", "map_overlap"
                ),
                debug_visualize=config.PRINT_IMAGES,
            )

        self._module = ObjectNavAgentModule(
            config, instance_memory=self.instance_memory
        )
        self.num_sem_categories = config.AGENT.SEMANTIC_MAP.num_sem_categories
        if config.NO_GPU:
            self.device = torch.device("cpu")
            self.module = self._module
        else:
            self.device_id = device_id
            self.device = torch.device(f"cuda:{self.device_id}")
            self._module = self._module.to(self.device)
            # Use DataParallel only as a wrapper to move model inputs to GPU
            self.module = DataParallel(self._module, device_ids=[self.device_id])

        self.visualize = config.VISUALIZE or config.PRINT_IMAGES
        self.use_dilation_for_stg = config.AGENT.PLANNER.use_dilation_for_stg
        self.semantic_map = Categorical2DSemanticMapState(
            device=self.device,
            num_environments=self.num_environments,
            num_sem_categories=config.AGENT.SEMANTIC_MAP.num_sem_categories,
            map_resolution=config.AGENT.SEMANTIC_MAP.map_resolution,
            map_size_cm=config.AGENT.SEMANTIC_MAP.map_size_cm,
            global_downscaling=config.AGENT.SEMANTIC_MAP.global_downscaling,
            record_instance_ids=getattr(
                config.AGENT.SEMANTIC_MAP, "record_instance_ids", False
            ),
            max_instances=getattr(config.AGENT.SEMANTIC_MAP, "max_instances", 0),
            evaluate_instance_tracking=getattr(
                config.ENVIRONMENT, "evaluate_instance_tracking", False
            ),
            instance_memory=self.instance_memory,
        )
        agent_radius_cm = config.AGENT.radius * 100.0
        agent_cell_radius = int(
            np.ceil(agent_radius_cm / config.AGENT.SEMANTIC_MAP.map_resolution)
        )
        self.planner = DiscretePlanner(
            turn_angle=config.ENVIRONMENT.turn_angle,
            collision_threshold=config.AGENT.PLANNER.collision_threshold,
            step_size=config.AGENT.PLANNER.step_size,
            obs_dilation_selem_radius=config.AGENT.PLANNER.obs_dilation_selem_radius,
            goal_dilation_selem_radius=config.AGENT.PLANNER.goal_dilation_selem_radius,
            map_size_cm=config.AGENT.SEMANTIC_MAP.map_size_cm,
            map_resolution=config.AGENT.SEMANTIC_MAP.map_resolution,
            visualize=config.VISUALIZE,
            print_images=config.PRINT_IMAGES,
            dump_location=config.DUMP_LOCATION,
            exp_name=config.EXP_NAME,
            agent_cell_radius=agent_cell_radius,
            min_obs_dilation_selem_radius=config.AGENT.PLANNER.min_obs_dilation_selem_radius,
            map_downsample_factor=config.AGENT.PLANNER.map_downsample_factor,
            map_update_frequency=config.AGENT.PLANNER.map_update_frequency,
            discrete_actions=config.AGENT.PLANNER.discrete_actions,
            min_goal_distance_cm=min_goal_distance_cm,
            continuous_angle_tolerance=continuous_angle_tolerance,
        )
        self.one_hot_encoding = torch.eye(
            config.AGENT.SEMANTIC_MAP.num_sem_categories, device=self.device
        )

        self.goal_update_steps = self._module.goal_update_steps
        self.timesteps = None
        self.timesteps_before_goal_update = None
        self.episode_panorama_start_steps = None
        self.last_poses = None
        self.closest_goal_map = None
        self.verbose = config.AGENT.PLANNER.verbose

        self.evaluate_instance_tracking = getattr(
            config.ENVIRONMENT, "evaluate_instance_tracking", False
        )
        self.one_hot_instance_encoding = None
        if self.evaluate_instance_tracking:
            self.one_hot_instance_encoding = torch.eye(
                config.AGENT.SEMANTIC_MAP.max_instances + 1, device=self.device
            )
        self.config = config

        self._itm = None
        self.text_prompt = "Seems like there is a target_object ahead."

        self.set_frontier_time = 0
        self.found_end_recep = False

        self.mllm_time = 0
        self.arrive = False
        self.after_arrive = 0

        self.vis = np.zeros((960,960))

        self.voxel_clip_map = VoxelClipMap()
        self.clip_model = MyClip()
        self.nav_to_goal_waypoint = []
        self.pick_waypoint = []
        self.nav_to_recep_waypoint = []
        self.place_waypoint = []
        self.waypoint_chain_set = []
        self._skip_chain_for_start_recep_goal = False

    def _nav_to_recep_flag(self) -> bool:
        nav_to_recep = self.get_nav_to_recep()
        if nav_to_recep is None:
            return False
        if torch.is_tensor(nav_to_recep):
            return bool(nav_to_recep.flatten()[0].item())
        if isinstance(nav_to_recep, (list, tuple, np.ndarray)):
            return bool(nav_to_recep[0])
        return bool(nav_to_recep)

    def _tensor_map_to_numpy(self, map_tensor) -> np.ndarray:
        if torch.is_tensor(map_tensor):
            arr = map_tensor.detach().cpu().numpy()
        else:
            arr = np.asarray(map_tensor)
        return arr.reshape(480, 480)

    def _category_local_map(self, category_tensor) -> np.ndarray:
        if category_tensor is None:
            return np.zeros((480, 480), dtype=bool)
        category = int(category_tensor.flatten()[0].item())
        if category < 0:
            category = 3
        return (
            self.semantic_map.local_map[0, MC.NON_SEM_CHANNELS + category, :, :]
            .detach()
            .cpu()
            .numpy()
            > 0
        )

    def _waypoint_sample_stride_cells(self) -> int:
        return max(1, int(round(20.0 / self.config.AGENT.SEMANTIC_MAP.map_resolution)))

    def _height_map_numpy(self) -> np.ndarray:
        return (
            self.semantic_map.local_map[0, MC.HEIGHT_MAP, :, :]
            .detach()
            .cpu()
            .numpy()
        )

    def _height_at(self, point: Tuple[int, int]) -> float:
        height_map = self._height_map_numpy()
        r, c = int(point[0]), int(point[1])
        if 0 <= r < height_map.shape[0] and 0 <= c < height_map.shape[1]:
            return float(height_map[r, c])
        return 0.0

    def _with_ground_z(self, points: List[Tuple[int, int]]) -> List[Tuple[int, int, float]]:
        return [(int(r), int(c), 0.0) for r, c in points]

    def _with_height_z(self, points: List[Tuple[int, int]]) -> List[Tuple[int, int, float]]:
        return [(int(r), int(c), self._height_at((int(r), int(c)))) for r, c in points]

    def _points_from_mask(self, mask: np.ndarray, stride: Optional[int] = None) -> List[Tuple[int, int]]:
        if stride is None:
            stride = self._waypoint_sample_stride_cells()
        ys, xs = np.nonzero(mask)
        if stride > 1:
            keep = (ys % stride == 0) & (xs % stride == 0)
            ys, xs = ys[keep], xs[keep]
        return [(int(y), int(x)) for y, x in zip(ys.tolist(), xs.tolist())]

    def _mask_center_point(self, mask: np.ndarray) -> Optional[Tuple[int, int]]:
        if mask.sum() == 0:
            return None
        dist = distance_transform_edt(mask.astype(np.uint8))
        return tuple(int(v) for v in np.unravel_index(np.argmax(dist), dist.shape))

    def _current_location_point(self) -> Optional[Tuple[int, int, float]]:
        cur_loc = (
            self.semantic_map.local_map[0, MC.CURRENT_LOCATION, :, :]
            .detach()
            .cpu()
            .numpy()
            > 0
        )
        if cur_loc.sum() == 0:
            return None
        ys, xs = np.nonzero(cur_loc)
        return int(round(float(ys.mean()))), int(round(float(xs.mean()))), 0.0

    def _beam_extend_chains(
        self,
        chains: List[Dict[str, Any]],
        next_points: List[Tuple[int, int, float]],
        set_name: str,
        origin: Optional[Tuple[int, int, float]] = None,
        beam_size: int = 6,
    ) -> List[Dict[str, Any]]:
        def _mean_score(stage_scores: List[float]) -> float:
            if len(stage_scores) == 0:
                return 0.0
            return float(sum(stage_scores) / len(stage_scores))

        if len(next_points) == 0:
            return chains
        if len(chains) == 0:
            if origin is None:
                return [
                    {
                        "origin": None,
                        "points": [pt],
                        "sets": [set_name],
                        "score": 0.0,
                        "stage_scores": [0.0],
                    }
                    for pt in next_points[:beam_size]
                ]
            expansion = []
            for pt in next_points:
                dist = float(np.linalg.norm(np.asarray(origin) - np.asarray(pt)))
                expansion.append((dist, pt))
            expansion = sorted(expansion, key=lambda x: x[0])[:beam_size]
            max_dist = max(dist for dist, _ in expansion)
            chains = []
            for dist, pt in expansion:
                score_delta = 1.0 if max_dist <= 1e-6 else 1.0 - dist / max_dist
                stage_scores = [float(score_delta)]
                chains.append(
                    {
                        "origin": origin,
                        "points": [pt],
                        "sets": [set_name],
                        "score": _mean_score(stage_scores),
                        "stage_scores": stage_scores,
                    }
                )
            return sorted(chains, key=lambda x: x["score"], reverse=True)[:beam_size]

        expansion = []
        for chain in chains:
            prev = chain["points"][-1]
            for pt in next_points:
                dist = float(np.linalg.norm(np.asarray(prev) - np.asarray(pt)))
                expansion.append((dist, chain, pt))
        if len(expansion) == 0:
            return chains

        expansion = sorted(expansion, key=lambda x: x[0])[:beam_size]
        max_dist = max(dist for dist, _, _ in expansion)
        expanded_chains = []
        for dist, chain, pt in expansion:
            score_delta = 1.0 if max_dist <= 1e-6 else 1.0 - dist / max_dist
            stage_scores = chain.get("stage_scores", []) + [float(score_delta)]
            expanded_chains.append(
                {
                    "points": chain["points"] + [pt],
                    "sets": chain["sets"] + [set_name],
                    "origin": chain.get("origin"),
                    "score": _mean_score(stage_scores),
                    "stage_scores": stage_scores,
                }
            )
        return sorted(expanded_chains, key=lambda x: x["score"], reverse=True)[:beam_size]

    def _update_waypoint_chain_set(self):
        origin = self._current_location_point()
        if self._nav_to_recep_flag():
            ordered_sets = [
                ("nav_to_recep_waypoint", self.nav_to_recep_waypoint),
                ("place_waypoint", self.place_waypoint),
            ]
        else:
            ordered_sets = [
                ("nav_to_goal_waypoint", self.nav_to_goal_waypoint),
                ("pick_waypoint", self.pick_waypoint),
                ("nav_to_recep_waypoint", self.nav_to_recep_waypoint),
                ("place_waypoint", self.place_waypoint),
            ]
        chains = []
        for set_name, points in ordered_sets:
            if len(points) == 0:
                break
            chains = self._beam_extend_chains(
                chains,
                points,
                set_name,
                origin=origin,
                beam_size=6,
        )
        self.waypoint_chain_set = chains
        self._module.waypoint_chain_set = chains

    def _chain_action_name(self, set_name: str) -> str:
        if set_name == "pick_waypoint":
            return "pick at"
        if set_name == "place_waypoint":
            return "place at"
        return "navigate to"

    def _chain_point_yaw(
        self,
        chain_points: List[Tuple[int, int, float]],
        idx: int,
        origin: Optional[Tuple[int, int, float]],
    ) -> Optional[float]:
        if len(chain_points) == 0:
            return None
        cur = np.asarray(chain_points[idx][:2], dtype=np.float32)
        if idx == 0:
            if origin is None:
                return None
            prev = np.asarray(origin[:2], dtype=np.float32)
        else:
            prev = np.asarray(chain_points[idx - 1][:2], dtype=np.float32)
        delta = cur - prev
        if float(np.linalg.norm(delta)) < 1e-6:
            return None
        return float(np.arctan2(delta[1], delta[0]))

    def _select_chain_goal_point(
        self,
        chain: Dict[str, Any],
        current_point: Optional[Tuple[int, int, float]],
        is_nav_to_recep: bool,
    ) -> Optional[Tuple[int, int, float]]:
        points = chain.get("points", [])
        sets = chain.get("sets", [])
        if len(points) == 0 or len(points) != len(sets):
            return None
        nav_goal_point = points[0]
        if current_point is None:
            return nav_goal_point
        current_xy = np.asarray(current_point[:2], dtype=np.float32)
        nav_xy = np.asarray(nav_goal_point[:2], dtype=np.float32)
        one_meter_cells = float(100.0 / self.config.AGENT.SEMANTIC_MAP.map_resolution)
        if float(np.linalg.norm(current_xy - nav_xy)) > one_meter_cells:
            return nav_goal_point
        target_set = "place_waypoint" if is_nav_to_recep else "pick_waypoint"
        if target_set in sets:
            target_idx = sets.index(target_set)
            target_point = points[target_idx]
            target_xy = np.asarray(target_point[:2], dtype=np.float32)
            if float(np.linalg.norm(current_xy - target_xy)) <= one_meter_cells:
                return target_point
        return nav_goal_point

    def _build_chain_token_candidates(self, lmb) -> List[Dict[str, Any]]:
        chain_infos = []
        for chain in self.waypoint_chain_set[:6]:
            points = chain.get("points", [])
            sets = chain.get("sets", [])
            if len(points) == 0 or len(points) != len(sets):
                continue
            actions = []
            for idx, (point, set_name) in enumerate(zip(points, sets)):
                yaw = self._chain_point_yaw(points, idx, chain.get("origin"))
                if yaw is None:
                    continue
                global_xy = (int(point[0]) + int(lmb[0]), int(point[1]) + int(lmb[2]))
                token = self.voxel_clip_map.token_map.gather_tokens(
                    global_xy,
                    yaw,
                    max_len_m=2.0,
                )[0]
                actions.append(
                    {
                        "action": self._chain_action_name(set_name),
                        "token": token,
                        "point": point,
                        "set": set_name,
                    }
                )
            if len(actions) > 0:
                chain_infos.append(
                    {
                        "label": len(chain_infos),
                        "actions": actions,
                        "chain": chain,
                    }
                )
        return chain_infos

    def _frontier_center_points(self) -> List[Tuple[int, int]]:
        frontier_map = np.asarray(self._module.detect_frontier_map > 0, dtype=np.uint8)
        num_labels, labels = cv2.connectedComponents(frontier_map, connectivity=8)
        centers = []
        for lab in range(1, num_labels):
            ys, xs = np.where(labels == lab)
            if len(xs) == 0:
                continue
            centers.append((int(round(float(ys.mean()))), int(round(float(xs.mean())))))
        return centers

    def _end_recep_map(self, end_recep_goal_category) -> np.ndarray:
        end_recep = (
            self.semantic_map.local_map[0, MC.FALSE_RECEP_MAP, :, :]
            .detach()
            .cpu()
            .numpy()
            > 0
        )
        if end_recep.sum() == 0:
            end_recep = self._category_local_map(end_recep_goal_category)
        return end_recep

    def _ground_points_near_mask(
        self,
        target_mask: np.ndarray,
        radius_m: float = 0.5,
        record_region: bool = True,
    ) -> List[Tuple[int, int]]:
        if target_mask.sum() == 0:
            return []
        obstacle = (
            self.semantic_map.local_map[0, MC.OBSTACLE_MAP, :, :]
            .detach()
            .cpu()
            .numpy()
            > 0
        )
        explored = (
            self.semantic_map.local_map[0, MC.EXPLORED_MAP, :, :]
            .detach()
            .cpu()
            .numpy()
            > 0
        )
        dilated_obstacle = cv2.dilate(
            obstacle.astype(np.uint8),
            skimage.morphology.disk(3),
            iterations=1,
        ).astype(bool)
        dist_to_target = distance_transform_edt(~target_mask)
        radius_cells = int(
            round(radius_m * 100.0 / self.config.AGENT.SEMANTIC_MAP.map_resolution)
        )
        ground_near_target = (
            (dist_to_target <= radius_cells)
            & explored
            & (~dilated_obstacle)
            & (~target_mask)
        )
        if record_region:
            self._module.waypoint_sample_region = ground_near_target
        return self._points_from_mask(ground_near_target)

    def _update_waypoint_sets(
        self,
        goal_map,
        found_goal,
        start_recep_goal_category,
        end_recep_goal_category,
    ):
        goal_np = self._tensor_map_to_numpy(goal_map[0, 0]) > 0
        found_goal_now = bool(found_goal.flatten()[0].item())
        is_nav_to_recep = self._nav_to_recep_flag()

        start_recep_map = self._category_local_map(start_recep_goal_category)
        goal_is_start_recep = (
            start_recep_map.sum() > 0 and np.logical_and(goal_np, start_recep_map).sum() > 0
        )
        self._skip_chain_for_start_recep_goal = bool(goal_is_start_recep and not is_nav_to_recep)
        if is_nav_to_recep:
            self.nav_to_goal_waypoint = []
            self._module.waypoint_sample_region = None
        elif goal_is_start_recep:
            self.nav_to_goal_waypoint = self._with_ground_z(
                self._ground_points_near_mask(start_recep_map, radius_m=0.5)
            )
        elif found_goal_now:
            self.nav_to_goal_waypoint = self._with_ground_z(self._points_from_mask(goal_np))
            self._module.waypoint_sample_region = None
        else:
            self.nav_to_goal_waypoint = self._with_ground_z(self._frontier_center_points())
            self._module.waypoint_sample_region = None

        object_center = (
            self._mask_center_point(goal_np)
            if found_goal_now and not is_nav_to_recep
            else None
        )
        self.pick_waypoint = [] if object_center is None else self._with_height_z([object_center])

        end_recep_map = self._end_recep_map(end_recep_goal_category)
        if end_recep_map.sum() > 0:
            self.nav_to_recep_waypoint = self._with_ground_z(
                self._ground_points_near_mask(
                    end_recep_map,
                    radius_m=0.5,
                    record_region=is_nav_to_recep,
                )
            )
            self.place_waypoint = self._with_height_z(self._points_from_mask(end_recep_map))
        else:
            self.nav_to_recep_waypoint = self._with_ground_z(self._frontier_center_points())
            self.place_waypoint = []

        self._module.waypoint_sets = {
            "nav_to_goal_waypoint": self.nav_to_goal_waypoint,
            "pick_waypoint": self.pick_waypoint,
            "nav_to_recep_waypoint": self.nav_to_recep_waypoint,
            "place_waypoint": self.place_waypoint,
        }
        if self._skip_chain_for_start_recep_goal:
            self.waypoint_chain_set = []
            self._module.waypoint_chain_set = []
        else:
            self._update_waypoint_chain_set()






    # ------------------------------------------------------------------
    # Inference methods to interact with vectorized simulation
    # environments
    # ------------------------------------------------------------------
    @torch.no_grad()
    def prepare_planner_inputs(
        self,
        obs: torch.Tensor,
        pose_delta: torch.Tensor,
        cosine_value,
        object_goal_category: torch.Tensor = None,
        start_recep_goal_category: torch.Tensor = None,
        end_recep_goal_category: torch.Tensor = None,
        instance_id: torch.Tensor = None,
        nav_to_recep: torch.Tensor = None,
        camera_pose: torch.Tensor = None,
        semantic_max_val: Optional[List[int]] = None,
        obstacle_locations: torch.Tensor = None,
        free_locations: torch.Tensor = None,
    ) -> Tuple[List[dict], List[dict]]:
        """Prepare low-level planner inputs from an observation - this is
        the main inference function of the agent that lets it interact with
        vectorized environments.

        This function assumes that the agent has been initialized.

        Args:
            obs: current frame containing (RGB, depth, segmentation) of shape
             (num_environments, 3 + 1 + num_sem_categories, frame_height, frame_width)
            pose_delta: sensor pose delta (dy, dx, dtheta) since last frame
             of shape (num_environments, 3)
            object_goal_category: semantic category of small object goals
            start_recep_goal_category: semantic category of start receptacle goals
            end_recep_goal_category: semantic category of end receptacle goals
            camera_pose: camera extrinsic pose of shape (num_environments, 4, 4)

        Returns:
            planner_inputs: list of num_environments planner inputs dicts containing
                obstacle_map: (M, M) binary np.ndarray local obstacle map
                 prediction
                sensor_pose: (7,) np.ndarray denoting global pose (x, y, o)
                 and local map boundaries planning window (gx1, gx2, gy1, gy2)
                goal_map: (M, M) binary np.ndarray denoting goal location
            vis_inputs: list of num_environments visualization info dicts containing
                explored_map: (M, M) binary np.ndarray local explored map
                 prediction
                semantic_map: (M, M) np.ndarray containing local semantic map
                 predictions
        """
        dones = torch.tensor([False] * self.num_environments) 
        update_global = torch.tensor(
            [
                self.timesteps_before_goal_update[e] == 0
                for e in range(self.num_environments)
            ]
        )

        if obstacle_locations is not None:
            obstacle_locations = obstacle_locations.unsqueeze(1)
        if free_locations is not None:
            free_locations = free_locations.unsqueeze(1)
        if object_goal_category is not None:
            object_goal_category = object_goal_category.unsqueeze(1)
        if start_recep_goal_category is not None:
            start_recep_goal_category = start_recep_goal_category.unsqueeze(1)
        if end_recep_goal_category is not None:
            end_recep_goal_category = end_recep_goal_category.unsqueeze(1)

        if instance_id is not None:
            instance_id = instance_id.unsqueeze(1)
 
        (
            goal_map,
            found_goal,
            frontier_map,
            self.semantic_map.local_map,
            self.semantic_map.global_map,
            seq_local_pose,
            seq_global_pose,
            seq_lmb,
            seq_origins,
            goal_reach_mask,
        ) = self.module(
            obs.unsqueeze(1),
            pose_delta.unsqueeze(1),
            dones.unsqueeze(1),
            update_global.unsqueeze(1),
            camera_pose.unsqueeze(1),
            self.semantic_map.local_map,
            self.semantic_map.global_map,
            self.semantic_map.local_pose,
            self.semantic_map.global_pose,
            self.semantic_map.lmb,
            self.semantic_map.origins,
            value = cosine_value,
            seq_object_goal_category=object_goal_category,
            seq_start_recep_goal_category=start_recep_goal_category,
            seq_end_recep_goal_category=end_recep_goal_category,
            seq_instance_id=instance_id,
            seq_nav_to_recep=nav_to_recep,
            semantic_max_val=semantic_max_val,
            seq_obstacle_locations=obstacle_locations,
            seq_free_locations=free_locations,
        )
        raw_goal_map = goal_map.clone()
        raw_found_goal = found_goal.clone()
        p = (seq_local_pose[0,0,:2] - seq_origins[0,0,:2])
        
        self.vis = self.vis + self.semantic_map.global_map[0,MC.CURRENT_LOCATION,:,:].cpu().numpy()
        start_sample = time.perf_counter()
        self._update_waypoint_sets(
            goal_map,
            found_goal,
            start_recep_goal_category,
            end_recep_goal_category,
        )

        if self._skip_chain_for_start_recep_goal:
            lmb = seq_lmb[0, 0, :].cpu().numpy()
            lmb_i = [int(v) for v in lmb]
            self.semantic_map.local_map[0, MC.CACHE_GOAL, :, :] = 0
            self.semantic_map.global_map[
                0,
                MC.CACHE_GOAL,
                lmb_i[0] : lmb_i[1],
                lmb_i[2] : lmb_i[3],
            ] = 0
            self.mllm_time = 0
        elif self.timesteps[0] > 12 and self.mllm_time == 0:
            lmb = seq_lmb[0, 0, :].cpu().numpy()
            lmb_i = [int(v) for v in lmb]
            try:
                chain_candidates = self._build_chain_token_candidates(lmb)
            except Exception as e:
                print(f"[waypoint_chain_mllm] failed to build chain tokens: {e}")
                chain_candidates = []
            cnt = len(chain_candidates)
            print(f"[waypoint_chain_mllm] timestep={self.timesteps[0]} cnt={cnt}")
            goal_updated = False
            if cnt > 0:
                end_sample = time.perf_counter()
                if cnt > 1:
                    try:
                        ind, reason = self.voxel_clip_map.get_chain_answer(
                            chain_candidates,
                            self.goal_name,
                        )
                    except Exception as e:
                        ind, reason = 0, f"chain VLM failed; fallback to 0: {e}"
                else:
                    ind, reason = 0, "single chain candidate"
                    print("[waypoint_chain_mllm] single candidate; skip VLM call")
                if ind < 0 or ind >= cnt:
                    ind, reason = 0, f"invalid chain index {ind}; fallback to 0"
                selected_chain = chain_candidates[ind]["chain"]
                if len(selected_chain.get("points", [])) > 0:
                    point = self._select_chain_goal_point(
                        selected_chain,
                        current_point=self._current_location_point(),
                        is_nav_to_recep=self._nav_to_recep_flag(),
                    )
                    if point is None:
                        point = selected_chain["points"][0]
                    goal_map_change = np.zeros((480, 480))
                    r, c = int(point[0]), int(point[1])
                    if 0 <= r < 480 and 0 <= c < 480:
                        goal_map_change[r, c] = 1
                        goal_map_change = torch.from_numpy(goal_map_change).to(
                            device=goal_map.device,
                            dtype=goal_map.dtype,
                        )
                        goal_map = goal_map_change.unsqueeze(0).unsqueeze(0)
                        self.semantic_map.local_map[0, MC.CACHE_GOAL, :, :] = goal_map_change
                        self.semantic_map.global_map[
                            0,
                            MC.CACHE_GOAL,
                            lmb_i[0] : lmb_i[1],
                            lmb_i[2] : lmb_i[3],
                        ] = goal_map_change
                        goal_updated = True
                end_reason = time.perf_counter()
                print(f'sample: {end_sample-start_sample:.4f}  reason: {end_reason-end_sample:.4f}')
            else:
                print("[waypoint_chain_mllm] no valid chain candidates; keep goal_map unchanged")
            if not goal_updated:
                self.semantic_map.local_map[0, MC.CACHE_GOAL, :, :] = 0
                self.semantic_map.global_map[
                    0,
                    MC.CACHE_GOAL,
                    lmb_i[0] : lmb_i[1],
                    lmb_i[2] : lmb_i[3],
                ] = 0
            self.mllm_time = 6
        elif self.timesteps[0] > 12 and self.mllm_time > 0:
            goal_map_change = self.semantic_map.local_map[:, MC.CACHE_GOAL, :, :]
            if goal_map_change.sum() > 0:
                goal_map = goal_map_change.unsqueeze(0)
            self.mllm_time -= 1
 

        cache_goal_active = bool(
            self.semantic_map.local_map[0, MC.CACHE_GOAL, :, :].sum().item() > 0
        )
        if self.set_frontier_time > 0 and not found_goal and not cache_goal_active:
            goal_map = frontier_map
            self.set_frontier_time -= 1

        self.semantic_map.local_pose = seq_local_pose[:, -1]
        self.semantic_map.global_pose = seq_global_pose[:, -1]
        self.semantic_map.lmb = seq_lmb[:, -1]
        self.semantic_map.origins = seq_origins[:, -1]

        nav_goal_map = goal_map.squeeze(1).cpu().numpy()
        semantic_goal_map = raw_goal_map.squeeze(1).cpu().numpy()
        self._last_semantic_goal_map = semantic_goal_map
        found_goal = raw_found_goal.squeeze(1).cpu()

        for e in range(self.num_environments):
            self.semantic_map.update_frontier_map(e, frontier_map[e][0].cpu().numpy())
            if found_goal[e] or self.timesteps_before_goal_update[e] == 0:
                self.semantic_map.update_global_goal_for_env(e, nav_goal_map[e])
                if self.timesteps_before_goal_update[e] == 0:
                    self.timesteps_before_goal_update[e] = self.goal_update_steps
            self.timesteps[e] = self.timesteps[e] + 1
            self.timesteps_before_goal_update[e] = (
                self.timesteps_before_goal_update[e] - 1
            )
        if debug_frontier_map:
            import matplotlib.pyplot as plt

            plt.subplot(131)
            plt.imshow(self.semantic_map.get_frontier_map(e))
            plt.subplot(132)
            plt.imshow(frontier_map[e][0].cpu().numpy())
            plt.subplot(133)
            plt.imshow(self.semantic_map.get_goal_map(e))
            plt.show()

        planner_inputs = [
            {
                "obstacle_map": self.semantic_map.get_obstacle_map(e),
                "goal_map": self.semantic_map.get_goal_map(e),
                "frontier_map": self.semantic_map.get_frontier_map(e),
                "sensor_pose": self.semantic_map.get_planner_pose_inputs(e),
                "found_goal": found_goal[e].item(),
            }
            for e in range(self.num_environments)
        ]
        if self.visualize:
            vis_inputs = [
                {
                    "explored_map": self.semantic_map.get_explored_map(e),
                    "semantic_map": self.semantic_map.get_semantic_map(e),
                    "been_close_map": self.semantic_map.get_been_close_map(e),
                    "timestep": self.timesteps[e],
                }
                for e in range(self.num_environments)
            ]
            if self.record_instance_ids:
                for e in range(self.num_environments):
                    vis_inputs[e]["instance_map"] = self.semantic_map.get_instance_map(
                        e
                    )

        else:
            vis_inputs = [{} for e in range(self.num_environments)]
        return planner_inputs, vis_inputs, goal_reach_mask

    def reset_vectorized(self):
        """Initialize agent state."""
        self.timesteps = [0] * self.num_environments
        self.timesteps_before_goal_update = [0] * self.num_environments
        self.last_poses = [np.zeros(3)] * self.num_environments
        self.semantic_map.init_map_and_pose()
        self.episode_panorama_start_steps = self.panorama_start_steps
        self._module.semantic_map_module._map = np.zeros((960, 960))  # TODO
        self._module.semantic_map_module._value_map = np.zeros((960, 960, 1),np.float32)
        self._module.semantic_map_module._map_recp = np.zeros((960, 960))  # TODO
        self._module.semantic_map_module._value_map_recp = np.zeros((960, 960, 1),np.float32)
        self._module.semantic_map_module.reset()
        if self.record_instance_ids:
            self.instance_memory.reset()
        self.closest_goal_map = [None] * self.num_environments
        self.planner.reset()
        self.set_frontier_time = 0
        self.found_end_recep = False
        self.mllm_time = 0
        self.mllm_choose_target = None
        self.arrive = False
        self.after_arrive = 0
        self.nav_to_goal_waypoint = []
        self.pick_waypoint = []
        self.nav_to_recep_waypoint = []
        self.place_waypoint = []
        self.waypoint_chain_set = []
        self._skip_chain_for_start_recep_goal = False
        self._module.waypoint_sets = {
            "nav_to_goal_waypoint": [],
            "pick_waypoint": [],
            "nav_to_recep_waypoint": [],
            "place_waypoint": [],
        }
        self._module.waypoint_chain_set = []

    def reset_vectorized_for_env(self, e: int):
        """Initialize agent state for a specific environment."""
        self.timesteps[e] = 0
        self.timesteps_before_goal_update[e] = 0
        self.last_poses[e] = np.zeros(3)
        self.semantic_map.init_map_and_pose_for_env(e)
        self.episode_panorama_start_steps = self.panorama_start_steps
        if self.record_instance_ids:
            self.instance_memory.reset_for_env(e)
        self.planner.reset()
        self.set_frontier_time = 0
        self.found_end_recep = False
        self.mllm_time = 0
        self.mllm_choose_target = None
        self.arrive = False
        self.after_arrive = 0
        self.nav_to_goal_waypoint = []
        self.pick_waypoint = []
        self.nav_to_recep_waypoint = []
        self.place_waypoint = []
        self.waypoint_chain_set = []
        self._module.waypoint_sets = {
            "nav_to_goal_waypoint": [],
            "pick_waypoint": [],
            "nav_to_recep_waypoint": [],
            "place_waypoint": [],
        }
        self._module.waypoint_chain_set = []


    # ---------------------------------------------------------------------
    # Inference methods to interact with the robot or a single un-vectorized
    # simulation environment
    # ---------------------------------------------------------------------

    def reset(self):
        """Initialize agent state."""
        self.reset_vectorized()
        self.planner.reset()
        if self.verbose:
            print("ObjectNavAgent reset")

    def get_nav_to_recep(self):
        return None
    
    def preprocess_vlfm(self, obs: Observations, camera_pose):
        tensor_dict = {}

        tensor_dict["compass"] = torch.from_numpy(obs.compass).unsqueeze(0).to(self.device)
        tensor_dict["depth"] = torch.from_numpy(obs.depth).unsqueeze(0).to(self.device)
        tensor_dict["gps"] = torch.from_numpy(obs.gps).unsqueeze(0).to(self.device)
        tensor_dict["rgb"] = torch.from_numpy(obs.rgb).unsqueeze(0).to(self.device)

        import trimesh.transformations as tra
        angles = torch.Tensor(
            [tra.euler_from_matrix(p[:3, :3].cpu(), "rzyx") for p in camera_pose]
        )
        tilt = angles[:, 1]
        yaw = angles[:, -1]
        roll = angles[:, 0]

        tensor_dict["heading"] = roll.unsqueeze(0).to(self.device)
        return tensor_dict


    


    def act(self, obs: Observations, sim = None) -> Tuple[DiscreteNavigationAction, Dict[str, Any]]:
        """Act end-to-end."""
        if self.get_timing:
            t0 = time.time()
        self.goal_name = obs.task_observations['goal_name']


        # 1 - Obs preprocessing
        (
            obs_preprocessed,
            pose_delta,
            object_goal_category,
            start_recep_goal_category,
            end_recep_goal_category,
            instance_id,
            goal_name,
            camera_pose,
        ) = self._preprocess_obs(obs)
        obstacle_locations = None      
        free_locations = None     


        text_goals = goal_name[0].split()
        obj_name = text_goals[1]
        start_recp = text_goals[3]
        end_recp = text_goals[5]
        self.obj_name = obj_name
        self.start_recp = start_recp
        self.end_recp = end_recp
        cosine = [0.0, 0.0]


        if self.get_timing:
            t1 = time.time()
            print(f"[Agent] Obs preprocessing time: {t1 - t0:.2f}")

        semantic_max_val = None
        if "semantic_max_val" in obs.task_observations:
            semantic_max_val = obs.task_observations["semantic_max_val"] 

        if self.timesteps[0] == 1:
            self.voxel_clip_map.reset()
            self.voxel_clip_map.set_camera_pose(camera_pose)
        self.voxel_clip_map.update_voxel_map(
            obs.rgb,
            obs_preprocessed[:,3,:,:].float(),
            camera_pose,
            obs.gps,
        )
        

        # 2 - Semantic mapping + policy
        planner_inputs, vis_inputs, goal_reach_mask = self.prepare_planner_inputs(
            obs_preprocessed,
            pose_delta,
            cosine_value=cosine,
            object_goal_category=object_goal_category,
            start_recep_goal_category=start_recep_goal_category,
            end_recep_goal_category=end_recep_goal_category,
            instance_id=instance_id,
            camera_pose=camera_pose,
            nav_to_recep=self.get_nav_to_recep(),
            semantic_max_val=semantic_max_val,
            obstacle_locations=obstacle_locations,
            free_locations=free_locations,
        )

        if self.get_timing:
            t2 = time.time()
            print(f"[Agent] Semantic mapping and policy time: {t2 - t1:.2f}")

        # 3 - Planning
        closest_goal_map = None
        short_term_goal = None
        dilated_obstacle_map = None
        if planner_inputs[0]["found_goal"]:
            self.episode_panorama_start_steps = 0
        if self.timesteps[0] < self.episode_panorama_start_steps:
            action = DiscreteNavigationAction.TURN_RIGHT
        elif self.timesteps[0] > self.max_steps:
            action = DiscreteNavigationAction.STOP
        else:
            height_map = self.semantic_map.local_map[0,MC.HEIGHT_MAP,:,:].cpu().numpy()
            (
                action,
                closest_goal_map,
                short_term_goal,
                dilated_obstacle_map,
                change_goal_to_frontier,
            ) = self.planner.plan(
                **planner_inputs[0],
                height_map=height_map,
                use_dilation_for_stg=self.use_dilation_for_stg,
                timestep=self.timesteps[0],
                get_nav_to_recep = self.get_nav_to_recep()[0],
                debug=self.verbose,
                goal_reach_mask = goal_reach_mask,
            )
            if action == DiscreteNavigationAction.STOP and self.get_nav_to_recep()[0]:
                idx_map = self._module.semantic_map_module.idx_map
                if idx_map is not None:
                    heigt_map = self.semantic_map.local_map[0,MC.HEIGHT_MAP,:,:]
                    place_info = {}
                    place_info['idx_map'] = idx_map
                    place_info['height_map'] = height_map
                    self.place_info = place_info

            self.closest_goal_map[0] = closest_goal_map

            if change_goal_to_frontier and not planner_inputs[0]['found_goal']:
                self.set_frontier_time = 20
                goal_idx = np.argwhere(planner_inputs[0]['goal_map']!=0)
                if goal_idx.shape[0] > 0:
                    dists = np.sum((goal_idx - np.array(short_term_goal)) ** 2, axis=1)
                    closest_idx = np.argmin(dists)
                    closest_goal_point = goal_idx[closest_idx]

                goal_map = planner_inputs[0]['goal_map']
                y, x = map(int, closest_goal_point)

                structure = np.ones((3,3), np.uint8)  
                labs, num = nd_label(goal_map.astype(np.uint8), structure=structure)
                lab_id = labs[y, x]
                assert lab_id != 0

                comp_mask = (labs == lab_id)
                comp_coords = np.column_stack(np.nonzero(comp_mask))
                comp_mask = torch.from_numpy(comp_mask).to(device=self.semantic_map.local_map.device)
                self.semantic_map.local_map[0, MC.BEEN_CLOSE_MAP,:,:] += comp_mask
                self.semantic_map.local_map[0, MC.BEEN_CLOSE_MAP,:,:] = (self.semantic_map.local_map[0, MC.BEEN_CLOSE_MAP,:,:] > 0).to(torch.float32)

                
                self.semantic_map.local_map[0,MC.BEEN_CLOSE_MAP,closest_goal_point[0]-2:closest_goal_point[0]+3,closest_goal_point[1]-2:closest_goal_point[1]+3]=1
                self.planner.change_goal_to_frontier = False


        if self.get_timing:
            t3 = time.time()
            print(f"[Agent] Planning time: {t3 - t2:.2f}")
            print(f"[Agent] Total time: {t3 - t0:.2f}")

        vis_inputs[0]["goal_name"] = obs.task_observations["goal_name"]
        if self.visualize:
            vis_inputs[0]["semantic_frame"] = obs.task_observations["semantic_frame"]
            vis_inputs[0]["closest_goal_map"] = self.closest_goal_map[0]
            vis_inputs[0]["third_person_image"] = obs.third_person_image
            vis_inputs[0]["short_term_goal"] = None
            vis_inputs[0]["dilated_obstacle_map"] = dilated_obstacle_map
            vis_inputs[0]["semantic_map_config"] = self.config.AGENT.SEMANTIC_MAP
            vis_inputs[0]["instance_memory"] = self.instance_memory

        info = {
            **planner_inputs[0],
            **vis_inputs[0],
            "nav_goal_map": planner_inputs[0]["goal_map"],
            "goal_map": getattr(self, "_last_semantic_goal_map", [planner_inputs[0]["goal_map"]])[0],
            "short_term_goal": short_term_goal,
        }
        self.prev_action = action

        return action, info

    def _preprocess_obs(self, obs: Observations):
        """Take a home-robot observation, preprocess it to put it into the correct format for the
        semantic map."""
        rgb = torch.from_numpy(obs.rgb).to(self.device)
        depth = (
            torch.from_numpy(obs.depth).unsqueeze(-1).to(self.device) * 100.0
        )  # m to cm
        instance_id = obs.task_observations.get("instance_id", None)
        if self.store_all_categories_in_map:
            semantic = obs.semantic
            obj_goal_idx = obs.task_observations["object_goal"]
            if "start_recep_goal" in obs.task_observations:
                start_recep_idx = obs.task_observations["start_recep_goal"]
            if "end_recep_goal" in obs.task_observations:
                end_recep_idx = obs.task_observations["end_recep_goal"]
        else:
            semantic = np.full_like(obs.semantic, 4)
            obj_goal_idx, start_recep_idx, end_recep_idx = 1, 2, 3

            semantic[
                obs.semantic == obs.task_observations["object_goal"]
            ] = obj_goal_idx
            if "start_recep_goal" in obs.task_observations:
                semantic[
                    obs.semantic == obs.task_observations["start_recep_goal"]
                ] = start_recep_idx
            if "end_recep_goal" in obs.task_observations:
                semantic[
                    obs.semantic == obs.task_observations["end_recep_goal"]
                ] = end_recep_idx
            if (semantic == end_recep_idx).sum() > 500:
                self.found_end_recep = True
                if self.get_nav_to_recep():
                    self.start_mllm_policy_place = True
                    mask = semantic == end_recep_idx


            
            if (semantic == obj_goal_idx).sum() > 30:
                self.start_mllm_policy_pick = True
                obj_mask = (semantic == obj_goal_idx).astype(np.uint8)
                dist = distance_transform_edt(obj_mask)
                i_c, j_c = np.unravel_index(np.argmax(dist), dist.shape)
                self.point_list.append((i_c,j_c))

        semantic = self.one_hot_encoding[torch.from_numpy(semantic).to(self.device)]

        obs_preprocessed = torch.cat([rgb, depth, semantic], dim=-1)
        if self.record_instance_ids:
            instances = obs.task_observations["instance_map"]
            instance_ids = np.unique(instances)
            instance_id_to_idx = {
                instance_id: idx for idx, instance_id in enumerate(instance_ids)
            }
            instances = torch.from_numpy(
                np.vectorize(instance_id_to_idx.get)(instances)
            ).to(self.device)
            # create a one-hot encoding
            instances = torch.eye(len(instance_ids), device=self.device)[instances]
            obs_preprocessed = torch.cat([obs_preprocessed, instances], dim=-1)

        if self.evaluate_instance_tracking:
            gt_instance_ids = (
                torch.from_numpy(obs.task_observations["gt_instance_ids"])
                .to(self.device)
                .long()
            )
            gt_instance_ids = self.one_hot_instance_encoding[gt_instance_ids]
            obs_preprocessed = torch.cat([obs_preprocessed, gt_instance_ids], dim=-1)

        obs_preprocessed = obs_preprocessed.unsqueeze(0).permute(0, 3, 1, 2)

        curr_pose = np.array([obs.gps[0], obs.gps[1], obs.compass[0]])
        pose_delta = torch.tensor(
            pu.get_rel_pose_change(curr_pose, self.last_poses[0])
        ).unsqueeze(0)
        self.last_poses[0] = curr_pose
        object_goal_category = None
        end_recep_goal_category = None
        if (
            "object_goal" in obs.task_observations
            and obs.task_observations["object_goal"] is not None
        ):
            if self.verbose:
                print("object goal =", obs.task_observations["object_goal"])
            object_goal_category = torch.tensor(obj_goal_idx).unsqueeze(0)
        start_recep_goal_category = None
        if (
            "start_recep_goal" in obs.task_observations
            and obs.task_observations["start_recep_goal"] is not None
        ):
            if self.verbose:
                print(
                    "start_recep goal =",
                    obs.task_observations["start_recep_goal"],
                )
            start_recep_goal_category = torch.tensor(start_recep_idx).unsqueeze(0)
        if (
            "end_recep_goal" in obs.task_observations
            and obs.task_observations["end_recep_goal"] is not None
        ):
            if self.verbose:
                print("end_recep goal =", obs.task_observations["end_recep_goal"])
            end_recep_goal_category = torch.tensor(end_recep_idx).unsqueeze(0)
        if (
            "instance_id" in obs.task_observations
            and obs.task_observations["instance_id"] is not None
        ):
            instance_id = torch.tensor(instance_id).unsqueeze(0)
        goal_name = [obs.task_observations["goal_name"]]
        if self.verbose:
            print("[ObjectNav] Goal name: ", goal_name)

        camera_pose = obs.camera_pose

        if camera_pose is not None:
            camera_pose = torch.tensor(np.asarray(camera_pose)).unsqueeze(0)

        return (
            obs_preprocessed,
            pose_delta,
            object_goal_category,
            start_recep_goal_category,
            end_recep_goal_category,
            instance_id,
            goal_name,
            camera_pose,
        )
    
    def get_habitat_world_point(self, episode_world_pt):
        relative_pt = np.array([episode_world_pt[1], 0.0, -episode_world_pt[0]])
        goal_position_world = self.habitat_origin + quaternion_rotate_vector(
            self.rotation_world_start,
            relative_pt
        )
        return goal_position_world
    
    def local_map_px_to_world(self, lmb, pt):

        if torch.is_tensor(lmb):
            lmb = lmb.cpu().numpy()
        if torch.is_tensor(pt):
            pt = pt.cpu().numpy()
        
        delta_origin_x = (480 - lmb[0]) 
        delta_origin_y = (480 - lmb[2]) 
        return ((pt[0] - delta_origin_x) / 20.0, (pt[1] - delta_origin_y) / 20.0)
    


def angle_deg_up0_right_minus90(p2, p1): 
    r1, c1 = p1
    r2, c2 = p2
    dr = r2 - r1
    dc = c2 - c1
    ang = np.degrees(np.arctan2(dc, -dr))  
    ang = (ang + 180) % 360 - 180
    return ang







    
