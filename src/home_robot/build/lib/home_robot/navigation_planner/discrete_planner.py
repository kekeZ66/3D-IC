# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import math
import os
import shutil
import time
from typing import List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import skimage.morphology

import home_robot.utils.pose as pu
from home_robot.core.interfaces import (
    ContinuousNavigationAction,
    DiscreteNavigationAction,
)
from home_robot.utils.geometry import xyt_global_to_base

from .fmm_planner import FMMPlanner

CM_TO_METERS = 0.01


def add_boundary(mat: np.ndarray, value=1) -> np.ndarray:
    h, w = mat.shape
    new_mat = np.zeros((h + 2, w + 2)) + value
    new_mat[1 : h + 1, 1 : w + 1] = mat
    return new_mat


def remove_boundary(mat: np.ndarray, value=1) -> np.ndarray:
    return mat[value:-value, value:-value]


class DiscretePlanner:
    """
    This class translates planner inputs into a discrete low-level action
    using an FMM planner.

    This is a wrapper used to navigate to a particular object/goal location.
    """

    def __init__(
        self,
        turn_angle: float,
        collision_threshold: float,
        step_size: int,
        obs_dilation_selem_radius: int,
        goal_dilation_selem_radius: int,
        map_size_cm: int,
        map_resolution: int,
        visualize: bool,
        print_images: bool,
        dump_location: str,
        exp_name: str,
        min_goal_distance_cm: float = 50.0,
        min_obs_dilation_selem_radius: int = 1,
        agent_cell_radius: int = 1,
        map_downsample_factor: float = 1.0,
        map_update_frequency: int = 1,
        goal_tolerance: float = 0.01,
        discrete_actions: bool = True,
        continuous_angle_tolerance: float = 30.0,
    ):
        """
        Arguments:
            turn_angle (float): agent turn angle (in degrees)
            collision_threshold (float): forward move distance under which we
             consider there's a collision (in meters)
            obs_dilation_selem_radius: radius (in cells) of obstacle dilation
             structuring element
            obs_dilation_selem_radius: radius (in cells) of goal dilation
             structuring element
            map_size_cm: global map size (in centimeters)
            map_resolution: size of map bins (in centimeters)
            visualize: if True, render planner internals for debugging
            print_images: if True, save visualization as images
        """
        self.discrete_actions = discrete_actions
        self.visualize = visualize
        self.print_images = print_images
        self.default_vis_dir = f"{dump_location}/images/{exp_name}"
        os.makedirs(self.default_vis_dir, exist_ok=True)

        self.map_size_cm = map_size_cm
        self.map_resolution = map_resolution
        self.map_shape = (
            self.map_size_cm // self.map_resolution,
            self.map_size_cm // self.map_resolution,
        )
        self.turn_angle = turn_angle
        self.collision_threshold = collision_threshold
        self.step_size = step_size
        self.start_obs_dilation_selem_radius = obs_dilation_selem_radius
        self.goal_dilation_selem_radius = goal_dilation_selem_radius
        self.min_obs_dilation_selem_radius = min_obs_dilation_selem_radius
        self.agent_cell_radius = agent_cell_radius
        self.goal_tolerance = goal_tolerance
        self.continuous_angle_tolerance = continuous_angle_tolerance

        self.vis_dir = None
        self.collision_map = None
        self.visited_map = None
        self.visited_count_map = None
        self.col_width = None
        self.stuck_time = 0
        self.last_pose = None
        self.curr_pose = None
        self.last_action = None
        self.timestep = 0
        self.curr_obs_dilation_selem_radius = None
        self.obs_dilation_selem = None
        self.min_goal_distance_cm = min_goal_distance_cm
        self.dd = None

        self.map_downsample_factor = map_downsample_factor
        self.map_update_frequency = map_update_frequency
        self.stuck = False
        self.move_ahead_flag = 0

    def reset(self):
        self.vis_dir = self.default_vis_dir
        self.collision_map = np.zeros(self.map_shape)
        self.visited_map = np.zeros(self.map_shape)
        self.visited_count_map = np.zeros(self.map_shape)
        self.col_width = 1
        self.last_pose = None
        self.curr_pose = [
            self.map_size_cm / 100.0 / 2.0,
            self.map_size_cm / 100.0 / 2.0,
            0.0,
        ]
        self.last_action = None
        self.timestep = 1
        self.curr_obs_dilation_selem_radius = self.start_obs_dilation_selem_radius
        self.obs_dilation_selem = skimage.morphology.disk(
            self.curr_obs_dilation_selem_radius
        )
        self.goal_dilation_selem = skimage.morphology.disk(
            self.goal_dilation_selem_radius
        )
        self.stuck = False
        self.move_ahead_flag = 0


    def set_vis_dir(self, scene_id: str, episode_id: str):
        # print("default vis dir: ", self.default_vis_dir)
        self.vis_dir = os.path.join(self.default_vis_dir, f"{scene_id}_{episode_id}")
        shutil.rmtree(self.vis_dir, ignore_errors=True)
        os.makedirs(self.vis_dir, exist_ok=True)

    def disable_print_images(self):
        self.print_images = False

    def plan(
        self,
        obstacle_map: np.ndarray,
        goal_map: np.ndarray,
        frontier_map: np.ndarray,
        sensor_pose: np.ndarray,
        found_goal: bool,
        debug: bool = True,
        use_dilation_for_stg: bool = False,
        timestep: int = None,
    ) -> Tuple[DiscreteNavigationAction, np.ndarray]:
        """Plan a low-level action.

        Args:
            obstacle_map: (M, M) binary local obstacle map prediction
            goal_map: (M, M) binary array denoting goal location
            sensor_pose: (7,) array denoting global pose (x, y, o)
             and local map boundaries planning window (gx1, gx2, gy1, gy2)
            found_goal: whether we found the object goal category

        Returns:
            action: low-level action
            closest_goal_map: (M, M) binary array denoting closest goal
             location in the goal map in geodesic distance
        """
        # Reset timestep using argument; useful when there are timesteps where the discrete planner is not invoked
        if timestep is not None:
            self.timestep = timestep

        self.last_pose = self.curr_pose
        obstacle_map = np.rint(obstacle_map)

        start_x, start_y, start_o, gx1, gx2, gy1, gy2 = sensor_pose    # 从全局大地图上裁出的一块子图，供本次局部规划使用
        gx1, gx2, gy1, gy2 = int(gx1), int(gx2), int(gy1), int(gy2)
        planning_window = [gx1, gx2, gy1, gy2]

        start = [         # 机器人当前位置->局部窗口内的像素坐标
            int(start_y * 100.0 / self.map_resolution - gx1),
            int(start_x * 100.0 / self.map_resolution - gy1),
        ]
        start = pu.threshold_poses(start, obstacle_map.shape)
        start = np.array(start)

        if debug:
            print()
            print("--- Planning ---")
            print("Found goal:", found_goal)
            print("Goal points provided:", np.any(goal_map > 0))

        self.curr_pose = [start_x, start_y, start_o]   # 标记visited map 已访问过的位置，免重复
        self.visited_map[gx1:gx2, gy1:gy2][
            start[0] - 0 : start[0] + 1, start[1] - 0 : start[1] + 1
        ] = 1

        self.visited_count_map[gx1:gx2, gy1:gy2][
            start[0] - 0 : start[0] + 1, start[1] - 0 : start[1] + 1
        ] += 1
        if self.visited_count_map[gx1:gx2, gy1:gy2][start[0],start[1]] > 25:
            self.stuck = True
            print("get out of stuck")
        else:
            if self.visited_count_map[gx1:gx2, gy1:gy2][start[0],start[1]] < 8:
                self.stuck = False

        # Check collisions if we have just moved and are uncertain
        if self.last_action == DiscreteNavigationAction.MOVE_FORWARD or (
            type(self.last_action) == ContinuousNavigationAction
            and np.linalg.norm(self.last_action.xyt[:2]) > 0
        ):
            self._check_collision()

        try:
            # High-level goal -> short-term goal
            # Extracts a local waypoint
            # Defined by the step size - should be relatively close to the robot
            (
                short_term_goal,
                closest_goal_map,
                replan,
                stop,
                closest_goal_pt,
                dilated_obstacles,
            ) = self._get_short_term_goal(
                found_goal,
                obstacle_map,
                np.copy(goal_map),
                start,
                planning_window,
                plan_to_dilated_goal=use_dilation_for_stg,
                frontier_map=frontier_map,
            )
        except Exception as e:
            print("Warning! Planner crashed with error:", e)
            return (
                DiscreteNavigationAction.STOP,
                np.zeros(goal_map.shape),
                (0, 0),
                np.zeros(goal_map.shape),
            )
        # Short term goal is in cm, start_x and start_y are in m
        if debug:
            print("Current pose:", start)
            print("Short term goal:", short_term_goal)
            print(
                "  - delta =",
                short_term_goal[0] - start[0],
                short_term_goal[1] - start[1],
            )
            dist_to_short_term_goal = np.linalg.norm(
                start - np.array(short_term_goal[:2])
            )
            print(
                "Distance (m):",
                dist_to_short_term_goal * self.map_resolution * CM_TO_METERS,
            )
            print("Replan:", replan)
        # t1 = time.time()
        # print(f"[Planning] get_short_term_goal() time: {t1 - t0}")

        # We were not able to find a path to the high-level goal
        if replan and not stop:
            # Clean collision map
            self.collision_map *= 0
            # Reduce obstacle dilation
            if self.curr_obs_dilation_selem_radius > self.min_obs_dilation_selem_radius:
                self.curr_obs_dilation_selem_radius -= 1
                self.obs_dilation_selem = skimage.morphology.disk(
                    self.curr_obs_dilation_selem_radius
                )
                if debug:
                    print(
                        f"reduced obs dilation to {self.curr_obs_dilation_selem_radius}"
                    )

            if found_goal:
                if debug:
                    print(
                        "ERROR: Could not find a path to the high-level goal. Trying to explore more..."
                    )
                (
                    short_term_goal,
                    closest_goal_map,
                    replan,
                    stop,
                    closest_goal_pt,
                    dilated_obstacles,
                ) = self._get_short_term_goal(
                    found_goal,
                    obstacle_map,
                    frontier_map,
                    start,
                    planning_window,
                    plan_to_dilated_goal=True,
                )
                if debug:
                    print("--- after replanning to frontier ---")
                    print("goal =", short_term_goal)
                found_goal = False
                if replan:
                    print("Nowhere left to explore. Stopping.")
                    # Calling the STOP action here will cause the agent to try grasping
                    #  TODO separate out STOP_SUCCESS and STOP_FAILURE actions
                    return (
                        DiscreteNavigationAction.STOP,
                        closest_goal_map,
                        short_term_goal,
                        dilated_obstacles,
                    )

        # Normalize agent angle
        angle_agent = pu.normalize_angle(start_o)

        # If we found a short term goal worth moving towards...
        stg_x, stg_y = short_term_goal
        relative_stg_x, relative_stg_y = stg_x - start[0], stg_y - start[1]
        angle_st_goal = math.degrees(math.atan2(relative_stg_x, relative_stg_y))
        relative_angle_to_stg = pu.normalize_angle(angle_agent - angle_st_goal)

        # Compute angle to the final goal
        goal_x, goal_y = closest_goal_pt
        angle_goal = math.degrees(math.atan2(goal_x - start[0], goal_y - start[1]))
        relative_angle_to_closest_goal = pu.normalize_angle(angle_agent - angle_goal)

        if debug:
            # Actual metric distance to goal
            distance_to_goal = np.linalg.norm(np.array([goal_x, goal_y]) - start)
            distance_to_goal_cm = distance_to_goal * self.map_resolution
            # Display information
            print("-----------------")
            print("Found reachable goal:", found_goal)
            print("Stop:", stop)
            print("Angle to goal:", relative_angle_to_closest_goal)
            print("Distance to goal", distance_to_goal)
            print(
                "Distance in cm:",
                distance_to_goal_cm,
                ">",
                self.min_goal_distance_cm,
            )

            m_relative_stg_x, m_relative_stg_y = [
                CM_TO_METERS * self.map_resolution * d
                for d in [relative_stg_x, relative_stg_y]
            ]
            print("continuous actions for exploring")
            print("agent angle =", angle_agent)
            print("angle stg goal =", angle_st_goal)
            print("angle final goal =", relative_angle_to_closest_goal)
            print(
                m_relative_stg_x, m_relative_stg_y, "rel ang =", relative_angle_to_stg
            )
            print("-----------------")

        action = self.get_action(
            relative_stg_x,
            relative_stg_y,
            relative_angle_to_stg,
            relative_angle_to_closest_goal,
            start_o,
            found_goal,
            stop,
            debug,
        )

        self.last_action = action
        return action, closest_goal_map, short_term_goal, dilated_obstacles

    def get_action(
        self,
        relative_stg_x: float,
        relative_stg_y: float,
        relative_angle_to_stg: float,
        relative_angle_to_closest_goal: float,
        start_compass: float,
        found_goal: bool,
        stop: bool,
        debug: bool,
    ):
        """
        Gets discrete/continuous action given short-term goal. Agent orients to closest goal if found_goal=True and stop=True
        """
        # Short-term goal -> deterministic local policy   
        # 
        if self.stuck:
            if self.last_action == DiscreteNavigationAction.MOVE_FORWARD and self.col_width == 1:
                self.move_ahead_flag = 1
            if self.move_ahead_flag:
                action = DiscreteNavigationAction.MOVE_FORWARD
                self.move_ahead_flag = 1 - self.move_ahead_flag
            else:
                action = DiscreteNavigationAction.TURN_RIGHT
                self.move_ahead_flag = 1 - self.move_ahead_flag
            return action         

        if not (found_goal and stop):
            if self.discrete_actions:
                if relative_angle_to_stg > self.turn_angle / 2.0:
                    action = DiscreteNavigationAction.TURN_RIGHT
                elif relative_angle_to_stg < -self.turn_angle / 2.0:
                    action = DiscreteNavigationAction.TURN_LEFT
                else:
                    action = DiscreteNavigationAction.MOVE_FORWARD
            else:
                # Use the short-term goal to set where we should be heading next
                m_relative_stg_x, m_relative_stg_y = [
                    CM_TO_METERS * self.map_resolution * d
                    for d in [relative_stg_x, relative_stg_y]
                ]
                if np.abs(relative_angle_to_stg) > self.turn_angle / 2.0:
                    # Must return commands in radians and meters
                    relative_angle_to_stg = math.radians(relative_angle_to_stg)
                    action = ContinuousNavigationAction([0, 0, -relative_angle_to_stg])
                else:
                    # Must return commands in radians and meters
                    relative_angle_to_stg = math.radians(relative_angle_to_stg)
                    xyt_global = [
                        m_relative_stg_y,
                        m_relative_stg_x,
                        -relative_angle_to_stg,
                    ]

                    xyt_local = xyt_global_to_base(
                        xyt_global, [0, 0, math.radians(start_compass)]
                    )
                    xyt_local[
                        2
                    ] = (
                        -relative_angle_to_stg
                    )  # the original angle was already in base frame
                    action = ContinuousNavigationAction(xyt_local)
        else:
            # Try to orient towards the goal object - or at least any point sampled from the goal
            # object.
            if debug:
                print()
                print("----------------------------")
                print(">>> orient towards the goal:", relative_angle_to_closest_goal)
            if self.discrete_actions:
                if relative_angle_to_closest_goal > 2 * self.turn_angle / 3.0:
                    action = DiscreteNavigationAction.TURN_RIGHT
                elif relative_angle_to_closest_goal < -2 * self.turn_angle / 3.0:
                    action = DiscreteNavigationAction.TURN_LEFT
                else:
                    action = DiscreteNavigationAction.STOP
            elif (
                np.abs(relative_angle_to_closest_goal) > self.continuous_angle_tolerance
            ):
                if debug:
                    print("Continuous rotation towards goal point")
                relative_angle_to_closest_goal = math.radians(
                    relative_angle_to_closest_goal
                )
                action = ContinuousNavigationAction(
                    [0, 0, -relative_angle_to_closest_goal]
                )
            else:
                action = DiscreteNavigationAction.STOP
                if debug:
                    print("!!! DONE !!!")

        return action

    def _get_short_term_goal(
        self,
        found_goal,
        obstacle_map: np.ndarray,
        goal_map: np.ndarray,
        start: List[int],
        planning_window: List[int],
        plan_to_dilated_goal=False,
        frontier_map=None,
        visualize=False,
    ) -> Tuple[Tuple[int, int], np.ndarray, bool, bool]:
        """Get short-term goal.
        1. 准备局部子图
        2. 对障碍物做膨胀，构造可通行图traversible
        3. 创建FMM
        4. 取短期目标


        Args:
            obstacle_map: (M, M) binary local obstacle map prediction
            goal_map: (M, M) binary array denoting goal location
            start: start location (x, y)
            planning_window: local map boundaries (gx1, gx2, gy1, gy2)
            plan_to_dilated_goal: for objectnav; plans to dialted goal points instead of explicitly checking reach.

        Returns:
            short_term_goal: short-term goal position (x, y) in map
            closest_goal_map: (M, M) binary array denoting closest goal
             location in the goal map in geodesic distance
            replan: binary flag to indicate we couldn't find a plan to reach
             the goal
            stop: binary flag to indicate we've reached the goal
        """
        gx1, gx2, gy1, gy2 = planning_window
        (x1, y1,) = (
            0,
            0,
        )
        x2, y2 = obstacle_map.shape
        obstacles = obstacle_map[x1:x2, y1:y2]

        # Dilate obstacles
        dilated_obstacles = cv2.dilate(obstacles, self.obs_dilation_selem, iterations=1)

        # Create inverse map of obstacles - this is territory we assume is traversible
        # Traversible is now the map
        traversible = 1 - dilated_obstacles
        traversible[self.collision_map[gx1:gx2, gy1:gy2][x1:x2, y1:y2] == 1] = 0
        traversible[self.visited_map[gx1:gx2, gy1:gy2][x1:x2, y1:y2] == 1] = 1
        agent_rad = self.agent_cell_radius
        # traversible[
        #     int(start[0] - x1) - agent_rad : int(start[0] - x1) + agent_rad + 1,
        #     int(start[1] - y1) - agent_rad : int(start[1] - y1) + agent_rad + 1,
        # ] = 1
        traversible = add_boundary(traversible)
        goal_map = add_boundary(goal_map, value=0)
        planner = FMMPlanner(
            traversible,
            step_size=self.step_size,
            vis_dir=self.vis_dir,
            visualize=self.visualize,
            print_images=self.print_images,
            goal_tolerance=self.goal_tolerance,
        )
        if plan_to_dilated_goal:  # 到目标
            # Compute dilated goal map for use with simulation code - use this to compute closest goal
            dilated_goal_map = cv2.dilate(
                goal_map, self.goal_dilation_selem, iterations=1
            )
            # Set multi goal to the dilated goal map
            # We will now try to find a path to any of these spaces
            self.dd = planner.set_multi_goal(
                dilated_goal_map,
                self.timestep,
                self.dd,
                self.map_downsample_factor,
                self.map_update_frequency,
            )
            goal_distance_map, closest_goal_pt = self.get_closest_traversible_goal(
                traversible, goal_map, start, dilated_goal_map=dilated_goal_map
            )
        else:   # 到frontier
            if found_goal:
                navigable_goal_map = planner._find_within_distance_to_multi_goal(
                    goal_map,
                    self.min_goal_distance_cm / self.map_resolution,
                    timestep=self.timestep,
                    vis_dir=self.vis_dir,
                    min_distance_only=True
                )
            else:
                navigable_goal_map = planner._find_within_distance_to_multi_goal(
                    goal_map,
                    self.min_goal_distance_cm / self.map_resolution,
                    timestep=self.timestep,
                    vis_dir=self.vis_dir,
                )        
            if not np.any(navigable_goal_map):
                frontier_map = add_boundary(frontier_map, value=0)
                navigable_goal_map = frontier_map
            self.dd = planner.set_multi_goal(
                navigable_goal_map,
                self.timestep,
                self.dd,
                self.map_downsample_factor,
                self.map_update_frequency,
            )
            goal_distance_map, closest_goal_pt = self.get_closest_goal(goal_map, start)

        self.timestep += 1

        state = [start[0] - x1 + 1, start[1] - y1 + 1]
        # This is where we create the planner to get the trajectory to this state
        stg_x, stg_y, replan, stop = planner.get_short_term_goal(
            state, continuous=(not self.discrete_actions)
        )
        stg_x, stg_y = stg_x + x1 - 1, stg_y + y1 - 1
        short_term_goal = int(stg_x), int(stg_y)

        if visualize:
            print("Start visualizing")
            plt.figure(1)
            plt.subplot(131)
            _navigable_goal_map = navigable_goal_map.copy()
            _navigable_goal_map[int(stg_x), int(stg_y)] = 1
            plt.imshow(np.flipud(_navigable_goal_map))
            plt.plot(stg_x, stg_y, "bx")
            plt.plot(start[0], start[1], "rx")
            plt.subplot(132)
            plt.imshow(np.flipud(planner.fmm_dist))
            plt.subplot(133)
            plt.imshow(np.flipud(planner.traversible))
            plt.show()
            print("Done visualizing.")

        if visualize:
            import os
            os.makedirs(self.vis_dir, exist_ok=True)

            fig, ax = plt.subplots(figsize=(6, 6))

            # 1) 障碍物灰度底图（1=障碍黑，0=可行白）
            ax.imshow(obstacles, cmap="gray", vmin=0, vmax=1,
                    origin="lower", interpolation="nearest")

            # 2) 叠加 goal_map（黄色半透明）
            goal_rgba = np.zeros((goal_map.shape[0], goal_map.shape[1], 4), dtype=float)
            goal_rgba[goal_map == 1] = [1.0, 1.0, 0.0, 0.6]
            ax.imshow(goal_rgba, origin="lower", interpolation="nearest")

            if not plan_to_dilated_goal:
                navigable_goal_rgba = np.zeros((navigable_goal_map.shape[0], navigable_goal_map.shape[1], 4), dtype=float)
                navigable_goal_rgba[navigable_goal_map == 1] = [0.0, 0.0, 1.0, 0.8]
                ax.imshow(navigable_goal_rgba, origin="lower", interpolation="nearest")
            
            collision_map = self.collision_map[gx1:gx2, gy1:gy2][x1:x2, y1:y2]
            collision_map = add_boundary(collision_map, value=0)
            collision_rgba = np.zeros((collision_map.shape[0], collision_map.shape[1], 4), dtype=float)
            collision_rgba[collision_map == 1] = [0.0, 1.0, 0.0, 0.7]
            ax.imshow(collision_rgba, origin="lower", interpolation="nearest")


            # 3) start（红点）
            ax.scatter(start[0], start[1], s=10, c="red", edgecolors="k", linewidths=0.5, label="start")

            # 4) short-term goal（紫色叉号）
            ax.scatter(stg_x, stg_y, s=80, c="purple", marker="x", linewidths=2, label="short-term goal")

            ax.set_title("Planning Window: obstacles (gray) + goal (yellow)")
            ax.set_aspect("equal")
            ax.set_xlim([-0.5, obstacles.shape[1]-0.5])
            ax.set_ylim([-0.5, obstacles.shape[0]-0.5])
            ax.legend(loc="upper right")
            plt.tight_layout()

            out_path = os.path.join(self.vis_dir, f"planning_vis_t{self.timestep:06d}.png")
            # out_path = os.path.join(self.vis_dir, f"planning_vis.png")
            fig.savefig(out_path, dpi=200, bbox_inches="tight")
            plt.close(fig)
            print(f"[viz] saved to {out_path}")



        return (
            short_term_goal,
            goal_distance_map,
            replan,
            stop,
            closest_goal_pt,
            dilated_obstacles,
        )

    def get_closest_traversible_goal(
        self, traversible, goal_map, start, dilated_goal_map=None
    ):
        """Old version of the get_closest_goal function, which takes into account the distance along geometry to a goal object. This will tell us the closest point on the goal map, both for visualization and for orienting towards it to grasp. Uses traversible to sort this out."""

        # NOTE: this is the old version - before adding goal dilation
        # vis_planner = FMMPlanner(traversible)
        # TODO How to do this without the overhead of creating another FMM planner?
        traversible_ = traversible.copy()
        if dilated_goal_map is None:
            traversible_[goal_map == 1] = 1
        else:
            traversible_[dilated_goal_map == 1] = 1
        vis_planner = FMMPlanner(traversible_)
        curr_loc_map = np.zeros_like(goal_map)
        # Update our location for finding the closest goal
        curr_loc_map[start[0], start[1]] = 1
        # curr_loc_map[short_term_goal[0], short_term_goal]1]] = 1
        vis_planner.set_multi_goal(curr_loc_map)
        fmm_dist_ = vis_planner.fmm_dist.copy()
        # find closest point on non-dilated goal map
        goal_map_ = goal_map.copy()
        goal_map_[goal_map_ == 0] = 10000
        fmm_dist_[fmm_dist_ == 0] = 10000
        closest_goal_map = (goal_map_ * fmm_dist_) == (goal_map_ * fmm_dist_).min()
        closest_goal_map = remove_boundary(closest_goal_map)
        closest_goal_pt = np.unravel_index(
            closest_goal_map.argmax(), closest_goal_map.shape
        )
        return closest_goal_map, closest_goal_pt

    def get_closest_goal(self, goal_map, start):
        """closest goal, avoiding any obstacles."""
        empty = np.ones_like(goal_map)
        empty_planner = FMMPlanner(empty)
        empty_planner.set_goal(start)
        dist_map = empty_planner.fmm_dist * goal_map
        dist_map[dist_map == 0] = 10000
        closest_goal_map = dist_map == dist_map.min()
        closest_goal_map = remove_boundary(closest_goal_map)
        closest_goal_pt = np.unravel_index(
            closest_goal_map.argmax(), closest_goal_map.shape
        )
        return closest_goal_map, closest_goal_pt

    def _check_collision(self):
        """Check whether we had a collision and update the collision map."""
        x1, y1, t1 = self.last_pose
        x2, y2, _ = self.curr_pose
        buf = 4
        length = 2

        # You must move at least 5 cm when doing forward actions
        # Otherwise we assume there has been a collision
        if abs(x1 - x2) < 0.05 and abs(y1 - y2) < 0.05:
            self.stuck_time += 1
            self.col_width += 2
            if self.col_width == 7:
                length = 4
                buf = 3
            self.col_width = min(self.col_width, 5) # param
            if self.stuck_time > 20:
                self.col_width = 10
                length = 10
                buf = 5
        else:
            self.col_width = 1
            self.stuck_time = 0

        dist = pu.get_l2_distance(x1, x2, y1, y2)

        if dist < self.collision_threshold:
            # We have a collision
            width = self.col_width

            # Add obstacles to the collision map
            for i in range(length * 2):
                for j in range(width * 2):
                    wx = x1 + 0.025 * (
                        (i + buf) * np.cos(np.deg2rad(t1))
                        + (j - width // 2) * np.sin(np.deg2rad(t1))
                    )
                    wy = y1 + 0.025 * (
                        (i + buf) * np.sin(np.deg2rad(t1))
                        - (j - width // 2) * np.cos(np.deg2rad(t1))
                    )
                    r, c = wy, wx
                    r, c = int(r * 100 / self.map_resolution), int(
                        c * 100 / self.map_resolution
                    )
                    [r, c] = pu.threshold_poses([r, c], self.collision_map.shape)
                    self.collision_map[r, c] = 1
