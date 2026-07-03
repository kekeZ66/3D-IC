# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from collections import defaultdict
from typing import Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import skimage.morphology
import torch
import torch.nn as nn
import trimesh.transformations as tra
from skimage import measure
from torch import IntTensor, Tensor
from torch.nn import functional as F

import home_robot.mapping.map_utils as mu
import home_robot.utils.depth as du
import home_robot.utils.pose as pu
import home_robot.utils.rotation as ru
from home_robot.mapping.instance import InstanceMemory
from home_robot.mapping.semantic.constants import MapConstants as MC
from home_robot.utils.spot import draw_circle_segment, fill_convex_hull
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import warnings


from frontier_exploration.frontier_detection import detect_frontier_waypoints
from frontier_exploration.utils.fog_of_war import reveal_fog_of_war

# For debugging input and output maps - shows matplotlib visuals
debug_maps = False


class Categorical2DSemanticMapModule(nn.Module):
    """
    This class is responsible for updating a dense 2D semantic map with one channel
    per object category, the local and global maps and poses, and generating
    map features — it is a stateless PyTorch module with no trainable parameters.

    Map proposed in:
    Object Goal Navigation using Goal-Oriented Semantic Exploration
    https://arxiv.org/pdf/2007.00643.pdf
    https://github.com/devendrachaplot/Object-Goal-Navigation
    """

    # If true, display point cloud visualizations using Open3d
    debug_mode = False

    def __init__(
        self,
        frame_height: int,
        frame_width: int,
        camera_height: int,
        hfov: int,
        num_sem_categories: int,
        map_size_cm: int,
        map_resolution: int,
        vision_range: int,
        explored_radius: int,
        been_close_to_radius: int,
        global_downscaling: int,
        du_scale: int,
        cat_pred_threshold: float,
        exp_pred_threshold: float,
        map_pred_threshold: float,
        min_depth: float = 0.5,
        max_depth: float = 3.5,
        must_explore_close: bool = False,
        min_obs_height_cm: int = 25,
        dilate_obstacles: bool = True,
        dilate_iter: int = 1,
        dilate_size: int = 3,
        target_blacklisting_radius: int = None,
        record_instance_ids: bool = False,
        evaluate_instance_tracking: bool = False,
        instance_memory: Optional[InstanceMemory] = None,
        max_instances: int = 0,
        instance_association: str = "map_overlap",
        dilation_for_instances: int = 5,
        padding_for_instance_overlap: int = 5,
        exploration_type="default",
        gaze_width=30,
        gaze_distance=3,
    ):
        """
        Arguments:
            frame_height: first-person frame height
            frame_width: first-person frame width
            camera_height: camera sensor height (in metres)
            hfov: horizontal field of view (in degrees)
            num_sem_categories: number of semantic segmentation categories
            map_size_cm: global map size (in centimetres)
            map_resolution: size of map bins (in centimeters)
            vision_range: diameter of the circular region of the local map
             that is visible by the agent located in its center (unit is
             the number of local map cells)
            explored_radius: radius (in centimeters) of region of the visual cone
             that will be marked as explored
            been_close_to_radius: radius (in centimeters) of been close to region
            target_blacklisting_radius: radius (in centimeters) of region
             around target that will be blacklisted (if invalid target)
            global_downscaling: ratio of global over local map
            du_scale: frame downscaling before projecting to point cloud
            cat_pred_threshold: number of depth points to be in bin to
             classify it as a certain semantic category
            exp_pred_threshold: number of depth points to be in bin to
             consider it as explored
            map_pred_threshold: number of depth points to be in bin to
             consider it as obstacle
            must_explore_close: reduce the distance we need to get to things to make them work
            min_obs_height_cm: minimum height of obstacles (in centimetres)
            record_instance_ids: whether to record instance ids in the 2d semantic map
            exploration_type: how to define explored area
            gaze_width: hfov in degrees for use with the gaze based exploration
            gaze_distance: depth to be considered explored with gaze based exploration
        """
        super().__init__()

        self.screen_h = frame_height
        self.screen_w = frame_width
        self.camera_matrix = du.get_camera_matrix(self.screen_w, self.screen_h, hfov)
        self.num_sem_categories = num_sem_categories
        self.must_explore_close = must_explore_close

        self.map_size_parameters = mu.MapSizeParameters(
            map_resolution, map_size_cm, global_downscaling
        )
        self.resolution = map_resolution
        self.global_map_size_cm = map_size_cm
        self.global_downscaling = global_downscaling
        self.local_map_size_cm = self.global_map_size_cm // self.global_downscaling
        self.global_map_size = self.global_map_size_cm // self.resolution
        self.local_map_size = self.local_map_size_cm // self.resolution
        self.xy_resolution = self.z_resolution = map_resolution
        self.vision_range = vision_range
        self.explored_radius = explored_radius
        self.been_close_to_radius = been_close_to_radius
        if target_blacklisting_radius is not None:
            self.target_blacklisting_radius = target_blacklisting_radius
        self.du_scale = du_scale
        self.cat_pred_threshold = cat_pred_threshold
        self.exp_pred_threshold = exp_pred_threshold
        self.map_pred_threshold = map_pred_threshold

        self.max_depth = max_depth * 100.0
        self.min_depth = min_depth * 100.0
        self.agent_height = camera_height * 100.0
        self.max_voxel_height = int(360 / self.z_resolution)
        self.min_voxel_height = int(-40 / self.z_resolution)
        self.min_obs_height_cm = min_obs_height_cm
        self.min_mapped_height = int(
            self.min_obs_height_cm / self.z_resolution - self.min_voxel_height
        )
        self.max_mapped_height = int(
            (self.agent_height + 1) / self.z_resolution - self.min_voxel_height
        )
        self.shift_loc = [self.vision_range * self.xy_resolution // 2, 0, np.pi / 2.0]

        # For cleaning up maps
        self.dilate_obstacles = dilate_obstacles
        self.dilate_kernel = np.ones((dilate_size, dilate_size))
        self.dilate_size = dilate_size
        self.dilate_iter = dilate_iter
        self.record_instance_ids = record_instance_ids
        self.instance_association = instance_association
        self.padding_for_instance_overlap = padding_for_instance_overlap
        self.dilation_for_instances = dilation_for_instances
        self.instance_memory = instance_memory
        self.max_instances = max_instances
        self.evaluate_instance_tracking = evaluate_instance_tracking
        self.exploration_type = exploration_type
        self.gaze_width = gaze_width
        self.gaze_distance = gaze_distance
        self.frontiers = np.array([])
        self.hfov = hfov # rad
        self._min_confidence = 0.25
        self._confidence_masks: Dict[Tuple[float, float], np.ndarray] = {}
        self._map = np.zeros((self.global_map_size, self.global_map_size))
        self._value_map = np.zeros((self.global_map_size, self.global_map_size, 1),np.float32)
        self._map_recp = np.zeros((self.global_map_size, self.global_map_size))
        self._value_map_recp = np.zeros((self.global_map_size, self.global_map_size, 1),np.float32)
        self.roll = None

    @torch.no_grad()
    def forward(
        self,
        seq_obs: Tensor,
        seq_pose_delta: Tensor,
        seq_dones: Tensor,
        seq_update_global: Tensor,
        seq_camera_poses: Tensor,
        init_local_map: Tensor,
        init_global_map: Tensor,
        init_local_pose: Tensor,
        init_global_pose: Tensor,
        init_lmb: Tensor,
        init_origins: Tensor,
        values: float,
        seq_obstacle_locations: Optional[Tensor] = None,
        seq_free_locations: Optional[Tensor] = None,
        blacklist_target: bool = False,
        semantic_max_val: Optional[int] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, IntTensor, Tensor]:
        """Update maps and poses with a sequence of observations and generate map
        features at each time step.

        Arguments:
            seq_obs: sequence of frames containing (RGB, depth, segmentation)
             of shape (batch_size, sequence_length, 3 + 1 + num_sem_categories,
             frame_height, frame_width)
            seq_pose_delta: sequence of delta in pose since last frame of shape
             (batch_size, sequence_length, 3)
            seq_dones: sequence of (batch_size, sequence_length) binary flags
             that indicate episode restarts
            seq_update_global: sequence of (batch_size, sequence_length) binary
             flags that indicate whether to update the global map and pose
            seq_camera_poses: sequence of (batch_size, sequence_length, 4, 4) extrinsic camera
             matrices
            init_local_map: initial local map before any updates of shape
             (batch_size, MC.NON_SEM_CHANNELS + num_sem_categories, M, M)
            init_global_map: initial global map before any updates of shape
             (batch_size, MC.NON_SEM_CHANNELS + num_sem_categories, M * ds, M * ds)
            init_local_pose: initial local pose before any updates of shape
             (batch_size, 3)
            init_global_pose: initial global pose before any updates of shape
             (batch_size, 3)
            init_lmb: initial local map boundaries of shape (batch_size, 4)
            init_origins: initial local map origins of shape (batch_size, 3)

        Returns:
            seq_map_features: sequence of semantic map features of shape
             (batch_size, sequence_length, 2 * MC.NON_SEM_CHANNELS + num_sem_categories, M, M)
            final_local_map: final local map after all updates of shape
             (batch_size, MC.NON_SEM_CHANNELS + num_sem_categories, M, M)
            final_global_map: final global map after all updates of shape
             (batch_size, MC.NON_SEM_CHANNELS + num_sem_categories, M * ds, M * ds)
            seq_local_pose: sequence of local poses of shape
             (batch_size, sequence_length, 3)
            seq_global_pose: sequence of global poses of shape
             (batch_size, sequence_length, 3)
            seq_lmb: sequence of local map boundaries of shape
             (batch_size, sequence_length, 4)
            seq_origins: sequence of local map origins of shape
             (batch_size, sequence_length, 3)
        """
        batch_size, sequence_length = seq_obs.shape[:2]
        device, dtype = seq_obs.device, seq_obs.dtype

        map_features_channels = 2 * MC.NON_SEM_CHANNELS + self.num_sem_categories
        if self.record_instance_ids:
            map_features_channels += self.num_sem_categories
        if self.evaluate_instance_tracking:
            map_features_channels += self.max_instances + 1
        seq_map_features = torch.zeros(
            batch_size,
            sequence_length,
            map_features_channels,
            self.local_map_size,
            self.local_map_size,
            device=device,
            dtype=dtype,
        )
        seq_local_pose = torch.zeros(batch_size, sequence_length, 3, device=device)
        seq_global_pose = torch.zeros(batch_size, sequence_length, 3, device=device)
        seq_lmb = torch.zeros(
            batch_size, sequence_length, 4, device=device, dtype=torch.int32
        )
        seq_origins = torch.zeros(batch_size, sequence_length, 3, device=device)

        local_map, local_pose = init_local_map.clone(), init_local_pose.clone()
        global_map, global_pose = init_global_map.clone(), init_global_pose.clone()
        lmb, origins = init_lmb.clone(), init_origins.clone()
        for t in range(sequence_length):
            # Reset map and pose for episodes done at time step t
            for e in range(batch_size):
                if seq_dones[e, t]:
                    mu.init_map_and_pose_for_env(
                        e,
                        local_map,
                        global_map,
                        local_pose,
                        global_pose,
                        lmb,
                        origins,
                        self.map_size_parameters,
                    )

            local_map, local_pose = self._update_local_map_and_pose(
                seq_obs[:, t],
                seq_pose_delta[:, t],
                local_map,
                local_pose,
                seq_camera_poses[:, t],
                origins,
                lmb,
                seq_obstacle_locations[:, t]
                if seq_obstacle_locations is not None
                else None,
                seq_free_locations[:, t] if seq_free_locations is not None else None,
                blacklist_target,
                semantic_max_val=semantic_max_val,
            )
            for e in range(batch_size):
                if seq_update_global[e, t]:
                    self._update_global_map_and_pose_for_env(
                        e, local_map, global_map, local_pose, global_pose, lmb, origins
                    )


            # self.visualize_frontiers(local_map)
            # value_mask = make_local_fov_mask(current_map=local_map, local_pose=local_pose)
            # self._get_confidence_mask(fov=np.deg2rad(self.hfov))
            lmb_np = lmb.squeeze(0).cpu().numpy()
            depth = seq_obs[:,t][:, 3, :, :].float() / 100.0
            visible_mask = self._process_current_data(depth.cpu().numpy(),np.deg2rad(self.hfov),0.5,5.0,global_pose)
            local_map_frontiers, local_map, global_map = self.update_frontiers(local_map,global_map,local_pose,origins.squeeze(0).cpu().numpy(),lmb_np)
            visible_mask = np.flipud(visible_mask)
            self._fuse_value_map(visible_mask, np.array(values[0]))
            self._fuse_value_map_recp(visible_mask, np.array(values[1]))
            # value_save_cv2(np.flipud(self._value_map.reshape((960,960))), "/home/zkm/home-robot/datadump_1/debug_value_map/value_map.png")
            value_map_channel = torch.from_numpy(self._value_map).to(global_map.device,global_map.dtype).squeeze(2).unsqueeze(0)
            global_map[:,MC.VALUE_MAP,:,:] = value_map_channel
            
            # lmb_np = lmb.squeeze(0).cpu().numpy()
            x1, x2, y1, y2 = lmb_np[0], lmb_np[1], lmb_np[2], lmb_np[3] # TODO:function
            local_value_map = self._value_map.reshape((960,960))[x1:x2,y1:y2]
            local_value_map_channel = torch.from_numpy(local_value_map).to(local_map.device, local_map.dtype).unsqueeze(0)
            local_map[:,MC.VALUE_MAP,:,:] = local_value_map_channel
            
            # --------------
            value_map_channel = torch.from_numpy(self._value_map_recp).to(global_map.device,global_map.dtype).squeeze(2).unsqueeze(0)
            global_map[:,MC.VALUE_MAP_PLACE,:,:] = value_map_channel
            
            # lmb_np = lmb.squeeze(0).cpu().numpy()
            x1, x2, y1, y2 = lmb_np[0], lmb_np[1], lmb_np[2], lmb_np[3] # TODO:function
            local_value_map = self._value_map_recp.reshape((960,960))[x1:x2,y1:y2]
            local_value_map_channel = torch.from_numpy(local_value_map).to(local_map.device, local_map.dtype).unsqueeze(0)
            local_map[:,MC.VALUE_MAP_PLACE,:,:] = local_value_map_channel
            # value_save_cv2(np.flipud(self._value_map_recp.reshape((960,960))), "/home/zkm/home-robot/datadump_1/debug_value_map/value_map_recp.png")

            # ----------------


            # self.save_current_map_vis(global_map ,step=t, map_resolution_cm=self.resolution,local_pose=global_pose[0].cpu().numpy(),theta_in_radians=False)
            # self.save_current_map_vis(local_map ,step=t, map_resolution_cm=self.resolution,local_pose=local_pose[0].cpu().numpy(),theta_in_radians=False)

            seq_local_pose[:, t] = local_pose
            seq_global_pose[:, t] = global_pose
            seq_lmb[:, t] = lmb
            seq_origins[:, t] = origins
            seq_map_features[:, t] = self._get_map_features(local_map, global_map)

            
        return (
            seq_map_features,
            local_map,
            global_map,
            seq_local_pose,
            seq_global_pose,
            seq_lmb,
            seq_origins,
        )

    def _aggregate_instance_map_channels_per_category(
        self, curr_map, num_instance_channels
    ):
        """Aggregate map channels for instances (input: one binary channel per instance in [0, 1])
        by category (output: one channel per category containing instance IDs)."""

        # first extract instance channels
        top_down_instance_one_hot = curr_map[
            :,
            (MC.NON_SEM_CHANNELS + self.num_sem_categories) : (
                MC.NON_SEM_CHANNELS + self.num_sem_categories + num_instance_channels
            ),
            :,
            :,
        ]
        # now we convert the top down instance map to get a map for storing instances per channel
        top_down_instances_per_category = torch.zeros(
            curr_map.shape[0],
            self.num_sem_categories,
            curr_map.shape[2],
            curr_map.shape[3],
            device=curr_map.device,
            dtype=curr_map.dtype,
        )

        if num_instance_channels > 0:
            # loop over envs
            # TODO Can we vectorize this across envs? (Only needed if we use multiple envs)
            for i in range(top_down_instance_one_hot.shape[0]):
                # create category id to instance id list mapping
                category_id_to_instance_id_list = defaultdict(list)
                # retrieve unprocessed instances
                unprocessed_instances = (
                    self.instance_memory.get_unprocessed_instances_per_env(i)
                )
                # loop over unprocessed instances
                for instance_id, instance in unprocessed_instances.items():
                    category_id_to_instance_id_list[instance.category_id].append(
                        instance_id
                    )

                # loop over categories
                # TODO Can we vectorize this across categories? (Only needed if speed bottleneck)
                for category_id in category_id_to_instance_id_list.keys():
                    if len(category_id_to_instance_id_list[category_id]) == 0:
                        continue
                    # get the instance ids for this category
                    instance_ids = category_id_to_instance_id_list[category_id]
                    # create a tensor by slicing top_down_instance_one_hot using the instance ids
                    instance_one_hot = top_down_instance_one_hot[i, instance_ids]
                    # add a channel with all values equal to 1e-5 as the first channel
                    instance_one_hot = torch.cat(
                        (
                            1e-5 * torch.ones_like(instance_one_hot[:1]),
                            instance_one_hot,
                        ),
                        dim=0,
                    )
                    # get the instance id map using argmax
                    instance_id_map = instance_one_hot.argmax(dim=0)
                    # add a zero to start of instance ids
                    instance_id = [0] + instance_ids
                    # update the ids using the list of instance ids
                    instance_id_map = torch.tensor(
                        instance_id, device=instance_id_map.device
                    )[instance_id_map]
                    # update the per category instance map
                    top_down_instances_per_category[i, category_id] = instance_id_map

        curr_map = torch.cat(
            (
                curr_map[:, : MC.NON_SEM_CHANNELS + self.num_sem_categories],
                top_down_instances_per_category,
                curr_map[
                    :,
                    MC.NON_SEM_CHANNELS
                    + self.num_sem_categories
                    + num_instance_channels :,
                ],
            ),
            dim=1,
        )

        return curr_map

    def _update_local_map_and_pose(  # noqa: C901
        self,
        obs: Tensor,
        pose_delta: Tensor,
        prev_map: Tensor,
        prev_pose: Tensor,
        camera_pose: Tensor,
        origins: Tensor,
        lmb: Tensor,
        obstacle_locations: Optional[Tensor] = None,
        free_locations: Optional[Tensor] = None,
        blacklist_target: bool = False,
        debug: bool = False,
        semantic_max_val: Optional[int] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Update local map and sensor pose given a new observation using parameter-free
        differentiable projective geometry.

        Args:
            obs: current frame containing (rgb, depth, segmentation) of shape
             (batch_size, 3 + 1 + num_sem_categories, frame_height, frame_width)
            pose_delta: delta in pose since last frame of shape (batch_size, 3)
            prev_map: previous local map of shape
             (batch_size, MC.NON_SEM_CHANNELS + num_sem_categories, M, M)
            prev_pose: previous pose of shape (batch_size, 3)
            camera_pose: current camera poseof shape (batch_size, 4, 4)

        Returns:
            current_map: current local map updated with current observation
             and location of shape (batch_size, MC.NON_SEM_CHANNELS + num_sem_categories, M, M)
            current_pose: current pose updated with pose delta of shape (batch_size, 3)
        """
        if semantic_max_val is None:
            semantic_max_val = self.num_sem_categories
        batch_size, obs_channels, h, w = obs.size()
        device, dtype = obs.device, obs.dtype
        if camera_pose is not None: # yaw pitch roll
            # TODO: make consistent between sim and real
            # hab_angles = pt.matrix_to_euler_angles(camera_pose[:, :3, :3], convention="YZX")
            # angles = pt.matrix_to_euler_angles(camera_pose[:, :3, :3], convention="ZYX")
            angles = torch.Tensor(
                [tra.euler_from_matrix(p[:3, :3].cpu(), "rzyx") for p in camera_pose]
            )

            # For habitat - pull x angle
            # tilt = angles[:, -1]
            # For real robot
            tilt = angles[:, 1]
            # angles gives roll, pitch, yaw
            yaw = angles[:, -1]
            roll = angles[:, 0]
            self.roll = roll.cpu().numpy()[0]
            camera_x = camera_pose[:, 0, 3] * -100
            camera_y = camera_pose[:, 1, 3] * -100
            # Get the agent pose
            # hab_agent_height = camera_pose[:, 1, 3] * 100
            agent_pos = camera_pose[:, :3, 3] * 100
            agent_height = agent_pos[:, 2]

            if debug:
                print("tilt", tilt)
                print("agent_height", agent_height)
                print()
        else:
            yaw = 0
            roll = 0
            camera_x = None
            camera_y = None
            tilt = torch.zeros(batch_size)
            agent_height = self.agent_height

        if not isinstance(yaw, torch.Tensor):
            yaw = torch.tensor(yaw)
        depth = obs[:, 3, :, :].float()
        depth[depth > self.max_depth] = 0

        point_cloud_t = du.get_point_cloud_from_z_t(
            depth, self.camera_matrix, device, scale=self.du_scale
        )

        if self.debug_mode:
            from home_robot.utils.point_cloud import show_point_cloud

            rgb = obs[:, :3, :: self.du_scale, :: self.du_scale].permute(0, 2, 3, 1)
            xyz = point_cloud_t[0].reshape(-1, 3)
            rgb = rgb[0].reshape(-1, 3)
            print("-> Showing point cloud in camera coords")
            show_point_cloud(
                (xyz / 100.0).cpu().numpy(),
                (rgb / 255.0).cpu().numpy(),
                orig=np.zeros(3),
            )

        point_cloud_base_coords = du.transform_camera_view_t(
            point_cloud_t, agent_height, torch.rad2deg(tilt).cpu().numpy(), device
        )

        # Show the point cloud in base coordinates for debugging
        if self.debug_mode:
            print()
            print("------------------------------")
            print("agent angles =", angles)
            print("agent tilt   =", tilt)
            print("agent height =", agent_height, "preset =", self.agent_height)
            xyz = point_cloud_base_coords[0].reshape(-1, 3)
            print("-> Showing point cloud in base coords")
            show_point_cloud(
                (xyz / 100.0).cpu().numpy(),
                (rgb / 255.0).cpu().numpy(),
                orig=np.zeros(3),
            )

        point_cloud_map_coords = du.transform_pose_t(
            point_cloud_base_coords, self.shift_loc, device
        )

        if self.debug_mode:
            xyz = point_cloud_base_coords[0].reshape(-1, 3)
            print("-> Showing point cloud in map coords")
            show_point_cloud(
                (xyz / 100.0).cpu().numpy(),
                (rgb / 255.0).cpu().numpy(),
                orig=np.zeros(3),
            )

        voxel_channels = 1 + self.num_sem_categories
        num_instance_channels = 0
        if self.record_instance_ids:
            num_instance_channels = obs_channels - 4 - self.num_sem_categories
            if self.evaluate_instance_tracking:
                num_instance_channels -= self.max_instances + 1
            voxel_channels += num_instance_channels
        if self.evaluate_instance_tracking:
            voxel_channels += self.max_instances + 1

        init_grid = torch.zeros(
            batch_size,
            voxel_channels,
            self.vision_range,
            self.vision_range,
            self.max_voxel_height - self.min_voxel_height,
            device=device,
            dtype=torch.float32,
        )
        feat = torch.ones(
            batch_size,
            voxel_channels,
            self.screen_h // self.du_scale * self.screen_w // self.du_scale,
            device=device,
            dtype=torch.float32,
        )

        semantic_channels = obs[:, 4 : 4 + self.num_sem_categories, :, :]

        current_pose = pu.get_new_pose_batch(prev_pose.clone(), pose_delta)

        current_pose = pu.get_new_pose_batch(prev_pose.clone(), pose_delta)

        if self.record_instance_ids:
            instance_channels = obs[
                :,
                4
                + self.num_sem_categories : 4
                + self.num_sem_categories
                + num_instance_channels,
                :,
                :,
            ]
            global_pose = current_pose + origins

            if camera_x is None:
                camera_x = global_pose[:, 0]
            if camera_y is None:
                camera_y = global_pose[:, 1]
            global_pose = np.array([camera_x.item(), camera_y.item(), roll.item()])
            absolute_point_cloud = du.transform_pose_t(
                point_cloud_base_coords, global_pose, device
            )

            if num_instance_channels > 0:
                self.instance_memory.process_instances(
                    instance_channels,
                    absolute_point_cloud,
                    torch.concat([current_pose + origins, lmb], axis=1)
                    .cpu()
                    .float(),  # store the global pose
                    image=obs[:, :3, :, :],
                    semantic_channels=semantic_channels,
                    background_class_labels=[0, semantic_max_val],
                )

        feat[:, 1:, :] = nn.AvgPool2d(self.du_scale)(obs[:, 4:, :, :]).view(
            batch_size, obs_channels - 4, h // self.du_scale * w // self.du_scale
        )

        XYZ_cm_std = point_cloud_map_coords.float()
        XYZ_cm_std[..., :2] = XYZ_cm_std[..., :2] / self.xy_resolution
        XYZ_cm_std[..., :2] = (
            (XYZ_cm_std[..., :2] - self.vision_range // 2.0) / self.vision_range * 2.0
        )
        XYZ_cm_std[..., 2] = XYZ_cm_std[..., 2] / self.z_resolution
        XYZ_cm_std[..., 2] = (
            (
                XYZ_cm_std[..., 2]
                - (self.max_voxel_height + self.min_voxel_height) // 2.0
            )
            / (self.max_voxel_height - self.min_voxel_height)
            * 2.0
        )
        XYZ_cm_std = XYZ_cm_std.permute(0, 3, 1, 2)
        XYZ_cm_std = XYZ_cm_std.view(
            XYZ_cm_std.shape[0],
            XYZ_cm_std.shape[1],
            XYZ_cm_std.shape[2] * XYZ_cm_std.shape[3],
        )

        voxels = du.splat_feat_nd(init_grid, feat, XYZ_cm_std).transpose(2, 3)

        agent_height_proj = voxels[
            ..., self.min_mapped_height : self.max_mapped_height
        ].sum(4)
        all_height_proj = voxels.sum(4)

        fp_map_pred = agent_height_proj[:, 0:1, :, :]

        # +rows is away from the camera, with the camra origin at row 0
        # +cols is to the right of the image frame, the camera origin is at num_cols/2
        # so the camera origin is at [0,num_cols/2]

        # self.local_map_size_cm
        # plt.imshow(fp_exp_pred[0,0].cpu())
        # plt.pause(0.01)

        fp_map_pred = fp_map_pred / self.map_pred_threshold
        # uses depth point projections but limits the fov and distance
        if self.exploration_type == "default":
            fp_exp_pred = all_height_proj[:, 0:1, :, :]
            fp_exp_pred = fp_exp_pred / self.exp_pred_threshold
        elif self.exploration_type == "hull":
            fp_exp_pred = all_height_proj[:, 0:1, :, :]
            fp_exp_pred = fp_exp_pred / self.exp_pred_threshold
            fp_exp_pred = fp_exp_pred.clip(0, 1)
            # set the current agent position as 1
            fp_exp_pred[:, :, 0, fp_exp_pred.shape[-1] // 2] = 1

            # fill convex hull
            filled = fill_convex_hull(fp_exp_pred[0, 0].cpu())
            assert fp_exp_pred.shape[:2] == (1, 1)
            fp_exp_pred[0, 0] = torch.tensor(filled)
        # uses a fixed cone infront of the camerea
        elif self.exploration_type == "gaze":
            fp_exp_pred = torch.zeros_like(fp_map_pred)
            view_image = torch.zeros(fp_map_pred.shape[-2:])
            # get the desired radius in cells
            dist = self.gaze_distance * 100 / self.resolution
            view_image = draw_circle_segment(
                view_image, (0, fp_exp_pred.shape[-1] // 2), dist, 0, self.gaze_width
            )
            fp_exp_pred[..., :, :] = view_image
        # uses depth point projections but limits the fov and distance using the code
        elif self.exploration_type == "gaze_projected":
            fp_exp_pred = all_height_proj[:, 0:1, :, :]
            fp_exp_pred = fp_exp_pred / self.exp_pred_threshold
            view_image = torch.zeros(fp_map_pred.shape[-2:])
            # get the desired radius in cells
            dist = self.gaze_distance * 100 / self.resolution
            view_image = (
                draw_circle_segment(
                    view_image,
                    (0, fp_exp_pred.shape[-1] // 2),
                    dist,
                    0,
                    self.gaze_width,
                )
                / 255
            )
            fp_exp_pred *= view_image.to(fp_exp_pred.device)
        else:
            raise Exception("not implemented")

        num_channels = MC.NON_SEM_CHANNELS + self.num_sem_categories
        if self.record_instance_ids:
            num_channels += num_instance_channels

        if self.evaluate_instance_tracking:
            num_channels += self.max_instances + 1

        agent_view = torch.zeros(
            batch_size,
            num_channels,
            self.local_map_size_cm // self.xy_resolution,
            self.local_map_size_cm // self.xy_resolution,
            device=device,
            dtype=dtype,
        )

        # Update agent view from the fp_map_pred
        if self.dilate_obstacles:
            for i in range(fp_map_pred.shape[0]):
                env_map = fp_map_pred[i, 0].cpu().numpy()
                # TODO: remove if not used
                # env_map_eroded = cv2.erode(
                #     env_map, self.dilate_kernel, self.dilate_iter
                # )
                # filt = cv2.filter2D(env_map, -1, self.dilate_kernel)
                median_filtered = cv2.medianBlur(env_map, self.dilate_size)

                # TODO: remove debug code
                # plt.subplot(121); plt.imshow(env_map)
                # plt.subplot(122); plt.imshow(env_map_eroded)
                # plt.show()
                # breakpoint()
                # fp_map_pred[i, 0] = torch.tensor(env_map_eroded)
                fp_map_pred[i, 0] = torch.tensor(median_filtered)

        x1 = self.local_map_size_cm // (self.xy_resolution * 2) - self.vision_range // 2
        x2 = x1 + self.vision_range
        y1 = self.local_map_size_cm // (self.xy_resolution * 2)
        y2 = y1 + self.vision_range
        agent_view[:, MC.OBSTACLE_MAP : MC.OBSTACLE_MAP + 1, y1:y2, x1:x2] = fp_map_pred
        agent_view[:, MC.EXPLORED_MAP : MC.EXPLORED_MAP + 1, y1:y2, x1:x2] = fp_exp_pred

        agent_view[:, MC.NON_SEM_CHANNELS :, y1:y2, x1:x2] = (
            all_height_proj[:, 1:] / self.cat_pred_threshold
        )

        st_pose = current_pose.clone().detach()
        st_pose[:, :2] = -(
            (
                st_pose[:, :2] * 100.0 / self.xy_resolution
                - self.local_map_size_cm // (self.xy_resolution * 2)
            )
            / (self.local_map_size_cm // (self.xy_resolution * 2))
        )
        st_pose[:, 2] = 90.0 - (st_pose[:, 2])

        rot_mat, trans_mat = ru.get_grid(st_pose, agent_view.size(), dtype)
        rotated = F.grid_sample(agent_view, rot_mat, align_corners=True)
        translated = F.grid_sample(rotated, trans_mat, align_corners=True)

        # Clamp to [0, 1] after transform agent view to map coordinates
        translated = torch.clamp(translated, min=0.0, max=1.0).float()

        # update instance channels
        if self.record_instance_ids:
            translated = self._aggregate_instance_map_channels_per_category(
                translated, num_instance_channels
            )

        # Aggregate by taking the max of the previous map and current map — this is robust
        # to false negatives in one frame but makes it impossible to remove false positives
        maps = torch.cat((prev_map.unsqueeze(1), translated.unsqueeze(1)), 1)
        current_map, _ = torch.max(maps, 1)

        # Aggregate by trusting the current map — this is not robust to false negatives in
        # one frame, but it makes it possible to remove false positives
        # TODO Implement this properly for num_environments > 1
        # current_mask = translated[0, 1, :, :] > 0
        # current_map = prev_map.clone()
        # current_map[0, :, current_mask] = translated[0, :, current_mask]

        # Set people as not obstacles for planning
        # TODO Handle people more cleanly
        # TODO Implement this properly for num_environments > 1
        # people_mask = (
        #     skimage.morphology.binary_dilation(
        #         current_map[0, 5 + 11, :, :].cpu().numpy(), skimage.morphology.disk(2)
        #     )
        #     * 1.0
        # )
        # current_map[0, 0, :, :] *= 1 - torch.from_numpy(people_mask).to(device)

        if self.record_instance_ids:
            # overwrite channels containing instance IDs
            current_map[
                :,
                MC.NON_SEM_CHANNELS
                + self.num_sem_categories : MC.NON_SEM_CHANNELS
                + 2 * self.num_sem_categories,
            ] = translated[
                :,
                MC.NON_SEM_CHANNELS
                + self.num_sem_categories : MC.NON_SEM_CHANNELS
                + 2 * self.num_sem_categories,
            ]

        # Reset current location
        current_map[:, MC.CURRENT_LOCATION, :, :].fill_(0.0)
        curr_loc = current_pose[:, :2]
        curr_loc = (curr_loc * 100.0 / self.xy_resolution).int()

        for e in range(batch_size):
            x, y = curr_loc[e]
            current_map[
                e,
                MC.CURRENT_LOCATION : MC.CURRENT_LOCATION + 2,
                y - 2 : y + 3,
                x - 2 : x + 3,
            ].fill_(1.0)

            # Set a disk around the agent to explored
            # This is around the current agent - we just sort of assume we know where we are
            try:
                radius = self.explored_radius // self.resolution
                explored_disk = torch.from_numpy(skimage.morphology.disk(radius))
                current_map[
                    e,
                    MC.EXPLORED_MAP,
                    y - radius : y + radius + 1,
                    x - radius : x + radius + 1,
                ][explored_disk == 1] = 1

                # Record the region the agent has been close to using a disc centered at the agent
                radius = self.been_close_to_radius // self.resolution
                been_close_disk = torch.from_numpy(skimage.morphology.disk(radius))

                current_map[
                    e,
                    MC.BEEN_CLOSE_MAP,
                    y - radius : y + radius + 1,
                    x - radius : x + radius + 1,
                ][been_close_disk == 1] = 1

                if blacklist_target:
                    # Record the region the agent has been close to using a disc centered at the agent
                    radius = self.target_blacklisting_radius // self.resolution
                    been_close_disk = torch.from_numpy(skimage.morphology.disk(radius))

                    current_map[
                        e,
                        MC.BLACKLISTED_TARGETS_MAP,
                        y - radius : y + radius + 1,
                        x - radius : x + radius + 1,
                    ][been_close_disk == 1] = 1

            except IndexError:

                pass

        if debug_maps:
            current_map = current_map.cpu()
            explored = current_map[0, MC.EXPLORED_MAP].numpy()
            been_close = current_map[0, MC.BEEN_CLOSE_MAP].numpy()
            obstacles = current_map[0, MC.OBSTACLE_MAP].numpy()
            plt.subplot(331)
            plt.axis("off")
            plt.title("explored")
            plt.imshow(explored)
            plt.subplot(332)
            plt.axis("off")
            plt.title("been close")
            plt.imshow(been_close)
            plt.subplot(233)
            plt.axis("off")
            plt.imshow(been_close * explored)
            plt.subplot(334)
            plt.axis("off")
            plt.title("obstacles")
            plt.imshow(obstacles)
            plt.subplot(335)
            plt.axis("off")
            plt.title("obstacles_eroded")

            obs_eroded = cv2.erode(obstacles, np.ones((5, 5)), iterations=5)
            plt.imshow(obs_eroded)
            plt.subplot(336)
            plt.axis("off")
            plt.imshow(been_close * obstacles)
            plt.subplot(337)
            plt.axis("off")
            rgb = obs[0, :3, :: self.du_scale, :: self.du_scale].permute(1, 2, 0)
            plt.imshow(rgb.cpu().numpy())
            plt.subplot(338)
            plt.imshow(depth[0].cpu().numpy())
            plt.axis("off")
            plt.subplot(339)
            seg = np.zeros_like(depth[0].cpu().numpy())
            for i in range(4, obs_channels):
                seg += (i - 4) * obs[0, i].cpu().numpy()
                print("class =", i, np.sum(obs[0, i].cpu().numpy()), "pts")
            plt.imshow(seg)
            plt.axis("off")
            plt.show()

            print("Non semantic channels =", MC.NON_SEM_CHANNELS)
            print("map shape =", current_map.shape)
            breakpoint()

        if self.must_explore_close:
            current_map[:, MC.EXPLORED_MAP] = (
                current_map[:, MC.EXPLORED_MAP] * current_map[:, MC.BEEN_CLOSE_MAP]
            )
            current_map[:, MC.OBSTACLE_MAP] = (
                current_map[:, MC.OBSTACLE_MAP] * current_map[:, MC.BEEN_CLOSE_MAP]
            )
        return current_map, current_pose

    def _update_global_map_instances_for_one_channel(
        self,
        env_id: int,
        global_instances: Tensor,
        local_map: Tensor,
        x_range: tuple,
        y_range: tuple,
        max_instance_id: int,
    ) -> Tensor:
        """
        Update one instance channels in the global map from one instance channels in the local map:
        aggregate local instances with existing global instances or create new global instances.

        Args:
            global_instances (Tensor): The global map tensor.
            local_map (Tensor): The local map tensor.
            x_range (tuple): The range of indices in the x-axis for the local map in the global map.
            y_range (tuple): The range of indices in the y-axis for the local map in the global map.

        Returns:
            Tensor: The updated global instances tensor.

        """
        p = self.padding_for_instance_overlap  # default: 1
        d = self.dilation_for_instances  # default: 0

        H = global_instances.shape[0]
        W = global_instances.shape[1]

        x1, x2 = x_range
        y1, y2 = y_range

        # padding added on each side
        t_p = min(x1, p)
        b_p = min(H - x2, p)
        l_p = min(y1, p)
        r_p = min(W - y2, p)

        # the indices of the padded local_map in the global map
        x_start = x1 - t_p
        x_end = x2 + b_p
        y_start = y1 - l_p
        y_end = y2 + r_p

        local_map = torch.round(local_map)

        # pad the local map
        extended_local_map = F.pad(local_map.float(), (l_p, r_p), mode="replicate")
        extended_local_map = F.pad(
            extended_local_map.transpose(1, 0), (t_p, b_p), mode="replicate"
        ).transpose(1, 0)

        self.instance_dilation_selem = skimage.morphology.disk(d)
        # dilate the extended local map
        if d > 0:
            extended_dilated_local_map = torch.round(
                torch.tensor(
                    cv2.dilate(
                        extended_local_map.cpu().numpy(),
                        self.instance_dilation_selem,
                        iterations=1,
                    ),
                    device=local_map.device,
                    dtype=local_map.dtype,
                )
            )
        else:
            extended_dilated_local_map = torch.clone(extended_local_map)
        # Get the instances from the global map within the local map's region
        global_instances_within_local = global_instances[x_start:x_end, y_start:y_end]

        instance_mapping = self._get_local_to_global_instance_mapping(
            env_id,
            extended_dilated_local_map,
            global_instances_within_local,
            max_instance_id,
            torch.unique(extended_local_map),
        )

        # Update the global map with the associated instances from the local map
        global_instances_in_local = np.vectorize(instance_mapping.get)(
            local_map.cpu().numpy()
        )
        global_instances[x1:x2, y1:y2] = torch.maximum(
            global_instances[x1:x2, y1:y2],
            torch.tensor(
                global_instances_in_local,
                dtype=torch.int64,
                device=global_instances.device,
            ),
        )
        return global_instances

    def _get_local_to_global_instance_mapping(
        self,
        env_id: int,
        extended_local_labels: Tensor,
        global_instances_within_local: Tensor,
        max_instance_id: int,
        local_instance_ids: Tensor,
    ) -> dict:
        """
        Creates a mapping of local instance IDs to global instance IDs.

        Args:
            extended_local_labels: Labels of instances in the extended local map.
            global_instances_within_local: Instances from the global map within the local map's region.
            max_instance_id: The number of instance ids that are used up
            local_instance_ids: The local instance ids for which local to global mapping is to be determined
        Returns:
            A mapping of local instance IDs to global instance IDs.
        """
        instance_mapping = {}

        # Associate instances in the local map with corresponding instances in the global map
        for local_instance_id in local_instance_ids:
            if local_instance_id == 0:
                # ignore 0 as it does not correspond to an instance
                continue
            # pixels corresponding to
            local_instance_pixels = extended_local_labels == local_instance_id

            # Check for overlapping instances in the global map
            overlapping_instances = global_instances_within_local[local_instance_pixels]
            unique_overlapping_instances = torch.unique(overlapping_instances)

            unique_overlapping_instances = unique_overlapping_instances[
                unique_overlapping_instances != 0
            ]
            if len(unique_overlapping_instances) >= 1:
                # If there is a corresponding instance in the global map, pick the first one and associate it
                global_instance_id = int(unique_overlapping_instances[0].item())
                instance_mapping[local_instance_id.item()] = global_instance_id
            else:
                # If there are no corresponding instances, create a new instance
                global_instance_id = max_instance_id + 1
                instance_mapping[local_instance_id.item()] = global_instance_id
                max_instance_id += 1
            # update the id in instance memory
            self.instance_memory.add_view_to_instance(
                env_id, int(local_instance_id.item()), global_instance_id
            )
        instance_mapping[0.0] = 0
        return instance_mapping

    def _update_global_map_instances(
        self, e: int, global_map: Tensor, local_map: Tensor, lmb: Tensor
    ) -> Tensor:
        """
        Update instance channels in the global map from instance channels in the local map:
        aggregate local instances with existing global instances or create new global instances.

        Args:
            e (int): The index of the environment.
            global_map (Tensor): The global map tensor.
            local_map (Tensor): The local map tensor.
            lmb (Tensor): The tensor containing the ranges of indices for the local map in the global map.

        Returns:
            Tensor: The updated global map tensor.
        """
        # TODO Can we vectorize this across categories? (Only needed if speed bottleneck)
        for i in range(self.num_sem_categories):
            if (
                torch.sum(
                    local_map[e, MC.NON_SEM_CHANNELS + i + self.num_sem_categories]
                )
                > 0
            ):
                max_instance_id = (
                    torch.max(
                        global_map[
                            e,
                            MC.NON_SEM_CHANNELS
                            + self.num_sem_categories : MC.NON_SEM_CHANNELS
                            + 2 * self.num_sem_categories,
                        ]
                    )
                    .int()
                    .item()
                )
                # if the local map has any object instances, update the global map with instance ids
                instances = self._update_global_map_instances_for_one_channel(
                    e,
                    global_map[e, MC.NON_SEM_CHANNELS + i + self.num_sem_categories],
                    local_map[e, MC.NON_SEM_CHANNELS + i + self.num_sem_categories],
                    (lmb[e, 0], lmb[e, 1]),
                    (lmb[e, 2], lmb[e, 3]),
                    max_instance_id,
                )
                global_map[
                    e, i + MC.NON_SEM_CHANNELS + self.num_sem_categories
                ] = instances

        return global_map

    def _update_global_map_and_pose_for_env(
        self,
        e: int,
        local_map: Tensor,
        global_map: Tensor,
        local_pose: Tensor,
        global_pose: Tensor,
        lmb: Tensor,
        origins: Tensor,
    ):
        """Update global map and pose and re-center local map and pose for a
        particular environment.
        """

        if self.record_instance_ids and self.instance_association == "map_overlap":
            global_map = self._update_global_map_instances(
                e, global_map, local_map, lmb
            )
            global_map[
                e,
                : MC.NON_SEM_CHANNELS + self.num_sem_categories,
                lmb[e, 0] : lmb[e, 1],
                lmb[e, 2] : lmb[e, 3],
            ] = local_map[e, : MC.NON_SEM_CHANNELS + self.num_sem_categories]
            global_map[
                e,
                MC.NON_SEM_CHANNELS + 2 * self.num_sem_categories :,
                lmb[e, 0] : lmb[e, 1],
                lmb[e, 2] : lmb[e, 3],
            ] = local_map[e, MC.NON_SEM_CHANNELS + 2 * self.num_sem_categories :]
        else:
            global_map[e, :, lmb[e, 0] : lmb[e, 1], lmb[e, 2] : lmb[e, 3]] = local_map[
                e
            ]
        global_pose[e] = local_pose[e] + origins[e]
        mu.recenter_local_map_and_pose_for_env(
            e,
            local_map,
            global_map,
            local_pose,
            global_pose,
            lmb,
            origins,
            self.map_size_parameters,
        )

    def _get_map_features(self, local_map: Tensor, global_map: Tensor) -> Tensor:
        """Get global and local map features.

        Arguments:
            local_map: local map of shape
             (batch_size, MC.NON_SEM_CHANNELS + num_sem_categories, M, M)
            global_map: global map of shape
             (batch_size, MC.NON_SEM_CHANNELS + num_sem_categories, M * ds, M * ds)

        Returns:
            map_features: semantic map features of shape
             (batch_size, 2 * MC.NON_SEM_CHANNELS + num_sem_categories, M, M)
        """
        map_features_channels = 2 * MC.NON_SEM_CHANNELS + self.num_sem_categories

        if self.record_instance_ids:
            map_features_channels += self.num_sem_categories
        if self.evaluate_instance_tracking:
            map_features_channels += self.max_instances + 1

        map_features = torch.zeros(
            local_map.size(0),
            map_features_channels,
            self.local_map_size,
            self.local_map_size,
            device=local_map.device,
            dtype=local_map.dtype,
        )

        # Local obstacles, explored area, and current and past position
        map_features[:, 0 : MC.NON_SEM_CHANNELS, :, :] = local_map[
            :, 0 : MC.NON_SEM_CHANNELS, :, :
        ]
        # Global obstacles, explored area, and current and past position
        map_features[
            :, MC.NON_SEM_CHANNELS : 2 * MC.NON_SEM_CHANNELS, :, :
        ] = nn.MaxPool2d(self.global_downscaling)(
            global_map[:, 0 : MC.NON_SEM_CHANNELS, :, :]
        )
        # Local semantic categories
        map_features[:, 2 * MC.NON_SEM_CHANNELS :, :, :] = local_map[
            :, MC.NON_SEM_CHANNELS :, :, :
        ]

        if debug_maps:
            plt.subplot(131)
            plt.imshow(local_map[0, 7])  # second object = cup
            plt.subplot(132)
            plt.imshow(local_map[0, 6])  # first object = chair
            # This is the channel in MAP FEATURES mode
            plt.subplot(133)
            plt.imshow(map_features[0, 12])
            plt.show()

        return map_features.detach()




    def update_frontiers(self, current_map, global_map, local_pose ,origins,lmb,save_path="/home/zkm/home-robot/datadump_1/map_debug/frontiers.png"):
        cm = current_map[0].cpu().numpy() if current_map.ndim == 4 else current_map.cpu().numpy()
        explored = cm[MC.EXPLORED_MAP] > 0
        obstacles = cm[MC.OBSTACLE_MAP] > 0

        navigable_map = 1 - cv2.dilate(
            obstacles.astype(np.uint8),
            np.ones((3,3),np.uint8),
            iterations=1,
        ).astype(bool)

        agent_pixel_location = (local_pose.squeeze().cpu().numpy() * (100.0 / self.xy_resolution)).astype(np.int)
        new_explored_area = reveal_fog_of_war(
            top_down_map = navigable_map.astype(np.uint8),
            current_fog_of_war_mask = np.zeros_like(navigable_map, dtype=np.uint8),
            current_point = agent_pixel_location[:2],
            current_angle = -self.roll,
            fov = 42.0,
            max_line_len = 3.5 * (100.0 / self.xy_resolution) # TODO
        )
         
        # update vlfm explore map
        x1, x2, y1, y2 = lmb[0], lmb[1], lmb[2], lmb[3] # TODO:function
        gm = global_map[0].cpu().numpy() if global_map.ndim == 4 else global_map.cpu().numpy()    # ????
        global_vlfm_explore_map = (gm[MC.VLFM_EXPLORE] > 0).astype(np.uint8)
        local_vlfm_explore_map = global_vlfm_explore_map[x1:x2,y1:y2]
        local_vlfm_explore_map += new_explored_area
        local_vlfm_explore_map = local_vlfm_explore_map > 0
        local_vlfm_explore_map_channel = torch.from_numpy(local_vlfm_explore_map).to(current_map.device, current_map.dtype).unsqueeze(0)
        current_map[:,MC.VLFM_EXPLORE,:,:] = local_vlfm_explore_map_channel


        # vis = local_vlfm_explore_map * 255
        # vis = np.flipud(vis)
        # import matplotlib.pyplot as plt

        # plt.imshow(vis, cmap="gray", origin="upper")
        # plt.axis("off")
        # plt.savefig("/home/zkm/home-robot/datadump_1/debug_value_map/fog_of_war.png",
        #             bbox_inches="tight", pad_inches=0, dpi=150)
        # plt.close()

        

        explored_area = cv2.dilate(
            local_vlfm_explore_map.astype(np.uint8),
            np.ones((3, 3), np.uint8),
            iterations=1,
        )
        frontiers = detect_frontier_waypoints(
            navigable_map.astype(np.uint8),
            explored_area,
            600,
        )
        self.frontiers = frontiers
        # frontiers = np.round(frontiers).astype(int)
        # return frontiers
        # frontiers = frontiers[0]
        # frontiers = np.array(frontiers)
        


        # vis = np.full((*explored.shape, 3), 255, np.uint8)   # 白底
        # vis[explored] = (200, 200, 200)  # 浅灰：已探索
        # vis[obstacles] = (64, 64, 64)    # 深灰：障碍

        # 绿色画frontiers点
        frontiers_map = np.zeros((explored.shape))
        frontiers_map_global = np.zeros((self.global_map_size, self.global_map_size))
        pixels_per_meter = int(100 / self.xy_resolution)
        for frontier in frontiers:
            frontier = frontier.reshape(-1,2)
            for x, y in frontier:  
                # if 0 <= x < vis.shape[0] and 0 <= y < vis.shape[1]:
                frontiers_map[y][x] = 1
                frontiers_map_global[int(y+origins[1]*pixels_per_meter)][int(x+origins[0]*pixels_per_meter)] = 1
                    # cv2.circle(vis, (x, y), radius=2, color=(0, 255, 0), thickness=-1)  
        # frontiers_map = np.flipud(frontiers_map).copy()
        # frontiers_map_global = np.flipud(frontiers_map_global).copy()
        frontiers_tensor = torch.from_numpy(frontiers_map).to(
            device=current_map.device,
            dtype=current_map.dtype
        )
        frontiers_global_tensor = torch.from_numpy(frontiers_map_global).to(
            device=global_map.device,
            dtype=global_map.dtype
        )
        if current_map.shape[0] == 1:
            frontiers_tensor = frontiers_tensor.unsqueeze(0)
            frontiers_global_tensor = frontiers_global_tensor.unsqueeze(0)
        current_map[:, MC.FRONTIERS, :, :] = frontiers_tensor
        global_map[:,MC.FRONTIERS,:,:] = frontiers_global_tensor


        # cv2.imwrite(save_path, vis)
        return frontiers, current_map, global_map

    def save_current_map_vis(
        self,
        current_map,                 # [C,H,W] 或 [1,C,H,W]，torch/np都行
        step:int,
        save_dir:str="/home/zkm/home-robot/datadump_1/map_debug",
        map_resolution_cm:int=None,  # 若用 local_pose 必须传（通常是 self.resolution）
        local_pose=None,             # 可选: [x_m, y_m, theta]（theta弧度或角度都可，见下）
        theta_in_radians:bool=True,  # local_pose[2] 是否是弧度
        flip_vertical:bool=True,
        agent_radius_px:int=3,
        arrow_len_px:int=20,
        filename_prefix:str="local",
        # === 新增参数 ===
        hfov_deg: float = 42.0,      # 水平视场角（度）；给了就画扇形
        fov_range_m: float = 3.5,   # 扇形半径（米），优先级高于 fov_range_px
        fov_range_px: int = None,    # 扇形半径（像素），两者都没给时自动估一个
        fov_alpha: float = 0.28      # 扇形透明度
    ):
        import os, cv2, numpy as np
        from home_robot.mapping.semantic.constants import MapConstants as MC
        import torch

        # 统一成 numpy [C,H,W]
        if isinstance(current_map, torch.Tensor):
            cm = current_map.detach().cpu().float().numpy()
        else:
            cm = np.asarray(current_map, dtype=np.float32)
        if cm.ndim == 4:
            cm = cm[0]
        assert cm.ndim == 3, f"current_map should be [C,H,W], got {cm.shape}"

        obst = cm[MC.OBSTACLE_MAP] > 0
        expl = cm[MC.EXPLORED_MAP] > 0
        frontiers = cm[MC.FRONTIERS] > 0
        cur_loc_mask = cm[MC.CURRENT_LOCATION] > 0
        value_map = cm[MC.VALUE_MAP]
        H, W = obst.shape

        if flip_vertical:
            obst = np.flipud(obst); expl = np.flipud(expl); cur_loc_mask = np.flipud(cur_loc_mask); frontiers = np.flipud(frontiers)
            value_map = np.flipud(value_map)

        COL_BG   = (255,255,255)
        COL_OBS  = (64,64,64)
        COL_EXP  = (200,200,200)
        COL_AGT  = (0,0,255)
        COL_FOV  = (255, 200, 120)   # 浅蓝（BGR，带点青色），看着不刺眼
        COL_FRONTIERS = (255, 0, 0)

        left  = np.full((H,W,3), COL_BG,  dtype=np.uint8)
        # right = np.full((H,W,3), COL_BG,  dtype=np.uint8)
        
        left[expl] = COL_EXP
        left[obst]  = COL_OBS
        frontier_vis_kernel = np.ones((5,5), np.uint8)
        fmask = cv2.dilate(frontiers.astype(np.uint8)*255, frontier_vis_kernel, iterations=1).astype(bool)


        # === 计算像素位置与朝向 ===
        cx = cy = None
        yaw_img = None

        if local_pose is not None:
            assert map_resolution_cm is not None, "map_resolution_cm 必须提供（像素换算要用）"
            x_m, y_m, th = float(local_pose[0]), float(local_pose[1]), float(local_pose[2])
            cx = int(round(x_m * 100.0 / map_resolution_cm))
            cy = self.local_map_size - int(round(y_m * 100.0 / map_resolution_cm)) ###########换成local需要改，目前可视化global

            yaw = th if theta_in_radians else np.deg2rad(th)
            # 图像坐标：x 右、y 下；使用 ey = cy - R*sin(yaw) 的惯例
            yaw_img = yaw
        else:
            ys, xs = np.where(cur_loc_mask)
            if len(xs) > 0:
                cx = int(round(xs.mean()))
                cy = int(round(ys.mean()))
            # 没有 local_pose 就不画箭头/FOV

        # 画机器人
        if cx is not None and 0 <= cx < W and 0 <= cy < H:
            cv2.circle(left,  (cx, cy), agent_radius_px, COL_AGT, -1, lineType=cv2.LINE_AA)
            # cv2.circle(right, (cx, cy), agent_radius_px, COL_AGT, -1, lineType=cv2.LINE_AA)

            if yaw_img is not None:
                ex = int(round(cx + arrow_len_px * np.cos(yaw_img)))
                ey = int(round(cy - arrow_len_px * np.sin(yaw_img)))  # 注意 y 反号
                ex = np.clip(ex, 0, W-1); ey = np.clip(ey, 0, H-1)
                cv2.arrowedLine(left,  (cx, cy), (ex, ey), COL_AGT, 2, tipLength=0.25, line_type=cv2.LINE_AA)
                # cv2.arrowedLine(right, (cx, cy), (ex, ey), COL_AGT, 2, tipLength=0.25, line_type=cv2.LINE_AA)

            # === 画 HFOV 扇形（浅蓝，半透明） ===
            if hfov_deg is not None:
                # 半径像素
                if fov_range_px is not None:
                    R = int(fov_range_px)
                elif fov_range_m is not None:
                    R = int(round(fov_range_m * 100.0 / map_resolution_cm))
                else:
                    R = max(arrow_len_px * 4, min(H, W) // 6)  # 合理默认

                half = np.deg2rad(hfov_deg) / 2.0
                a1, a2 = yaw_img - half, yaw_img + half

                def draw_fov(wimg):
                    overlay = wimg.copy()
                    # 构造扇形多边形：中心 + 圆弧采样点
                    num = 64
                    angs = np.linspace(a1, a2, num)
                    pts = [(cx, cy)]
                    for t in angs:
                        px = int(round(cx + R * np.cos(t)))
                        py = int(round(cy - R * np.sin(t)))  # y 反号
                        px = np.clip(px, 0, W-1); py = np.clip(py, 0, H-1)
                        pts.append((px, py))
                    pts = np.array(pts, dtype=np.int32)
                    cv2.fillPoly(overlay, [pts], COL_FOV, lineType=cv2.LINE_AA)
                    cv2.addWeighted(overlay, fov_alpha, wimg, 1.0 - fov_alpha, 0, dst=wimg)

                draw_fov(left)
                # draw_fov(right)

        # draw frontier points
        left[fmask] = COL_FRONTIERS

        right = np.clip(value_map, 0.0, 0.6) / 0.6 * 255.0
        right = right.astype(np.uint8)

        right = cv2.applyColorMap(right, cv2.COLORMAP_INFERNO)
        right[fmask] = COL_FRONTIERS


        vis = np.concatenate([left, right], axis=1)
        os.makedirs(save_dir, exist_ok=True)
        out_path = os.path.join(save_dir, f"{filename_prefix}_map_{step:05d}.png")
        cv2.imwrite(out_path, vis)
        return out_path, vis

    
    def _get_blank_cone_mask(self, fov: float, max_depth: float) -> np.ndarray:
        """Generate a FOV cone without any obstacles considered"""
        pixels_per_meter = int(100 / self.xy_resolution)
        size = int(max_depth * pixels_per_meter)
        cone_mask = np.zeros((size * 2 + 1, size * 2 + 1))
        cone_mask = cv2.ellipse(  # type: ignore
            cone_mask,
            (size, size),  # center_pixel
            (size, size),  # axes lengths
            0,  # angle circle is rotated
            -np.rad2deg(fov) / 2 + 90,  # start_angle
            np.rad2deg(fov) / 2 + 90,  # end_angle
            1,  # color
            -1,  # thickness
        )
        return cone_mask

    def _get_confidence_mask(
        self, 
        fov: float, 
        max_depth: float = 3.5
    ) -> np.ndarray:
        """Generate a FOV cone with central values weighted more heavily"""
        if (fov, max_depth) in self._confidence_masks:
            return self._confidence_masks[(fov, max_depth)].copy()
        cone_mask = self._get_blank_cone_mask(fov, max_depth)
        adjusted_mask = np.zeros_like(cone_mask).astype(np.float32)
        for row in range(adjusted_mask.shape[0]):
            for col in range(adjusted_mask.shape[1]):
                horizontal = abs(row - adjusted_mask.shape[0] // 2)
                vertical = abs(col - adjusted_mask.shape[1] // 2)
                angle = np.arctan2(vertical, horizontal)
                angle = remap(angle, 0, fov / 2, 0, np.pi / 2)
                confidence = np.cos(angle) ** 2
                confidence = remap(confidence, 0, 1, self._min_confidence, 1)
                adjusted_mask[row, col] = confidence
        adjusted_mask = adjusted_mask * cone_mask
        self._confidence_masks[(fov, max_depth)] = adjusted_mask.copy()

        return adjusted_mask

    def _process_current_data(self, depth: np.ndarray, fov: float, min_depth: float, max_depth: float, global_pose) -> np.ndarray:
        """Using the FOV and depth, return the visible portion of the FOV.

        Args:
            depth: The depth image to use for determining the visible portion of the
                FOV.
        Returns:
            A mask of the visible portion of the FOV.
        """
        # Squeeze out the channel dimension if depth is a 3D array
        if depth.shape[0] == 1:
            depth = depth.squeeze(0)
        # Squash depth image into one row with the max depth value for each column
        depth_row = np.max(depth, axis=0)

        # Create a linspace of the same length as the depth row from -fov/2 to fov/2
        angles = np.linspace(-fov / 2, fov / 2, len(depth_row))

        # Assign each value in the row with an x, y coordinate depending on 'angles'
        # and the max depth value for that column
        x = depth_row
        y = depth_row * np.tan(angles)

        # Get blank cone mask
        cone_mask = self._get_confidence_mask(fov, max_depth)

        # Convert the x, y coordinates to pixel coordinates
        pixels_per_meter = int(100 / self.xy_resolution)
        x = (x * pixels_per_meter + cone_mask.shape[0] / 2).astype(int)
        y = (y * pixels_per_meter + cone_mask.shape[1] / 2).astype(int)

        # Create a contour from the x, y coordinates, with the top left and right
        # corners of the image as the first two points
        last_row = cone_mask.shape[0] - 1
        last_col = cone_mask.shape[1] - 1
        start = np.array([[0, last_col]])
        end = np.array([[last_row, last_col]])
        contour = np.concatenate((start, np.stack((y, x), axis=1), end), axis=0)

        # Draw the contour onto the cone mask, in filled-in black
        visible_mask = cv2.drawContours(cone_mask, [contour], -1, 0, -1)  # type: ignore

        h, w = visible_mask.shape[:2]
        cx, cy = w / 2.0, h / 2.0
        theta = global_pose.squeeze(0).cpu().numpy()[2]
        theta += 90
        visible_mask_rot_matrix = cv2.getRotationMatrix2D(center=(cx, cy), angle=theta, scale=1.0)
        visible_mask_rotated = cv2.warpAffine(
            visible_mask, visible_mask_rot_matrix, (w, h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0
        )

        x = int(round(global_pose.squeeze(0).cpu().numpy()[0] * 100.0 / self.xy_resolution))
        y = self.global_map_size - int(round(global_pose.squeeze(0).cpu().numpy()[1] * 100.0 / self.xy_resolution))
        cx, cy = int(w // 2), int(h // 2)
        x1 = x - cx
        y1 = y - cy
        x2 = x1 + w
        y2 = y1 + h

        global_visible_mask = np.zeros((self.global_map_size, self.global_map_size))
        xs1, ys1 = max(0, x1), max(0, y1)
        xs2, ys2 = min(self.global_map_size, x2), min(self.global_map_size, y2)
        rx1, ry1 = xs1 - x1, ys1 - y1
        rx2, ry2 = rx1 + (xs2 - xs1), ry1 + (ys2 - ys1)

        global_visible_mask[ys1:ys2, xs1:xs2] = np.maximum(
            global_visible_mask[ys1:ys2, xs1:xs2],
            visible_mask_rotated[ry1:ry2, rx1:rx2]
        )


        # import matplotlib
        # matplotlib.use("Agg") 
        # import matplotlib.pyplot as plt
        # import os
        # os.makedirs("/home/zkm/home-robot/datadump_1/debug_value_map/", exist_ok=True)
        # fig = plt.figure(figsize=(6, 6), dpi=150)
        # ax = plt.axes([0, 0, 1, 1])
        # im = ax.imshow(global_visible_mask, cmap="inferno", vmin=0.0, vmax=1.0, origin="upper")
        # ax.set_axis_off()
        # cax = fig.add_axes([0.86, 0.1, 0.03, 0.8])
        # fig.colorbar(im, cax=cax)
        # plt.savefig("/home/zkm/home-robot/datadump_1/debug_value_map/visible_mask_global_map.png", bbox_inches="tight", pad_inches=0, dpi=150)
        # plt.close(fig)

        return global_visible_mask


    def _fuse_value_map(self, new_map: np.ndarray, values: np.ndarray):
        decision_threshold = 0.35
        value_channels = 1
        new_map_mask = np.logical_and(new_map < decision_threshold, new_map < self._map)
        new_map[new_map_mask] = 0 

        confidence_denominator = self._map + new_map
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            weight_1 = self._map / confidence_denominator
            weight_2 = new_map / confidence_denominator

        weight_1_channeled = np.repeat(np.expand_dims(weight_1, axis=2), value_channels, axis=2)
        weight_2_channeled = np.repeat(np.expand_dims(weight_2, axis=2), value_channels, axis=2)

        self._value_map = self._value_map * weight_1_channeled + values * weight_2_channeled
        self._map = self._map * weight_1 + new_map * weight_2

        # Because confidence_denominator can have 0 values, any nans in either the
        # value or confidence maps will be replaced with 0
        self._value_map = np.nan_to_num(self._value_map)
        self._map = np.nan_to_num(self._map)     


    def _fuse_value_map_recp(self, new_map: np.ndarray, values: np.ndarray):
        decision_threshold = 0.35
        value_channels = 1
        new_map_mask = np.logical_and(new_map < decision_threshold, new_map < self._map_recp)
        new_map[new_map_mask] = 0 

        confidence_denominator = self._map_recp + new_map
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            weight_1 = self._map_recp / confidence_denominator
            weight_2 = new_map / confidence_denominator

        weight_1_channeled = np.repeat(np.expand_dims(weight_1, axis=2), value_channels, axis=2)
        weight_2_channeled = np.repeat(np.expand_dims(weight_2, axis=2), value_channels, axis=2)

        self._value_map_recp = self._value_map_recp * weight_1_channeled + values * weight_2_channeled
        self._map_recp = self._map_recp * weight_1 + new_map * weight_2

        # Because confidence_denominator can have 0 values, any nans in either the
        # value or confidence maps will be replaced with 0
        self._value_map_recp = np.nan_to_num(self._value_map_recp)
        self._map_recp = np.nan_to_num(self._map_recp)     




def remap(value: float, from_low: float, from_high: float, to_low: float, to_high: float) -> float:
    """Maps a value from one range to another.

    Args:
        value (float): The value to be mapped.
        from_low (float): The lower bound of the input range.
        from_high (float): The upper bound of the input range.
        to_low (float): The lower bound of the output range.
        to_high (float): The upper bound of the output range.

    Returns:
        float: The mapped value.
    """
    return (value - from_low) * (to_high - to_low) / (from_high - from_low) + to_low


# def value_save(img, path):
#     import matplotlib
#     matplotlib.use("Agg") 
#     import matplotlib.pyplot as plt
#     import os
#     os.makedirs("/home/zkm/home-robot/datadump_1/debug_value_map/", exist_ok=True)
#     fig = plt.figure(figsize=(6, 6), dpi=150)
#     ax = plt.axes([0, 0, 1, 1])
#     im = ax.imshow(img, cmap="inferno", vmin=0.0, vmax=0.6, origin="upper")
#     ax.set_axis_off()
#     cax = fig.add_axes([0.86, 0.1, 0.03, 0.8])
#     fig.colorbar(im, cax=cax)
#     # plt.savefig("/home/zkm/home-robot/datadump_1/debug_value_map/visible_mask_global_map.png", bbox_inches="tight", pad_inches=0, dpi=150)
#     plt.savefig(path, bbox_inches="tight", pad_inches=0, dpi=150)

#     plt.close(fig)
import os
def value_save_cv2(img, path):
    os.makedirs("/home/zkm/home-robot/datadump_1/debug_value_map/", exist_ok=True)

    # 归一化到 0~255
    img_norm = np.clip(img, 0.0, 0.6) / 0.6 * 255.0
    img_norm = img_norm.astype(np.uint8)

    # 应用 colormap (cv2 支持 inferno 对应的 COLORMAP_INFERNO)
    img_color = cv2.applyColorMap(img_norm, cv2.COLORMAP_INFERNO)

    # 保存
    cv2.imwrite(path, img_color)