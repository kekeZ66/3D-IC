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
from scipy.ndimage import distance_transform_edt
from home_robot.utils.nav_utils import save_pc_base_to_ply



from frontier_exploration.frontier_detection import detect_frontier_waypoints
from frontier_exploration.utils.fog_of_war import reveal_fog_of_war

import os, time, psutil
from kornia.contrib import connected_components

NCPU = os.cpu_count() or 1
PROC = psutil.Process()

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

        self.local_frontier_map = np.zeros((self.local_map_size, self.local_map_size))
        self.global_frontier_map = np.zeros((self.global_map_size, self.global_map_size))
        self.local_vlfm_explore_map = np.zeros((self.local_map_size, self.local_map_size))
        self.global_vlfm_explore_map = np.zeros((self.global_map_size, self.global_map_size))

        self.centers = None
        self.choose_point_flag = False
        self.idx_map = None

        self.enable_world_pcd = False          
        self.world_pcd_stride = 10           
        self.world_pcd_voxel_m = 0.05        
        self.world_pcd_max_points = 2_000_000 
        self._world_pcd_step = 0          
        self.world_pcd_xyz = np.zeros((0, 3), dtype=np.float32)
        self.world_pcd_rgb = np.zeros((0, 3), dtype=np.uint8)
    
    def reset(self):
        self.local_frontier_map = np.zeros((self.local_map_size, self.local_map_size))
        self.global_frontier_map = np.zeros((self.global_map_size, self.global_map_size))
        self.local_vlfm_explore_map = np.zeros((self.local_map_size, self.local_map_size))
        self.global_vlfm_explore_map = np.zeros((self.global_map_size, self.global_map_size))
        self.centers = None
        self.choose_point_flag = False
        self.idx_map = None

        # reset world pcd
        self._world_pcd_step = 0
        self.world_pcd_xyz = np.zeros((0, 3), dtype=np.float32)
        self.world_pcd_rgb = np.zeros((0, 3), dtype=np.uint8)


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
            
            
            lmb_np = lmb.squeeze(0).cpu().numpy()
            depth = seq_obs[:,t][:, 3, :, :].float() / 100.0
            visible_mask = self._process_current_data(depth.cpu().numpy(),np.deg2rad(self.hfov),0.5,5.0,global_pose)
            local_map_frontiers = self.update_frontiers(local_map,global_map,local_pose,origins.squeeze(0).cpu().numpy(),lmb_np)
            visible_mask = np.flipud(visible_mask)
            if values[0] > 0:
                self._fuse_value_map(visible_mask, np.array(values[0]))
            if values[1] > 0:
                self._fuse_value_map_recp(visible_mask, np.array(values[1]))

            lmb_np = lmb.squeeze(0).cpu().numpy()
            x1, x2, y1, y2 = lmb_np[0], lmb_np[1], lmb_np[2], lmb_np[3] 
            local_value_map = self._value_map.reshape((960,960))[x1:x2,y1:y2]
            self._value_map_local = local_value_map

            
            local_value_map = self._value_map_recp.reshape((960,960))[x1:x2,y1:y2]
            self._value_map_recp_local = local_value_map
 
            local_map[0,MC.FALSE_RECEP_MAP,:,:] = filter_end_recep_with_height(local_map[0,MC.NON_SEM_CHANNELS + 3,:,:], local_map[0,MC.HEIGHT_MAP,:,:], torch.from_numpy(self.local_vlfm_explore_map).to(device=local_map.device,dtype = torch.uint8))
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
        if camera_pose is not None: 
            angles = torch.Tensor(
                [tra.euler_from_matrix(p[:3, :3].cpu(), "rzyx") for p in camera_pose]
            )


            tilt = angles[:, 1]
            yaw = angles[:, -1]
            roll = angles[:, 0]
            self.roll = roll.cpu().numpy()[0]
            camera_x = camera_pose[:, 0, 3] * -100
            camera_y = camera_pose[:, 1, 3] * -100

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


        point_cloud_map_coords = du.transform_pose_t(
            point_cloud_base_coords, self.shift_loc, device
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


        flag = True
        if getattr(self, "enable_world_pcd", False):
            self._world_pcd_step += 1
            if flag:
                point_cloud_full_t = du.get_point_cloud_from_z_t(
                    depth, self.camera_matrix, device
                )  

                point_cloud_full_base = du.transform_camera_view_t(
                    point_cloud_full_t,
                    agent_height,
                    torch.rad2deg(tilt).cpu().numpy(),
                    device,
                )  

                global_pose_cm = np.array([camera_x.item(), camera_y.item(), roll.item()], dtype=np.float32)
                point_cloud_full_world_cm = du.transform_pose_t(point_cloud_full_base, global_pose_cm, device)  
                pc_world_m = (point_cloud_full_world_cm[0] / 100.0)

                rgb_img = obs[0, :3, :, :].permute(1, 2, 0).contiguous().to(torch.uint8)  

                d = depth[0] 
                valid = (d > 0) & (d < self.max_depth) 

                xyz = pc_world_m[valid].reshape(-1, 3).detach().cpu().numpy().astype(np.float32)
                col = rgb_img[valid].reshape(-1, 3).detach().cpu().numpy().astype(np.uint8)

                if xyz.shape[0] > 0:
                    self.world_pcd_xyz = np.concatenate([self.world_pcd_xyz, xyz], axis=0)
                    self.world_pcd_rgb = np.concatenate([self.world_pcd_rgb, col], axis=0)

                    vox = getattr(self, "world_pcd_voxel_m", 0.02)
                    self.world_pcd_xyz, self.world_pcd_rgb = self._voxel_downsample_numpy(
                        self.world_pcd_xyz, self.world_pcd_rgb, vox
                    )

                    max_n = getattr(self, "world_pcd_max_points", 2_000_000)


        if self.choose_point_flag:
            point_cloud_full_t = du.get_point_cloud_from_z_t(depth, self.camera_matrix, device)
            point_cloud_full_base = du.transform_camera_view_t(point_cloud_full_t, agent_height, torch.rad2deg(tilt).cpu().numpy(), device)
            idx_map = self.pixel2map_indices_current(point_cloud_full_base, current_pose)
            self.idx_map = idx_map
            debug_img = np.zeros(
                (self.local_map_size_cm // self.xy_resolution,
                self.local_map_size_cm // self.xy_resolution),
                dtype=np.uint8,
            )

            yy = idx_map[..., 0].cpu().numpy()
            xx = idx_map[..., 1].cpu().numpy()
            valid = (xx >= 0) & (yy >= 0)
            h_mm = idx_map[..., 2].cpu().numpy()
            h_min, h_max = 0.0, 1500.0
            h8 = np.clip((h_mm - h_min) / (h_max - h_min) * 255.0, 0, 255).astype(np.uint8)
            debug_img[xx[valid], yy[valid]] = h8[valid]
            debug_img = 255 - debug_img
            debug_img = np.flipud(debug_img)

            # debug map to rgb
            mask = (depth >= 250) & (depth < 255)
            mask2d = mask[0] > 0
            mask2d = mask[0]  
            pc = point_cloud_full_base[0]
            z = pc[..., 2]
            mask_valid = (z < 1.0) & mask2d

            cc_mask = mask_valid.detach().cpu().numpy().astype(np.uint8)
            num_labels, labels = cv2.connectedComponents(cc_mask, connectivity=8)
            centers = []
            for lab in range(1, num_labels):   
                region = (labels == lab)
                if region.sum() < 300:
                    continue
                print(region.sum())

                ys, xs = np.where(region)
                sort_idx = np.argsort(xs)
                ys = ys[sort_idx]
                xs = xs[sort_idx]
                idx = np.linspace(int(0.15 * (len(ys) - 1)), int(0.85 * (len(ys) - 1)), (region.sum() // 1000) + 1, dtype=int)
                
                for k in idx:
                    i_pt, j_pt = int(ys[k]), int(xs[k])
                    centers.append((i_pt, j_pt))
            self.centers = centers

            rgb = obs[0, :3, :, :]
            rgb_np = rgb.cpu().numpy().transpose(1, 2, 0).astype(np.uint8) 
            for (i, j) in centers:
                cv2.circle(rgb_np, (j, i), radius=3, color=(0, 0, 255), thickness=-1)      
            rgb_bgr = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR)
 

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
                    .float(),  
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


        B, C, X, Y, Z = voxels.shape
        device = voxels.device
        occ_voxels = voxels[:, 0:1, :, :, :] 
        z_indices = torch.arange(
            self.min_voxel_height,
            self.max_voxel_height,
            device=device,
            dtype=torch.float32,
        ).view(1, 1, 1, 1, -1)                  
        z_cm = z_indices * self.z_resolution    

        occ_slice = occ_voxels[..., self.min_mapped_height:self.max_mapped_height]  
        z_cm_slice = z_cm[..., self.min_mapped_height:self.max_mapped_height]      

        occ_mask = (occ_slice > 0.1).float()
        height_candidates = occ_mask * z_cm_slice   

        height_cm, _ = height_candidates.max(dim=-1)   

        HEIGHT_MAX_CM = 150.0
        height_norm = torch.clamp(height_cm / HEIGHT_MAX_CM, 0.0, 1.0)


        agent_height_proj = voxels[
            ..., self.min_mapped_height : self.max_mapped_height
        ].sum(4)
        all_height_proj = voxels.sum(4)

        fp_map_pred = agent_height_proj[:, 0:1, :, :]


        fp_map_pred = fp_map_pred / self.map_pred_threshold
        if self.exploration_type == "default":
            fp_exp_pred = all_height_proj[:, 0:1, :, :]
            fp_exp_pred = fp_exp_pred / self.exp_pred_threshold
        elif self.exploration_type == "hull":
            fp_exp_pred = all_height_proj[:, 0:1, :, :]
            fp_exp_pred = fp_exp_pred / self.exp_pred_threshold
            fp_exp_pred = fp_exp_pred.clip(0, 1)
            fp_exp_pred[:, :, 0, fp_exp_pred.shape[-1] // 2] = 1

            filled = fill_convex_hull(fp_exp_pred[0, 0].cpu())
            assert fp_exp_pred.shape[:2] == (1, 1)
            fp_exp_pred[0, 0] = torch.tensor(filled)
        elif self.exploration_type == "gaze":
            fp_exp_pred = torch.zeros_like(fp_map_pred)
            view_image = torch.zeros(fp_map_pred.shape[-2:])
            dist = self.gaze_distance * 100 / self.resolution
            view_image = draw_circle_segment(
                view_image, (0, fp_exp_pred.shape[-1] // 2), dist, 0, self.gaze_width
            )
            fp_exp_pred[..., :, :] = view_image
        elif self.exploration_type == "gaze_projected":
            fp_exp_pred = all_height_proj[:, 0:1, :, :]
            fp_exp_pred = fp_exp_pred / self.exp_pred_threshold
            view_image = torch.zeros(fp_map_pred.shape[-2:])
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

        if self.dilate_obstacles:
            for i in range(fp_map_pred.shape[0]):
                env_map = fp_map_pred[i, 0].cpu().numpy()               
                median_filtered = cv2.medianBlur(env_map, self.dilate_size)
                fp_map_pred[i, 0] = torch.tensor(median_filtered)

        x1 = self.local_map_size_cm // (self.xy_resolution * 2) - self.vision_range // 2
        x2 = x1 + self.vision_range
        y1 = self.local_map_size_cm // (self.xy_resolution * 2)
        y2 = y1 + self.vision_range
        agent_view[:, MC.OBSTACLE_MAP : MC.OBSTACLE_MAP + 1, y1:y2, x1:x2] = fp_map_pred
        agent_view[:, MC.EXPLORED_MAP : MC.EXPLORED_MAP + 1, y1:y2, x1:x2] = fp_exp_pred
        agent_view[:, MC.HEIGHT_MAP : MC.HEIGHT_MAP+1, y1:y2, x1:x2] = height_norm


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

        translated = torch.clamp(translated, min=0.0, max=1.0).float()

        if self.record_instance_ids:
            translated = self._aggregate_instance_map_channels_per_category(
                translated, num_instance_channels
            )


        maps = torch.cat((prev_map.unsqueeze(1), translated.unsqueeze(1)), 1)
        current_map, _ = torch.max(maps, 1)


        if self.record_instance_ids:
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
        explored = current_map[:,MC.EXPLORED_MAP,:,:].squeeze().to(torch.bool).cpu().numpy()
        obstacles = current_map[:,MC.OBSTACLE_MAP,:,:].squeeze().to(torch.bool).cpu().numpy()
        navigable_map = 1 - cv2.dilate(
            obstacles.astype(np.uint8),
            np.ones((3,3),np.uint8),
            iterations=1,
        ).astype(bool)

        agent_pixel_location = (local_pose.squeeze().cpu().numpy() * (100.0 / self.xy_resolution)).astype(np.int)
        tmp = agent_pixel_location[0]
        agent_pixel_location[0] = agent_pixel_location[1]
        agent_pixel_location[1] = tmp
        new_explored_area = reveal_fog_of_war(
            top_down_map = navigable_map.astype(np.uint8),
            current_fog_of_war_mask = np.zeros_like(navigable_map, dtype=np.uint8),
            current_point = agent_pixel_location[:2],
            current_angle = -self.roll,
            fov = 42.0,
            max_line_len = 3.5 * (100.0 / self.xy_resolution) 
        )
        
        x1, x2, y1, y2 = lmb[0], lmb[1], lmb[2], lmb[3] 
        global_vlfm_explore_map = self.global_vlfm_explore_map
        local_vlfm_explore_map = global_vlfm_explore_map[x1:x2,y1:y2]
        local_vlfm_explore_map += new_explored_area # update
        local_vlfm_explore_map = local_vlfm_explore_map > 0
        self.local_vlfm_explore_map = local_vlfm_explore_map
        self.global_vlfm_explore_map[x1:x2, y1:y2] = local_vlfm_explore_map


        

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

        frontiers_map = np.zeros((explored.shape))
        frontiers_map_global = np.zeros((self.global_map_size, self.global_map_size))
        pixels_per_meter = int(100 / self.xy_resolution)
        for frontier in frontiers:
            frontier = frontier.reshape(-1,2)
            for x, y in frontier:  
                frontiers_map[y][x] = 1
                frontiers_map_global[int(y+origins[1]*pixels_per_meter)][int(x+origins[0]*pixels_per_meter)] = 1
                  
        self.local_frontier_map = frontiers_map
        self.global_frontier_map = frontiers_map_global
 
        return frontiers

    
    def _get_blank_cone_mask(self, fov: float, max_depth: float) -> np.ndarray:
        """Generate a FOV cone without any obstacles considered"""
        pixels_per_meter = int(100 / self.xy_resolution)
        size = int(max_depth * pixels_per_meter)
        cone_mask = np.zeros((size * 2 + 1, size * 2 + 1))
        cone_mask = cv2.ellipse(  
            cone_mask,
            (size, size),
            (size, size),  
            0, 
            -np.rad2deg(fov) / 2 + 90, 
            np.rad2deg(fov) / 2 + 90, 
            1,  
            -1,  
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

        if ys2 > ys1 and xs2 > xs1:
            global_visible_mask[ys1:ys2, xs1:xs2] = np.maximum(
                global_visible_mask[ys1:ys2, xs1:xs2],
                visible_mask_rotated[ry1:ry2, rx1:rx2]
            )


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


        self._value_map_recp = np.nan_to_num(self._value_map_recp)
        self._map_recp = np.nan_to_num(self._map_recp)     


    def pixel2map_indices_current(
        self,
        point_cloud_full_base: torch.Tensor, 
        current_pose: torch.Tensor,        
    ) -> torch.Tensor:

        device = point_cloud_full_base.device
        B, H_img, W_img, _ = point_cloud_full_base.shape
        assert B == 1

        pc_xy_m = point_cloud_full_base[0, ..., :3] / 100.0 

        pose_xy = current_pose[0, :2]         
        theta = current_pose[0, 2] * torch.pi / 180.0  

        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)

        bx = pc_xy_m[..., 0]
        by = pc_xy_m[..., 1]

        wx = cos_t * bx - sin_t * by + pose_xy[0]
        wy = sin_t * bx + cos_t * by + pose_xy[1]

        x_idx = torch.round(wx * 100.0 / self.xy_resolution)
        x_idx = 480 - x_idx
        y_idx = torch.round(wy * 100.0 / self.xy_resolution)

        H_map = self.local_map_size_cm // self.xy_resolution
        W_map = self.local_map_size_cm // self.xy_resolution

        x_idx = x_idx.long().clamp(0, W_map - 1)
        y_idx = y_idx.long().clamp(0, H_map - 1)

        h_mm = pc_xy_m[:,:,2]
        h_mm = (h_mm * 1000).long()

        idx_map = torch.stack([y_idx, x_idx, h_mm], dim=-1)  
        return idx_map



    def _base_cm_to_world_m(
        self,
        pc_base_cm: torch.Tensor,   
        world_pose_m_deg: torch.Tensor,  
    ) -> torch.Tensor:
        """
        return: [H,W,3] world xyz in meters
        """
        if pc_base_cm.dim() == 4:
            pc_base_cm = pc_base_cm[0]
        if world_pose_m_deg.dim() == 2:
            world_pose_m_deg = world_pose_m_deg[0]

        pc_m = pc_base_cm / 100.0 

        bx = pc_m[..., 0]
        by = pc_m[..., 1]
        bz = pc_m[..., 2]

        x = world_pose_m_deg[0]
        y = world_pose_m_deg[1]
        yaw = world_pose_m_deg[2] * torch.pi / 180.0

        cos_t = torch.cos(yaw)
        sin_t = torch.sin(yaw)

        wx = cos_t * bx - sin_t * by + x
        wy = sin_t * bx + cos_t * by + y
        wz = bz

        return torch.stack([wx, wy, wz], dim=-1)
    
    def _voxel_downsample_numpy(self, xyz: np.ndarray, rgb: np.ndarray, voxel: float):
        """
        xyz: (N,3) float32 meters
        rgb: (N,3) uint8
        return: downsampled xyz,rgb
        """
        if xyz.shape[0] == 0:
            return xyz, rgb
        q = np.floor(xyz / voxel).astype(np.int32)
        _, keep = np.unique(q, axis=0, return_index=True)
        return xyz[keep], rgb[keep]




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






def filter_end_recep_with_height(end_recep_map: torch.Tensor,
                                 height_map: torch.Tensor,
                                 explored_map: torch.Tensor) -> torch.Tensor:

    assert end_recep_map.shape == height_map.shape

    device = end_recep_map.device
    H, W = end_recep_map.shape

    base_mask = end_recep_map > 0.5 

    h = height_map
    cond1 = (h >= 0.1) & (h <= 0.47)

    h4 = h.unsqueeze(0).unsqueeze(0)  

    h7_max = F.max_pool2d(h4, kernel_size=3, stride=1, padding=3 // 2) 
    h7_min = -F.max_pool2d(-h4, kernel_size=3, stride=1, padding=3 // 2)
    h7_max = h7_max.squeeze(0).squeeze(0)  
    h7_min = h7_min.squeeze(0).squeeze(0)
    cond2 = (h7_max - h7_min) <= 0.1

    h21_min = -F.max_pool2d(-h4, kernel_size=11, stride=1, padding=11 // 2)
    h21_min = h21_min.squeeze(0).squeeze(0)
    cond3 = h21_min < 0.1

    e = explored_map.to(dtype=torch.float32).unsqueeze(0).unsqueeze(0) 
    e21_max = F.max_pool2d(e, kernel_size=11, stride=1, padding=11 // 2)
    e21_max = e21_max.squeeze(0).squeeze(0) 
    cond4 = e21_max > 0.5

    final_mask = base_mask & cond1 & cond2 & cond3 & cond4

    filtered_map = final_mask.to(end_recep_map.dtype)

    return filtered_map


