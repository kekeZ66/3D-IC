# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np
import scipy
import skimage.morphology
import torch
import torch.nn as nn
from sklearn.cluster import DBSCAN
import torch.nn.functional as F

from home_robot.mapping.semantic.constants import MapConstants as MC
from home_robot.utils.morphology import binary_dilation, binary_erosion


class ObjectNavFrontierExplorationPolicy(nn.Module):
    """
    Policy to select high-level goals for Object Goal Navigation:
    go to object goal if it is mapped and explore frontier (closest
    unexplored region) otherwise.
    """

    def __init__(
        self,
        exploration_strategy: str,
        num_sem_categories: int,
        explored_area_dilation_radius=10,
    ):
        super().__init__()
        assert exploration_strategy in ["seen_frontier", "been_close_to_frontier"] # seen_frontier in heuristic baseline
        self.exploration_strategy = exploration_strategy

        self.dilate_explored_kernel = nn.Parameter(
            torch.from_numpy(skimage.morphology.disk(explored_area_dilation_radius))
            .unsqueeze(0)
            .unsqueeze(0)
            .float(),
            requires_grad=False,
        )
        self.select_border_kernel = nn.Parameter(
            torch.from_numpy(skimage.morphology.disk(1))
            .unsqueeze(0)
            .unsqueeze(0)
            .float(),
            requires_grad=False,
        )
        self.num_sem_categories = num_sem_categories
        self.local_pose = None
        self.frontiers = np.array([])

    @property
    def goal_update_steps(self):
        return 1

    def reach_single_category(self, map_features, category):
        # if the goal is found, reach it
        goal_map, found_goal = self.reach_goal_if_in_map(map_features, category)
        # otherwise, do frontier exploration
        goal_map = self.explore_otherwise_recp(map_features, goal_map, found_goal)
        return goal_map, found_goal

    def reach_object_recep_combination(
        self, map_features, object_category, recep_category
    ):
        # First check if object (small goal) and recep category are in the same cell of the map. if found, set it as a goal
        goal_map, found_goal = self.reach_goal_if_in_map(
            map_features,
            recep_category,
            small_goal_category=object_category,
        )
        # Then check if the recep category exists in the map. if found, set it as a goal
        goal_map, found_rec_goal = self.reach_goal_if_in_map(
            map_features,
            recep_category,
            reject_visited_regions=True,
            goal_map=goal_map,
            found_goal=found_goal,
        )
        # Otherwise, set closest frontier as the goal
        goal_map = self.explore_otherwise(map_features, goal_map, found_goal, found_rec_goal)
        return goal_map, found_goal

    def forward(
        self,
        map_features,
        frontiers,
        object_category=None,
        start_recep_category=None,
        end_recep_category=None,
        instance_id=None,
        nav_to_recep=None,
        seq_local_pose = None
    ):
        """
        Arguments:
            map_features: semantic map features of shape
             (batch_size, 9 + num_sem_categories, M, M)
            object_category: object goal category
            start_recep_category: start receptacle category
            end_recep_category: end receptacle category
            nav_to_recep: If both object_category and recep_category are specified, whether to navigate to receptacle
        Returns:
            goal_map: binary map encoding goal(s) of shape (batch_size, M, M)
            found_goal: binary variables to denote whether we found the object
            goal category of shape (batch_size,)
        """
        self.local_pose = seq_local_pose
        self.frontiers = frontiers
        assert (
            object_category is not None
            or end_recep_category is not None
            or instance_id is not None
        )
        if instance_id is not None:
            instance_map = map_features[0][
                2 * MC.NON_SEM_CHANNELS
                + self.num_sem_categories : 2 * MC.NON_SEM_CHANNELS
                + 2 * self.num_sem_categories,
                :,
                :,
            ]
            if len(instance_map) != 0:
                inst_map_idx = instance_map == instance_id
                inst_map_idx = torch.argmax(torch.sum(inst_map_idx, axis=(1, 2)))
                goal_map = (
                    (instance_map[inst_map_idx] == instance_id)
                    .to(torch.float)
                    .unsqueeze(0)
                )
                if torch.sum(goal_map) == 0:
                    found_goal = torch.tensor([0])
                else:
                    found_goal = torch.tensor([1])
            else:
                # try to navigate to instance without an instance map -- explore
                # create an empty goal map
                batch_size, _, height, width = map_features.shape
                device = map_features.device
                goal_map = torch.zeros((batch_size, height, width), device=device)
                found_goal = torch.tensor([0])

            goal_map = self.explore_otherwise(map_features, goal_map, found_goal)
            return goal_map, found_goal

        elif object_category is not None and start_recep_category is not None:
            if nav_to_recep is None or end_recep_category is None:
                nav_to_recep = torch.tensor([0] * map_features.shape[0])

            # batch，object + recp / recp ---> pick or place
            # there is at least one instance in the batch where the goal is object
            if nav_to_recep.sum() < map_features.shape[0]:
                goal_map_o, found_goal_o = self.reach_object_recep_combination(
                    map_features, object_category, start_recep_category
                )
            # there is at least one instance in the batch where the goal is receptacle
            elif nav_to_recep.sum() > 0:
                goal_map_r, found_goal_r = self.reach_single_category(
                    map_features, end_recep_category
                )
            # some instances in batch may be navigating to objects (before pick skill) and some may be navigating to recep (before place skill)
            if nav_to_recep.sum() == 0:
                return goal_map_o, found_goal_o
            elif nav_to_recep.sum() == map_features.shape[0]:
                return goal_map_r, found_goal_r
            else:
                goal_map = (
                    goal_map_o * nav_to_recep.view(-1, 1, 1)
                    + (1 - nav_to_recep).view(-1, 1, 1) * goal_map_o
                )
                found_goal = (
                    found_goal_r * nav_to_recep + (1 - nav_to_recep) * found_goal_r
                )
                return goal_map, found_goal
        else:
            # Here, the goal is specified by a single object or receptacle to navigate to with no additional constraints (eg. the given object can be on any receptacle)
            goal_category = (
                object_category if object_category is not None else end_recep_category
            )
            return self.reach_single_category(map_features, goal_category)

    def cluster_filtering(self, m):
        # m is a 480x480 goal map
        if not m.any():
            return m
        device = m.device

        # cluster goal points
        k = DBSCAN(eps=4, min_samples=1)
        m = m.cpu().numpy()
        data = np.array(m.nonzero()).T
        k.fit(data)

        # mask all points not in the largest cluster
        mode = scipy.stats.mode(k.labels_, keepdims=True).mode.item()
        mode_mask = (k.labels_ != mode).nonzero()
        x = data[mode_mask]

        m_filtered = np.copy(m)
        m_filtered[x] = 0.0
        m_filtered = torch.tensor(m_filtered, device=device)

        return m_filtered

    def reach_goal_if_in_map(
        self,
        map_features,
        goal_category,
        small_goal_category=None,
        reject_visited_regions=False,
        goal_map=None,
        found_goal=None,
    ):
        """If the desired goal is in the semantic map, reach it."""
        batch_size, _, height, width = map_features.shape
        device = map_features.device
        if goal_map is None and found_goal is None: 
            goal_map = torch.zeros((batch_size, height, width), device=device)
            found_goal_current = torch.zeros(
                batch_size, dtype=torch.bool, device=device
            )
        else: # 找recp
            # crate a fresh map
            found_goal_current = torch.clone(found_goal)
        for e in range(batch_size):
            # if the category goal was not found previously
            if not found_goal_current[e]:
                # the category to navigate to
                category_map = map_features[
                    e, goal_category[e] + 2 * MC.NON_SEM_CHANNELS, :, :
                ]
                if small_goal_category is not None:
                    # additionally check if the category has the required small object on it
                    category_map = (
                        category_map
                        * map_features[
                            e, small_goal_category[e] + 2 * MC.NON_SEM_CHANNELS, :, :
                        ]
                    )
                if reject_visited_regions:
                    # remove the receptacles that the already been close to
                    category_map = category_map * (
                        1 - map_features[e, MC.BEEN_CLOSE_MAP, :, :]
                    )
                # if the desired category is found with required constraints, set goal for navigation
                if (category_map == 1).sum() > 0:
                    goal_map[e] = category_map == 1
                    found_goal_current[e] = True
        return goal_map, found_goal_current

    def get_frontier_map(self, map_features):
        # Select unexplored area
        if self.exploration_strategy == "seen_frontier":
            frontier_map = (map_features[:, [MC.EXPLORED_MAP], :, :] == 0).float()
        elif self.exploration_strategy == "been_close_to_frontier":
            frontier_map = (map_features[:, [MC.BEEN_CLOSE_MAP], :, :] == 0).float()
        else:
            raise Exception("not implemented")

        # Dilate explored area
        frontier_map = 1 - binary_dilation(
            1 - frontier_map, self.dilate_explored_kernel
        )

        # Select the frontier
        frontier_map = (
            binary_dilation(frontier_map, self.select_border_kernel) - frontier_map
        )

        return frontier_map

    def explore_otherwise(self, map_features, goal_map, found_goal, found_rec_goal):
     
        batch_size = map_features.shape[0]
        for e in range(batch_size):
            if not found_rec_goal[e]:


                frontier_waypoints = map_features[:, [MC.FRONTIERS], :, :]
                value_map = map_features[:, [MC.VALUE_MAP], :, :]
                local_value_max = F.max_pool2d(value_map, kernel_size=5, stride=1, padding=2)
                max_score = 0
                best_frontier = np.array([])
                for frontier in self.frontiers:
                    coords = frontier.reshape(-1,2)
                    if frontier.shape[0] < 11:
                        continue
                    # len_f = frontier.shape[0] // 3
                    # coords = coords[len_f:-len_f]
                    ys, xs = coords[:,0], coords[:,1]
                    frontier_scores = local_value_max[0][0][xs, ys]
                    frontier_score = frontier_scores.mean()
                    if frontier_score > max_score:
                        best_frontier = frontier
                
                if len(best_frontier) == 0:
                    frontier_map = self.get_frontier_map(map_features)
                    goal_map[e] = frontier_map[e]
                    continue
                

                out = torch.zeros_like(value_map)
                best_frontier = best_frontier.reshape(-1, 2)
                len_f = best_frontier.shape[0] // 2
                best_frontier = best_frontier[len_f - 1: -len_f + 1]
                ys, xs = best_frontier[:,0], best_frontier[:,1]
                # device = out.device
                # ys = torch.as_tensor(ys, dtype=torch.int, device=device)
                # xs = torch.as_tensor(xs, dtype=torch.int, device=device)
                out[0,0,xs,ys] = 1.0
                # kernel = torch.ones((1,1,3,3), device=out.device, dtype=out.dtype)
                # out = F.conv2d(out, kernel, stride=1, padding=1)
                out = (out > 0).float()
                goal_map = out.squeeze(0)
                # else:
                #     """Explore closest unexplored region otherwise."""
                #     frontier_map = self.get_frontier_map(map_features)

                #     goal_map[e] = frontier_map[e]

        
        self.save_current_map_vis(map_features,goal_map, local_pose=self.local_pose.squeeze(0))



        return goal_map
    
    def explore_otherwise_recp(self, map_features, goal_map, found_goal):

        batch_size = map_features.shape[0]
        for e in range(batch_size):
            if not found_goal[e]:
            # if not found_rec_goal[e]:

                frontier_waypoints = map_features[:, [MC.FRONTIERS], :, :]
                value_map = map_features[:, [MC.VALUE_MAP_PLACE], :, :]
                local_value_max = F.max_pool2d(value_map, kernel_size=5, stride=1, padding=2)
                max_score = 0
                best_frontier = np.array([])
                for frontier in self.frontiers:
                    coords = frontier.reshape(-1,2)
                    if frontier.shape[0] < 11:
                        continue
                    # len_f = frontier.shape[0] // 3
                    # coords = coords[len_f:-len_f]
                    ys, xs = coords[:,0], coords[:,1]
                    frontier_scores = local_value_max[0][0][xs, ys]
                    frontier_score = frontier_scores.mean()
                    if frontier_score > max_score:
                        best_frontier = frontier
                
                if len(best_frontier) == 0:
                    frontier_map = self.get_frontier_map(map_features)
                    goal_map[e] = frontier_map[e]
                    continue
                

                out = torch.zeros_like(value_map)
                best_frontier = best_frontier.reshape(-1, 2)
                len_f = best_frontier.shape[0] // 2
                best_frontier = best_frontier[len_f - 1: -len_f + 1]
                ys, xs = best_frontier[:,0], best_frontier[:,1]
                # device = out.device
                # ys = torch.as_tensor(ys, dtype=torch.int, device=device)
                # xs = torch.as_tensor(xs, dtype=torch.int, device=device)
                out[0,0,xs,ys] = 1.0
                # kernel = torch.ones((1,1,3,3), device=out.device, dtype=out.dtype)
                # out = F.conv2d(out, kernel, stride=1, padding=1)
                out = (out > 0).float()
                goal_map = goal_map +  out.squeeze(0)
                # else:
                #     """Explore closest unexplored region otherwise."""
                #     frontier_map = self.get_frontier_map(map_features)

                #     goal_map[e] = frontier_map[e]

        
        
        self.save_current_map_vis(map_features,goal_map, local_pose=self.local_pose.squeeze(0),change_value=True)
        return goal_map

    
    def save_current_map_vis(
        self,
        current_map,                 # [C,H,W] 或 [1,C,H,W]，torch/np都行
        goal_map,
        save_dir:str="/home/zkm/home-robot/datadump_1/map_debug",
        map_resolution_cm:int=5,  # 若用 local_pose 必须传（通常是 self.resolution）
        local_pose=None,             # 可选: [x_m, y_m, theta]（theta弧度或角度都可，见下）
        theta_in_radians:bool=False,  # local_pose[2] 是否是弧度
        flip_vertical:bool=True,
        agent_radius_px:int=3,
        arrow_len_px:int=20,
        filename_prefix:str="local",
        # === 新增参数 ===
        hfov_deg: float = 42.0,      # 水平视场角（度）；给了就画扇形
        fov_range_m: float = 3.5,   # 扇形半径（米），优先级高于 fov_range_px
        fov_range_px: int = None,    # 扇形半径（像素），两者都没给时自动估一个
        fov_alpha: float = 0.28,      # 扇形透明度
        change_value: bool = False
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
        expl = cm[MC.VLFM_EXPLORE] > 0
        frontiers = cm[MC.FRONTIERS] > 0
        cur_loc_mask = cm[MC.CURRENT_LOCATION] > 0
        if change_value:
            value_map = cm[MC.VALUE_MAP_PLACE]
        else:
            value_map = cm[MC.VALUE_MAP]
        goal_map = goal_map.reshape(480,480).cpu().numpy()
        goal_map = goal_map > 0
        H, W = obst.shape

        if flip_vertical:
            obst = np.flipud(obst); expl = np.flipud(expl); cur_loc_mask = np.flipud(cur_loc_mask); frontiers = np.flipud(frontiers)
            value_map = np.flipud(value_map)
            goal_map = np.flipud(goal_map)

        COL_BG   = (255,255,255)
        COL_OBS  = (64,64,64)
        COL_EXP  = (200,200,200)
        COL_AGT  = (0,0,255)
        COL_FOV  = (255, 200, 120)   # 浅蓝（BGR，带点青色），看着不刺眼
        COL_FRONTIERS = (255, 0, 0)
        COL_GOAL = (120,213,168)

        left  = np.full((H,W,3), COL_BG,  dtype=np.uint8)
        # right = np.full((H,W,3), COL_BG,  dtype=np.uint8)
        
        left[expl] = COL_EXP
        left[obst]  = COL_OBS
        frontier_vis_kernel = np.ones((3,3), np.uint8)
        fmask = cv2.dilate(frontiers.astype(np.uint8)*255, frontier_vis_kernel, iterations=1).astype(bool)


        # === 计算像素位置与朝向 ===
        cx = cy = None
        yaw_img = None

        if local_pose is not None:
            local_pose = local_pose.squeeze(0).cpu().numpy()
            x_m, y_m, th = float(local_pose[0]), float(local_pose[1]), float(local_pose[2])
            cx = int(round(x_m * 100.0 / map_resolution_cm))
            cy = 480 - int(round(y_m * 100.0 / map_resolution_cm)) 
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
        left[goal_map] = COL_GOAL


        vis = np.concatenate([left, right], axis=1)
        os.makedirs(save_dir, exist_ok=True)
        out_path = os.path.join(save_dir, f"{filename_prefix}_map_policy_device1.png")
        cv2.imwrite(out_path, vis)
        return out_path, vis
