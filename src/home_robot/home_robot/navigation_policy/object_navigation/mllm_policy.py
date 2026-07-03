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
from kornia.contrib import connected_components
import cv2
import torch.nn.functional as F



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
        self.value_map_obj = None
        self.value_map_recep = None
        self.detect_frontier_map = None
        self.goal_reach_mask = None

    @property
    def goal_update_steps(self):
        return 1

    def reach_single_category(self, map_features, category):
        goal_map, found_goal = self.reach_goal_if_in_map(map_features, category)
        
        false_recep = map_features[:,MC.FALSE_RECEP_MAP,:,:]
        if found_goal and false_recep.sum() == 0:
            labels = connected_components(goal_map).to(torch.int64)
            uniq, inv = torch.unique(labels, return_inverse=True)
            labels_seq = inv.view_as(labels)
            bg_id = (uniq == 0).nonzero(as_tuple=False).item()
            K = uniq.numel()
            counts = torch.bincount(labels_seq.view(-1), minlength=K)
            min_pixels = 20
            keep_ids = torch.nonzero(counts >= min_pixels).flatten()
            keep_ids = keep_ids[keep_ids != bg_id]
            lut = torch.zeros(K, dtype=torch.bool, device = goal_map.device)
            if keep_ids.numel() > 0:
                lut[keep_ids] = True
            keep_mask = lut[labels_seq].to(goal_map.dtype)

            goal_map = keep_mask
        
            if goal_map.max() == 0:
                found_goal[0] = False


        goal_map = self.explore_otherwise_recp(map_features, goal_map, found_goal)
        return goal_map, found_goal

    def reach_object_recep_combination(
        self, map_features, object_category, recep_category
    ):
        goal_map, found_goal = self.reach_goal_if_in_map(
            map_features,
            recep_category,
            small_goal_category=object_category,
        )
        goal_map, found_rec_goal = self.reach_goal_if_in_map(
            map_features,
            recep_category,
            reject_visited_regions=True,
            goal_map=goal_map,
            found_goal=found_goal,
        )
        if not found_goal and found_rec_goal:
            explored_mask = torch.from_numpy(self.vlfm_explored_map).to(goal_map.device, dtype=goal_map.dtype)
            g = goal_map.unsqueeze(1)
            e = explored_mask.to(goal_map.dtype).unsqueeze(0).unsqueeze(0)
            dil = F.max_pool2d(e.float(), kernel_size=19, stride=1, padding=9) > 0
            out = (g * dil.to(g.dtype)).squeeze(1)

            labels = connected_components(out).to(torch.int64)
            uniq, inv = torch.unique(labels, return_inverse=True)
            labels_seq = inv.view_as(labels)
            bg_id = (uniq == 0).nonzero(as_tuple=False).item()
            K = uniq.numel()
            counts = torch.bincount(labels_seq.view(-1), minlength=K)
            min_pixels = 10
            keep_ids = torch.nonzero(counts >= min_pixels).flatten()
            keep_ids = keep_ids[keep_ids != bg_id]
            lut = torch.zeros(K, dtype=torch.bool, device = goal_map.device)
            if keep_ids.numel() > 0:
                lut[keep_ids] = True
            keep_mask = lut[labels_seq].to(goal_map.dtype)

            goal_map = keep_mask
        
            if goal_map.max() == 0:
                found_rec_goal[0] = False
        
        self.goal_reach_mask = None
        if found_goal:
            obst = map_features[:,MC.OBSTACLE_MAP,:,:,]
            obst = F.max_pool2d(obst, kernel_size=3, stride=1, padding=1)
            frontier_vis_kernel = np.ones((5,5), np.uint8)
            fmask = cv2.dilate(self.detect_frontier_map.astype(np.uint8)*255, frontier_vis_kernel, iterations=1).astype(bool)
            fmap = torch.from_numpy(fmask).to(obst.device, obst.dtype)
            fmap = fmap.unsqueeze(0)
            unreachable_mask = (obst + fmap) > 0
            local_pose = self.local_pose.squeeze(0).squeeze(0)
            x_m = local_pose[0].item()
            y_m = local_pose[0].item()
            map_resolution_cm = 5
            cx = int(round(x_m * 100.0 / map_resolution_cm))
            cy = int(round(y_m * 100.0 / map_resolution_cm)) 
            
            unreachable_mask = unreachable_mask.to(torch.float32)
            trasveribale = 1 - unreachable_mask
            labels = connected_components(trasveribale).to(torch.int64)
            goal_reach_mask = (labels == labels[0][cx][cy]).to(torch.float32)
            goal_reach_mask = F.max_pool2d(goal_reach_mask, kernel_size=3, stride=1, padding=1)
            self.goal_reach_mask = goal_reach_mask


        self.map_frontier_pt = []
        goal_map = self.explore_otherwise(map_features, goal_map, found_goal, found_rec_goal)
        return goal_map, found_goal

    def forward(
        self,
        map_features,
        frontiers,
        value_map_obj,
        value_map_recep,
        frontiers_map,
        vlfm_explored_map,
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
        self.value_map_obj = value_map_obj
        self.value_map_recep = value_map_recep
        self.detect_frontier_map = frontiers_map
        self.vlfm_explored_map = vlfm_explored_map
        assert (
            object_category is not None
            or end_recep_category is not None
            or instance_id is not None
        )
        end_recep_category[0] = -1
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
                batch_size, _, height, width = map_features.shape
                device = map_features.device
                goal_map = torch.zeros((batch_size, height, width), device=device)
                found_goal = torch.tensor([0])

            goal_map = self.explore_otherwise(map_features, goal_map, found_goal)
            return goal_map, found_goal

        elif object_category is not None and start_recep_category is not None:
            if nav_to_recep is None or end_recep_category is None:
                nav_to_recep = torch.tensor([0] * map_features.shape[0])

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
        else: 
            found_goal_current = torch.clone(found_goal)
        for e in range(batch_size):
            if not found_goal_current[e]:
                # the category to navigate to
                category_map = map_features[
                    e, goal_category[e] + 2 * MC.NON_SEM_CHANNELS, :, :
                ]
                if goal_category[e] == -1:
                    category_map = map_features[e, 3 + 2 * MC.NON_SEM_CHANNELS,:,:]
                if small_goal_category is not None:
                    category_map = (
                        category_map
                        * map_features[
                            e, small_goal_category[e] + 2 * MC.NON_SEM_CHANNELS, :, :
                        ]
                    )
                if reject_visited_regions:
                    category_map = category_map * (
                        1 - map_features[e, MC.BEEN_CLOSE_MAP, :, :]
                    )
                if (category_map == 1).sum() > 0:
                    goal_map[e] = category_map == 1
                    found_goal_current[e] = True
        return goal_map, found_goal_current

    def get_frontier_map(self, map_features):
        if self.exploration_strategy == "seen_frontier":
            frontier_map = (map_features[:, [MC.EXPLORED_MAP], :, :] == 0).float()
        elif self.exploration_strategy == "been_close_to_frontier":
            frontier_map = (map_features[:, [MC.BEEN_CLOSE_MAP], :, :] == 0).float()
        else:
            raise Exception("not implemented")

        frontier_map = 1 - binary_dilation(
            1 - frontier_map, self.dilate_explored_kernel
        )

        frontier_map = (
            binary_dilation(frontier_map, self.select_border_kernel) - frontier_map
        )

        return frontier_map

    def explore_otherwise(self, map_features, goal_map, found_goal, found_rec_goal):
     
        batch_size = map_features.shape[0]
        for e in range(batch_size):
            if not found_rec_goal[e]:

                value_map = self.value_map_obj
                max_score = 0
                score_list = []
                best_frontier = np.array([])
                ft_idx = -1
                ft_idx_list = []
                for frontier in self.frontiers:
                    ft_idx += 1
                    coords = frontier.reshape(-1,2)
                    if frontier.shape[0] < 5:
                        continue
                    ys, xs = coords[:,0], coords[:,1]
                    frontier_scores = value_map[xs, ys]
                    frontier_score = frontier_scores.max()
                    score_list.append(frontier_score)
                    ft_idx_list.append(ft_idx)

                if len(score_list) == 0:
                    frontier_map = self.get_frontier_map(map_features)
                    goal_map[e] = frontier_map[e]
                    continue
                score_list = np.array(score_list)
                ft_idx_list = np.array(ft_idx_list)

                sort_idx = np.argsort(-score_list)
                score_list = score_list[sort_idx]
                ft_idx_list = ft_idx_list[sort_idx]

                high_scores = score_list[:(len(score_list)+1)//2]
                
                cut = 0
                for i in range(len(high_scores)-1):
                    if score_list[i] > score_list[i+1] + 0.02: 
                        break
                    cut += 1 

                filtered_frontiers_idx = ft_idx_list[:cut+1].tolist()

                out = torch.zeros_like(goal_map)
                for frontier_idx in filtered_frontiers_idx:
                    choose_frontier = self.frontiers[frontier_idx]            
                    choose_frontier = choose_frontier.reshape(-1, 2)
                    len_f = choose_frontier.shape[0] // 2
                    self.map_frontier_pt.append(choose_frontier[len_f])
                    choose_frontier = choose_frontier[len_f - 1: -len_f + 1]
                    ys, xs = choose_frontier[:,0], choose_frontier[:,1]
                    out[0,xs,ys] = 1.0



                out = (out > 0).float()
                goal_map = out
        return goal_map
    
    def explore_otherwise_recp(self, map_features, goal_map, found_goal):

        batch_size = map_features.shape[0]
        for e in range(batch_size):
            if not found_goal[e]:
                value_map = self.value_map_recep
                max_score = 0
                best_frontier = np.array([])
                for frontier in self.frontiers:
                    coords = frontier.reshape(-1,2)
                    if frontier.shape[0] < 11:
                        continue

                    ys, xs = coords[:,0], coords[:,1]
                    frontier_scores = value_map[xs, ys]
                    frontier_score = frontier_scores.mean()
                    if frontier_score >= max_score:
                        best_frontier = frontier
                        max_score = frontier_score
                
                if len(best_frontier) == 0:
                    frontier_map = self.get_frontier_map(map_features)
                    goal_map[e] = frontier_map[e]
                    continue
                

                out = torch.zeros_like(goal_map)
                best_frontier = best_frontier.reshape(-1, 2)
                len_f = best_frontier.shape[0] // 2
                best_frontier = best_frontier[len_f - 1: -len_f + 1]
                ys, xs = best_frontier[:,0], best_frontier[:,1]

                out[0,xs,ys] = 1.0

                out = (out > 0).float()
                goal_map = goal_map + out


        
        
        return goal_map

    


    