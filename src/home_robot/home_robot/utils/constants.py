# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


MIN_DEPTH_REPLACEMENT_VALUE = 10000
MAX_DEPTH_REPLACEMENT_VALUE = 10001

ignore_keys = {'articulated_agent_force.accum',
               'articulated_agent_force.instant',
               'force_terminate',
               'robot_collisions.total_collisions',
               'robot_collisions.robot_obj_colls',
               'num_steps',
               'object_at_rest',
               'robot_collisions.robot_scene_colls',
               'does_want_terminate', 
               'obj_anywhere_on_goal.0',
               'picked_object_linear_vel',
               'picked_object_angular_vel',
               'navmesh_collision',
               'pick_goal_iou_coverage',''
               'robot_collisions.obj_scene_colls',  
               'is_curr_skill_GAZE_AT_OBJ','is_curr_skill_EXPLORE','is_curr_skill_GAZE_AT_REC','is_curr_skill_NAV_TO_INSTANCE','ovmm_nav_to_place_succ','ee_to_rest_distance',
               'ovmm_object_to_place_goal_distance.0','pick_success', 'ovmm_nav_orient_to_pick_succ','ovmm_nav_orient_to_place_succ', 'ovmm_placement_stability'
              }