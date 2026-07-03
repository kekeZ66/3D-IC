import torch
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Dict, Any, List, Tuple

@torch.no_grad()
def dilate_wall_mask_torch(wall_mask: torch.Tensor, r: int) -> torch.Tensor:
    """
    wall_mask: (H,W) 或 (1,1,H,W)；bool / 0-1 / 0-255 都行（非零视为 True）
    r: 膨胀半径（像素）
    return: 同 shape 的 bool wall_dilated
    """
    if r <= 0:
        return wall_mask.bool()

    # reshape -> (1,1,H,W)
    if wall_mask.dim() == 2:
        x = wall_mask[None, None]
    elif wall_mask.dim() == 4:
        x = wall_mask
        assert x.size(0) == 1 and x.size(1) == 1
    else:
        raise ValueError("wall_mask must be (H,W) or (1,1,H,W)")

    x = (x != 0).float()  # 非零=墙

    k = 2 * r + 1
    y = F.max_pool2d(x, kernel_size=k, stride=1, padding=r)  # dilation
    y = (y > 0)

    return y[0, 0] if wall_mask.dim() == 2 else y



@torch.no_grad()
def los_exists_visible(
    wall_mask: torch.Tensor,   # (H,W) bool
    obj_mask: torch.Tensor,    # (H,W) bool
    cand_hw: torch.Tensor,     # (N,2) long, [h,w]
    K: int = 32,               # obj 上采样点数
    samples_per_pixel: int = 2,
) -> torch.Tensor:
    """
    exists 语义：对每个候选点，只要存在一个 obj 点与其连线不穿墙 => visible=True
    返回: visible (N,) bool
    """
    device = wall_mask.device
    H, W = wall_mask.shape
    wall = wall_mask.bool()

    oy, ox = torch.where(obj_mask.bool())
    if oy.numel() == 0:
        return torch.zeros((cand_hw.size(0),), dtype=torch.bool, device=device)

    # 从 obj 像素里随机/均匀抽 K 个点（这里用等间隔抽样，确定性）
    idx = torch.linspace(0, oy.numel() - 1, steps=min(K, oy.numel()), device=device).long()
    seed_y = oy[idx].float()
    seed_x = ox[idx].float()

    N = cand_hw.size(0)
    visible = torch.zeros((N,), dtype=torch.bool, device=device)

    for i in range(N):
        y1 = cand_hw[i, 0].float()
        x1 = cand_hw[i, 1].float()

        # 对 K 个 seed，只要有一个不被挡住就行
        ok = False
        for j in range(seed_y.numel()):
            y0, x0 = seed_y[j], seed_x[j]
            steps = int(torch.max((y1 - y0).abs(), (x1 - x0).abs()).item() * samples_per_pixel) + 1
            if steps <= 1:
                ok = True
                break

            t = torch.linspace(0, 1, steps, device=device)
            yy = (y0 + (y1 - y0) * t).round().long().clamp(0, H - 1)
            xx = (x0 + (x1 - x0) * t).round().long().clamp(0, W - 1)

            # 跳过起点一点点，避免 seed 自身贴墙误判
            yy = yy[1:]
            xx = xx[1:]

            if not wall[yy, xx].any():
                ok = True
                break

        visible[i] = ok

    return visible



import numpy as np
from typing import Optional

def los_exists_visible_mask_numpy(
    wall_mask: np.ndarray,
    goal_mask: np.ndarray,
    cand_mask: np.ndarray,
    K: int = 32,
    samples_per_pixel: int = 2,
    radius: Optional[int] = 50, 
    skip_start: int = 1,
) -> np.ndarray:

    wall = (wall_mask != 0)
    goal = (goal_mask != 0)
    cand = (cand_mask != 0)

    H, W = wall.shape
    assert goal.shape == (H, W) and cand.shape == (H, W)

    oy, ox = np.where(goal)
    if oy.size == 0:
        return np.zeros((H, W), dtype=bool)

    # 采样 K 个 goal seeds（等间隔）
    m = min(K, oy.size)
    idx = np.linspace(0, oy.size - 1, m).astype(np.int64)
    seed_y = oy[idx].astype(np.float32)
    seed_x = ox[idx].astype(np.float32)

    # 取候选点坐标
    cy, cx = np.where(cand)
    if cy.size == 0:
        return np.zeros((H, W), dtype=bool)

    # 可选：只保留离 goal 中心 radius 内的候选（加速+更符合你的需求）
    if radius is not None:
        gy0 = float(oy.mean())
        gx0 = float(ox.mean())
        dy = cy.astype(np.float32) - gy0
        dx = cx.astype(np.float32) - gx0
        keep = (dy * dy + dx * dx) <= float(radius * radius)
        cy = cy[keep]
        cx = cx[keep]

    visible_mask = np.zeros((H, W), dtype=bool)

    for i in range(cy.size):
        y1 = float(cy[i])
        x1 = float(cx[i])

        ok = False
        for j in range(m):
            y0 = float(seed_y[j])
            x0 = float(seed_x[j])

            steps = int(max(abs(y1 - y0), abs(x1 - x0)) * samples_per_pixel) + 1
            if steps <= 1:
                ok = True
                break

            t = np.linspace(0.0, 1.0, steps, dtype=np.float32)
            yy = np.rint(y0 + (y1 - y0) * t).astype(np.int64)
            xx = np.rint(x0 + (x1 - x0) * t).astype(np.int64)

            yy = np.clip(yy, 0, H - 1)
            xx = np.clip(xx, 0, W - 1)

            if skip_start > 0 and yy.size > skip_start:
                yy = yy[skip_start:]
                xx = xx[skip_start:]

            # 线上不撞墙 => 可见
            if not wall[yy, xx].any():
                ok = True
                break

        if ok:
            visible_mask[int(y1), int(x1)] = True

    return visible_mask


import numpy as np
import open3d as o3d

def save_pc_base_to_ply(point_cloud_base_coords, path, rgb_down=None, cm_to_m=True):
    """
    支持两种输入：
    1) torch/np [1,H,W,3] 或 [H,W,3] 规则点云（单帧）
    2) np [N,3] 或 torch [N,3] 点集（融合后的全局点云）

    rgb_down 支持：
    - 单帧： [1,3,H,W] / [H,W,3]
    - 点集： [N,3]
    """

    # ---- to numpy ----
    if hasattr(point_cloud_base_coords, "detach"):
        pc = point_cloud_base_coords.detach().cpu().numpy()
    else:
        pc = np.asarray(point_cloud_base_coords)

    # ---- flatten points ----
    valid = None
    if pc.ndim == 4:          # [1,H,W,3]
        pc = pc[0]
    if pc.ndim == 3:          # [H,W,3]
        pts = pc.reshape(-1, 3)
        valid = np.isfinite(pts).all(axis=1) & (np.linalg.norm(pts, axis=1) > 1e-6)
        pts = pts[valid]
    elif pc.ndim == 2 and pc.shape[1] == 3:   # [N,3]
        pts = pc
        valid = np.isfinite(pts).all(axis=1) & (np.linalg.norm(pts, axis=1) > 1e-6)
        pts = pts[valid]
    else:
        raise ValueError(f"Unsupported pc shape: {pc.shape}")

    # ---- unit ----
    if cm_to_m:
        pts = pts / 100.0

    # ---- build pcd ----
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))

    # ---- color ----
    if rgb_down is not None:
        if hasattr(rgb_down, "detach"):
            rgb = rgb_down.detach().cpu().numpy()
        else:
            rgb = np.asarray(rgb_down)

        # 单帧颜色
        if rgb.ndim == 4:            # [1,3,H,W]
            rgb = rgb[0].transpose(1, 2, 0)   # [H,W,3]
        if rgb.ndim == 3:            # [H,W,3]
            rgb = rgb.reshape(-1, 3)
            rgb = rgb[valid]         # valid 来自展平后的点
        elif rgb.ndim == 2 and rgb.shape[1] == 3:  # [N,3]
            rgb = rgb[valid]         # valid 来自 N 点过滤
        else:
            raise ValueError(f"Unsupported rgb shape: {rgb.shape}")

        rgb = np.clip(rgb, 0, 255) / 255.0
        pcd.colors = o3d.utility.Vector3dVector(rgb.astype(np.float64))

    o3d.io.write_point_cloud(path, pcd)



import numpy as np
import cv2
from scipy.ndimage import distance_transform_edt

def connected_component_centers(
    goal_map: np.ndarray,
    connectivity: int = 8,
    min_area: int = 0,
    topk: int = None,
    mode: str = "dt",   # "dt"=最大内点中心(推荐), "centroid"=质心
):
    """
    goal_map: (H,W) numpy, values in {0,1} or bool
    returns: list of dicts, each has:
      - label, area
      - center_rc (row,col)
      - centroid_rc (row,col)
      - dt_center_rc (row,col)  (mode="dt" 时这个就是 center_rc)
      - bbox (rmin,cmin,rmax,cmax)
    """
    assert goal_map.ndim == 2
    mask = (goal_map > 0).astype(np.uint8)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=connectivity
    )
    comps = []
    for lab in range(1, num_labels):  # 0 is background
        area = int(stats[lab, cv2.CC_STAT_AREA])
        if area < min_area:
            continue

        # bbox
        x = int(stats[lab, cv2.CC_STAT_LEFT])
        y = int(stats[lab, cv2.CC_STAT_TOP])
        w = int(stats[lab, cv2.CC_STAT_WIDTH])
        h = int(stats[lab, cv2.CC_STAT_HEIGHT])
        bbox = (y, x, y + h - 1, x + w - 1)  # (rmin,cmin,rmax,cmax)

        # centroid from OpenCV is (cx, cy) = (col, row)
        cx, cy = centroids[lab]
        centroid_rc = (int(round(cy)), int(round(cx)))

        # dt-center: point farthest from boundary (max inscribed circle center)
        region = (labels == lab).astype(np.uint8)
        dist = distance_transform_edt(region)  # float64
        rr, cc = np.unravel_index(np.argmax(dist), dist.shape)
        dt_center_rc = (int(rr), int(cc))

        if mode == "centroid":
            center_rc = centroid_rc
        else:  # "dt"
            center_rc = dt_center_rc

        comps.append({
            "label": lab,
            "area": area,
            "center_rc": center_rc,
            "centroid_rc": centroid_rc,
            "dt_center_rc": dt_center_rc,
            "bbox": bbox,
        })

    # 按面积排序（大→小）
    comps.sort(key=lambda d: d["area"], reverse=True)
    if topk is not None:
        comps = comps[:topk]
    return comps


import numpy as np
from scipy.ndimage import distance_transform_edt

def nearest_one_point_edt(map01: np.ndarray, pt_rc):
    """
    map01: (H,W) 0/1 或 bool
    pt_rc: (r,c)
    return: (r_near, c_near) 或 None
    """
    m = (map01 > 0)
    if not m.any():
        return None

    # edt 默认计算到“0”的距离，所以要对 m 取反：
    # (~m) 的 0 区域就是 m==1 的位置
    dist, inds = distance_transform_edt(~m, return_indices=True)
    r0, c0 = int(pt_rc[0]), int(pt_rc[1])

    r_near = int(inds[0, r0, c0])
    c_near = int(inds[1, r0, c0])
    return r_near, c_near




def sample_points_per_cc_dict(
    goal_map: np.ndarray,
    pixels_per_point: int = 50,
    min_cc_pixels: int = 20,
    alpha: float = 0.15,
    candidate_factor: int = 20,
) -> Dict[int, Dict[str, Any]]:
    """
    goal_map: (H,W) float/bool，非零视为前景
    返回:
      {
        label_id: {
          "centroid": (cy, cx),          # float
          "area": area,                  # int
          "points": [(y,x), ...]         # List[Tuple[int,int]]
        },
        ...
      }
    规则：连通域像素数 < min_cc_pixels => 忽略
    """
    fg = (goal_map > 0).astype(np.uint8)
    if fg.sum() == 0:
        return {}

    num_labels, labels = cv2.connectedComponents(fg, connectivity=8)

    out: Dict[int, Dict[str, Any]] = {}

    for lab in range(1, num_labels):
        region = (labels == lab)
        area = int(region.sum())
        if area < min_cc_pixels:
            continue

        ys, xs = np.where(region)
        coords = np.stack([ys, xs], axis=1)  # (N,2) in (y,x)

        cy, cx = coords.mean(axis=0)  # float centroid

        # 每 pixels_per_point 个像素采样 1 个点（至少1个）
        k = max(1, int(np.ceil(area / float(pixels_per_point))))

        # 1) 中心性：离边界越远越中心
        dist = cv2.distanceTransform(region.astype(np.uint8), cv2.DIST_L2, 5)
        dvals = dist[ys, xs]

        # 2) 质心惩罚：更靠近质心更好
        r = np.sqrt((coords[:, 0] - cy) ** 2 + (coords[:, 1] - cx) ** 2)

        # 综合分数：越大越“中心”
        score = dvals - alpha * r

        # 候选点：取分数最高的一批，再做 FPS 保证均匀
        cand_n = min(len(coords), max(k * candidate_factor, k))
        cand_idx = np.argsort(score)[::-1][:cand_n]
        cand = coords[cand_idx]  # (M,2)

        # FPS：先取最中心的，再取与已选集合最远的
        picked = [0]
        if k > 1:
            dmin = np.sum((cand - cand[0]) ** 2, axis=1).astype(np.float32)
            for _ in range(1, k):
                i = int(np.argmax(dmin))
                picked.append(i)
                di = np.sum((cand - cand[i]) ** 2, axis=1).astype(np.float32)
                dmin = np.minimum(dmin, di)

        pts = cand[picked]
        pts_list: List[Tuple[int, int]] = [(int(y), int(x)) for y, x in pts]

        out[int(lab)] = {
            "centroid": (float(cy), float(cx)),
            "area": area,
            "points": pts_list,
        }

    return out



import math
import torch
def map_indices_to_base_xy(
    y_idx: torch.Tensor,              # scalar int/long OR tensor
    x_idx: torch.Tensor,              # scalar int/long OR tensor
    current_pose: torch.Tensor,       # [1,3] (x,y,theta_deg)
    xy_resolution_cm: float = 5.0,
    x_flip_const: int = 480,          # 你 forward 写死的 480
    device=None,
):
    """
    Inverse of pixel2map_indices_current's XY part.
    Returns base (bx,by) in meters.
    """
    if device is None:
        device = current_pose.device

    # pose
    px, py, theta_deg = current_pose[0].tolist()
    theta = math.radians(theta_deg)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)

    # ensure tensors
    y_idx = torch.as_tensor(y_idx, device=device, dtype=torch.float32)
    x_idx = torch.as_tensor(x_idx, device=device, dtype=torch.float32)

    # undo x flip: x_idx0 = 480 - x_idx
    x_idx0 = float(x_flip_const) - x_idx

    # grid -> world meters (approx inverse of round)
    wx = x_idx0 * (xy_resolution_cm / 100.0)
    wy = y_idx  * (xy_resolution_cm / 100.0)

    # world -> base: [bx;by] = R^T * ([wx;wy] - [px;py])
    dx = wx - px
    dy = wy - py

    bx =  cos_t * dx + sin_t * dy
    by = -sin_t * dx + cos_t * dy

    return bx, by
