import torch
import numpy as np
import cv2
from typing import Optional, Union
import math


class SparseTokenMap:
    def __init__(self, device, map_size=960, z_size=20, token_dim=1280, background_id=0):
        self.device = torch.device(device)
        self.map_size = map_size
        self.z_size = z_size
        self.token_dim = token_dim
        self.bg = int(background_id)

        # voxel -> embedding id
        self.voxels = torch.zeros((map_size, map_size, z_size), device=self.device, dtype=torch.int64)

        # embedding pool
        self.embeddings = torch.zeros((1, token_dim), device=self.device, dtype=torch.bfloat16)
        # reference count per embedding id (how many voxels point to it)
        self.counts = torch.zeros((1,), device=self.device, dtype=torch.int64)
        self.counts[0] = map_size * map_size * z_size  # background occupies everything initially
    
    def reset(self):

        self.voxels.zero_()          
        self.embeddings = self.embeddings[:1].zero_()   
        self.counts = self.counts[:1].zero_()
        self.counts[0] = self.map_size * self.map_size * self.z_size

    @torch.no_grad()
    def _ensure_capacity(self, new_total: int):
        """ensure embeddings/counts length >= new_total"""
        cur = self.embeddings.size(0)
        if new_total <= cur:
            return
        extra = new_total - cur
        self.embeddings = torch.cat(
            [self.embeddings, torch.zeros((extra, self.token_dim), device=self.device, dtype=self.embeddings.dtype)],
            dim=0
        )
        self.counts = torch.cat(
            [self.counts, torch.zeros((extra,), device=self.device, dtype=self.counts.dtype)],
            dim=0
        )

    @torch.no_grad()
    def update_many(
        self,
        pts_all: torch.Tensor,        
        patch_id: torch.Tensor,      
        patch_token: torch.Tensor, 
        uv: torch.Tensor,  
        *,
        reuse_free: bool = True,     
        dedup_voxels: bool = True,    
    ):
        """
        Parallel update:
          - all points are updated in one shot
          - counts decremented for overwritten voxels, incremented for new assignments
        """
        if pts_all.numel() == 0:
            return

        pts_all = pts_all.to(self.device, torch.int64)
        patch_id = patch_id.to(self.device, torch.int64)
        patch_token = patch_token.to(self.device, self.embeddings.dtype)

        x = pts_all[:, 1].clamp(0, self.map_size - 1)
        y = pts_all[:, 0].clamp(0, self.map_size - 1)
        z = pts_all[:, 2].clamp(0, self.z_size - 1)

        if dedup_voxels:
            key = (x * (self.map_size * self.z_size) + y * self.z_size + z)  
            order = torch.argsort(key) 
            key_s = key[order]  

            uv = uv.to(self.device)
            u = uv[:, 0]
            v = uv[:, 1]
            dist2 = (u - 240) ** 2 + (v - 320) ** 2
            dist2_s = dist2[order]

            change = torch.ones_like(key_s, dtype=torch.bool)
            change[1:] = key_s[1:] != key_s[:-1]
            starts = torch.where(change)[0]
            ends = torch.empty_like(starts)
            ends[:-1] = starts[1:]
            ends[-1] = key_s.numel()

            keep_sorted = []
            for st, ed in zip(starts.tolist(), ends.tolist()):
                j = torch.argmin(dist2_s[st:ed]).item() + st
                keep_sorted.append(j)
            keep_sorted = torch.tensor(keep_sorted, device=self.device, dtype=torch.long)

            keep = order[keep_sorted]
            x, y, z = x[keep], y[keep], z[keep]
            patch_id = patch_id[keep]

        M = x.numel()

        old_ids = self.voxels[x, y, z]  # (M,)
        dec = torch.bincount(old_ids, minlength=self.counts.numel()).to(self.counts.dtype)
        self.counts -= dec

        uniq_patch, inv = torch.unique(patch_id, return_inverse=True)  
        Kuniq = uniq_patch.numel()

        tok_u = patch_token[uniq_patch] 

        if reuse_free:
            free = torch.where(self.counts == 0)[0]
            free = free[free != self.bg]
            if free.numel() >= Kuniq:
                new_ids = free[:Kuniq]
            else:
                need = Kuniq - free.numel()
                start = self.embeddings.size(0)
                self._ensure_capacity(start + need)
                app_ids = torch.arange(start, start + need, device=self.device, dtype=torch.int64)
                new_ids = torch.cat([free, app_ids], dim=0)
        else:
            start = self.embeddings.size(0)
            self._ensure_capacity(start + Kuniq)
            new_ids = torch.arange(start, start + Kuniq, device=self.device, dtype=torch.int64)

        self.embeddings[new_ids] = tok_u

        new_id_per_point = new_ids[inv]  # (M,)
        self.voxels[x, y, z] = new_id_per_point

        inc_per_patch = torch.bincount(inv, minlength=Kuniq).to(self.counts.dtype)  # (Kuniq,)
        self.counts[new_ids] += inc_per_patch

    def debug_visualize(self):
        floor = self.voxels[:,:,:3].sum(dim=2) > 0
        recep = self.voxels[:,:,3:17].sum(dim=2) > 0
        wall = self.voxels[:,:,17:].sum(dim=2) > 0
        vis = np.ones((960,960)) * 255
        vis = vis - floor.cpu().numpy() * 100
        vis = vis - recep.cpu().numpy() * 100
        vis = vis - wall.cpu().numpy() * 50


    @torch.no_grad()
    def gather_tokens(
        self,
        origin_xy,                  
        yaw: float,                 
        *,
        hfov_deg: float = 90.0,
        Wt: int = 32,               
        xyz_resolution_cm: float = 5.0,
        max_len_m: float = 3.5,
        step_vox: float = 1.0,
        thickness: int = 1,
        out_path: str = "",
        return_tokens: bool = True,
    ):
        device = self.device
        bg = int(self.bg)
        D = self.token_dim

        if not torch.is_tensor(origin_xy):
            origin_xy = torch.tensor(origin_xy, device=device, dtype=torch.int64)
        else:
            origin_xy = origin_xy.to(device=device, dtype=torch.int64)

        ox, oy = int(origin_xy[0].item()), int(origin_xy[1].item())
        ox = max(0, min(self.map_size - 1, ox))
        oy = max(0, min(self.map_size - 1, oy))

        max_len_cm = max_len_m * 100.0
        max_range_vox = int(max_len_cm / float(xyz_resolution_cm) + 1e-6)   
        max_steps = int(max_range_vox / max(step_vox, 1e-6))

        hfov = math.radians(hfov_deg)
        if Wt == 1:
            angles = [yaw]
        else:
            angles = [yaw + (((u / (Wt - 1)) - 0.5) * hfov) for u in range(Wt)]

        bg_tok = self.embeddings[bg]
        tokens_zw = bg_tok.view(1, 1, D).repeat(self.z_size, Wt, 1).clone()

        occ = (self.voxels != bg).any(dim=2)  
        vis = np.ones((self.map_size, self.map_size), dtype=np.uint8) * 255
        vis[occ.detach().cpu().numpy()] = 180
        vis_rgb = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

        hit_xy = []  

        hitted_id_list = []
        for z in range(self.z_size):             
            for u in range(Wt):                      
                ang = angles[u]
                dx = math.cos(ang)
                dy = math.sin(ang)

                hit_id = bg
                hx = hy = None
                f_hit_id = None

                for s in range(1, max_steps + 1):
                    fx = ox + dx * (s * step_vox)
                    fy = oy + dy * (s * step_vox)
                    xi = int(round(fx))
                    yi = int(round(fy))
                    if xi < 0 or xi >= self.map_size or yi < 0 or yi >= self.map_size:
                        break

                    vid = int(self.voxels[xi, yi, z].item())
                    if vid != bg:
                        if vid not in hitted_id_list:
                            hit_id = vid
                            hx, hy = xi, yi
                            break
                        elif f_hit_id is None:
                            f_hit_id = vid
                            hx, hy = xi, yi
                    
                if hit_id == bg and f_hit_id is not None:
                    hit_id = f_hit_id
                hitted_id_list.append(hit_id)
                    

                tokens_zw[z, u] = self.embeddings[hit_id]
                if hx is not None:
                    hit_xy.append((hx, hy))

        for (hx, hy) in hit_xy:
            vis_rgb[hx, hy] = [255,0,0] 
        vis_rgb[ox-1:ox+1,oy-1:oy+1] = [0,0,255]           
        vis_rgb_out = np.flipud(vis_rgb)
        cv2.imwrite(out_path, vis_rgb_out)

        seq = tokens_zw.reshape(self.z_size * Wt, D)

        if return_tokens:
            return seq, tokens_zw, vis_rgb_out
        else:
            return vis_rgb_out






