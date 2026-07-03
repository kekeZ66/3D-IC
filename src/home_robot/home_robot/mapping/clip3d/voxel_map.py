import torch
import numpy as np
import trimesh.transformations as tra
import home_robot.utils.depth as du
import cv2
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from PIL import Image
from sklearn.cluster import DBSCAN
from home_robot.mapping.clip3d.spare_token_map import SparseTokenMap
from home_robot.mllm.promptsv3 import CHAIN_SELECTION_PROMPT
from typing import Optional, List, Dict, Any

import re

class VoxelClipMap:
    
    def __init__(self):
        
        self.xyz_resolution = 5
        self.token_dim = 1280
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.vlm_device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        self.max_depth = 350
        self.shift = torch.tensor([480,480,0]).to(self.device)
        

        self.screen_h = 640
        self.screen_w = 480
        self.hfov = 42
        self.camera_matrix = du.get_camera_matrix(self.screen_w, self.screen_h, self.hfov)
        
        self.token_map = SparseTokenMap(device = self.device, token_dim=self.token_dim)

        self.vlm_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "/path/to/model", dtype="auto", device_map={"": 1}
        )
        self.vlm_model.eval()
        self.processor = AutoProcessor.from_pretrained("/path/to/processor")



        self.init_camera_x = None
        self.init_camera_y = None

    def set_camera_pose(self, init_camera_pose):
        self.init_camera_x = init_camera_pose[:, 0, 3] * -100
        self.init_camera_y = init_camera_pose[:, 1, 3] * -100 

    def reset(self):  
        self.token_map.reset()

    def update_voxel_map(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        camera_pose: torch.Tensor,
        gps: np.ndarray,
    ):
        camera_pose = camera_pose.to(self.device)
        angles = torch.Tensor([tra.euler_from_matrix(p[:3, :3].cpu(), "rzyx") for p in camera_pose])
        tilt, yaw, roll = angles[:, 1], angles[:, 2], angles[:, 0]
        camera_x = camera_pose[:, 0, 3] * -100
        camera_y = camera_pose[:, 1, 3] * -100
        agent_pos = camera_pose[:, :3, 3] * 100
        agent_height = agent_pos[:, 2]

        depth[depth > self.max_depth] = 0
        point_cloud_full_t = du.get_point_cloud_from_z_t(depth, self.camera_matrix, self.device)
        point_cloud_full_base = du.transform_camera_view_t(point_cloud_full_t, agent_height, torch.rad2deg(tilt).cpu().numpy(), self.device)
        global_pose_cm = np.array([gps[0] * 100, gps[1] * 100, roll.item() + np.pi / 2], dtype=np.float32)
        point_cloud_full_map = du.transform_pose_t(point_cloud_full_base, global_pose_cm, self.device)


        img = Image.fromarray(rgb, "RGB")
        vision_model_input = self.processor.image_processor(img)
        vision_model_input = vision_model_input.to(self.vlm_device)
        with torch.no_grad():
            self.vlm_model.visual.token_list = None
            image_embeds = self.vlm_model.get_image_features(**vision_model_input, return_merger=False)

        z = point_cloud_full_map[:,:,:,2]
        

        H, W = 640, 480
        valid = (depth > 0) & (depth <= self.max_depth) & (z < 100)

        _, Ht, Wt = vision_model_input["image_grid_thw"][0].tolist() 
        patch_size = 14


        B, H, W = valid.shape
        assert B == 1

        xx, yy = torch.where(valid[0])                        
        pc = point_cloud_full_map[0, xx, yy, :]               

        vox = torch.floor(pc / self.xyz_resolution).to(torch.int64)  
        pts = vox + self.shift.view(1,3).to(vox.device, vox.dtype)  

        ph = (xx // patch_size).to(torch.int64).clamp(0, Ht - 1)          
        pw = (yy // patch_size).to(torch.int64).clamp(0, Wt - 1)
        patch_id = ph * Wt + pw                         

        uv = torch.stack([yy, xx], dim=1)
        if image_embeds.shape[0] != 1564:
            return
        self.token_map.update_many(
            pts_all=pts,
            patch_id=patch_id,
            patch_token=image_embeds.to(self.device),
            uv = uv,
            reuse_free=True,
            dedup_voxels=True,   
        )


    def dbscan_centers_torch(self, pts_t: torch.Tensor, eps: float, min_samples: int = 10, robust=False):

        pts = pts_t.detach().float().cpu().numpy()
        labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(pts)

        centers = []
        for k in sorted(set(labels)):
            if k == -1:
                continue
            c = pts[labels == k]
            if robust:
                centers.append(np.median(c, axis=0))
            else:
                centers.append(c.mean(axis=0))

        centers = np.stack(centers, axis=0) if len(centers) else np.zeros((0,3), dtype=np.float32)
        return centers, labels
    


    def get_chain_answer(
        self,
        chain_info_list: List[Dict[str, Any]],
        goal_name: str,
    ):
        if isinstance(goal_name, (list, tuple)):
            goal_name = goal_name[0] if len(goal_name) > 0 else ""
        goal_name = str(goal_name)
        goal_parts = goal_name.split()
        target_object = goal_parts[1] if len(goal_parts) > 1 else "object"
        end_recep = goal_parts[5] if len(goal_parts) > 5 else "receptacle"

        img = np.zeros((280, 448, 3), dtype=np.uint8)
        img = Image.fromarray(img, "RGB")
        content = []
        input_images = []
        token_list = []
        plans_text = ""
        img_cnt = 1

        for chain in chain_info_list:
            action_texts = []
            for action in chain.get("actions", []):
                token = action.get("token")
                if token is None:
                    continue
                action_name = action.get("action", "navigate to")
                action_texts.append(f"{action_name} Image {img_cnt}")
                content.append({"type": "text", "text": f"Image {img_cnt}"})
                content.append({"type": "image", "image": img})
                token_list.append(token)
                input_images.append(img)
                img_cnt += 1
            if len(action_texts) > 0:
                plan_id = int(chain["label"]) + 1
                plans_text += f"plan_{plan_id}: " + ", ".join(action_texts) + "\n"

        if len(token_list) == 0 or len(plans_text) == 0:
            return 0, None

        prompt = CHAIN_SELECTION_PROMPT.format(
            instruction=f"place the {target_object} on the {end_recep}",
            target_object=target_object,
            target_receptacle=end_recep,
            interaction_chains=plans_text,
        )
        content.append({"type": "text", "text": prompt})
        self.vlm_model.visual.token_list = [token.to(self.vlm_device) for token in token_list]
        messages = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text],
            images=input_images,
            videos=None,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.vlm_device)
        try:
            generated_ids = self.vlm_model.generate(**inputs, max_new_tokens=128)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
        except Exception as e:
            raise
        chain_ids = {int(chain["label"]) for chain in chain_info_list}
        vlm_scores = parse_chain_scores(output_text)
        combined_scores = []
        for chain in chain_info_list:
            label = int(chain["label"])
            plan_id = label + 1
            if plan_id not in vlm_scores:
                continue
            chain_score = float(chain.get("chain", {}).get("score", 0.0))
            vlm_score = float(np.clip(vlm_scores[plan_id], 0.0, 1.0))
            combined_scores.append((chain_score + vlm_score, label, chain_score, vlm_score))

        if len(combined_scores) > 0:
            combined_scores.sort(key=lambda x: x[0], reverse=True)
            return int(combined_scores[0][1]), output_text

        choice = parse_chain_answer(output_text)
        if choice is not None and choice > 0:
            choice -= 1
        if choice not in chain_ids:
            return 0, output_text
        return int(choice), output_text




def parse_answer_reason(text: Any) -> Dict[str, Any]:
    if isinstance(text, list):
        text = "\n".join(str(x) for x in text)
    elif isinstance(text, dict):
        text = str(text)
    else:
        text = str(text)

    text = text.strip()

    m_a = re.search(r'(?im)^\s*answer\s*[:=\-]\s*(.+?)\s*$', text)
    answer = m_a.group(1).strip() if m_a else None

    m_r = re.search(r'(?ims)^\s*reason\s*[:=\-]\s*(.*)\s*$', text)
    reason = m_r.group(1).strip() if m_r else None

    if isinstance(answer, str):
        answer = answer.strip().strip('"').strip("'")
        if re.fullmatch(r'\d+', answer):
            answer = int(answer)

    if isinstance(reason, str):
        reason = reason.strip().strip('"').strip("'")

    return {"answer": answer, "reason": reason}


def parse_chain_answer(text: Any):
    if isinstance(text, list):
        text = "\n".join(str(x) for x in text)
    elif isinstance(text, dict):
        text = json.dumps(text)
    else:
        text = str(text)

    try:
        data = json.loads(text)
        for key in ("best_plan", "answer", "plan", "chain"):
            if key in data:
                return int(data[key])
    except Exception:
        pass

    for key in ("best_plan", "answer", "plan", "chain"):
        m = re.search(rf'(?im)"?{key}"?\s*[:=\-]\s*"?(\d+)"?', text)
        if m:
            return int(m.group(1))

    m = re.search(r"\b(\d+)\b", text)
    return int(m.group(1)) if m else None


def parse_chain_scores(text: Any) -> Dict[int, float]:
    if isinstance(text, list):
        text = "\n".join(str(x) for x in text)
    elif isinstance(text, dict):
        text = json.dumps(text)
    else:
        text = str(text)

    scores = {}
    try:
        data = json.loads(text)
        for key, value in data.items():
            m = re.fullmatch(r"plan_(\d+)", str(key).strip())
            if m:
                scores[int(m.group(1))] = float(value)
        if len(scores) > 0:
            return scores
    except Exception:
        pass

    for m in re.finditer(r'(?im)"?plan_(\d+)"?\s*[:=]\s*"?([0-9]*\.?[0-9]+)"?', text):
        scores[int(m.group(1))] = float(m.group(2))
    return scores
