import open_clip
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np

class MyClip:
    _instance = None
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, device="cuda"):
        if getattr(self, "initialized", False):
            return
        self.device = device
        pretrained = "laion2b_s32b_b82k"
        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
            "ViT-L-14", pretrained=pretrained
        )
        self.clip_model = self.clip_model.to(self.device).eval()
        self.clip_tokenizer = open_clip.get_tokenizer("ViT-L-14")
        self.initialized = True

    @torch.inference_mode()
    def get_text_feature(self, text_queries):
        tok = self.clip_tokenizer(text_queries).to(self.device)
        feat = self.clip_model.encode_text(tok)
        return feat / feat.norm(dim=-1, keepdim=True)

    @torch.inference_mode()
    def get_image_feature(self, image: Image.Image):
        x = self.clip_preprocess(image).unsqueeze(0).to(self.device)
        feat = self.clip_model.encode_image(x)
        return feat / feat.norm(dim=-1, keepdim=True)

    def tensor_to_pil(self, img: torch.Tensor) -> Image.Image:
        img = img.detach().cpu()
        if img.ndim != 3:
            raise ValueError(f"tensor_to_pil expects 3D tensor, got {img.shape}")

        if img.shape[0] == 3 and img.shape[-1] != 3:
            img = img.permute(1, 2, 0)

        if img.dtype != torch.uint8:
            if img.max() <= 1.0:
                img = (img.clamp(0, 1) * 255.0).to(torch.uint8)
            else:
                img = img.clamp(0, 255).to(torch.uint8)

        return Image.fromarray(img.numpy(), mode="RGB")
    
    @torch.inference_mode()
    def get_image_tokens(
        self,
        image,
        *,
        input_is_pil: bool = False,
        normalize: bool = True,
        return_projected: bool = False,
    ):
        if input_is_pil:
            if not isinstance(image, Image.Image):
                raise TypeError("input_is_pil=True but image is not a PIL.Image")
            x = self.clip_preprocess(image).unsqueeze(0)  # CPU tensor float
        else:
            if isinstance(image, np.ndarray):
                x, _meta = self.my_preprocess_np(image, out_size=224)   
            elif torch.is_tensor(image):
                t = image.detach().cpu().numpy()
                x, _meta = self.my_preprocess_np(t, out_size=224)
            else:
                raise TypeError("input_is_pil=False but image must be np.ndarray or torch.Tensor")

        x = x.to(self.device, non_blocking=True)

        feats = {}
        def _hook(_module, _inp, out):
            feats["ln_post"] = out.detach()

        h = self.clip_model.visual.ln_post.register_forward_hook(_hook)
        _ = self.clip_model.encode_image(x)
        h.remove()

        tok = feats["ln_post"]
        cls_token = tok[:, 0, :]
        patch_tokens = tok[:, 1:, :]

        if return_projected and getattr(self.clip_model.visual, "proj", None) is not None:
            proj = self.clip_model.visual.proj
            cls_token = cls_token @ proj
            patch_tokens = patch_tokens @ proj

        if normalize:
            cls_token = F.normalize(cls_token, dim=-1)
            patch_tokens = F.normalize(patch_tokens, dim=-1)

        return cls_token, patch_tokens
    
    def my_preprocess_np(self, img_np: np.ndarray, out_size: int = 224):
        """
        img_np: 640, 480, 3
        """
        if not isinstance(img_np, np.ndarray):
            raise TypeError(f"expect np.ndarray, got {type(img_np)}")

        t = torch.from_numpy(img_np)  # (H,W,3)

        # dtype -> float in [0,1]
        if t.dtype == torch.uint8:
            t = t.float() / 255.0
        else:
            t = t.float()
            if t.max() > 1.5:
                t = t / 255.0
        t = t.clamp(0, 1)

        H, W, C = t.shape  # HWC
        if C != 3:
            raise ValueError(f"expect last dim 3, got {C}")

        t = t.permute(2, 0, 1).unsqueeze(0)  # (1,3,H,W)

        scale = min(out_size / W, out_size / H)
        new_w = int(round(W * scale))
        new_h = int(round(H * scale))

        t = F.interpolate(t, size=(new_h, new_w), mode="bicubic", align_corners=False)

        pad_left = (out_size - new_w) // 2
        pad_right = out_size - new_w - pad_left
        pad_top = (out_size - new_h) // 2
        pad_bottom = out_size - new_h - pad_top

        t = F.pad(t, (pad_left, pad_right, pad_top, pad_bottom), value=0.5)

        mean = torch.tensor([0.5, 0.5, 0.5], dtype=t.dtype).view(1, 3, 1, 1)
        std  = torch.tensor([0.5, 0.5, 0.5], dtype=t.dtype).view(1, 3, 1, 1)
        t = (t - mean) / std  # (1,3,out,out)

        meta = {
            "orig_hw": (H, W),
            "out_size": out_size,
            "scale": float(scale),
            "new_hw": (new_h, new_w),
            "pad_left": int(pad_left),
            "pad_top": int(pad_top),
            "pad_right": int(pad_right),
            "pad_bottom": int(pad_bottom),
        }
        return t, meta



 