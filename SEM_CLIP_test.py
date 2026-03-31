import os
import csv
from dataclasses import dataclass
from typing import List, Dict, Tuple, Iterable
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import open_clip


# =========================
# USER CONFIG
# =========================
@dataclass
class Config:
    ckpt_path: str = r"sem_clip_classifier.pt"        # path to your trained checkpoint
    input_dir: str = r"/test_images_all/"     # folder containing images
    output_csv: str = r"/sem_clip_results_out.csv"         # where to write CSV
    recursive: bool = True                            # include subfolders
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    valid_exts: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


CONFIG = Config()


# ----------------- Prompt builder (same as training) -----------------
class PromptBuilder:
    def __init__(self):
        self.templates = [
            "a SEM image of {state}.",
            "a dark SEM image of {state}.",
            "a blurry SEM image of {state}.",
            "a high-magnification SEM image of {state}.",
            "a SEM micrograph of {state} for anomaly detection."
        ]

        self.states = {

            "lines": [
                "line morphology",
                "line-shaped ",
                "lamellar structures",
                "linear pattern "
            ],

            "dots": [
                "dots morphology",
                "hexagonally ordered dark spots",
                "inverse perpendicular cylinders",
                "dots structure",
                "spot pattern"
            ],

            "mixed": [
                "mixed line and dot structure",
                "combination of lines and dots",
                "line and dot morphology",
                "mixed region"
            ],
        }

    def class_prompts(self, classes: List[str]) -> Dict[str, List[str]]:
        return {c: [t.format(state=s) for s in self.states[c] for t in self.templates] for c in classes}


# --------- Robust CLIP featurizer (timm & OpenAI ViT supported) ---------
class CLIPFeaturizer(nn.Module):
    """
    Captures token tensors after blocks [2,5,8,11]; each output is [B, 1+T, C].
    Supports:
      - timm-style ViT: visual.trunk.blocks
      - OpenAI-style ViT: visual.transformer.resblocks
    """
    def __init__(self, clip_model: nn.Module):
        super().__init__()
        self.clip = clip_model
        self.visual = clip_model.visual
        self.capture_idx = [2, 5, 8, 11]

        self.mode = None
        if hasattr(self.visual, "trunk") and hasattr(self.visual.trunk, "blocks"):
            self.mode = "timm"
            self.blocks = self.visual.trunk.blocks
            self.patch_embed = self.visual.trunk.patch_embed
            self.cls_token = self.visual.trunk.cls_token
            self.pos_embed = self.visual.trunk.pos_embed
            self.pos_drop = self.visual.trunk.pos_drop
            self.norm_pre = getattr(self.visual.trunk, "norm_pre", None)
            self.norm = self.visual.trunk.norm
        elif hasattr(self.visual, "transformer") and hasattr(self.visual.transformer, "resblocks"):
            self.mode = "openai"
            self.resblocks = self.visual.transformer.resblocks
            self.conv1 = self.visual.conv1
            self.class_embedding = self.visual.class_embedding
            self.positional_embedding = getattr(self.visual, "positional_embedding", None)
            self.ln_pre = getattr(self.visual, "ln_pre", None)
            self.ln_post = getattr(self.visual, "ln_post", None)
        else:
            raise AssertionError("Unsupported CLIP visual backbone (need ViT).")

        for p in self.clip.parameters():
            p.requires_grad_(False)

    def _img_dtype(self):
        if self.mode == "timm":
            proj = getattr(self.patch_embed, "proj", None)
            if proj is not None and hasattr(proj, "weight"):
                return proj.weight.dtype
            return torch.float32
        else:
            return self.conv1.weight.dtype

    def forward(self, images: torch.Tensor):
        if self.mode == "timm":
            images = images.to(dtype=self._img_dtype())
            x = self.patch_embed(images)                     # [B, T, C]
            B = x.shape[0]
            cls_tok = self.cls_token.expand(B, -1, -1)       # [B,1,C]
            x = torch.cat([cls_tok, x], dim=1)               # [B,1+T,C]
            x = x + self.pos_embed
            x = self.pos_drop(x)
            if self.norm_pre is not None:
                x = self.norm_pre(x)
            outs = []
            for i, blk in enumerate(self.blocks):
                x = blk(x)
                if i in self.capture_idx:
                    outs.append(x.clone())
            outs = [self.norm(o) for o in outs]
            return outs

        # OpenAI-style path
        images = images.to(dtype=self._img_dtype())
        x = self.conv1(images)                               # [B,C',H',W']
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)  # [B,T,C]
        cls = self.class_embedding.to(x.dtype).expand(x.shape[0], 1, -1)
        x = torch.cat([cls, x], dim=1)                       # [B,1+T,C]
        if self.positional_embedding is not None:
            x = x + self.positional_embedding.to(x.dtype)
        if self.ln_pre is not None:
            x = self.ln_pre(x)
        outs = []
        for i, blk in enumerate(self.resblocks):
            x = blk(x)
            if i in self.capture_idx:
                outs.append(x.clone())
        if self.ln_post is not None:
            outs = [self.ln_post(o) for o in outs]
        return outs


# ------------- Classifier head (same as training, robust dim inference) -------------
class SemClipClassifier(nn.Module):
    def __init__(self, clip_model: nn.Module, classnames: List[str], alpha: float = 0.8):
        super().__init__()
        self.clip = clip_model
        self.classnames = classnames
        self.alpha = alpha

        v = self.clip.visual

        def _get_image_hw(visual):
            s = getattr(visual, "image_size", None) or getattr(visual, "input_resolution", None) or 224
            if isinstance(s, int):
                return s, s
            if isinstance(s, (tuple, list)) and len(s) >= 2:
                return int(s[0]), int(s[1])
            return 224, 224

        # Infer token width from several sources; probe only if needed
        if hasattr(v, "trunk") and hasattr(v.trunk, "embed_dim"):
            img_width = int(v.trunk.embed_dim)
        elif hasattr(v, "embed_dim"):
            img_width = int(v.embed_dim)
        elif hasattr(v, "width"):
            img_width = int(v.width)
        elif hasattr(v, "conv1") and hasattr(v.conv1, "out_channels"):
            img_width = int(v.conv1.out_channels)
        elif hasattr(v, "patch_embed") and hasattr(getattr(v.patch_embed, "proj", None), "out_channels"):
            img_width = int(v.patch_embed.proj.out_channels)
        else:
            H, W = _get_image_hw(v)
            device = next(v.parameters()).device
            dtype = next(v.parameters()).dtype
            dummy = torch.zeros((1, 3, H, W), device=device, dtype=dtype)
            with torch.no_grad():
                tokens = CLIPFeaturizer(self.clip)(dummy)
            img_width = int(tokens[0].shape[-1])
        self.img_width = img_width

        tp = getattr(self.clip, "text_projection", None)
        self.txt_width = int(tp.shape[-1]) if tp is not None else int(getattr(self.clip, "embed_dim", 512))

        self.m = 4
        self.transform = nn.Linear(self.img_width, self.txt_width, bias=True)
        self.cls_head = nn.Sequential(
            nn.LayerNorm(self.m * self.img_width),
            nn.Dropout(0.1),
            nn.Linear(self.m * self.img_width, len(classnames)),
        )

        for p in self.clip.parameters():
            p.requires_grad_(False)

    def build_text_embeddings(self, tokenizer, prompts_by_class: Dict[str, List[str]], device):
        class_embeds = []
        for cname in self.classnames:
            texts = prompts_by_class[cname]
            tok = tokenizer(texts).to(device)
            with torch.no_grad():
                txt = self.clip.encode_text(tok)        # [P, D]
                txt = F.normalize(txt, dim=-1)
            e = F.normalize(txt.mean(dim=0, keepdim=True), dim=-1)  # [1, D]
            class_embeds.append(e)
        return torch.cat(class_embeds, dim=0)            # [K, D]

    def forward_sem_logits(self, stage_tokens: List[torch.Tensor], text_emb: torch.Tensor):
        sims_per_stage = []
        for x in stage_tokens:
            patches = x[:, 1:, :]                 # [B, T, C]
            patches = self.transform(patches)     # [B, T, D]
            patches = F.normalize(patches, dim=-1)
            sims = patches @ text_emb.t()         # [B, T, K]
            sims_per_stage.append(sims.mean(dim=1))
        sims_stack = torch.stack(sims_per_stage, dim=0)   # [m, B, K]
        return sims_stack.max(dim=0).values               # [B, K]

    def forward_cls_logits(self, stage_tokens: List[torch.Tensor]):
        cls_list = [x[:, 0, :] for x in stage_tokens]     # [B, C] x 4
        concat = torch.cat(cls_list, dim=-1)              # [B, 4*C]
        return self.cls_head(concat)                      # [B, K]

    def forward_mixed_probs(self, stage_tokens: List[torch.Tensor], text_emb: torch.Tensor):
        logits_S = self.forward_sem_logits(stage_tokens, text_emb)
        logits_C = self.forward_cls_logits(stage_tokens)
        P_S = F.softmax(logits_S, dim=-1)
        P_C = F.softmax(logits_C, dim=-1)
        return (1.0 - self.alpha) * P_S + self.alpha * P_C


# ---------------------- Loader & inference helpers ----------------------
@torch.no_grad()
def load_model_and_text(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    classes = ckpt["classes"]
    alpha = ckpt.get("alpha", 0.8)
    model_name = ckpt.get("model_name", "ViT-B-16")
    pretrained = ckpt.get("pretrained", "laion2b_s34b_b88k")
    saved_prompts = ckpt.get("prompts", None)

    clip_model, _, preprocess = open_clip.create_model_and_transforms(model_name=model_name, pretrained=pretrained)
    clip_model.to(device).eval()
    tokenizer = open_clip.get_tokenizer(model_name)

    if saved_prompts is None:
        prompts = PromptBuilder().class_prompts(classes)
    else:
        prompts = saved_prompts

    clf = SemClipClassifier(clip_model, classes, alpha=alpha).to(device).eval()
    clf.load_state_dict(ckpt["state_dict"], strict=True)

    text_emb = clf.build_text_embeddings(tokenizer, prompts, device)
    feat = CLIPFeaturizer(clip_model).to(device).eval()
    return feat, clf, preprocess, classes, text_emb


def iter_images(root: str, exts: Tuple[str, ...], recursive: bool = True) -> Iterable[str]:
    exts_lower = tuple(e.lower() for e in exts)

    if recursive:
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                if fn.lower().endswith(exts_lower):
                    yield os.path.join(dirpath, fn)
    else:
        for fn in os.listdir(root):
            p = os.path.join(root, fn)
            if os.path.isfile(p) and fn.lower().endswith(exts_lower):
                yield p


@torch.no_grad()
def classify_one(path: str, feat: CLIPFeaturizer, clf: SemClipClassifier, preprocess, text_emb, device) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    img_t = preprocess(img).unsqueeze(0).to(device)
    stages = feat(img_t)                                # list of 4 [1,1+T,C]
    probs = clf.forward_mixed_probs(stages, text_emb)   # [1, K]
    return probs.squeeze(0).detach().cpu()              # [K]


def main(cfg: Config):
    device = torch.device(cfg.device)
    feat, clf, preprocess, classes, text_emb = load_model_and_text(cfg.ckpt_path, device)

    os.makedirs(os.path.dirname(cfg.output_csv) or ".", exist_ok=True)

    # CSV header: image_path, predicted_class, predicted_prob, then one column per class (in checkpoint order)
    header = ["image_path", "predicted_class", "predicted_prob"] + [f"prob_{c}" for c in classes]

    total, ok, fail = 0, 0, 0
    with open(cfg.output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for path in iter_images(cfg.input_dir, cfg.valid_exts, recursive=cfg.recursive):
            total += 1
            try:
                probs = classify_one(path, feat, clf, preprocess, text_emb, device)
                top_idx = int(torch.argmax(probs).item())
                top_class = classes[top_idx]
                top_prob = float(probs[top_idx].item())
                row = [path, top_class, f"{top_prob:.6f}"] + [f"{float(p):.6f}" for p in probs.tolist()]
                writer.writerow(row)
                ok += 1
            except Exception as e:
                # Write a row with error info (optional); or skip entirely.
                writer.writerow([path, "ERROR", "0.0"] + [""] * len(classes))
                print(f"[WARN] Failed on {path}: {e}")
                fail += 1

    print(f"Done. Wrote: {cfg.output_csv}")
    print(f"Images processed: {total} | ok: {ok} | failed: {fail}")
    print("Class order in CSV:", classes)


if __name__ == "__main__":
    main(CONFIG)
