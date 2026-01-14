# sem_clip_pycharm.py
# Requires: torch>=2.0, torchvision, timm, open_clip_torch, pillow

import os
from dataclasses import dataclass
from typing import List, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import open_clip


# ==========
# CONFIG
# ==========
@dataclass
class Config:
    data_root: str = r"data"  # <-- set to your dataset root (see folder layout below)
    save_path: str = "sem_clip_classifier.pt"
    model_name: str = "ViT-B-16"
    pretrained: str = "laion2b_s34b_b88k"

    # Training
    epochs: int = 20
    batch_size: int = 8
    lr: float = 3e-4
    weight_decay: float = 1e-4
    alpha: float = 0.8  # mixture weight between P_S and P_C (Eq. 23)
    num_workers: int = 0  # Windows/PyCharm friendly; bump to 2-4 on Linux

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


CONFIG = Config(
    data_root=r"/home/oren/Documents/Models/Order_phase/CLIP_class/test4/data/",  #
    epochs=20,
    batch_size=8,
    lr=3e-4,
    alpha=0.8,
    num_workers=0,
    device="cuda" if torch.cuda.is_available() else "cpu",
)

# Expected layout:
# data/
#   train/
#     lines/  *.png|jpg
#     dots/   *.png|jpg
#     mixed/  *.png|jpg
#   val/
#     lines/  *.png|jpg
#     dots/   *.png|jpg
#     mixed/  *.png|jpg


# ----------------------------
# Prompt builder
# ----------------------------
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
            # 'lines' ≈ scratches/linear marks (paper uses scratch as example)
            "lines": [
                "line morphology",
                "line-shaped ",
                "lamellar structures",
                "linear pattern "
            ],
            # 'dots' ≈ particles/spot-like (paper references particulate matter)
            "dots": [
                "dots morphology",
                "hexagonally ordered dark spots",
                "inverse perpendicular cylinders",
                "dots structure",
                "spot pattern"
            ],
            # 'mixed' ≈ co-occurrence of both morphologies
            "mixed": [
                "mixed line and dot structure",
                "combination of lines and dots",
                "line and dot morphology",
                "mixed region"
            ],
        }

    def prompts_for_class(self, cls_name: str) -> List[str]:
        out = []
        for s in self.states[cls_name]:
            for t in self.templates:
                out.append(t.format(state=s))
        return out

    def class_prompts(self, classes: List[str]) -> Dict[str, List[str]]:
        return {c: self.prompts_for_class(c) for c in classes}


# ----------------------------
# CLIP multi-level feature extractor
# ----------------------------
class CLIPFeaturizer(nn.Module):
    """
    Works with both timm-style ViT (visual.trunk.blocks) and OpenAI-style ViT (visual.transformer.resblocks).
    Captures tokens after blocks [2,5,8,11]; returns 4 tensors, each [B, 1+T, C].
    """
    def __init__(self, clip_model: nn.Module):
        super().__init__()
        self.clip = clip_model
        self.visual = clip_model.visual
        self.capture_idx = [2, 5, 8, 11]

        # Detect layout
        self.mode = None
        # timm wrapper (common in open_clip)
        if hasattr(self.visual, "trunk") and hasattr(self.visual.trunk, "blocks"):
            self.mode = "timm"
            self.blocks = self.visual.trunk.blocks
            self.patch_embed = self.visual.trunk.patch_embed
            self.cls_token = self.visual.trunk.cls_token
            self.pos_embed = self.visual.trunk.pos_embed
            self.pos_drop = self.visual.trunk.pos_drop
            self.norm_pre = getattr(self.visual.trunk, "norm_pre", None)
            self.norm = self.visual.trunk.norm
        # OpenAI-style CLIP ViT (no timm trunk)
        elif hasattr(self.visual, "transformer") and hasattr(self.visual.transformer, "resblocks"):
            self.mode = "openai"
            self.resblocks = self.visual.transformer.resblocks
            # names used by open_clip’s OpenAI-style ViT
            self.conv1 = self.visual.conv1
            self.class_embedding = self.visual.class_embedding
            # positional embedding name differs from timm
            self.positional_embedding = getattr(self.visual, "positional_embedding", None)
            self.ln_pre = getattr(self.visual, "ln_pre", None)
            self.ln_post = getattr(self.visual, "ln_post", None)
        else:
            raise AssertionError(
                "Unsupported CLIP visual backbone: expected ViT with either "
                "`visual.trunk.blocks` (timm) or `visual.transformer.resblocks` (OpenAI)."
            )

        # Freeze CLIP
        for p in self.clip.parameters():
            p.requires_grad_(False)

    def _dtype_for_images(self):
        if self.mode == "timm":
            try:
                return self.patch_embed.proj.weight.dtype
            except Exception:
                return torch.float32
        else:
            try:
                return self.conv1.weight.dtype
            except Exception:
                return torch.float16 if next(self.visual.parameters()).dtype == torch.float16 else torch.float32

    def forward(self, images: torch.Tensor) -> List[torch.Tensor]:
        if self.mode == "timm":
            x = self.patch_embed(images)                             # [B, T, C]
            B, T, C = x.shape
            cls_tok = self.cls_token.expand(B, -1, -1)               # [B, 1, C]
            x = torch.cat([cls_tok, x], dim=1)                       # [B, 1+T, C]
            x = x + self.pos_embed
            x = self.pos_drop(x)
            if self.norm_pre is not None:
                x = self.norm_pre(x)

            outs = []
            for i, blk in enumerate(self.blocks):
                x = blk(x)
                if i in self.capture_idx:
                    outs.append(x.clone())
            outs = [self.norm(o) for o in outs]                      # final norm like ViT forward_features
            return outs

        # OpenAI-style forward
        dtype = self._dtype_for_images()
        x = images.to(dtype=dtype)
        # conv1 -> patchify
        x = self.conv1(x)                                            # [B, C', H', W']
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)   # [B, T, C]
        # prepend CLS
        cls = self.class_embedding.to(x.dtype).expand(x.shape[0], 1, -1)  # [B,1,C]
        x = torch.cat([cls, x], dim=1)                               # [B, 1+T, C]
        # positional embedding (if present)
        if self.positional_embedding is not None:
            x = x + self.positional_embedding.to(x.dtype)
        # ln_pre (if present)
        if self.ln_pre is not None:
            x = self.ln_pre(x)

        outs = []
        for i, blk in enumerate(self.resblocks):
            x = blk(x)
            if i in self.capture_idx:
                outs.append(x.clone())

        # ln_post for consistency
        if self.ln_post is not None:
            outs = [self.ln_post(o) for o in outs]
        return outs


# ----------------------------
# SEM-CLIP Classifier
# ----------------------------
class SemClipClassifier(nn.Module):
    def __init__(self, clip_model: nn.Module, classnames: List[str], alpha: float = 0.8):
        super().__init__()
        self.clip = clip_model
        self.classnames = classnames
        self.alpha = alpha

        v = self.clip.visual

        # ---------- Helper to get (H, W) safely ----------
        def _get_image_hw(visual):
            s = getattr(visual, "image_size", None)
            if s is None:
                s = getattr(visual, "input_resolution", None)  # some OpenAI CLIP expose this
            if s is None:
                return 224, 224
            if isinstance(s, int):
                return int(s), int(s)
            if isinstance(s, (tuple, list)):
                if len(s) == 2:
                    return int(s[0]), int(s[1])
                # defensive fallback: use first as both
                return int(s[0]), int(s[0])
            # last resort
            return 224, 224

        # ---------- Infer image token width robustly ----------
        img_width = None
        # timm-style ViT in open_clip
        if hasattr(v, "trunk") and hasattr(v.trunk, "embed_dim"):
            img_width = int(v.trunk.embed_dim)
        # some variants expose embed_dim directly
        elif hasattr(v, "embed_dim"):
            img_width = int(v.embed_dim)
        # OpenAI-style ViT exposes "width"
        elif hasattr(v, "width"):
            img_width = int(v.width)
        # OpenAI-style patchifier conv: out_channels == token width
        elif hasattr(v, "conv1") and hasattr(v.conv1, "out_channels"):
            img_width = int(v.conv1.out_channels)
        # timm-style patch embed conv
        elif hasattr(v, "patch_embed") and hasattr(getattr(v.patch_embed, "proj", None), "out_channels"):
            img_width = int(v.patch_embed.proj.out_channels)
        else:
            # Careful probe with a dummy image; handle tuple image_size correctly
            try:
                H, W = _get_image_hw(v)
                device = next(v.parameters()).device
                dtype = next(v.parameters()).dtype
                dummy = torch.zeros((1, 3, int(H), int(W)), device=device, dtype=dtype)
                with torch.no_grad():
                    # Use the featurizer to get token shape; it does not depend on this classifier
                    tokens = CLIPFeaturizer(self.clip)(dummy)
                img_width = int(tokens[0].shape[-1])
            except Exception as e:
                raise RuntimeError("Could not infer image token width from CLIP visual.") from e

        self.img_width = img_width

        # ---------- Text projection dim (embed dim) ----------
        tp = getattr(self.clip, "text_projection", None)
        if tp is None:
            # fallback: many models also expose clip_model.embed_dim
            self.txt_width = int(getattr(self.clip, "embed_dim", 512))
        else:
            # tp can be [width, embed_dim] or a 1-D vector
            self.txt_width = int(tp.shape[-1])

        # ---------- Heads ----------
        self.transform = nn.Linear(self.img_width, self.txt_width, bias=True)
        self.m = 4
        self.cls_head = nn.Sequential(
            nn.LayerNorm(self.m * self.img_width),
            nn.Dropout(0.1),
            nn.Linear(self.m * self.img_width, len(classnames)),
        )

        # Freeze CLIP
        for p in self.clip.parameters():
            p.requires_grad_(False)


    def build_text_embeddings(self, tokenizer, prompts_by_class: Dict[str, List[str]], device):
        class_embeds = []
        for cname in self.classnames:
            texts = prompts_by_class[cname]
            tok = tokenizer(texts).to(device)
            with torch.no_grad():
                txt = self.clip.encode_text(tok)  # [P, D], already projected
                txt = F.normalize(txt, dim=-1)
            e = F.normalize(txt.mean(dim=0, keepdim=True), dim=-1)  # [1, D]
            class_embeds.append(e)
        return torch.cat(class_embeds, dim=0)  # [K, D]

    def forward_sem_logits(self, stage_tokens: List[torch.Tensor], text_emb: torch.Tensor) -> torch.Tensor:
        sims_per_stage = []
        for x in stage_tokens:
            patches = x[:, 1:, :]                 # [B, T, C]
            patches = self.transform(patches)     # [B, T, D]
            patches = F.normalize(patches, dim=-1)
            sims = patches @ text_emb.t()         # [B, T, K]
            stage_score = sims.mean(dim=1)        # [B, K]
            sims_per_stage.append(stage_score)
        sims_stack = torch.stack(sims_per_stage, dim=0)  # [m, B, K]
        return sims_stack.max(dim=0).values              # [B, K]

    def forward_cls_logits(self, stage_tokens: List[torch.Tensor]) -> torch.Tensor:
        cls_list = [x[:, 0, :] for x in stage_tokens]    # [B, C] × 4
        concat = torch.cat(cls_list, dim=-1)             # [B, 4*C]
        return self.cls_head(concat)                     # [B, K]

    def forward_mixed_probs(self, stage_tokens: List[torch.Tensor], text_emb: torch.Tensor):
        logits_S = self.forward_sem_logits(stage_tokens, text_emb)
        logits_C = self.forward_cls_logits(stage_tokens)
        P_S = F.softmax(logits_S, dim=-1)
        P_C = F.softmax(logits_C, dim=-1)
        probs = (1.0 - self.alpha) * P_S + self.alpha * P_C
        return probs, {"logits_S": logits_S, "logits_C": logits_C, "P_S": P_S, "P_C": P_C}


# ----------------------------
# Training utils
# ----------------------------
def build_dataloaders(data_root: str, preprocess, batch_size=8, num_workers=0):
    # Ensure grayscale images are treated as RGB (CLIP expects 3 channels)
    preprocess_rgb = transforms.Compose(
        [transforms.Lambda(lambda im: im.convert("RGB"))] + list(preprocess.transforms)
    )
    train_ds = datasets.ImageFolder(os.path.join(data_root, "train"), transform=preprocess_rgb)
    val_ds = datasets.ImageFolder(os.path.join(data_root, "val"), transform=preprocess_rgb)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, train_ds.classes


def mixture_nll_loss(probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    eps = 1e-8
    pick = probs.gather(1, targets.view(-1, 1)).clamp_min(eps)  # [B,1]
    return -torch.log(pick).mean()


@torch.no_grad()
def evaluate(featurizer: CLIPFeaturizer,
             model: SemClipClassifier,
             text_emb: torch.Tensor,
             val_loader: DataLoader,
             device: torch.device):
    featurizer.eval()
    model.eval()
    correct, total = 0, 0
    for imgs, y in val_loader:
        imgs = imgs.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        stages = featurizer(imgs)
        probs, _ = model.forward_mixed_probs(stages, text_emb)
        pred = probs.argmax(dim=-1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return {"acc": correct / max(total, 1)}


def train_in_pycharm(cfg: Config):
    device = torch.device(cfg.device)
    print(f"Using device: {device}")

    # 1) CLIP backbone & preprocess
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        model_name=cfg.model_name,
        pretrained=cfg.pretrained
    )
    clip_model.to(device).eval()
    tokenizer = open_clip.get_tokenizer(cfg.model_name)

    # 2) Data
    train_loader, val_loader, classes = build_dataloaders(
        cfg.data_root, preprocess, batch_size=cfg.batch_size, num_workers=cfg.num_workers
    )
    expected = {"lines", "dots", "mixed"}
    if set(classes) != expected:
        raise ValueError(f"Found classes {classes}; expected exactly {sorted(list(expected))}")

    # 3) Prompts -> text embeddings
    pb = PromptBuilder()
    prompts = pb.class_prompts(classes)

    # 4) Models
    featurizer = CLIPFeaturizer(clip_model).to(device)
    model = SemClipClassifier(clip_model, classes, alpha=cfg.alpha).to(device)

    # 5) Optimizer (only train transformation + cls head)
    params = list(model.transform.parameters()) + list(model.cls_head.parameters())
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    # 6) Text embeddings
    text_emb = model.build_text_embeddings(tokenizer, prompts, device)

    # 7) Train loop
    best = 0.0
    for epoch in range(1, cfg.epochs + 1):
        featurizer.train(False)
        model.train(True)

        running = 0.0
        for imgs, y in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            stages = featurizer(imgs)  # list of 4 [B,1+T,C]
            probs, _ = model.forward_mixed_probs(stages, text_emb)
            loss = mixture_nll_loss(probs, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            running += loss.item() * imgs.size(0)

        train_loss = running / max(len(train_loader.dataset), 1)
        metrics = evaluate(featurizer, model, text_emb, val_loader, device)
        acc = metrics["acc"]
        print(f"Epoch {epoch:03d} | train NLL {train_loss:.4f} | val acc {acc:.4f}")

        if acc > best:
            best = acc
            torch.save({
                "state_dict": model.state_dict(),
                "classes": classes,
                "alpha": cfg.alpha,
                "model_name": cfg.model_name,
                "pretrained": cfg.pretrained,
                "prompts": prompts,
            }, cfg.save_path)
            print(f"  ✓ Saved best model to {cfg.save_path} (acc={best:.4f})")

    print(f"Best val acc: {best:.4f}")


if __name__ == "__main__":
    # Tip: In PyCharm you can set breakpoints, inspect variables, etc.
    train_in_pycharm(CONFIG)
