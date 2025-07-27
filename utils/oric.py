import os
import json
import random
from typing import List, Dict, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from shapely.geometry import Polygon
from shapely.ops import unary_union
from pycocotools.coco import COCO
from transformers import CLIPModel, CLIPProcessor

from chatbot import Chatbot, DecodingArguments


class ORIC:
    QUESTION_TEMPLATE = [
        "Is there {object} in the image?",
        "Does the image contain {object}?",
        "Have you noticed {object} in the image?",
        "Can you see {object} in the image?",
    ]

    def __init__(
        self,
        coco,
        clip_model,
        clip_processor,
        device: torch,
        image_folder,
        reject_prompt_template,
        decoding_args,
    ):
        self.coco = coco
        self.model = clip_model.to(device)
        self.processor = clip_processor
        self.device = device
        self.image_folder = image_folder

        # load reject‑prompt template
        with open(reject_prompt_template, "r") as f:
            self.reject_template = f.read()

        # initialize ChatBot once
        self.chatbot = ChatBot(decoding_args)

    # -------------------------
    #  Sampling COCO images
    # -------------------------

    def extract_images(self, min_num_objects: int = 2) -> List[Dict]:
        ids = self.coco.getImgIds()
        sampled = []
        for img_id in tqdm(ids, desc="Sampling images"):
            anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=[img_id]))
            cats = {a["category_id"] for a in anns}
            if len(cats) < min_num_objects:
                continue
            info = self.coco.loadImgs([img_id])[0]
            file_name = info["file_name"]
            for a in anns:
                a["category_name"] = self.coco.cats[a["category_id"]]["name"]
                assert a["image_id"] == img_id
            sampled.append(
                {
                    "image_id": img_id,
                    "image_info": info,
                    "image_path": file_name,
                    "annotations": anns,
                }
            )
        return sampled

    # ----------------------------------------
    #  CLIP embeddings & similarity lookup
    # ----------------------------------------

    def precompute_embeddings(
        self, embedding_path: str, batch_size: int = 1000
    ) -> Tuple[torch.Tensor, List[int]]:
        if os.path.exists(embedding_path):
            data = torch.load(embedding_path)
            return data["embeddings"], data["image_ids"]

        img_ids = self.coco.getImgIds()
        infos = self.coco.loadImgs(img_ids)
        all_embs, all_ids = [], []

        for i in tqdm(range(0, len(infos), batch_size), desc="CLIP embed"):
            batch = infos[i : i + batch_size]
            imgs, ids = [], []
            for inf in batch:
                path = os.path.join(self.image_folder, inf["file_name"])
                img = Image.open(path).convert("RGB")
                imgs.append(img)
                ids.append(inf["id"])
            if not imgs:
                continue

            inp = self.processor(images=imgs, return_tensors="pt", padding=True)
            inp = {k: v.to(self.device) for k, v in inp.items()}
            with torch.no_grad():
                emb = self.model.get_image_features(**inp)
                emb = F.normalize(emb, dim=-1).cpu()
            all_embs.append(emb)
            all_ids.extend(ids)

        embeddings = torch.cat(all_embs, dim=0)
        torch.save({"embeddings": embeddings, "image_ids": all_ids}, embedding_path)
        return embeddings, all_ids

    # ----------------------------------------
    #  Extract similar images
    # ----------------------------------------

    def extract_similar_images(
        self,
        sampled: List[Dict],
        embedding_path: str,
        output_path: str,
        batch_size: int = 256,
    ) -> List[Dict]:
        embs, ids = self.precompute_embeddings(embedding_path)
        embs = embs.to(self.device)
        idx_map = {img_id: idx for idx, img_id in enumerate(ids)}
        seen: Set[Tuple[int, int]] = set()
        prog: Dict[int, Dict] = {}

        # load existing results if any
        if os.path.exists(output_path):
            with open(output_path) as f:
                saved = json.load(f)
            for item in saved:
                s, t = item["image_id"], item["similar_images"]["image_id"]
                seen.add(tuple(sorted([s, t])))
                prog[s] = item
            to_do = [x for x in sampled if x["image_id"] not in prog]
        else:
            to_do = sampled

        # batch loop
        for i in tqdm(range(0, len(to_do), batch_size), desc="Find similar"):
            batch = to_do[i : i + batch_size]
            imgs, bids = [], []
            for x in batch:
                p = os.path.join(self.image_folder, x["image_path"])
                im = Image.open(p).convert("RGB")
                inp = self.processor(images=im, return_tensors="pt", padding=True)
                imgs.append(inp["pixel_values"])
                bids.append(x["image_id"])
            if not imgs:
                continue

            pix = torch.cat(imgs, dim=0).to(self.device)
            with torch.no_grad():
                q_emb = F.normalize(
                    self.model.get_image_features(pixel_values=pix), dim=-1
                )

            sims = q_emb @ embs.T  # [B, N]
            dists = 1 - sims  # cosine distance
            for bi, img in enumerate(batch):
                sid = bids[bi]
                if sid in idx_map:
                    dists[bi, idx_map[sid]] = float("inf")
                min_d, min_i = torch.min(dists[bi], dim=0)
                tid = ids[min_i]
                key = tuple(sorted([sid, tid]))
                if key in seen:
                    continue
                seen.add(key)

                info = self.coco.loadImgs([tid])[0]
                ann = self.coco.loadAnns(self.coco.getAnnIds(imgIds=[tid]))
                for a in ann:
                    a["category_name"] = self.coco.cats[a["category_id"]]["name"]
                    assert a["image_id"] == tid

                img["similar_images"] = {
                    "image_id": tid,
                    "image_info": info,
                    "image_path": info["file_name"],
                    "cosine_dist": min_d.item(),
                    "annotations": ann,
                }
                prog[sid] = img

            # save incremental
            with open(output_path, "w") as f:
                json.dump(list(prog.values()), f, indent=4)

        return list(prog.values())

    # ---------------------------------------
    # Positive Q&A via area & reject prompt
    # ---------------------------------------

    @staticmethod
    def _union_area(bboxes: List[List[float]]) -> float:
        # Union area of multiple [x,y,w,h] boxes.
        if not bboxes:
            return 0.0
        polys = [
            Polygon([(x, y), (x + w, y), (x + w, y + h), (x, y + h)])
            for x, y, w, h in bboxes
        ]
        return unary_union(polys).area

    def _classify_by_area(
        self, anns: List[Dict], width: int, height: int
    ) -> Dict[str, List[str]]:
        # Split categories into 'background' (≥median area) vs 'target' (< median).
        total = width * height
        if total <= 0:
            return {"background": [], "target": []}
        by_cat = {}
        for a in anns:
            by_cat.setdefault(a["category_name"], []).append(a["bbox"])
        ratios = {c: self._union_area(bs) / total for c, bs in by_cat.items()}
        if not ratios:
            return {"background": [], "target": []}

        med = np.median(list(ratios.values()))
        out = {"background": [], "target": []}
        for c, r in ratios.items():
            out["background" if r >= med else "target"].append(c)
        return out

    def construct_positive_questions(
        self,
        image_path: str,
        anns: List[Dict],
        width: int,
        height: int,
        num_targets: int,
    ) -> Optional[Dict[str, List[str]]]:
        parts = self._classify_by_area(anns, width, height)
        bg, tg = parts["background"], parts["target"]
        if not bg or not tg:
            return None

        final: List[str] = []
        for obj in tg:
            prompt = self.reject_template.format(
                background_objects=f"[{','.join(bg)}]", target_objects=obj
            )
            resp = self.chatbot.call_model(
                {"text": prompt}, self.decoding_args, return_list=False
            ).lower()
            if "no" in resp:
                final.append(obj)
            if len(final) >= num_targets:
                break

        if len(final) < num_targets:
            return None

        pos_qs = {}
        for obj in final[:num_targets]:
            texts = [
                t.format(object=obj if obj[0] not in "aeiouAEIOU" else f"an {obj}")
                for t in self.QUESTION_TEMPLATE
            ]
            pos_qs[obj] = {"text": texts}
        return pos_qs

    # ----------------------------------------
    # Negative Q&A via CLIP text‑image score
    # ----------------------------------------

    def construct_negative_questions(
        self,
        all_objects: Set[str],
        similar_path: str,
        sim_anns: List[Dict],
        num_targets: int,
        chunk_size: int = 100,
    ) -> Dict[str, Dict]:
        existed = {a["category_name"] for a in sim_anns}
        candidates = list(all_objects - existed)

        # load image once
        img = Image.open(os.path.join(self.image_folder, similar_path)).convert("RGB")

        texts = [f"A image contains {o}" for o in candidates]
        scores: List[float] = []

        for i in range(0, len(texts), chunk_size):
            chunk = texts[i : i + chunk_size]
            inp = self.processor(
                text=chunk, images=img, return_tensors="pt", padding=True
            )
            inp = {k: v.to(self.device) for k, v in inp.items()}
            with torch.no_grad():
                out = self.model(**inp)
            # logits_per_image: [1, len(chunk)]
            probs = out.logits_per_image.softmax(dim=-1).cpu().numpy()[0]
            scores.extend(probs.tolist())

        arr = np.array(scores)
        idx = np.argsort(arr)[:num_targets]
        top20 = {candidates[i]: float(arr[i]) for i in np.argsort(arr)[:20]}

        neg_qs: Dict[str, Dict] = {}
        for i in idx:
            obj = candidates[i]
            texts = [
                t.format(object=obj if obj[0] not in "aeiouAEIOU" else f"an {obj}")
                for t in self.QUESTION_TEMPLATE
            ]
            neg_qs[obj] = {"clip_score": float(arr[i]), "text": texts}
        return neg_qs

    # ---------------------------
    # 5. Assemble final Q&A list
    # ---------------------------

    def extract_QA(
        self, sim_pairs: List[Dict], num_targets: int = 3, max_images: int = 750
    ) -> List[Dict]:
        all_objs = {c["name"] for c in self.coco.dataset["categories"]}
        seen: Set[Tuple[str, str, str]] = set()
        out, qid = [], 1
        limit = max_images * num_targets * 2

        for pair in tqdm(sim_pairs, desc="Building Q&A"):
            if qid > limit:
                break
            img, sim = pair["image_path"], pair["similar_images"]["image_path"]
            w, h = pair["image_info"]["width"], pair["image_info"]["height"]
            anns, sims = pair["annotations"], pair["similar_images"]["annotations"]

            pos = self.construct_positive_questions(
                imgs_path := img, anns=anns, width=w, height=h, num_targets=num_targets
            )
            if not pos:
                continue
            neg = self.construct_negative_questions(
                all_objs, sim, sims, num_targets=num_targets
            )

            # skip duplicates
            skip = False
            for obj in pos:
                if (img, obj, "yes") in seen:
                    skip = True
                    break
            for obj in neg:
                if obj == "top_20_similar_objects":
                    continue
                if (sim, obj, "no") in seen:
                    skip = True
                    break
            if skip:
                continue

            # emit positives
            for obj, item in pos.items():
                seen.add((img, obj, "yes"))
                out.append(
                    {
                        "id": qid,
                        "image": img,
                        "similar_image": sim,
                        "target_object": obj,
                        "question": item["text"],
                        "label": "yes",
                    }
                )
                qid += 1

            # emit negatives
            for obj, item in neg.items():
                if obj == "top_20_similar_objects":
                    continue
                seen.add((sim, obj, "no"))
                out.append(
                    {
                        "id": qid,
                        "image": sim,
                        "similar_image": img,
                        "target_object": obj,
                        "question": item["text"],
                        "label": "no",
                        "clip_score": item["clip_score"],
                        "top_20_similar_objects": neg["top_20_similar_objects"],
                    }
                )
                qid += 1

        return out
