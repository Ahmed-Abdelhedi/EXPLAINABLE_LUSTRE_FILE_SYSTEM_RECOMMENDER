from __future__ import annotations
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from ..semantic.labels import ACCESS_LABELS, HA_LABELS
from ..semantic.model import CategoricalBooleanMultiTaskXLMR
from .train_multitask import MultiTaskDataset, load_jsonl, make_collator

@torch.no_grad()
def predict_split(*, checkpoint_dir: Path, dataset_path: Path, batch_size: int=32):
    from transformers import AutoTokenizer
    config=json.loads((checkpoint_dir/"model_config.json").read_text(encoding="utf-8"))
    tokenizer=AutoTokenizer.from_pretrained(checkpoint_dir/"tokenizer", local_files_only=True)
    model=CategoricalBooleanMultiTaskXLMR(
        base_model_name=config["base_model_name"], dropout=float(config["dropout"])
    )
    model.load_state_dict(torch.load(checkpoint_dir/"model.pt",map_location="cpu",weights_only=True),strict=True)
    device="cuda" if torch.cuda.is_available() else "cpu"
    model.to(device); model.eval()

    records=load_jsonl(dataset_path)
    loader=DataLoader(
        MultiTaskDataset(records,tokenizer,int(config["max_length"])),
        batch_size=batch_size, shuffle=False, collate_fn=make_collator(tokenizer)
    )

    rows=[]; offset=0
    for batch in loader:
        n=int(batch["input_ids"].shape[0])
        mb={k:v.to(device) for k,v in batch.items()}
        out=model(input_ids=mb["input_ids"],attention_mask=mb["attention_mask"])
        hp=torch.softmax(out["ha_logits"],dim=-1).cpu()
        ap=torch.softmax(out["access_logits"],dim=-1).cpu()
        for i in range(n):
            r=records[offset+i]
            rows.append({
                "sample_id":r.sample_id,
                "text":r.text,
                "language":r.language,
                "ha_gold":r.ha_label,
                "access_gold":r.access_label,
                "ha_probabilities":{label:float(hp[i,j]) for j,label in enumerate(HA_LABELS)},
                "access_probabilities":{label:float(ap[i,j]) for j,label in enumerate(ACCESS_LABELS)},
            })
        offset += n
    return rows
