from __future__ import annotations
import argparse, json, math, random
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Dataset
from ..semantic.labels import ACCESS_TO_ID, HA_TO_ID
from ..semantic.model import CategoricalBooleanMultiTaskXLMR
from .dataset_schema import TrainingRecord

def seed_everything(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(TrainingRecord.from_dict(json.loads(line)))
    return rows

class MultiTaskDataset(Dataset):
    def __init__(self, records, tokenizer, max_length: int):
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = int(max_length)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        r = self.records[index]
        e = self.tokenizer(
            r.text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )
        return {
            "input_ids": e["input_ids"],
            "attention_mask": e["attention_mask"],
            "ha_label": HA_TO_ID[r.ha_label],
            "access_label": ACCESS_TO_ID[r.access_label],
        }

def make_collator(tokenizer):
    def collate(batch):
        padded = tokenizer.pad(
            {
                "input_ids": [x["input_ids"] for x in batch],
                "attention_mask": [x["attention_mask"] for x in batch],
            },
            return_tensors="pt",
        )
        padded["ha_labels"] = torch.tensor([x["ha_label"] for x in batch], dtype=torch.long)
        padded["access_labels"] = torch.tensor([x["access_label"] for x in batch], dtype=torch.long)
        return padded
    return collate

def macro_f1(gold, pred, num_classes: int) -> float:
    scores = []
    for class_id in range(num_classes):
        tp = sum(1 for y,p in zip(gold,pred) if y==class_id and p==class_id)
        fp = sum(1 for y,p in zip(gold,pred) if y!=class_id and p==class_id)
        fn = sum(1 for y,p in zip(gold,pred) if y==class_id and p!=class_id)
        precision = tp/(tp+fp) if tp+fp else 0.0
        recall = tp/(tp+fn) if tp+fn else 0.0
        scores.append(2*precision*recall/(precision+recall) if precision+recall else 0.0)
    return sum(scores)/len(scores)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ha_gold=[]; ha_pred=[]; ac_gold=[]; ac_pred=[]; losses=[]
    for batch in loader:
        batch = {k:v.to(device) for k,v in batch.items()}
        out = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            ha_labels=batch["ha_labels"],
            access_labels=batch["access_labels"],
        )
        losses.append(float(out["loss"].item()))
        ha_gold.extend(batch["ha_labels"].cpu().tolist())
        ac_gold.extend(batch["access_labels"].cpu().tolist())
        ha_pred.extend(out["ha_logits"].argmax(-1).cpu().tolist())
        ac_pred.extend(out["access_logits"].argmax(-1).cpu().tolist())
    ha_f1 = macro_f1(ha_gold, ha_pred, 4)
    ac_f1 = macro_f1(ac_gold, ac_pred, 4)
    return {
        "loss": sum(losses)/max(len(losses),1),
        "ha_macro_f1": ha_f1,
        "access_macro_f1": ac_f1,
        "composite_macro_f1": (ha_f1+ac_f1)/2.0,
    }

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--train", required=True)
    p.add_argument("--validation", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--base-model", default="FacebookAI/xlm-roberta-base")
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--gradient-accumulation", type=int, default=1)
    p.add_argument("--learning-rate", type=float, default=2e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=20260826)
    args=p.parse_args()

    seed_everything(args.seed)
    out_dir=Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer, get_linear_schedule_with_warmup
    tokenizer=AutoTokenizer.from_pretrained(args.base_model)
    train_records=load_jsonl(Path(args.train))
    val_records=load_jsonl(Path(args.validation))
    collator=make_collator(tokenizer)
    train_loader=DataLoader(
        MultiTaskDataset(train_records,tokenizer,args.max_length),
        batch_size=args.batch_size, shuffle=True, collate_fn=collator
    )
    val_loader=DataLoader(
        MultiTaskDataset(val_records,tokenizer,args.max_length),
        batch_size=args.batch_size, shuffle=False, collate_fn=collator
    )

    device="cuda" if torch.cuda.is_available() else "cpu"
    model=CategoricalBooleanMultiTaskXLMR(
        base_model_name=args.base_model,
        dropout=args.dropout,
    ).to(device)

    optimizer=torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    steps_per_epoch=math.ceil(len(train_loader)/args.gradient_accumulation)
    total_steps=steps_per_epoch*args.epochs
    scheduler=get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )
    scaler=torch.amp.GradScaler("cuda", enabled=(device=="cuda"))

    history=[]; best_score=-1.0; best_epoch=None
    for epoch in range(1,args.epochs+1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss=0.0
        for step,batch in enumerate(train_loader,1):
            batch={k:v.to(device) for k,v in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device=="cuda")):
                out=model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    ha_labels=batch["ha_labels"],
                    access_labels=batch["access_labels"],
                )
                loss=out["loss"]/args.gradient_accumulation
            scaler.scale(loss).backward()
            running_loss += float(loss.item())
            if step % args.gradient_accumulation == 0 or step == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

        metrics=evaluate(model,val_loader,device)
        metrics["epoch"]=epoch
        metrics["train_loss_accumulated"]=running_loss
        history.append(metrics)
        print(json.dumps(metrics,indent=2))

        if metrics["composite_macro_f1"] > best_score:
            best_score=metrics["composite_macro_f1"]; best_epoch=epoch
            torch.save(model.state_dict(), out_dir/"model.pt")
            tokenizer.save_pretrained(out_dir/"tokenizer")
            (out_dir/"model_config.json").write_text(json.dumps({
                "architecture":"CategoricalBooleanMultiTaskXLMR",
                "base_model_name":args.base_model,
                "dropout":args.dropout,
                "max_length":args.max_length,
                "ha_num_labels":4,
                "access_num_labels":4,
                "loss":"CE_HA + CE_ACCESS",
                "checkpoint_selection":"mean(HA_macro_F1, Access_macro_F1)",
                "seed":args.seed,
            },indent=2)+"\n",encoding="utf-8")
            (out_dir/"best_validation_metrics.json").write_text(
                json.dumps(metrics,indent=2)+"\n",encoding="utf-8"
            )

    metadata={
        "status":"TRAINING_COMPLETE_BEST_VALIDATION_CHECKPOINT_SAVED",
        "epochs_run":args.epochs,
        "best_epoch":best_epoch,
        "best_composite_macro_f1":best_score,
        "test_used":False,
        "final_holdout_used":False,
    }
    (out_dir/"training_metadata.json").write_text(json.dumps(metadata,indent=2)+"\n",encoding="utf-8")
    (out_dir/"training_history.json").write_text(json.dumps(history,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(metadata,indent=2))

if __name__=="__main__":
    main()
