from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

REQUIRED=[
    "model.pt","model_config.json","calibration.json",
    "best_validation_metrics.json","training_metadata.json",
]

def sha256(path: Path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--checkpoint-dir",required=True)
    p.add_argument("--output",required=True)
    args=p.parse_args()
    checkpoint=Path(args.checkpoint_dir); output=Path(args.output)
    missing=[name for name in REQUIRED if not (checkpoint/name).exists()]
    if not (checkpoint/"tokenizer").is_dir():
        missing.append("tokenizer/")
    if missing:
        raise FileNotFoundError("Missing: "+", ".join(missing))
    if output.exists(): output.unlink()
    included=[]
    with ZipFile(output,"w",ZIP_DEFLATED) as z:
        for name in REQUIRED:
            z.write(checkpoint/name,name); included.append(name)
        for path in sorted((checkpoint/"tokenizer").rglob("*")):
            if path.is_file():
                rel=Path("tokenizer")/path.relative_to(checkpoint/"tokenizer")
                z.write(path,rel); included.append(str(rel))
    print(json.dumps({
        "artifact":str(output),
        "sha256":sha256(output),
        "included_files":included,
        "test_metrics_in_artifact":False,
        "final_holdout_metrics_in_artifact":False,
    },indent=2))

if __name__=="__main__":
    main()
