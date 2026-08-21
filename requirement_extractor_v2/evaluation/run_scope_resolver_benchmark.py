from __future__ import annotations
import argparse,json
from pathlib import Path
from requirement_extractor_v2.conversation_scope_resolver import ConversationScopeResolver
from requirement_extractor_v2.models import ParamName
LABELS=["NEW_REQUIREMENT","ANSWER_TO_PREVIOUS_QUESTION","CORRECTION","OUT_OF_SCOPE"]
def load(p): return [json.loads(x) for x in Path(p).read_text(encoding="utf-8").splitlines() if x.strip()]
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--dataset",required=True);ap.add_argument("--output",default="scope_resolver_metrics.json");a=ap.parse_args();rows=load(a.dataset);rsv=ConversationScopeResolver();M={g:{p:0 for p in LABELS} for g in LABELS};details=[]
    for r in rows:
        c=r.get("context") or {};pf=ParamName(c["previous_question_field"]) if c.get("previous_question_field") else None;res=rsv.resolve(r["text"],previous_question_field=pf,requested_unit=c.get("requested_unit"),previous_question=c.get("previous_question"));g=r["expected_scope"];p=res.intent.value
        if g in M and p in LABELS:M[g][p]+=1
        details.append({"id":r["id"],"category":r["category"],"text":r["text"],"gold":g,"predicted":res.to_dict(),"correct":g==p})
    per={};total=sum(sum(x.values()) for x in M.values());correct=sum(M[l][l] for l in LABELS)
    for l in LABELS:
        tp=M[l][l];fp=sum(M[g][l] for g in LABELS if g!=l);fn=sum(M[l][p] for p in LABELS if p!=l);P=tp/(tp+fp) if tp+fp else 0;R=tp/(tp+fn) if tp+fn else 0;F=2*P*R/(P+R) if P+R else 0;per[l]={"precision":P,"recall":R,"f1":F,"support":sum(M[l].values())}
    metrics={"n_messages":total,"accuracy":correct/total if total else 0,"macro_f1":sum(x["f1"] for x in per.values())/len(LABELS),"per_class":per,"confusion_matrix":M}
    Path(a.output).write_text(json.dumps({"metrics":metrics,"details":details},ensure_ascii=False,indent=2),encoding="utf-8");print(json.dumps(metrics,ensure_ascii=False,indent=2))
if __name__=="__main__":main()
