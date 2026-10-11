from pathlib import Path
import sys,json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.audit import *
import pandas as pd

frame=pd.read_csv(ROOT/'results/token_counts.csv');rows=[];detail=[];front=[]
for tokenizer in ['cl100k_base','o200k_base','qwen2.5']:
    def matrix(split,column):
        return frame[frame.split==split].pivot(index='id',columns='language',values=column)[LANGS].to_numpy()
    c=matrix('devtest',tokenizer);h=matrix('devtest','chars4');by=matrix('devtest','bytes');dev=matrix('dev',tokenizer)
    for fraction in [.25,.5,.75]:
        B=int(fraction*c[:,0].sum());equal=np.repeat(B,len(LANGS));weighted=allocate(B*len(LANGS),dev.mean(axis=0))
        policies={'chars4':(h,equal),'exact':(c,equal),'bytes':(by,equal),'calibrated':(c,weighted)}
        for seed in range(100,130):
            order=np.random.default_rng(seed).permutation(len(c))
            front.append(dict(tokenizer=tokenizer,fraction=fraction,seed=seed,prefix=prefix_frontier(c,B*12,order)/len(c),
                              cherry=count_frontier(c,B*12)/len(c)))
            for batch in [1,8,64,len(c)]:
                if batch in [8,64] and fraction!=.5:continue
                for policy,(reservation,budget) in policies.items():
                    a,u=replay(c,reservation,budget,order,batch)
                    coverage=a.mean(axis=0)
                    rows.append(dict(tokenizer=tokenizer,fraction=fraction,seed=seed,batch=batch,policy=policy,
                        gap=coverage.max()-coverage.min(),admission=coverage.mean(),
                        budget_ratio=u.sum()/budget.sum(),max_key_ratio=(u/budget).max(),
                        min_coverage=coverage.min(),max_coverage=coverage.max()))
                    if fraction==.5 and batch==len(c):
                        for l,lang in enumerate(LANGS):
                            detail.append(dict(tokenizer=tokenizer,seed=seed,policy=policy,language=lang,coverage=coverage[l],
                                               budget=int(budget[l]),usage=int(u[l]),budget_ratio=u[l]/budget[l]))
    print('finished',tokenizer,flush=True)
pd.DataFrame(rows).to_csv(ROOT/'results/replays.csv',index=False)
pd.DataFrame(detail).to_csv(ROOT/'results/language_results.csv',index=False)
pd.DataFrame(front).to_csv(ROOT/'results/frontiers.csv',index=False)
print(pd.DataFrame(rows).query('fraction==.5 and batch==1012').groupby(['tokenizer','policy'])[['gap','admission','budget_ratio','max_key_ratio']].mean().round(4).to_string())
