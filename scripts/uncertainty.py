from pathlib import Path
import sys,json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.audit import *
import pandas as pd
f=pd.read_csv(ROOT/'results/token_counts.csv');out=[]
for tokenizer in ['cl100k_base','o200k_base','qwen2.5']:
    def mat(split,col):return f[f.split==split].pivot(index='id',columns='language',values=col)[LANGS].to_numpy()
    c=mat('devtest',tokenizer);h=mat('devtest','chars4');dev=mat('dev',tokenizer);B=int(.5*c[:,0].sum())
    eq=np.repeat(B,12);cal=allocate(12*B,dev.mean(axis=0));rng=np.random.default_rng(2026)
    for rep in range(1000):
        ids=rng.integers(0,len(c),len(c));cc=c[ids];hh=h[ids];order=rng.permutation(len(c))
        a,u=replay(cc,cc,eq,order,len(cc));b,v=replay(cc,cc,cal,order,len(cc));d,w=replay(cc,hh,eq,order,len(cc))
        ac=a.mean(axis=0);bc=b.mean(axis=0)
        out.append(dict(tokenizer=tokenizer,replicate=rep,gap_equal=ac.max()-ac.min(),gap_cal=bc.max()-bc.min(),
            gap_reduction=(ac.max()-ac.min())-(bc.max()-bc.min()),throughput_ratio=b.sum()/a.sum(),
            min_equal=ac.min(),min_cal=bc.min(),chars_budget_ratio=w.sum()/eq.sum()))
    print('bootstrap',tokenizer,flush=True)
pd.DataFrame(out).to_csv(ROOT/'results/bootstrap.csv',index=False)
rows=[]
for tokenizer in ['cl100k_base','o200k_base','qwen2.5']:
    def mat(split,col):return f[f.split==split].pivot(index='id',columns='language',values=col)[LANGS].to_numpy()
    c=mat('devtest',tokenizer);dev=mat('dev',tokenizer);B=int(.5*c[:,0].sum())
    for mode in ['baseline','nonenglish_down25','nonenglish_up25','english50_pool']:
        weights=dev.mean(axis=0).copy()
        if mode=='nonenglish_down25':weights[1:]*=.75
        if mode=='nonenglish_up25':weights[1:]*=1.25
        if mode=='english50_pool':weights[0]=weights[1:].sum()
        budgets=allocate(12*B,weights)
        for seed in range(100,130):
            a,u=replay(c,c,budgets,np.random.default_rng(seed).permutation(len(c)),1012)
            coverage=a.mean(axis=0)
            rows.append(dict(tokenizer=tokenizer,mode=mode,seed=seed,gap=coverage.max()-coverage.min(),
                             min_coverage=coverage.min(),admission=coverage.mean(),budget_ratio=u.sum()/budgets.sum()))
pd.DataFrame(rows).to_csv(ROOT/'results/sensitivity.csv',index=False)
