from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.audit import *
import pandas as pd
import tiktoken
from tokenizers import Tokenizer
from huggingface_hub import hf_hub_download
source=Path(sys.argv[1]);q=Tokenizer.from_file(hf_hub_download('Qwen/Qwen2.5-0.5B','tokenizer.json',revision='060db6499f32faf8b98477b0a26969ef7d8b9987'))
encs={'cl100k_base':tiktoken.get_encoding('cl100k_base'),'o200k_base':tiktoken.get_encoding('o200k_base'),'qwen2.5':q}
allrows=[];results=[]
for split in ['dev','devtest']:
 for lang in LANGS:
  lines=(source/split/(lang+'.'+split)).read_text().splitlines()
  for i in range(len(lines)//4):
   text='\n'.join(lines[4*i:4*i+4]);row=dict(split=split,id=i,language=lang,chars4=max(1,len(text)//4),bytes=len(text.encode()))
   for name,enc in encs.items():row[name]=len(enc.encode(text,add_special_tokens=False).ids) if name=='qwen2.5' else len(enc.encode_ordinary(text))
   allrows.append(row)
f=pd.DataFrame(allrows);f.to_csv(ROOT/'results/block_counts.csv',index=False)
for tok in encs:
 def mat(split,col):return f[f.split==split].pivot(index='id',columns='language',values=col)[LANGS].to_numpy()
 c=mat('devtest',tok);h=mat('devtest','chars4');by=mat('devtest','bytes');dev=mat('dev',tok);B=int(.5*c[:,0].sum());eq=np.repeat(B,12);cal=allocate(12*B,dev.mean(axis=0))
 for seed in range(100,130):
  order=np.random.default_rng(seed).permutation(len(c))
  for batch in [1,len(c)]:
   for policy,r,budget in [('chars4',h,eq),('exact',c,eq),('bytes',by,eq),('calibrated',c,cal)]:
    a,u=replay(c,r,budget,order,batch);coverage=a.mean(axis=0)
    results.append(dict(tokenizer=tok,seed=seed,batch=batch,policy=policy,gap=coverage.max()-coverage.min(),admission=coverage.mean(),budget_ratio=u.sum()/budget.sum()))
pd.DataFrame(results).to_csv(ROOT/'results/blocks.csv',index=False)
