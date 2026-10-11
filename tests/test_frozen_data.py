from src.audit import ROOT, LANGS
import pandas as pd

def test_frozen_counts_and_conservative_bytes():
    f=pd.read_csv(ROOT/'results/token_counts.csv')
    assert len(f)==2009*12
    assert not f.duplicated(['split','language','id']).any()
    assert set(f.language)==set(LANGS)
    for tok in ['cl100k_base','o200k_base','qwen2.5']:
        assert (f[tok]>0).all()
        assert (f[tok]<=f['bytes']).all()
    for split,n in [('dev',997),('devtest',1012)]:
        assert (f[f.split==split].groupby('language').size()==n).all()
