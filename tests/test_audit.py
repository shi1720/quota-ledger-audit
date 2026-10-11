import itertools
import numpy as np
from src.audit import allocate,replay,prefix_frontier,count_frontier,upstream_components

def test_upstream_estimator():
    obj,_=upstream_components()
    for text in ['hello','你好世界','नमस्ते दुनिया','ကမ္ဘာ','a'*100,'🙂'*40]:
        assert obj._estimate_input_and_output_tokens({'input':text},call_type='aembedding')==(max(1,len(text)//4),0)
        assert obj._estimate_input_and_output_tokens({'prompt':text,'max_tokens':128},call_type='atext_completion')==(max(1,len(text)//4),128)

def test_apportionment():
    rng=np.random.default_rng(4)
    for _ in range(1000):
        w=rng.uniform(.1,10,12);total=int(rng.integers(12,100000));b=allocate(total,w)
        assert b.sum()==total
        assert np.max(np.abs(b-total*w/w.sum()))<1

def test_safe_reservations():
    rng=np.random.default_rng(5)
    for _ in range(100):
        c=rng.integers(1,100,(31,4));r=c+rng.integers(0,100,c.shape);b=rng.integers(100,1000,4)
        for batch in [1,8,31]:
            a,u=replay(c,r,b,rng.permutation(31),batch);assert np.all(u<=b)

def test_error_bound():
    rng=np.random.default_rng(6)
    for _ in range(100):
        c=rng.integers(1,100,(31,4));r=rng.integers(1,100,c.shape);b=rng.integers(100,1000,4)
        a,u=replay(c,r,b,rng.permutation(31),31)
        assert np.all(u<=b+(a*np.maximum(c-r,0)).sum(axis=0))

def test_cherry_frontier_exhaustive():
    rng=np.random.default_rng(7)
    for _ in range(40):
        c=rng.integers(1,10,(4,2));B=int(rng.integers(1,60));best=0
        for bits in itertools.product([0,1],repeat=8):
            a=np.array(bits).reshape(4,2)
            if (a*c).sum()<=B:best=max(best,int(a.sum(axis=0).min()))
        assert count_frontier(c,B)==best

def test_prefix():
    c=np.array([[2,4],[3,5],[9,10]])
    assert prefix_frontier(c,14,np.arange(3))==2

def test_inversion_and_settlement():
    c=np.tile([4,8],(10,1));h=np.tile([4,2],(10,1));b=np.array([16,16])
    a,u=replay(c,h,b,np.arange(10),10)
    assert np.array_equal(a.sum(axis=0),[4,8]);assert np.array_equal(u,[16,64])
    a,u=replay(c,c,b,np.arange(10),10)
    assert np.array_equal(a.sum(axis=0),[4,2]);assert np.array_equal(u,[16,16])
