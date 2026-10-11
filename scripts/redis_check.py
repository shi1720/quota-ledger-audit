"""Run unchanged extracted upstream Lua in an isolated temporary Redis."""
from pathlib import Path
import sys,subprocess,tempfile,time,hashlib,json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.audit import *
import pandas as pd
import redis

def main():
    _,lua=upstream_components(); f=pd.read_csv(ROOT/'results/token_counts.csv');f=f[f.split=='devtest']
    checks=[];decisions=0
    with tempfile.TemporaryDirectory(prefix='quota-redis-',dir='/tmp') as tmp:
        sock=Path(tmp)/'redis.sock'
        process=subprocess.Popen(['redis-server','--port','0','--unixsocket',str(sock),'--save','','--appendonly','no'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        try:
            client=redis.Redis(unix_socket_path=str(sock))
            for _ in range(100):
                try:
                    if client.ping():break
                except redis.ConnectionError:time.sleep(.02)
            check_sha=client.script_load(lua['CHECK_AND_INCREMENT_BY_N_SCRIPT'])
            settle_sha=client.script_load(lua['WINDOW_GUARDED_TOKEN_INCREMENT_SCRIPT'])
            for tokenizer in ['cl100k_base','o200k_base','qwen2.5']:
                c=f.pivot(index='id',columns='language',values=tokenizer)[LANGS].to_numpy()
                h=f.pivot(index='id',columns='language',values='chars4')[LANGS].to_numpy();B=int(.5*c[:,0].sum())
                for policy,r in [('chars4',h),('exact',c)]:
                    for batch in [1,1012]:
                        for seed in [100,101]:
                            order=np.random.default_rng(seed).permutation(len(c));expected,usage=replay(c,r,np.repeat(B,12),order,batch)
                            actual=np.zeros_like(expected)
                            for l in range(12):
                                prefix=f't{tokenizer}:p{policy}:b{batch}:s{seed}:l{l}'
                                keys=[prefix+':window',prefix+':tokens']
                                for start in range(0,len(c),batch):
                                    ids=order[start:start+batch];pipe=client.pipeline(transaction=False)
                                    for i in ids:pipe.evalsha(check_sha,2,*keys,B,int(r[i,l]),3600,3600)
                                    responses=pipe.execute();pipe=client.pipeline(transaction=False)
                                    for i,response in zip(ids,responses):
                                        ok=response[0]==0;actual[i,l]=ok
                                        if ok:pipe.evalsha(settle_sha,2,*keys,str(response[2]),int(c[i,l]-r[i,l]),3600)
                                    pipe.execute()
                                observed=int(client.get(keys[1]) or 0)
                                assert observed==usage[l],(prefix,observed,usage[l])
                            assert np.array_equal(actual,expected)
                            decisions+=actual.size
                            checks.append(dict(tokenizer=tokenizer,policy=policy,batch=batch,seed=seed,
                                decisions=actual.size,digest=hashlib.sha256(actual.tobytes()).hexdigest()))
                            print(tokenizer,policy,batch,seed,'PASS',flush=True)
        finally:
            process.terminate();process.wait(timeout=10)
    (ROOT/'results/redis_validation.json').write_text(json.dumps({'decision_checks':decisions,'conditions':checks},indent=2)+'\n')
if __name__=='__main__':main()
