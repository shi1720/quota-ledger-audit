from pathlib import Path
import sys, hashlib, json, importlib.metadata, urllib.request, tarfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.audit import ROOT, LANGS, upstream_components
import pandas as pd
import tiktoken
from tokenizers import Tokenizer
from huggingface_hub import hf_hub_download

DATA=ROOT/'data';DATA.mkdir(exist_ok=True)
SOURCE=Path(sys.argv[1]) if len(sys.argv)>1 else ROOT/'data/flores200_dataset'
if not SOURCE.exists():
    archive=DATA/'flores200.tar.gz'
    urllib.request.urlretrieve('https://tinyurl.com/flores200dataset',archive)
    with tarfile.open(archive) as tar:
        for m in tar.getmembers():
            if m.isfile() and not m.name.startswith('/') and '..' not in Path(m.name).parts:
                tar.extract(m,DATA)
revision='060db6499f32faf8b98477b0a26969ef7d8b9987'
qpath=hf_hub_download('Qwen/Qwen2.5-0.5B','tokenizer.json',revision=revision)
qwen=Tokenizer.from_file(qpath)
encs={'cl100k_base':tiktoken.get_encoding('cl100k_base'),
      'o200k_base':tiktoken.get_encoding('o200k_base'),'qwen2.5':qwen}
harness,_=upstream_components(); rows=[]; hashes={}
for split in ['dev','devtest']:
    for lang in LANGS:
        p=SOURCE/split/(lang+'.'+split); b=p.read_bytes();hashes[str(p.relative_to(SOURCE))]=hashlib.sha256(b).hexdigest()
        texts=b.decode().splitlines()
        for i,text in enumerate(texts):
            inp,out=harness._estimate_input_and_output_tokens({'input':text},call_type='aembedding')
            assert out==0 and inp==max(1,len(text)//4)
            row={'split':split,'id':i,'language':lang,'chars':len(text),'bytes':len(text.encode()),'chars4':inp}
            for name,enc in encs.items():
                row[name]=len(enc.encode(text,add_special_tokens=False).ids) if name=='qwen2.5' else len(enc.encode_ordinary(text))
            rows.append(row)
        print(split,lang,len(texts),flush=True)
pd.DataFrame(rows).to_csv(ROOT/'results/token_counts.csv',index=False)
meta={'flores_sha256':hashes,'qwen_revision':revision,'qwen_tokenizer_sha256':hashlib.sha256(Path(qpath).read_bytes()).hexdigest(),
      'source_commit':(ROOT/'vendor/litellm.commit').read_text().strip(),
      'source_sha256':hashlib.sha256((ROOT/'vendor/parallel_request_limiter_v3.py').read_bytes()).hexdigest(),
      'packages':{p:importlib.metadata.version(p) for p in ['numpy','pandas','scipy','matplotlib','tiktoken','tokenizers','huggingface_hub','redis','pytest']}}
(ROOT/'results/provenance.json').write_text(json.dumps(meta,indent=2)+'\n')
