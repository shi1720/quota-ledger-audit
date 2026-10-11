"""Deterministic input-token quota audit. No model calls or private user data."""
from pathlib import Path
import ast
from collections.abc import Mapping
from typing import Final
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LANGS = ['eng_Latn','fra_Latn','deu_Latn','spa_Latn','arb_Arab','hin_Deva',
         'ben_Beng','zho_Hans','jpn_Jpan','swh_Latn','tel_Telu','mya_Mymr']
LABELS = ['English','French','German','Spanish','Arabic','Hindi','Bengali',
          'Chinese','Japanese','Swahili','Telugu','Burmese']


def upstream_components():
    """Extract upstream bodies unchanged; dependency scope intentionally narrow.

    The source file is vendored and SHA256 recorded. We execute only relevant
    estimator methods and pure constants, not the full package or HTTP proxy.
    Native Google translation and messages are outside the tested interface.
    """
    source = (ROOT/'vendor/parallel_request_limiter_v3.py').read_text()
    tree = ast.parse(source)
    scope = dict(Final=Final, Mapping=Mapping,
        DEFAULT_CHARS_PER_TOKEN=4, DEFAULT_MAX_TOKENS_ESTIMATE=4096,
        _TPM_FLOOR_FRACTION=4, RESPONSES_API_MIN_OUTPUT_TOKENS=16,
        RESPONSES_API_CALL_TYPES=('aresponses','responses'),
        EMBEDDING_API_CALL_TYPES=('aembedding','embedding'),
        TEXT_COMPLETION_API_CALL_TYPES=('atext_completion','text_completion'),
        GOOGLE_GENAI_NATIVE_CALL_TYPES=('generate_content','agenerate_content'))
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef)
             and n.name == '_parse_output_cap_value']
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
               and n.name == '_PROXY_MaxParallelRequestsHandler_v3')
    wanted = {'no_max_tokens_output_floor', '_is_embedding_request',
              '_get_explicit_output_cap','get_output_candidate_count',
              '_estimate_input_and_output_tokens'}
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
    for n in methods:
        # Ordinary function compilation; descriptor restored below.
        n.decorator_list = []
    for n in funcs+methods:
        exec(compile('from __future__ import annotations\n'+ast.unparse(n),
                     '<upstream-ast>', 'exec'), scope)
    class Harness:
        _translate_google_genai_native_request = staticmethod(lambda data, call_type: None)
    for name in wanted:
        f=scope[name]
        setattr(Harness, name, f if name=='_estimate_input_and_output_tokens' else staticmethod(f))
    lua={n.target.id:ast.literal_eval(n.value) for n in tree.body
         if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name)
         and n.target.id in ['CHECK_AND_INCREMENT_BY_N_SCRIPT','WINDOW_GUARDED_TOKEN_INCREMENT_SCRIPT']}
    return Harness(), lua


def allocate(total, weights):
    """Largest remainder integer apportionment, deterministic index tie breaks."""
    w=np.array(weights, dtype=float); x=total*w/w.sum(); b=np.floor(x).astype(int)
    remainder=int(total-b.sum())
    b[np.argsort(-(x-b),kind='stable')[:remainder]]+=1
    assert b.sum()==total
    return b


def replay(costs, reservations, budgets, order, batch_size):
    """Per-key quota, settle all admitted requests after each batch.

    Physical usage is token demand, not measured GPU time. Counter after
    settlement is actual admitted input tokens. Rejections are not retried.
    """
    costs=np.asarray(costs,dtype=np.int64); reservations=np.asarray(reservations,dtype=np.int64)
    n,L=costs.shape; used=np.zeros(L,dtype=np.int64); admitted=np.zeros((n,L),bool)
    for start in range(0,n,batch_size):
        delta=np.zeros(L,dtype=np.int64)
        for i in order[start:start+batch_size]:
            ok=used+reservations[i]<=budgets
            admitted[i]=ok
            used+=ok*reservations[i]
            delta+=ok*(costs[i]-reservations[i])
        used+=delta
    assert np.array_equal(used,(admitted*costs).sum(axis=0))
    return admitted,used


def prefix_frontier(costs, total, order):
    """Largest common prefix of aligned translated tasks within pooled budget."""
    cumulative=np.cumsum(costs[order].sum(axis=1))
    return int(np.searchsorted(cumulative,total,side='right'))


def count_frontier(costs,total):
    """Largest equal count allowing each language to cherry-pick cheap tasks."""
    cumulative=np.cumsum(np.sort(costs,axis=0),axis=0).sum(axis=1)
    return int(np.searchsorted(cumulative,total,side='right'))
