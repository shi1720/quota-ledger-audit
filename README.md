# Equal Quotas, Unequal Access

Reproducible research on multilingual input-token reservation and admission.
Author: Shivam Gupta. Prepared for ACM FAccT 2027 consideration; not accepted or peer reviewed.

This artifact uses real FLORES-200 translations and real tokenizer executions.
It does **not** run a language model, call a hosted inference service, or estimate
production harm. Redis validation executes unchanged Lua bodies extracted from
a pinned LiteLLM source file; it is component validation, not a full proxy test.

## Main result

At the central cl100k_base burst condition, character reservations admit 49.48%
of translated requests but commit 3.367 times the aggregate token quota. Exact
reservations respect the quota and expose a 45.80 percentage-point language
coverage gap. Development-data proportional quota allocation reduces this gap
to 0.56 points at the same total quota, while admitting about 40.4% fewer requests
than equal token quotas. The allocation improves a chosen minimum-access
objective; it is not a free throughput gain or a universal definition of fairness.

## Reproduce from frozen counts

Python 3.9+ is supported. The tested package versions are in requirements-lock.txt.

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements-lock.txt
.venv/bin/python -m pytest -q tests
.venv/bin/python scripts/run.py
.venv/bin/python scripts/uncertainty.py
.venv/bin/python scripts/figures.py
.venv/bin/python scripts/redis_check.py
```

The final command requires `redis-server` on PATH. It starts a private Unix-socket
instance with persistence disabled and terminates it in a `finally` block. On
macOS the temporary socket uses /tmp to stay under AF_UNIX path length limits.
No existing Redis instance is used or modified.

To regenerate text token counts (requires Internet access and downloads about
25 MB of source data plus tokenizer tables):

```sh
.venv/bin/python scripts/prepare.py
.venv/bin/python scripts/blocks.py data/flores200_dataset
```

A local path to the original extracted FLORES dataset may be passed as the first
argument to prepare.py instead. Model weights are never downloaded. Ordinary
text counts exclude special/chat tokens. No raw text is republished here.

## Files

- `paper/main.pdf`: anonymous manuscript, 13 pages main content, one endmatter
  page, references and appendices. `paper/named.pdf`: author version.
- `src/audit.py`: integer-ledger simulator, proportional apportionment, exact
  finite-workload frontiers, AST extraction of narrow upstream estimator methods.
- `results/token_counts.csv`: all per-sentence language/tokenizer counts.
- `results/replays.csv`: all 2,880 main replay conditions (34,974,720 decisions).
- `results/language_results.csv`: all central language-level outcomes.
- `results/bootstrap.csv`: 3,000 matched-sentence bootstrap replicates.
- `results/redis_validation.json`: admission-array hashes for 291,456 exact
  cross-runtime checks across 24 full-workload conditions.
- `results/provenance.json`: dataset hashes, source commit/hash, tokenizer revision,
  and package versions. `results/blocks.csv`: four-sentence sensitivity.
- `protocol/PROTOCOL.txt`: prospective local protocol, committed before results.
- `protocol/LITERATURE_REVIEW.txt`: contribution boundary and search limitations.
- `protocol/CLAIM_CHECKS.txt`: main claims mapped to data and calculations.
- `vendor/`: original pinned upstream source and its MIT license.

## Build manuscript

With Tectonic on PATH, from `paper/`:

```sh
tectonic -X compile main.tex
tectonic -X compile named.tex
```

The ACM class is extracted from official acmart 2.20 documented source and
retains its LPPL licensing. References use ACM-Reference-Format.bst. For anonymous
submission use main.pdf, never the named PDF or an identifying repository link.
The public repository is for named sharing and independent reproduction.

## Interpretation and integrity

This is a controlled audit, not a deployed-service prevalence estimate. Thirty
orders reuse the same benchmark. Bootstrap intervals assess benchmark-composition
sensitivity conditional on paired sentence resampling, not social-population
sampling. Full-burst settlement is an explicit stress endpoint. The calibrated
baseline uses development means and has no distribution-shift guarantee.

The intended 50%-English sensitivity was implemented as an entitlement-priority
stress test under balanced demand, not a skewed-production-arrival trace. This
protocol deviation is disclosed in the paper. Tokenizer inequity, fair serving,
weighted allocation, and reservation conservation have substantial prior work;
we claim a combined executable audit and findings, not invention of those ideas.

Codex (GPT-6) substantively supported research framing, literature search,
implementation, execution, analysis, figures, and manuscript preparation, as
disclosed in the required manuscript endmatter. The named human author remains
responsible for checking and submitting the work. No independent human review
of every claim is represented by this package.

## License and data

Original audit code: MIT, see LICENSE. Vendored LiteLLM code: its own MIT notice.
ACM formatting files: LPPL, see paper/LICENSE.acmart.txt. FLORES-200: CC BY-SA 4.0,
see the original Meta distribution and citation in the paper. Tokenizers retain
their upstream licenses. Count measurements and results are released under CC BY
4.0; this does not change any upstream dataset or code license.
