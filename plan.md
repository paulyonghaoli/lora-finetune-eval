This repo is currently much more focused than the agent repo: it benchmarks dense, BM25, and RRF hybrid retrieval with synthetic ML documents and standard IR metrics. The upgrade should turn it into a serious retrieval-evaluation lab that can plug into your RAG/agent stack.

Received app response
Thought for 32s

Yes. For hybrid-retrieval-benchmark, I would evolve it into:

Hybrid Retrieval Lab

A serious, reproducible benchmark framework for comparing:

BM25
dense embeddings
hybrid RRF
score fusion
query-adaptive routing
reranking
GraphRAG-style retrieval
agent-memory retrieval
long-context retrieval
multilingual retrieval

This repo fits your career story well because your portfolio already emphasizes LLM + vector retrieval, embeddings, Databricks SQL, and agentic analytics systems .

Your current repo already benchmarks dense, BM25, and hybrid RRF, and produces metrics/report/plots. The code currently uses sentence-transformers, rank_bm25, and RRF with configurable k_values, model_name, and rrf_k.

Updated plan
Phase 1 — Make it a real benchmark framework

Add:

typed data schemas
CLI args
dataset validation
per-query result export
statistical comparison
CI
unit tests
reproducibility metadata
Phase 2 — Add modern retrieval methods

Add:

weighted RRF
score normalization fusion
distribution-based score fusion
dense + sparse reranking
cross-encoder reranker
query-adaptive routing

Qdrant’s docs explicitly frame hybrid search as combining dense semantic retrieval with sparse exact-match retrieval and support RRF/score-fusion style approaches.

Phase 3 — Add stronger datasets

Move beyond synthetic ML docs:

BEIR datasets
small local “mini-BEIR” subset
enterprise-style RAG dataset
agent-memory retrieval dataset
long-document retrieval dataset

BEIR is a widely used heterogeneous IR benchmark for testing retrieval generalization across domains.

Phase 4 — Make it research-grade

Add:

ablation studies
confidence intervals
latency/quality Pareto frontier
failure clustering
query-type router analysis
“when hybrid helps vs hurts” report
Phase 5 — Connect to your agent repo

This is the big opportunity.

Your open-ended-task-agent needs a retrieval layer. This repo can become the retrieval evaluation engine for OETA-Lab:

OETA-Lab agent
  ↓
retrieval strategy selector
  ↓
hybrid-retrieval-benchmark
  ↓
retrieval quality + latency + failure report

That makes both repos stronger.

Claude Code prompt
You are working in my repo:
paulyonghaoli/hybrid-retrieval-benchmark

The repo currently compares:
- dense retrieval using sentence-transformers
- sparse BM25 using rank_bm25
- hybrid retrieval using Reciprocal Rank Fusion

Current outputs include:
- results/metrics.json
- results/report.md
- results/plots/

Long-term goal:
Transform this repo into “Hybrid Retrieval Lab” — a serious retrieval evaluation framework for RAG systems, agent memory systems, GraphRAG-style retrieval, and query-adaptive hybrid search.

Do NOT rewrite everything at once. Build a stable research-grade foundation first.

==================================================
PHASE 1 TARGET — BENCHMARK FOUNDATION
==================================================

Main goals:
1. Make the benchmark reproducible
2. Add typed schemas and validation
3. Add richer result exports
4. Add stronger fusion methods
5. Add query-adaptive routing skeleton
6. Add tests and CI
7. Keep the repo readable and interview-friendly

==================================================
PART 1 — PROJECT STRUCTURE CLEANUP
==================================================

Review the current codebase.

Keep existing behavior:
- python data/prepare_corpus.py
- python main.py
- dense / sparse / hybrid benchmark

But improve structure if needed.

Suggested package structure:

retrieval/
  dense.py
  sparse.py
  hybrid.py
  fusion.py
  rerank.py
  router.py

evaluation/
  benchmark.py
  metrics.py
  schemas.py
  analysis.py
  reporting.py

data/
  prepare_corpus.py

scripts/
  run_benchmark.py
  compare_runs.py

tests/

==================================================
PART 2 — TYPED DATA SCHEMAS
==================================================

Create evaluation/schemas.py.

Define dataclasses or Pydantic models:

Document:
- doc_id: str
- title: str
- text: str
- metadata: dict optional

Query:
- query_id: str
- query: str
- relevant_doc_ids: list[str]
- query_type: str
- metadata: dict optional

RetrievalResult:
- query_id
- strategy
- ranked_doc_ids
- scores optional
- latency_ms
- query_type

Validation requirements:
- doc_id unique
- query_id unique
- every relevant_doc_id exists in corpus
- query text not empty
- document text not empty
- query_type not empty

Add clear error messages.

Add tests for invalid corpus/query files.

==================================================
PART 3 — CLI CONFIGURATION
==================================================

Replace hardcoded main.py behavior with argparse while preserving default behavior.

Support:

python main.py --config config.yaml
python main.py --corpus data/corpus.jsonl --queries data/queries.jsonl
python main.py --out results
python main.py --strategies dense sparse hybrid
python main.py --k 1 3 5 10
python main.py --model all-MiniLM-L6-v2

Keep config.yaml as the default.

==================================================
PART 4 — RICH RESULT EXPORT
==================================================

Currently metrics.json is aggregate-heavy.

Add:
results/run_metadata.json
results/per_query_results.jsonl
results/metrics.json
results/report.md

run_metadata.json should include:
- timestamp
- git commit if available
- python version
- platform
- model name
- corpus size
- query count
- k values
- config

per_query_results.jsonl should include for every query and strategy:
- query_id
- query
- query_type
- relevant_doc_ids
- ranked_doc_ids
- metrics by k
- latency_ms

==================================================
PART 5 — MORE FUSION METHODS
==================================================

Create retrieval/fusion.py.

Implement:
1. Reciprocal Rank Fusion
2. Weighted Reciprocal Rank Fusion
3. CombSUM with min-max score normalization
4. Distribution-based score fusion if simple enough

Important:
- Dense and BM25 raw scores are not directly comparable.
- Normalize before score-based fusion.
- Keep RRF as default.

Update HybridRetriever to support:

fusion_method:
- rrf
- weighted_rrf
- combsum

Config example:
hybrid:
  fusion_method: rrf
  rrf_k: 60
  dense_weight: 1.0
  sparse_weight: 1.0

Add tests:
- RRF ranking deterministic
- weighted RRF changes ranking when weights change
- score normalization handles equal scores
- fusion handles missing documents

==================================================
PART 6 — RETURN SCORES, NOT ONLY DOC IDS
==================================================

Current DenseRetriever/SparseRetriever mostly return ranked doc IDs.

Refactor carefully so retrievers can return:
- doc_id
- score
- rank

while preserving old retrieve() behavior if possible.

Suggested:
retrieve_with_scores(query_text, k) -> list[ScoredDocument]

Keep retrieve(query_text, k) -> list[str] for compatibility.

Dense currently computes cosine similarity manually.
Sparse uses BM25 scores.
Hybrid should expose fused scores.

Add tests.

==================================================
PART 7 — QUERY ROUTER SKELETON
==================================================

Create retrieval/router.py.

Goal:
Implement query-adaptive retrieval strategy selection.

Define QueryFeatures:
- token_count
- has_rare_terms
- has_numbers
- has_acronyms
- exact_match_likelihood
- semantic_likelihood

Implement simple heuristic router:
- exact/named/acronym-heavy query -> sparse
- conceptual/long natural-language query -> dense
- mixed/uncertain -> hybrid

Create RouterRetriever:
- chooses strategy per query
- logs selected strategy and rationale

Add benchmark strategy:
- routed

Add metrics by router decision.

Add tests.

==================================================
PART 8 — RERANKER SKELETON
==================================================

Create retrieval/rerank.py.

Implement optional reranker interface:

BaseReranker:
- rerank(query, candidates, corpus, top_k)

Start with:
- NoOpReranker
- SimpleOverlapReranker

Optional future:
- CrossEncoderReranker using sentence-transformers CrossEncoder

Do not make heavy reranking mandatory.

Config:
reranker:
  enabled: false
  type: none
  top_n: 20

==================================================
PART 9 — STATISTICAL ANALYSIS
==================================================

Create evaluation/analysis.py.

Add:
- bootstrap confidence intervals for NDCG@k and MRR@k
- paired comparison between strategies
- win/loss/tie counts per query
- quality-latency Pareto analysis

Report:
- best average strategy
- fastest strategy
- best quality/latency tradeoff
- query types where hybrid helps
- query types where hybrid hurts

Add tests for bootstrap shape and paired comparison.

==================================================
PART 10 — REPORTING IMPROVEMENT
==================================================

Create evaluation/reporting.py.

Improve report.md with:
- executive summary
- dataset summary
- strategy comparison table
- per-query-type table
- latency table
- Pareto frontier
- failure examples:
  - queries where dense wins
  - queries where sparse wins
  - queries where hybrid wins
  - queries where all fail
- router analysis if routed strategy enabled
- methodology limitations

Keep it concise and portfolio-grade.

==================================================
PART 11 — PLOTS
==================================================

Improve plots:
- precision@k
- recall@k
- ndcg@k
- mrr@k
- latency by strategy
- quality vs latency scatter
- query-type heatmap

Do not overcomplicate styling.

==================================================
PART 12 — DATASET MODES
==================================================

Keep synthetic corpus mode.

Add dataset mode abstraction:

DatasetLoader:
- SyntheticDatasetLoader
- JsonlDatasetLoader
- future BeirDatasetLoader

Do not fully implement BEIR if it becomes large.
But create a clean interface and docs.

Add docs explaining future BEIR integration.

==================================================
PART 13 — RESEARCH DOCS
==================================================

Create docs/ROADMAP.md.

Position the repo as:

Hybrid Retrieval Lab:
A reproducible benchmark for dense, sparse, hybrid, routed, reranked, and graph-aware retrieval systems for RAG and agent memory.

Include roadmap:
v0: dense/BM25/RRF synthetic benchmark
v1: typed schemas + per-query exports
v2: fusion methods + router
v3: reranking + statistical analysis
v4: BEIR/min-BEIR integration
v5: GraphRAG retrieval
v6: agent-memory retrieval benchmark
v7: paper-ready experiments

Create docs/REFERENCES.md.

Include annotated references:
- BEIR
- BM25
- Reciprocal Rank Fusion
- Dense Passage Retrieval
- ColBERT / late interaction
- SPLADE
- BGE-M3
- GraphRAG
- Qdrant hybrid search docs
- Vespa hybrid search docs

For each:
- key idea
- what this repo borrows
- what remains future work

==================================================
PART 14 — CONNECTION TO OETA-LAB
==================================================

Create docs/OETA_INTEGRATION.md.

Explain how this benchmark can plug into open-ended-task-agent:

- evaluate retrieval strategies for agent memory
- choose retrieval strategy by query type
- feed retrieval diagnostics into verifier
- benchmark GraphRAG-lite retrieval
- measure retrieval latency vs answer quality

Add a simple integration contract:

Input:
- corpus.jsonl
- queries.jsonl

Output:
- per_query_results.jsonl
- retrieval diagnostics
- best strategy by query type

==================================================
PART 15 — TESTS
==================================================

Add pytest tests for:
- metrics correctness
- schema validation
- dense/sparse compatibility with tiny corpus
- RRF and weighted RRF
- score normalization
- router decisions
- per-query export
- report generation
- benchmark run on tiny dataset

Use tiny in-memory datasets where possible.

==================================================
PART 16 — CI
==================================================

Add GitHub Actions:

.github/workflows/ci.yml

Run:
- pip install -r requirements.txt
- pip install pytest
- pytest -v
- python data/prepare_corpus.py
- python main.py

If sentence-transformers model download makes CI flaky, create a tiny mock dense retriever test path or mark full benchmark as optional.

==================================================
PART 17 — README UPDATE
==================================================

Update README with:
- clearer positioning
- quick start
- CLI examples
- strategy list
- metrics list
- output files
- roadmap
- how this connects to RAG / agent memory
- limitations

Do not make README huge.
Point detailed material to docs/.

==================================================
ENGINEERING CONSTRAINTS
==================================================

- Keep offline/local execution possible.
- Avoid mandatory vector database dependency.
- Avoid huge datasets by default.
- Keep synthetic demo fast.
- Do not add secrets.
- Keep code readable.
- Add type hints.
- Prefer small modules.
- Preserve current default workflow.
- Make benchmark reproducible.
- Do not overfit to synthetic data.
- Avoid claiming hybrid is always best.

==================================================
FINAL COMMANDS TO RUN
==================================================

After implementation, run:

python data/prepare_corpus.py
python main.py
pytest -v

If CLI args implemented, also run:

python main.py --config config.yaml
python main.py --strategies dense sparse hybrid routed
python main.py --out results

Then summarize:
- files changed
- new architecture
- new retrieval methods
- tests added
- benchmark results
- remaining TODOs

Most important idea: this repo should not just say “hybrid is better.” It should answer:

When does dense retrieval win?
When does sparse retrieval win?
When does hybrid win?
When does hybrid hurt?
Can a router learn/select the right strategy?
How much latency do we pay for quality?

That is the difference between a toy benchmark and a serious retrieval-systems project.