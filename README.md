# Shopping Rules! Shopping Copilot — Context-Aware Conversational Product Search (No LLM Needed)

A next-generation conversational shopping agent built for **TikTok TechJam 2026 — Shopping Copilot: AI Conversational Search and Recommendations**.

---

## Project Overview

Traditional e-commerce search works well when users already know exactly what they want:

> "black leather boots size 9"

But real shopping conversations are often less structured:

> "I need something comfortable for travelling."

> "Maybe something waterproof too."

> "Actually, I'd rather have cotton."

These conversations require more than keyword matching.

Our Shopping Copilot combines **hybrid retrieval, conversational state tracking, proactive clarification, personalized context, adaptive orchestration, structured ranking, and local semantic reranking** to search a frozen catalog of 50,000 Amazon products.

The system is designed around three principles:

1. **Preserve recall early** — avoid prematurely removing potentially relevant products.
2. **Distill intent over time** — turn conversation history into structured shopping context.
3. **Increase precision progressively** — apply increasingly sophisticated ranking only to promising candidates.

---

# How Our Solution Addresses the Challenge

## 1. Intent-Aware Hybrid Search

Every turn is dynamically classified as either **Buying** or **Browsing**.

**Buying** requests contain stronger purchase intent or accumulated constraints and use a precision-oriented retrieval strategy.

**Browsing** requests are more exploratory and place greater emphasis on semantic discovery.

Intent is recalculated every turn, allowing the workflow to evolve naturally as the shopper becomes more specific.

Products are retrieved through multiple complementary signals:

* **BM25 / SQLite FTS5** for lexical relevance
* **FAISS + SentenceTransformer** for dense semantic similarity
* **Amazon category hierarchy** for structured product-type relevance

BM25 and FAISS candidates are combined using weighted **CombMNZ fusion**, with different weights for Buying and Browsing traffic.

---

## 2. Multi-Turn Dialogue State

The agent maintains structured shopping preferences across turns:

```text
category · material · color · size · style
brand · budget · feature · use_case
```

For example:

```text
User: "I need a jacket."
→ category = jacket

User: "Something waterproof."
→ category = jacket
→ feature = waterproof

User: "Black would be good."
→ category = jacket
→ feature = waterproof
→ color = black
```

The tracker also handles:

* preference accumulation
* negation
* "no preference" responses
* clarification answers
* preference changes
* intent overrides

Rather than blindly resetting conversation history when preferences change, conflicting information is rewritten while useful context is preserved.

---

## 3. Candidate-Aware Proactive Clarification

An ambiguous request such as:

> "I'm looking for shoes."

can correspond to thousands of plausible products.

The agent detects **over-generality** using:

* accumulated constraint density
* retrieval ambiguity
* candidate-pool characteristics

When clarification is useful, the system analyzes the current candidates and estimates which missing attribute can best narrow the search.

This creates a feedback loop:

```text
User request
     ↓
Initial retrieval
     ↓
Candidate uncertainty
     ↓
Best clarification question
     ↓
New user preference
     ↓
Refined retrieval
```

The policy combines candidate information gain with a high-value default question priority to avoid asking theoretically discriminative but practically unhelpful questions.

---

## 4. Context-Aware Candidate Reranking

Our retrieval diagnostics showed that finding relevant products was not the main bottleneck, but getting accurate rankings **was a challenge**.

We therefore preserve strong candidates from BM25, FAISS, and fused retrieval before applying a multi-signal reranker.

The reranker considers:

* normalized BM25 relevance
* dense FAISS similarity
* retriever agreement
* title, category, and feature coverage
* bigram phrase matching
* trigram phrase matching
* user profile coverage
* hierarchical category compatibility
* structured shopping constraints
* IDF-weighted term importance

Phrase matching is especially useful for product attributes such as:

```text
"water resistant"
"machine washable"
"quick drying"
"pull on closure"
```

where treating words independently loses useful meaning.

---

## 5. Personalized Context Distillation

Each evaluation session includes a historical user profile.

We use the supplied `preference_tags` as a lightweight long-term preference representation.

Historical preferences act only as a **weak ranking prior**:

```text
Current explicit intent
        >
Current session context
        >
Historical preferences
```

This prevents past behavior from overriding what the shopper explicitly requests now.

Personalization is also slightly stronger during exploratory Browsing than high-intent Buying.

---

## 6. Runtime Adaptive Orchestration

The agent estimates how specific the current shopping request has become.

Broad requests preserve larger candidate pools for diversity and recall, while highly constrained requests can send fewer candidates into expensive downstream reranking.

Importantly, first-stage BM25 and FAISS retrieval remain broad, preventing adaptive computation from sacrificing catalog coverage.

In local evaluation, this orchestration produced no degradation in recommendation quality, allowing runtime behavior to adapt while maintaining the same TechnicalScore.

---

## 7. Local Cross-Encoder Semantic Ranking

The strongest products from our deterministic reranker are passed to a lightweight local **CrossEncoder**:``ms-marco-MiniLM-L-6-v2``.

Unlike independent embeddings, the CrossEncoder jointly evaluates:

```text
(user intent, product information)
```

to estimate semantic relevance.

Only the strongest shortlist is processed, keeping inference practical.

CrossEncoder relevance is blended with the existing domain-aware ranking rather than replacing it entirely.

This produced one of our largest late-stage improvements in Hit Rate@10 and conversational efficiency.

---

# Architecture

```text
                 User + Profile
                       │
                       ▼
             Dialogue State Tracker
                       │
             ┌─────────┴─────────┐
             ▼                   ▼
       Intent Router       Context Distillation
     Buying / Browsing      Structured Slots
             │                   │
             └─────────┬─────────┘
                       ▼
              Runtime Orchestrator
                       │
          ┌────────────┴────────────┐
          ▼                         ▼
     BM25 / FTS5              FAISS Dense Search
          │                         │
          └────────────┬────────────┘
                       ▼
                  CombMNZ Fusion
                       │
                       ▼
             Candidate Preservation
                       │
             ┌─────────┴─────────┐
             ▼                   ▼
       Candidate-Aware       Multi-Signal
        Clarification         Reranker
                                   │
                                   ▼
                            Local CrossEncoder
                                   │
                                   ▼
                          Top-10 Recommendations
```

---

# Results

Evaluation uses the official deterministic TechJam evaluator over the 200-session public development set.

### Initial Baseline

| Metric         | Score |
| -------------- | ----: |
| Hit Rate@10    | 0.125 |
| MRR            | 0.068 |
| MTTC           | 9.810 |
| Efficiency     | 0.119 |
| TechnicalScore | 0.107 |

### Best Validated Full-Stack Result

| Metric         |     Score |
| -------------- | --------: |
| Hit Rate@10    | **0.775** |
| MRR            | **0.413** |
| MTTC           | **4.475** |
| Efficiency     | **0.653** |
| TechnicalScore | **0.642** |

---

# Engineering & Experimentation

We developed the agent through controlled experiments rather than adding every technique simultaneously.

Some major checkpoints were:

| Improvement                   | TechnicalScore |
| ----------------------------- | -------------: |
| Initial baseline              |          0.451 |
| Dialogue/state improvements   |          0.534 |
| Constraint-aware reranking    |          0.583 |
| Context distillation          |          0.594 |
| Candidate preservation        |          0.597 |
| Bigram matching               |          0.612 |
| Trigram matching              |          0.622 |
| Candidate-aware clarification |          0.625 |
| Profile personalization       |          0.627 |
| Category hierarchy            |          0.628 |
| IDF Coverage                  |          0.628 |
| Local CrossEncoder            |      **0.642** |

We also rejected techniques when experiments showed that they reduced performance.

For example, aggressive temporal slot decay weakened still-valid constraints from earlier conversation turns and significantly increased MTTC.

This iterative process helped us keep architectural complexity only when it provided useful behavior.

---

# Retrieval Diagnostics

We built a separate offline diagnostic pipeline to identify where relevant products were being lost.

It measures target presence through:

```text
BM25 → FAISS → Union → Fusion
→ Candidate Pool → Rerank@50
→ Rerank@20 → Top-10
```

Early diagnostics showed approximately:

* **97.5% BM25@250 target coverage**
* **98% BM25 + FAISS union coverage**

This revealed an important insight:

> Our main problem was not finding the purchased product — it was preserving and ranking it correctly.

That finding directly motivated candidate preservation and the multi-stage reranking architecture.

Ground-truth ASINs are used only for offline evaluation **after recommendations have been generated** and never influence agent inference.

---

# Technology Stack

### Development Tools

* Python
* VS Code
* Git / GitHub
* TechJam participant kit

### Libraries & Frameworks

* HuggingFace
* SentenceTransformers
* PyTorch
* FAISS
* SQLite FTS5
* NumPy
* scikit-learn
* python-dotenv

### Models

* SentenceTransformer `all-MiniLM-L6-v2`
* local MiniLM CrossEncoder semantic reranker `ms-marco-MiniLM-L-6-v2`

### APIs

**No paid external API is required.**

The recommendation pipeline runs locally using downloaded Hugging Face models. Hugging Face Hub is used only to obtain the SentenceTransformer and CrossEncoder model files.

Users should authenticate with the Hugging Face CLI before first use. Once the required models and FAISS index are cached locally, repeated evaluation does not require paid inference API calls.

This avoids per-request API costs and keeps the core recommendation pipeline locally executable.

---

# Dataset & Assets

The project uses the frozen competition dataset derived from:

**Amazon Reviews 2023 — Clothing, Shoes & Jewelry**

Competition resources include:

* 50,000 product catalog entries
* 200 public development sessions
* 800 private evaluation sessions
* official local evaluator
* official agent API contract

The catalog remains strictly read-only.

No synthetic products or ASINs are introduced.

---

# Repository Structure

```text
# Repository Structure

```text
techjam-participant-kit/
│
├── data/
│   ├── index_cache/                  # Cached FAISS vector index
│   ├── catalog.jsonl                 # Frozen 50,000-product catalog
│   ├── public_set.jsonl              # 200 public evaluation sessions
│   └── README.md                     # Dataset documentation
│
├── docs/
│   ├── agent_api_contract.json       # Required agent input/output contract
│   ├── baseline_results.json         # Organizer baseline results
│   ├── competition_specification.md  # Competition problem specification
│   ├── evaluation_config.json        # Official evaluation configuration
│   └── submission_rules.md           # Submission requirements
│
├── evaluator/
│   ├── __init__.py
│   └── local_evaluator.py            # Official deterministic evaluator
│
├── starter/
│   ├── __init__.py
│   ├── agent.py                      # Main Shopping Copilot agent
│   ├── clarification_policy.py       # Candidate-aware clarification policy
│   ├── eda.ipynb                     # Exploratory data analysis
│   ├── indexer.py                    # FAISS indexing and cache management
│   ├── orchestration.py              # Runtime adaptive candidate orchestration
│   ├── retrieval_diagnostics.py      # Retrieval/ranking diagnostic tools
│   ├── state_tracker_rulebased.py    # Multi-turn dialogue state tracker
│   └── test_state_tracker.py         # State-tracker tests
│
├── DATA_ATTRIBUTION.md               # Dataset attribution
├── README.md                         # Project documentation
├── requirements.txt                  # Python dependencies
├── SHA256SUMS                         # Dataset integrity checksums
│
├── results.json                      # Latest local evaluation results
├── retrieval_diagnostics.json        # Generated retrieval diagnostic results
│
├── catalog.jsonl.gz                  # Compressed catalog distribution
├── .env                              # Local environment variables (not committed)
└── .gitignore                        # Git exclusions
```
---

### Key Implementation Files

The core Shopping Copilot implementation is contained in `starter/`:

* **`agent.py`** — orchestrates intent routing, hybrid retrieval, candidate preservation, reranking, personalization, and response generation.
* **`state_tracker_rulebased.py`** — maintains multi-turn shopping preferences and handles accumulation, clarification responses, negation, and intent changes.
* **`indexer.py`** — builds and caches the SentenceTransformer + FAISS dense retrieval index.
* **`clarification_policy.py`** — detects over-general queries and selects candidate-aware clarification attributes.
* **`orchestration.py`** — dynamically adjusts downstream candidate processing based on current query specificity.
* **`retrieval_diagnostics.py`** — analyzes where ground-truth products are retained or lost across retrieval and ranking stages.

The official evaluator is kept separately under `evaluator/`, while organizer-provided specifications and API contracts remain under `docs/`.

---

# Setup & Installation

## 1. Clone the Repository

```bash
git clone https://github.com/edselllll/techjam2026edselllll.git
cd techjam-participant-kit
```

All commands below should be run from the **repository root** — the directory containing:

```text
starter/
evaluator/
data/
requirements.txt
README.md
```

## 2. Install Dependencies

Make sure Python and `pip` are installed, then upgrade `pip`:

```bash
python -m pip install --upgrade pip
```

Install the project dependencies:

```bash
pip install -r requirements.txt
```

Upgrade Hugging Face Hub to ensure the latest CLI is available:

```bash
pip install --upgrade huggingface_hub
```

## 3. Log In to Hugging Face

The project uses Hugging Face models for dense retrieval and local semantic reranking.

Authenticate with:

```bash
hf auth login
```

Follow the browser login flow or paste a Hugging Face access token when prompted.

You can verify the login with:

```bash
hf auth whoami
```

The project uses:

```text
SentenceTransformer: all-MiniLM-L6-v2
CrossEncoder:        cross-encoder/ms-marco-MiniLM-L-6-v2
```

The models will be downloaded automatically when required.

> Never commit Hugging Face access tokens or `.env` credentials to the repository.

## 4. Competition Data

The required competition files should be located under:

```text
data/
├── catalog.jsonl
├── public_set.jsonl
└── index_cache/
```

The supplied Amazon catalog is read-only and contains 50,000 products.

## 5. Run the Evaluator

Make sure your terminal is still at the **repository root**:

```text
techjam-participant-kit/techjam-conversational-search
```

Then run:

```bash
python evaluator/local_evaluator.py
```

The evaluator loads the agent from:

```text
starter/agent.py
```

and evaluates it against the public development sessions.

Evaluation results are written to:

```text
results.json
```

## 6. First Run

The first run may take longer because the project needs to load the Hugging Face models and, if no cached vector index exists, encode the product catalog and build the FAISS index.

The generated FAISS index is cached under:

```text
data/index_cache/
```

Subsequent runs reuse the cached index and should start significantly faster.

If the embedding model or catalog embedding representation is changed, rebuild the FAISS cache before evaluating again.

---

# Practicality

The architecture intentionally uses different levels of computation at different stages.

Cheap operations handle broad retrieval:

```text
SQLite BM25 + FAISS
```

More expensive semantic reasoning is reserved for a small candidate shortlist:

```text
CrossEncoder → Top candidates only
```

The system also runs entirely in memory during evaluation and requires no industrial vector database or hosted LLM service.

This makes the approach suitable for environments where latency, infrastructure complexity, and inference cost matter.

---

# Limitations & Future Improvements

The current dialogue state tracker is primarily rule-based, so unusual free-form descriptions may not map perfectly into structured attributes.

Amazon metadata is also incomplete for some products, particularly structured fields such as price.

The local CrossEncoder is a general relevance model rather than one trained specifically on Amazon shopping interactions.

Given more time, we would explore:

* product-specific learning-to-rank;
* stronger structured price reasoning;
* automatic feature normalization;
* confidence-calibrated clarification;
* latency profiling and optimization;
* domain-adapted semantic reranking;
* evaluation against a larger unseen development set.

We intentionally avoid aggressive optimization against individual public evaluation scenarios to reduce the risk of overfitting before private evaluation.

---

# Impact

The architecture is designed for a broader problem than finding one Amazon product.

The same approach can support conversational discovery in catalogs where users:

* begin with vague requirements;
* progressively reveal preferences;
* change their minds;
* require personalized recommendations;
* cannot express their needs as rigid filters.

Potential applications include fashion, electronics, marketplaces, travel discovery, content recommendation, and other large-catalog search systems.

The central idea is to transform search from:

```text
query → results
```

into:

```text
conversation
    → understanding
    → retrieval
    → clarification
    → refinement
    → recommendation
```

---
