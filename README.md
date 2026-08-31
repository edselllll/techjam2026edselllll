# Shopping Copilot

A conversational product search and recommendation agent built for **TikTok TechJam 2026 — Shopping Copilot: AI Conversational Search and Recommendations**.


## Architecture

```text
User Message + Profile
        ↓
Dialogue State Tracker
        ↓
Buying / Browsing Intent Router
        ↓
Context Distillation
        ↓
┌─────────────┬──────────────┐
│ BM25 / FTS5 │ FAISS Dense  │
└──────┬──────┴──────┬───────┘
       ↓             ↓
       CombMNZ Fusion
              ↓
   Candidate Preservation
              ↓
 Constraint-Aware Reranker
              ↓
  Local Cross-Encoder
              ↓
 Top-10 Recommendations
              ↕
 Candidate-Aware Clarification
```

## Key Features

### Intent-Aware Hybrid Retrieval

Each turn is dynamically classified as **Buying** or **Browsing**.

* **Buying** prioritizes precise lexical and constraint matching.
* **Browsing** gives greater weight to semantic retrieval and exploration.

Products are retrieved using:

* SQLite FTS5 / BM25 lexical search
* FAISS + SentenceTransformer dense search
* Amazon category hierarchy signals
* weighted CombMNZ fusion

Strong candidates from individual retrieval routes are preserved before final ranking to avoid premature loss of relevant products.

### Multi-Turn Dialogue State

The rule-based state tracker accumulates structured preferences such as:

```text
category, material, color, size, style,
brand, budget, feature, use_case
```

It supports incremental information accumulation, negation, no-preference responses, and preference changes without unnecessarily deleting useful conversation history.

### Proactive Clarification

The agent detects overly broad requests using current state and retrieval uncertainty.

When clarification is useful, the current candidate pool helps determine which missing attribute would best narrow the search.

This combines candidate information gain with a high-value default question strategy to reduce unnecessary conversational turns.

### Context-Aware Reranking

Retrieved candidates are reranked using a combination of:

* BM25 relevance
* FAISS similarity
* query and field coverage
* bigram and trigram phrase matching
* hierarchical category compatibility
* structured product constraints
* retriever agreement
* user preference tags

Explicit session requirements remain more important than historical user preferences.

### Local Semantic Reranking

A lightweight local CrossEncoder semantically reranks the strongest candidates from the deterministic ranking stage.

Only a small shortlist is processed, keeping semantic inference relatively lightweight while improving final recommendations.

### Runtime Orchestration

Query specificity and intent are used to dynamically adjust downstream candidate processing.

Broad requests preserve more candidates for diversity, while highly constrained requests can use smaller reranking pools without reducing first-stage retrieval recall.

## Results

Evaluation uses the official deterministic TechJam evaluator.

| Metric         | Initial | Current Best |
| -------------- | ------: | -----------: |
| Hit Rate@10    |   0.560 |    **0.775** |
| MRR            |   0.334 |    **0.412** |
| MTTC           |   7.450 |    **4.465** |
| Efficiency     |   0.355 |    **0.654** |
| TechnicalScore |   0.451 |    **0.642** |

This represents approximately a **42% improvement in TechnicalScore** over the initial baseline.

Current scenario performance:

| Scenario        |    Hit@10 |       MRR |      MTTC |
| --------------- | --------: | --------: | --------: |
| Boundary        |     0.800 |     0.424 |     4.400 |
| Browsing        | **0.863** | **0.486** | **3.775** |
| Buying          |     0.763 |     0.350 |     4.113 |
| Intent Override |     0.567 |     0.377 |     7.267 |

## Project Structure

```text
starter/
├── agent.py
├── indexer.py
├── state_tracker_rulebased.py
├── clarification_policy.py
└── orchestration.py

retrieval_diagnostics.py
data/
requirements.txt
README.md
```

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Place the official TechJam catalog and public evaluation data in the expected `data/` directory.

Run the organizer-provided local evaluator according to the participant-kit instructions.

The FAISS index is cached locally after construction so subsequent evaluations do not require re-embedding the full catalog.

## Development Approach

The system was developed through controlled ablation experiments rather than adding all components simultaneously.

Major improvements came from:

1. stronger dialogue-state handling;
2. hybrid candidate preservation;
3. constraint-aware reranking;
4. distilled conversational context;
5. bigram/trigram phrase matching;
6. candidate-aware clarification;
7. profile-tag personalization;
8. hierarchical category scoring;
9. local cross-encoder reranking.

Several approaches were tested and rejected when they reduced evaluation performance, including Reciprocal Rank Fusion and aggressive temporal slot decay.

Offline retrieval diagnostics also showed that BM25 + FAISS could retrieve the target product in approximately **98% of public sessions**, shifting development focus from retrieval recall toward ranking and conversational efficiency.

## Limitations

The state tracker relies partly on rule-based vocabularies and may miss unusual free-form constraints. Amazon metadata can also be incomplete, particularly for structured attributes such as price.

The local cross-encoder is a general relevance model rather than one trained specifically for Amazon shopping data.

Finally, development decisions were evaluated on the 200 public sessions, so the system avoids highly scenario-specific heuristics that could overfit the public evaluator.

## Future Improvements

Potential extensions include:

* stronger structured price and hard-constraint reasoning;
* IDF-weighted constraint matching;
* learned product-specific reranking;
* richer feature normalization;
* latency profiling and optimization;
* improved handling of free-form preference changes.

## Dataset

The project uses the frozen TechJam catalog derived from **Amazon Reviews 2023 — Clothing, Shoes & Jewelry**:

* 50,000 products
* 200 public development sessions
* 800 private evaluation sessions

The catalog remains read-only and no synthetic ASINs are introduced.

## Team

**[Team / Participant Name]**

Add team member roles and contributions here if applicable.

## Acknowledgements

Built for **TikTok TechJam 2026 — Shopping Copilot: AI Conversational Search and Recommendations**, using the official competition kit and Amazon Reviews 2023-derived catalog.
