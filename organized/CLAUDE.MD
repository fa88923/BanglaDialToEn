# Bengali Dialect-to-English Translation System
### ML Lab Project — Technical Overview

---

## Project Summary

An end-to-end pipeline that translates Bengali regional dialect text into English. The system uses a **"Route-then-Translate"** architecture: an automatic dialect classifier identifies the incoming dialect, then hot-swaps the appropriate fine-tuned translation adapter to generate a semantically accurate English output.

**Dialects Covered:** Barisal, Chittagong, Mymensingh, Noakhali, Sylhet  
**Target Languages:** Bengali dialects → English  
**Core Models:** NLLB-200 (translation) + BanglaBERT (dialect routing)  
**Fine-Tuning Method:** PEFT with DoRA adapters (one per dialect)

---

## Workflow Overview

```
Raw Dialect Text
      │
      ▼
[1] Preprocessing & Normalization
      │
      ▼
[2] BanglaBERT Dialect Router  ──→  Dialect Label (e.g., "Sylhet")
      │
      ▼
[3] NLLB-200 + Dialect Adapter (hot-swapped via PEFT multi-adapter API)
      │
      ▼
[4] English Translation Output
```

---

## Phase 0 — Base Model Selection

**Goal:** Identify the best pre-trained translation model for low-resource Bengali dialects.

| Model | Result |
|---|---|
| `facebook/mbart-large-50-many-to-many-mmt` | Struggled with low-resource regional dialects |
| `facebook/nllb-200-distilled-600M` | ✅ Selected — better handling of phonetic nuances in dialects like Chittagong and Sylhet |

> **Decision:** NLLB-200 distilled 600M was chosen as the base translation model.

---

## Phase 1 — Data Pipeline

### 1.1 Dataset Collection & Merging (`mergeDataset.ipynb`)

Dialect data was consolidated from multiple source datasets:

- **Sources:** Vashantor, ONUBAD, ChatgaiyyaAlap, BIDWESH, BanglaRegionalTextCorpus, ANUBHUTI, ANCHOLIK-NER
- Each source had different column naming conventions. A unified `make_df()` helper standardized all schemas to three columns:
  - `Standard_Bangla`
  - `dialect_speech`
  - `English_translation`
- Rows with missing dialect speech were dropped.
- **Output:** Five dialect-specific CSVs with ~7,800–15,900 rows each.

### 1.2 Deduplication & Leakage Removal (`RemovingDuplicates.ipynb`)

Two data quality issues were addressed:

- **Deduplication:** When duplicates existed, the copy retaining the English translation was kept. This reduced each dataset by roughly 20–35%.
- **Leakage Detection:** Overlaps between training and validation/test sets (sourced from Vashantor) were identified and removed. ~170–250 overlapping rows per dialect were found and dropped from val/test splits.

### 1.3 English Translation of Missing Labels (`translationOpenAI.ipynb`)

Many rows lacked English translations. GPT-4o-mini was used to fill these gaps:

- **Method:** Batches of 50 Standard Bangla sentences per API call, using a numbered prompt format for reliable parsing.
- **Output:** `translated_output.csv`

### 1.4 Filling Missing Translations (`merge.ipynb`)

The GPT-4o-mini translations were merged back into the dialect CSVs:

- A lookup dictionary was built from translated results.
- Unicode normalization (NFC + zero-width character removal) was applied before matching to handle Bangla encoding inconsistencies.
- Rows that still couldn't be matched were dropped.
- **Final dataset sizes:** ~5,100–13,000 rows per dialect.

### Data Pipeline Summary

```
Multi-source raw datasets
        │
        ▼
Merge & standardize schema
        │
        ▼
Deduplicate + remove val/test leakage
        │
        ▼
Translate missing English labels (GPT-4o-mini)
        │
        ▼
Fill & finalize clean dialect CSVs (×5)
```

---

## Phase 2 — Dialect Router Training

**Goal:** Build a classifier that automatically identifies the dialect of an input sentence.

- **Initial experiments:** Basic BERT models (underperformed on dialectal text)
- **Final model:** `csebuetnlp/banglabert` (BanglaBERT)
- **Training:** Fine-tuned on dialectal speech samples from the merged dataset
- **Result:** Classification accuracy of **96–98%**

> **Impact of misrouting:** When the classifier routes to the wrong dialect adapter, BERTScore F1 drops by ~14.5 points — confirming that dialect-specific specialization is critical.

---

## Phase 3 — Adapter Training (PEFT Evolution)

Fine-tuning was done using Parameter-Efficient Fine-Tuning (PEFT) to specialize the NLLB-200 model per dialect without full fine-tuning.

### 3.1 Stage 1 — Initial LoRA Experiment
- Tested LoRA (Low-Rank Adaptation) on a frozen NLLB-200 base.
- Goal: Verify that adapters can capture regional nuances.

> *[Add: initial LoRA results here]*

### 3.2 Stage 2 — Hyperparameter Tuning
- Exhaustive sweeps over learning rates, schedulers, and target modules.
- Optimized per-dialect configurations.

> *[Add: HP tuning trial results/table here]*

### 3.3 Stage 3 — DoRA Experimentation
- Tested **DoRA (Weight-Decomposed LoRA)**, which decouples directional and magnitude weight updates, as a potential improvement over standard LoRA.
- DoRA was evaluated per dialect — it was **not universally better**.
- **Result:** DoRA improved performance only for **Noakhali** and **Chittagong**; the remaining three dialects performed better with standard LoRA.

> *[Add: LoRA vs DoRA per-dialect comparison results here]*

### 3.4 Stage 4 — Final Adapter Training
- Trained **5 individual dialect adapters**, each using the method (LoRA or DoRA) that performed best for that dialect:

| Dialect | Method |
|---|---|
| Mymensingh | LoRA |
| Barisal | LoRA |
| Sylhet | LoRA |
| Noakhali | **DoRA** |
| Chittagong | **DoRA** |

- **Anchor Mixing:** 10% Standard Bangla sentences mixed into every training batch to prevent the model from drifting from its pre-trained foundation.

> *[Add: per-adapter final training metrics here]*

---

## Phase 4 — Final Pipeline Assembly

### Pipeline Steps

1. **Preprocessing:** Normalization fixes whitespace and punctuation while strictly preserving dialectal markers (unique pronouns, nasalization, etc.)
2. **Dialect Routing:** BanglaBERT router identifies the dialect — **96.52% accuracy**
3. **Adapter Hot-Swap:** The PEFT multi-adapter API loads the correct dialect adapter into NLLB-200
4. **Translation Generation:** Dialect-specific generation parameters applied (e.g., higher beam search for Sylhet)
5. **Output:** English translation

---

## Evaluation

### Metrics Used

**Translation Quality (Semantic)**
- **SacreBLEU** — Standardized BLEU for corpus-level precision
- **chrF++** — Character n-gram F-score; better suited for morphologically rich languages like Bengali
- **METEOR** — Recall-focused, accounts for synonym matching
- **BERTScore** — Contextual embedding similarity; captures semantic accuracy beyond exact word matches

**Error Analysis (Literal)**
- **WER** (Word Error Rate) — Word-level substitutions, deletions, insertions
- **CER** (Character Error Rate) — High-resolution character-level error tracking
- **TER** (Translation Edit Rate) — Human editing effort to match reference

### Key Results

| Metric | Value |
|---|---|
| Dialect Router Accuracy | 96.52% |
| Mymensingh SacreBLEU (tuned adapter) | 66.59 |
| Mymensingh SacreBLEU (baseline full fine-tune) | 11.05 |
| Mymensingh BLEU improvement | **+55.54** |
| Combined pipeline SacreBLEU (all dialects) | 43.37 |
| Combined pipeline chrF++ (all dialects) | 64.81 |
| BERTScore F1 drop on misrouting | ~14.5 points |

> *[Add: full per-dialect results table here]*  
> *[Add: per-dialect chrF++, METEOR, BERTScore breakdown here]*

---

## Architecture Summary

| Step | Component | Purpose |
|---|---|---|
| I | Model Selection | NLLB-200 chosen over mBART-50 |
| II | Data Scaling | Merged multi-source datasets per dialect |
| III | Router Training | BanglaBERT fine-tuned for dialect detection |
| IV | Adapter Evolution | LoRA → HP Tuning → DoRA → 5 final adapters |
| V | Pipeline Assembly | Route-then-Translate with hot-swappable adapters |