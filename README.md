# Bangla Dialect-to-English Translation System

A route-then-translate NLP system that automatically detects Bangladeshi regional dialects and translates them to English with high semantic accuracy using BanglaBERT + NLLB-200 + PEFT adapters.

## Problem & Motivation

Bangladesh has 5+ major regional dialects (Barisal, Chittagong, Mymensingh, Noakhali, Sylhet) with distinct phonetics, vocabulary, and grammar that differ significantly from Standard Bangla. Off-the-shelf translation systems like mBART-50 produce poor results for these low-resource dialects.

## Solution Overview

Our system uses a **two-stage pipeline**:

1. **Dialect Classification** → Detect which dialect the input belongs to (89.21% accuracy)
2. **Dialect-Specific Translation** → Route to the appropriate adapter and translate to English

## Architecture

```
Input Text → Preprocessing → BanglaBERT Classifier → PEFT Adapter Selection → NLLB-200 Translation → English Output
```

### Key Components

- **Router**: BanglaBERT fine-tuned on dialect text (89.21% classification accuracy on 5 dialects + Standard Bangla)
- **Translator**: NLLB-200 Distilled 600M with 5 dialect-specific adapters (LoRA or DoRA per dialect)
- **Optimization**: Hyperparameter tuning per dialect; anchor mixing prevents catastrophic forgetting

## Results

### Dialect Classification
| Approach | Accuracy | Macro-F1 |
|----------|----------|----------|
| BanglaBERT | **89.21%** | **87.1%** |
| Soft Voting TF-IDF | 85.2% | 84.1% |
| Grid Search Ensemble | 87.22% | 85.8% |

### Translation Quality (Per-Dialect Performance)

| Dialect | SacreBLEU | chrF++ | WER (%) | METEOR |
|---------|-----------|--------|---------|--------|
| **Mymensingh** | **57.95** | **74.54** | **33.01** | **77.99** |
| Barisal | 45.53 | 65.70 | 46.78 | 67.04 |
| Sylhet | 49.21 | 67.22 | 40.81 | 68.57 |
| Noakhali | 43.33 | 63.27 | 47.57 | 66.91 |
| Chittagong | 33.93 | 56.57 | 55.71 | 54.99 |
| **Combined** | **43.37** | **64.81** | — | — |

**Key Finding**: Mymensingh leads in accuracy, while Noakhali and Chittagong show higher error rates. DoRA adapters outperform LoRA only for Noakhali & Chittagong—benefits are dialect-specific, not universal.

## Dataset

Merged from 7 sources with ~50.7K total samples:
- **Vashantor** (primary large-scale study)
- ONUBAD, ChatgaiyyaAlap, BIDWESH
- BanglaRegionalTextCorpus, ANUBHUTI, ANCHOLIK-NER

After deduplication & leakage prevention (~8% reduction):
- Chittagong: 12.8K | Standard Bangla: 12.4K | Noakhali: 7.9K | Barisal: 7.1K | Sylhet: 5.6K | Mymensingh: 4.8K

## Evaluation Metrics

**Semantic Quality**: SacreBLEU, chrF++, METEOR, ROUGE (1, 2, L)  
**Error Analysis**: WER (Word Error Rate), CER (Character Error Rate)

## Adapter Training

Each dialect trained with optimized hyperparameters:
- **Mymensingh, Barisal, Sylhet**: LoRA (Learning Rate: 5e-4, Scheduler: linear/cosine)
- **Noakhali, Chittagong**: DoRA ⭐ (Weight-decomposed adapters for better performance)
- Beam size: 3–8 | Length penalty: 0.8–1.0

## Conclusion

1. **Route-then-Translate** achieves 89.21% dialect classification accuracy with BanglaBERT
2. PEFT adapters (LoRA/DoRA) provide massive translation gains over baseline
3. DoRA benefits are **empirical and dialect-specific** (Noakhali & Chittagong only)
4. **Combined pipeline: 43.37 SacreBLEU & 64.81 chrF++** across all 5 dialects

## Team

- Sheikh Iftikharun Nisa (2105069)
- Abony Kamal (2105072)
- Mahbuba Sayed Sinthia (2105078)
- Farhana Adri (2105089)

**CSE 330 Project** | University course on Machine Learning

## References

- Inspired by: **Vashantor** — First large-scale study on Bangla regional dialect translation and region detection
- NLLB-200: Facebook AI's open-source machine translation model
- PEFT: Parameter-Efficient Fine-Tuning (Hugging Face)
