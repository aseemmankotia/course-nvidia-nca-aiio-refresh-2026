# AI/ML/DL Fundamentals and GPU Architecture - Exam Cheat Sheet

## Learning Paradigms Comparison
| Paradigm | Data Type | Example Use Case | Training Signal |
|----------|-----------|------------------|----------------|
| Supervised | Labeled (X→Y pairs) | Image classification, regression | Loss from ground truth labels |
| Unsupervised | Unlabeled | Clustering, anomaly detection | Reconstruction error, cluster cohesion |
| Reinforcement | Environment + Rewards | Game playing, robotics | Cumulative reward signal |

**Exam Trap:** Unsupervised learning does NOT use labeled data. Questions often describe unlabeled data and ask which paradigm - answer is unsupervised.

## Training vs Inference Characteristics
| Aspect | Training | Inference |
|--------|----------|----------|
| Computation | Forward + Backward pass | Forward pass only |
| Memory | High (activations + gradients) | Low (activations only) |
| Latency | Hours to weeks | Milliseconds to seconds |
| Throughput | 100-1000 samples/sec | 1000-100,000 samples/sec |
| Precision | FP32 or FP16 mixed | INT8 or FP16 |
| Batch Size | 16-512 typical | 1 (real-time) or 8-128 (batch) |

**Exam Tip:** Inference is 10-100x faster because no backprop. Questions test understanding that training needs more memory and time.

## GPU Architecture Essentials
| Component | NVIDIA A100 Spec | Purpose | Exam Relevance |
|-----------|-----------------|---------|----------------|
| CUDA Cores | 6,912 | General parallel compute | Parallelism for matrix ops |
| Tensor Cores | 432 (Gen 3) | AI matrix operations | 10-20x faster than CUDA for DL |
| Memory | 40 GB or 80 GB | Store model + activations | Limits model/batch size |
| Memory Bandwidth | 1,555 GB/s (HBM2) | Data transfer rate | Critical for large models |
| FP16 Performance | 312 TFLOPS | Half-precision training | 2x faster than FP32 |
| INT8 Performance | 624 TOPS | Inference acceleration | 4x faster than FP32 |

**Why GPU Wins:** Thousands of cores execute matrix multiply in parallel. CPUs have 8-64 cores, GPUs have 2,000-10,000+ cores. Matrix ops are embarrassingly parallel.

## Training Hyperparameters
| Parameter | Typical Range | Too High Symptom | Too Low Symptom |
|-----------|--------------|------------------|----------------|
| Learning Rate | 0.0001 - 0.01 | Loss explodes/diverges | Training too slow, stuck |
| Batch Size | 16 - 512 | Out of memory | Slow, noisy gradients |
| Epochs | 10 - 200 | Overfitting | Underfitting |

**Exam Calculation:** Dataset 100K samples, batch size 200 → 500 iterations per epoch. If 50 epochs → 25,000 total weight updates.

## Model Evaluation Metrics
### Confusion Matrix (Binary Classification)
```
                Predicted
              Pos      Neg
Actual Pos | TP (180) | FN (80)  | → Recall = TP/(TP+FN) = 180/260 = 69%
Actual Neg | FP (20)  | TN (720) |
           ↓
    Precision = TP/(TP+FP) = 180/200 = 90%
```

### Key Formulas
- **Precision** = TP/(TP+FP) — "Of predicted positives, how many correct?"
- **Recall** = TP/(TP+FN) — "Of actual positives, how many caught?"
- **F1 Score** = 2×(P×R)/(P+R) — Harmonic mean, balances both
- **Accuracy** = (TP+TN)/(TP+TN+FP+FN) — Overall correctness

### When Each Metric Matters
- **High Precision:** Spam filter (don't block legitimate email), Ad targeting (don't waste budget)
- **High Recall:** Cancer detection (don't miss cases), Fraud detection (catch all fraud)
- **Balanced (F1):** Search relevance, general classification

**Exam Trap:** Precision and recall trade off. Lowering threshold → more positives → higher recall, lower precision. Questions test which metric matters for specific scenario.

## Precision Trade-offs (Inference)
| Data Type | Speed vs FP32 | Accuracy Loss | When to Use |
|-----------|--------------|---------------|-------------|
| FP32 | 1x (baseline) | 0% | Rare (training only) |
| FP16 | 2x faster | <0.5% | Most inference, some training |
| INT8 | 4x faster | 1-2% | Latency-critical inference |

**Exam Scenario:** "Must process 30 fps with <20ms latency" → INT8. "Medical diagnosis requiring highest accuracy" → FP16 or FP32.

## Data Preprocessing (Exam Mentions)
- **Normalization:** Scale features to 0-1 or -1 to 1 — prevents large features dominating
- **Augmentation:** Flip, rotate, crop images — increases training data diversity
- **Train/Val/Test Split:** Typical 70/15/15 or 80/10/10 — test set NEVER seen during training

**Exam Tip:** Questions may ask why accuracy is high on training but low on test data → overfitting (too many epochs, not enough data, no regularization).
