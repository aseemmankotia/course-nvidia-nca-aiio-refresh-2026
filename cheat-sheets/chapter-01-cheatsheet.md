# NVIDIA AI Ecosystem Quick Reference

## Core Software Stack

| Component | Purpose | Use When |
|-----------|---------|----------|
| **CUDA** | Parallel computing platform | Writing custom GPU kernels, foundation for all GPU compute |
| **cuDNN** | Deep learning primitives library | Training neural networks (used by frameworks internally) |
| **TensorRT** | Inference optimizer/runtime | Deploying trained models for low-latency production inference |
| **RAPIDS** | Data science acceleration | ETL pipelines, pandas/scikit-learn workloads on GPU |
| **Triton** | Inference serving | Multi-model serving, production deployment, A/B testing |

## CUDA vs cuDNN vs TensorRT

```
CUDA (Foundation Layer)
  ↑ builds on
cuDNN (Training Primitives)
  ↑ used during
Model Training
  ↓ outputs
Trained Model
  ↓ optimized by
TensorRT (Inference)
```

**Exam Key**: CUDA = foundation, cuDNN = training, TensorRT = inference

## NGC Catalog Artifact Types

1. **Containers**: Framework (PyTorch, TensorFlow), Inference (Triton), Data Science (RAPIDS)
2. **Pre-trained Models**: .pth, .onnx, .plan files (separate from containers)
3. **Helm Charts**: Kubernetes deployment templates
4. **SDKs/Resources**: Code samples, notebooks, documentation

**Exam Trap**: Containers ≠ models. Models are downloaded separately.

## NVIDIA AI Enterprise vs Free NGC

| Feature | Free NGC | AI Enterprise |
|---------|----------|---------------|
| Container Access | ✓ Public containers | ✓ + Enterprise-hardened |
| Support | Community forums | Enterprise SLA + NVIDIA experts |
| Security Patches | Community timeline | <30 day guarantee |
| Certifications | None | VMware, Red Hat, K8s |
| Long-term Support | No | Yes (LTS versions) |
| Cost | Free | Subscription required |

**Exam Focus**: Enterprise = support SLA + certified configs + security patches

## RAPIDS Libraries

| Library | Replaces | Use Case |
|---------|----------|----------|
| **cuDF** | pandas | GPU-accelerated DataFrames |
| **cuML** | scikit-learn | GPU machine learning (classical ML, not deep learning) |
| **cuGraph** | NetworkX | Graph analytics on GPU |
| **cuSignal** | SciPy.signal | Signal processing |

**Exam Note**: RAPIDS = data science, NOT deep learning. For DL use PyTorch/TF.

## Triton Inference Server

**Supported Protocols**: HTTP/REST, gRPC (dynamic batching works with both)

**Supported Backends**: TensorRT, ONNX Runtime, PyTorch, TensorFlow, Python

**Key Features**:
- Model versioning (multiple versions per model)
- Dynamic batching (server-side, automatic)
- Model ensemble (multi-model pipelines)
- Concurrent model execution
- GPU/CPU scheduling

**Exam Scenario**: "Multiple models, single endpoint, A/B testing" → Triton

## TensorRT Optimizations

1. **Layer Fusion**: Combine ops (conv + batch norm + ReLU → single op)
2. **Precision Calibration**: FP32 → FP16 or INT8 (lower latency)
3. **Kernel Auto-tuning**: Select fastest implementation per GPU
4. **Dynamic Tensor Memory**: Optimize memory usage

**Exam Key**: TensorRT optimizes trained models. Input = .onnx/.pth, Output = .plan (optimized engine)

## Common Exam Traps

1. **cuDNN for inference?** NO → cuDNN is for training. Use TensorRT for inference optimization.
2. **CUDA required for PyTorch?** Indirectly—PyTorch uses cuDNN which uses CUDA, but you don't write CUDA code.
3. **NGC containers include models?** NO → Models are separate artifacts to download.
4. **RAPIDS for deep learning?** NO → RAPIDS is for data science (pandas/sklearn). Use PyTorch/TF for DL.
5. **Triton needs TensorRT?** NO → Triton supports multiple backends (TensorRT is one option).
6. **AI Enterprise = free?** NO → Subscription required for enterprise features and support.

## Deployment Workflow

```
1. Pull NGC container (framework or Triton)
2. Download pre-trained model OR train custom model
3. Optimize with TensorRT (optional but recommended)
4. Deploy via Triton Inference Server
5. Send inference via HTTP/gRPC
```

**Exam Workflow Questions**: Test full pipeline understanding (NGC → train → optimize → serve)
