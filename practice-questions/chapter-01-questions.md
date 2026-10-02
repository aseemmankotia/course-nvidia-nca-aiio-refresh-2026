## Chapter 1: NVIDIA AI Ecosystem and Software Stack — Practice Questions

**Q1.** A data science team needs to accelerate ETL pipelines processing 500GB of CSV files daily on NVIDIA GPUs. The pipelines currently use pandas and scikit-learn on CPUs and take 6 hours to complete. Which NVIDIA solution BEST addresses this requirement?

A) Use RAPIDS cuDF and cuML libraries as drop-in replacements for pandas and scikit-learn
B) Deploy cuDNN to accelerate the pandas dataframe operations and scikit-learn models
C) Implement TensorRT to optimize the data processing inference workloads
D) Use CUDA C++ to rewrite the entire ETL pipeline from scratch

<details>
<summary>Answer</summary>

**Correct: A**

RAPIDS (specifically cuDF for dataframes and cuML for machine learning) provides GPU-accelerated, API-compatible alternatives to pandas and scikit-learn. This is the exact use case RAPIDS was designed for—accelerating data science workflows without requiring code rewrites.

**Why the others are wrong:**
- cuDNN is for deep learning neural network primitives, not general data processing or classical ML algorithms
- TensorRT optimizes trained neural networks for inference, not ETL pipelines or data transformations
- Rewriting in CUDA C++ would work but requires months of development versus RAPIDS' drop-in replacement approach

> ⚠️ Frequently missed on the exam — read the options carefully.

</details>

---

**Q2.** An ML engineer has a trained PyTorch ResNet-50 model that must serve predictions with <10ms latency at 1000 requests/second. The model currently runs on CPU with 45ms average latency. Which combination of NVIDIA technologies is MOST appropriate?

A) Use cuDNN to accelerate the PyTorch model and deploy on GPU
B) Deploy the PyTorch model directly in an NGC catalog container
C) Use CUDA to write custom kernels for each ResNet layer
D) Convert the model with TensorRT and deploy via Triton Inference Server

<details>
<summary>Answer</summary>

**Correct: D**

TensorRT optimizes trained models (layer fusion, precision calibration, kernel auto-tuning) specifically for low-latency inference. Triton Inference Server provides production-grade serving with dynamic batching and multi-model support. This combination is the standard NVIDIA inference stack.

**Why the others are wrong:**
- cuDNN accelerates training and is already used by PyTorch internally, but doesn't provide the optimizations needed for <10ms latency
- NGC containers package frameworks but don't optimize the model itself for inference performance
- Custom CUDA kernels would take months to develop and maintain versus TensorRT's automated optimizations

</details>

---

**Q3.** A company needs GPU-accelerated AI containers with enterprise support, security patches within 30 days, and access to NVIDIA experts for production deployments. Which licensing option meets these requirements?

A) NVIDIA AI Enterprise subscription
B) NGC catalog public containers with community support
C) CUDA Toolkit with developer forum support
D) Open-source TensorRT builds from GitHub

<details>
<summary>Answer</summary>

**Correct: A**

NVIDIA AI Enterprise provides: enterprise-grade containers with security patches, direct NVIDIA support with SLAs, access to NVIDIA AI experts, and certified configurations. This is the only option providing contractual support and enterprise features.

**Why the others are wrong:**
- NGC public containers are free but have no SLA, no guaranteed security patching timeline, and community-only support
- CUDA Toolkit provides development tools but no containers, no production support, and no enterprise SLA
- Open-source TensorRT has no support, no security patching SLA, and requires self-management

> ⚠️ Frequently missed on the exam — read the options carefully.

</details>

---

**Q4.** Which statement BEST describes the relationship between CUDA, cuDNN, and TensorRT in the NVIDIA software stack?

A) TensorRT replaces both CUDA and cuDNN for production deployments
B) cuDNN is required to use TensorRT for model optimization
C) CUDA is the foundation; cuDNN builds on CUDA for training; TensorRT builds on cuDNN for inference
D) CUDA and cuDNN are alternatives—use one or the other based on use case

<details>
<summary>Answer</summary>

**Correct: C**

This correctly describes the layered architecture: CUDA provides the parallel computing foundation, cuDNN adds optimized deep learning primitives on top of CUDA (used during training), and TensorRT optimizes trained models for inference (often using cuDNN kernels internally).

**Why the others are wrong:**
- TensorRT complements CUDA/cuDNN but doesn't replace them; it uses them internally for optimized inference
- TensorRT can work with models trained using cuDNN but doesn't require cuDNN to be installed separately for inference
- CUDA and cuDNN are not alternatives—cuDNN is built on top of CUDA and uses it

</details>

---

**Q5.** A research team needs to pull a container with a pre-trained BERT model, fine-tune it on custom data, then deploy for inference. Which NGC catalog artifact type should they use initially?

A) Helm chart for Kubernetes deployment
B) SDK container with training frameworks
C) Pre-trained model (.pth or .onnx file)
D) Triton Inference Server container

<details>
<summary>Answer</summary>

**Correct: C**

NGC provides pre-trained models as downloadable artifacts (.pth PyTorch, .onnx, etc.). The team needs the pre-trained BERT weights first, then can use a framework container for fine-tuning and eventually Triton for deployment.

**Why the others are wrong:**
- Helm charts are for deploying applications to Kubernetes, not for downloading model weights
- SDK containers provide the framework to train/fine-tune but don't include the pre-trained BERT model weights
- Triton containers are for serving inference, not for accessing pre-trained model files

</details>

---

**Q6.** An organization runs AI workloads on VMware vSphere and needs certified GPU-accelerated containers with guaranteed compatibility. Which NVIDIA offering provides this?

A) NVIDIA AI Enterprise with VMware certification
B) CUDA Toolkit installed directly on vSphere VMs
C) NGC public catalog containers deployed on vSphere
D) Community-built containers from Docker Hub

<details>
<summary>Answer</summary>

**Correct: A**

NVIDIA AI Enterprise includes VMware vSphere certification, tested and validated configurations, and support for running GPU workloads on VMware. This is the only option with guaranteed compatibility and enterprise support.

**Why the others are wrong:**
- CUDA Toolkit can be installed but provides no containers, no validation, and no VMware-specific support
- NGC public containers are not certified for vSphere and have no compatibility guarantees
- Community containers have no NVIDIA certification, testing, or support for vSphere environments

> ⚠️ Frequently missed on the exam — read the options carefully.

</details>

---

**Q7.** Triton Inference Server receives inference requests via HTTP and gRPC. A developer needs to send batch requests with dynamic batching enabled. Which protocol feature is MOST relevant?

A) Both protocols support dynamic batching—protocol choice doesn't affect batching behavior
B) HTTP/2 for request multiplexing and dynamic batch formation
C) gRPC streaming for continuous batch request submission
D) Dynamic batching requires a custom protocol extension not available in HTTP or gRPC

<details>
<summary>Answer</summary>

**Correct: A**

Dynamic batching is a server-side Triton feature that works regardless of client protocol. Triton accumulates requests from multiple clients (HTTP or gRPC) and batches them based on configured timeout and max batch size. The protocol doesn't affect batching behavior.

**Why the others are wrong:**
- HTTP/2 multiplexing helps with connection efficiency but Triton's dynamic batching is independent of this
- gRPC streaming can send continuous requests but dynamic batching happens server-side, not via streaming
- Dynamic batching is a built-in Triton feature accessible via standard HTTP and gRPC protocols

</details>

---

**Q8.** A team needs to deploy multiple models (ONNX, TensorRT, PyTorch) behind a single endpoint with automatic model versioning and A/B testing. Which component of the NVIDIA AI stack addresses this requirement?

A) TensorRT model optimizer with multiple backend support
B) NGC catalog model repository with version tagging
C) CUDA Multi-Process Service (MPS) for concurrent model execution
D) Triton Inference Server with model repository and ensemble support

<details>
<summary>Answer</summary>

**Correct: D**

Triton Inference Server supports multiple frameworks (ONNX, TensorRT, PyTorch, TensorFlow), model versioning, A/B testing via model repository configs, and ensemble pipelines. This is Triton's core feature set for production model serving.

**Why the others are wrong:**
- TensorRT optimizes individual models but doesn't provide serving infrastructure, versioning, or A/B testing
- NGC catalog stores and versions models but doesn't provide runtime serving or endpoint management
- CUDA MPS enables GPU sharing among processes but provides no model serving, versioning, or routing features

</details>

---

