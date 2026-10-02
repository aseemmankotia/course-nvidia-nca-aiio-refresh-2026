## Chapter 2: AI/ML/DL Fundamentals and GPU Architecture — Practice Questions

**Q1.** A financial services company is deploying a fraud detection model that must process transactions in real-time with sub-50ms latency. The model uses a deep neural network with 200 million parameters. During peak hours, the system receives 10,000 transactions per second. Which GPU configuration characteristic is MOST critical for meeting these requirements?

A) Tensor Cores with INT8 precision support to accelerate matrix operations while maintaining accuracy
B) High memory bandwidth (900+ GB/s) to rapidly load model weights for each inference
C) Large number of CUDA cores (10,000+) to maximize parallel throughput for batch processing
D) NVLink interconnect bandwidth for multi-GPU scaling of the training workload

<details>
<summary>Answer</summary>

**Correct: A**

Real-time inference with sub-50ms latency requires fastest possible matrix operations. Tensor Cores accelerate inference 10-20x vs CUDA cores alone. INT8 precision (vs FP32) provides 4x speedup with minimal accuracy loss for inference. This directly addresses latency requirement.

**Why the others are wrong:**
- Memory bandwidth helps but isn't the bottleneck for inference - model weights stay loaded in GPU memory, not reloaded per transaction. Bandwidth matters more for training with large batch updates.
- CUDA core count helps throughput but doesn't address the 50ms latency constraint. Latency is about speed per operation (Tensor Cores), not total parallelism. Also, 10K requests/sec can be batched, so raw core count isn't the primary bottleneck.
- NVLink is for multi-GPU training workloads where gradients sync between GPUs. This is an inference scenario (fraud detection on live transactions), not training. NVLink adds cost without benefiting single-model inference.

> ⚠️ Frequently missed on the exam — read the options carefully.

</details>

---

**Q2.** A machine learning team is evaluating a medical diagnosis model that classifies X-ray images as showing pneumonia (positive) or healthy (negative). The model achieves 95% precision and 70% recall on the test set. Hospital administrators are primarily concerned about missing actual pneumonia cases. Which action would BEST address this concern?

A) Increase the classification threshold to improve precision to 98%
B) Lower the classification threshold to increase recall above 85%
C) Focus on improving F1 score to balance precision and recall equally
D) Optimize for AUC-ROC to maximize overall model discrimination ability

<details>
<summary>Answer</summary>

**Correct: B**

Missing actual pneumonia cases = false negatives. Recall = TP/(TP+FN), so low recall (70%) means many false negatives. Lowering classification threshold makes model predict 'positive' more often, catching more true positives (increasing recall) at cost of more false positives (decreasing precision). This is the right trade-off when missing positives is dangerous.

**Why the others are wrong:**
- Increasing threshold makes model MORE conservative (predicts positive less often), which INCREASES precision but DECREASES recall - opposite of what's needed. Would miss even more pneumonia cases.
- F1 score balances precision and recall equally (harmonic mean). Hospital explicitly prioritizes not missing cases (recall), not balanced performance. Equal weighting doesn't address their stated concern.
- AUC-ROC measures overall discrimination at all thresholds but doesn't address the specific operating point. Can have high AUC but still miss cases if threshold is set wrong. Doesn't directly solve the recall problem.

> ⚠️ Frequently missed on the exam — read the options carefully.

</details>

---

**Q3.** An autonomous vehicle company is training a reinforcement learning model for lane-keeping. The model receives a reward of +1 for staying centered in the lane, 0 for minor deviations, and -10 for crossing lane markers. Training requires 100 million environment interactions. Which workload characteristic differentiates this from supervised learning training?

A) Reinforcement learning requires labeled training data with ground-truth steering angles for each frame
B) Reinforcement learning uses only forward passes without backpropagation during training
C) Reinforcement learning inference requires GPU acceleration while supervised learning inference can use CPU
D) Reinforcement learning training generates its own data through environment interaction and cannot pre-batch all training samples

<details>
<summary>Answer</summary>

**Correct: D**

Key difference: RL agent learns by trying actions and getting rewards (environment interaction), not from pre-labeled dataset. Training data is generated on-the-fly as model explores. Cannot pre-load all 100M samples because they don't exist yet - they're created during training based on model's current policy. This affects infrastructure (need simulation environment) and training pipeline (sequential, not pure batch).

**Why the others are wrong:**
- OPPOSITE is true. Reinforcement learning specifically does NOT require labeled data - that's supervised learning. RL uses rewards (scalar feedback), not labels. The question setup describes rewards (+1, 0, -10), not labels.
- RL still uses backpropagation to update neural network weights based on policy gradients or value function errors. It's not just forward passes. The learning algorithm (policy gradient, Q-learning, etc.) uses backprop to optimize the policy network.
- Both RL and supervised learning can benefit from GPU acceleration for inference if using deep neural networks. The hardware requirement depends on model architecture, not learning paradigm. This is not a distinguishing characteristic.

</details>

---

**Q4.** A research team is training a large language model with 70 billion parameters on a cluster of 64 NVIDIA A100 GPUs. Each GPU has 80GB of memory. During training with a batch size of 4 per GPU, they observe GPU memory usage at 75GB per GPU and training throughput of 120 samples/second across the cluster. Which factor is MOST likely the bottleneck preventing them from increasing batch size to 8 per GPU?

A) GPU compute capacity (TFLOPS) cannot process larger batches within the target iteration time
B) PCIe bandwidth between CPU and GPU cannot transfer larger batches fast enough
C) Activation memory during forward and backward passes would exceed 80GB GPU memory limit
D) NVLink bandwidth between GPUs cannot synchronize gradients for larger batch sizes

<details>
<summary>Answer</summary>

**Correct: C**

At 75GB with batch size 4, doubling to batch size 8 would approximately double activation memory (intermediate layer outputs stored for backprop). 75GB * 2 = 150GB, which exceeds 80GB GPU limit. Memory is the hard constraint. Model weights (70B params * 2 bytes = 140GB for FP16) are already distributed across GPUs; activations scale with batch size and are the limiting factor.

**Why the others are wrong:**
- Compute bottleneck would show as low GPU utilization (<90%) or high iteration time, not memory pressure at 75GB/80GB. A100 has 312 TFLOPS FP16, rarely the bottleneck vs memory for large models. Question states memory is nearly full.
- PCIe transfers data between CPU and GPU, but in distributed training, training data and model are already on GPUs. PCIe isn't the primary data path during training iterations - that's GPU-to-GPU (NVLink) for gradient sync. PCIe would bottleneck data loading between epochs, not batch size.
- NVLink bandwidth (600 GB/s per A100) is typically not the bottleneck for gradient synchronization in batch size 4 → 8 range. Gradient size is constant (model size), not dependent on batch size. NVLink issues would manifest as long synchronization delays, not memory overflow.

> ⚠️ Frequently missed on the exam — read the options carefully.

</details>

---

**Q5.** A computer vision team is deploying an object detection model for real-time video analysis. The model must process 30 frames per second with maximum 20ms latency per frame. They are comparing inference on NVIDIA T4 GPU (260 TOPS INT8) versus CPU (Intel Xeon, 16 cores). The model performs 500 GFLOPS of matrix operations per frame. Which statement BEST explains why GPU is required for this workload?

A) The 20ms latency requirement demands parallel execution of matrix operations that CPU sequential processing cannot achieve
B) CPUs cannot execute the matrix multiplication operations required by convolutional neural networks
C) GPUs have higher clock speeds (GHz) than CPUs, enabling faster completion of each matrix operation
D) GPU memory bandwidth is required to store the 30 frames per second of video data

<details>
<summary>Answer</summary>

**Correct: A**

500 GFLOPS per frame / 0.02 seconds = 25 TFLOPS required sustained performance. CPU with 16 cores might achieve 1-2 TFLOPS peak on matrix ops (much less sustained). GPU's thousands of cores (2560 CUDA cores in T4) execute matrix multiplications in parallel - critical for meeting 20ms deadline. Parallelism, not clock speed, is the key advantage for matrix-heavy workloads.

**Why the others are wrong:**
- CPUs CAN execute matrix multiplication (BLAS libraries, AVX-512 instructions). They're just much slower because they lack massive parallelism. This is about performance, not capability. The question asks why GPU is REQUIRED for the latency constraint, not whether CPU can theoretically do it.
- OPPOSITE is true. Modern CPUs often run at 3-4 GHz, while GPU cores run at 1-2 GHz. GPUs win through parallelism (thousands of slower cores), not clock speed. This is a fundamental misunderstanding of GPU architecture.
- Memory bandwidth matters but isn't the primary reason. Video frames are loaded once into memory; inference performs many operations on that data. 30 fps at 1920×1080×3 bytes = ~200 MB/s, well within both CPU and GPU bandwidth. The bottleneck is compute (matrix ops), not memory I/O for this workload.

</details>

---

**Q6.** A natural language processing team is training a sentiment analysis model using supervised learning. The training dataset contains 1 million customer reviews labeled as positive, negative, or neutral. During training, they use a batch size of 256, learning rate of 0.001, and train for 10 epochs. Which statement correctly describes the training process?

A) The model will perform exactly 3,906 weight updates (1M samples / 256 batch size) during training
B) Each epoch processes all 1 million samples exactly once, with the model seeing each review 10 times total across all epochs
C) The model will perform exactly 39,060 weight updates (3,906 per epoch × 10 epochs) during training
D) The learning rate of 0.001 determines how many samples are processed before updating model weights

<details>
<summary>Answer</summary>

**Correct: B**

This correctly describes epochs: 1 epoch = one complete pass through entire dataset. 10 epochs = model sees each of the 1M reviews exactly 10 times. Order may shuffle between epochs, but coverage is complete. This is the definition of epoch-based training.

**Why the others are wrong:**
- This is weight updates for ONE epoch only. 1,000,000 / 256 = 3,906 batches per epoch, so 3,906 updates per epoch. But training runs for 10 epochs, so this undercounts by 10x. Confuses single epoch with total training.
- While the math (3,906 × 10 = 39,060) is correct for total updates, the question asks which statement correctly describes the process. Option C describes the fundamental concept (what epochs mean), while this option just does arithmetic. Both are technically true, but C is the BEST answer because it explains the training paradigm, not just counts updates. In exam context, conceptual understanding beats calculation.
- Learning rate determines step size of weight updates (how much to adjust weights), NOT batch size or update frequency. Batch size (256) determines update frequency. This fundamentally confuses two hyperparameters. Learning rate of 0.001 means 'adjust weights by 0.1% of gradient direction', not sample count.

</details>

---

**Q7.** An e-commerce company deploys a product recommendation model that uses unsupervised learning to cluster similar products. The model runs inference every time a customer views a product page, returning 10 recommended products. Average inference latency is 15ms with p99 latency of 45ms. The team wants to reduce p99 latency to under 25ms. Which optimization approach is MOST appropriate for inference workload characteristics?

A) Increase training batch size from 512 to 2048 to improve model convergence and reduce inference latency
B) Add more labeled training data to improve model accuracy and reduce inference complexity
C) Use mixed precision (FP16) inference to reduce memory bandwidth and computation time for matrix operations
D) Increase the number of training epochs from 50 to 100 to ensure better model generalization at inference

<details>
<summary>Answer</summary>

**Correct: C**

Inference optimization requires reducing forward pass computation time. Mixed precision (FP16 vs FP32) provides ~2x speedup on Tensor Core GPUs: smaller data = less memory bandwidth, faster math operations. This directly reduces latency. Targets the inference workload, not training. Likely to bring p99 from 45ms toward 25ms target.

**Why the others are wrong:**
- Training batch size affects training speed and convergence, NOT inference latency. Once model is trained, batch size during training is irrelevant to inference performance. This confuses training hyperparameters with inference optimization.
- Unsupervised learning doesn't use labeled data - that's the definition of unsupervised. Also, adding training data affects model quality, not inference speed. Inference latency depends on model architecture and execution, not training data quantity. Two errors: wrong learning paradigm and wrong optimization target.
- Training epochs affect model accuracy and generalization, NOT inference latency. Inference latency is determined by model architecture (layer count, width) and execution (GPU, precision, batching). Number of training epochs doesn't change the forward pass computation. Another training vs inference confusion.

> ⚠️ Frequently missed on the exam — read the options carefully.

</details>

---

**Q8.** A healthcare analytics company is evaluating a diagnostic model with the following performance on a test set of 1,000 patients: 180 true positives, 20 false positives, 720 true negatives, 80 false negatives. Regulatory requirements mandate that the model must achieve at least 80% recall and 85% precision. Does this model meet the requirements?

A) Yes, the model achieves 90% precision and 69% recall, exceeding the precision requirement
B) No, the model achieves 69% recall and 90% precision, failing to meet the recall requirement
C) Yes, the model achieves 82% recall and 86% precision, meeting both requirements
D) No, the model achieves 90% recall and 82% precision, failing to meet the precision requirement

<details>
<summary>Answer</summary>

**Correct: B**

Recall = TP/(TP+FN) = 180/(180+80) = 180/260 = 69.2%. Precision = TP/(TP+FP) = 180/(180+20) = 180/200 = 90%. Recall of 69% fails the 80% requirement. Precision of 90% exceeds 85% requirement. Model fails overall because it doesn't meet BOTH requirements.

**Why the others are wrong:**
- Gets precision correct (90%) but recall wrong. Recall calculation shown above is 69%, not 69% stated here... wait, this option says 'exceeding precision requirement' but the model FAILS on recall. This is internally inconsistent and wrong conclusion (says 'Yes' when answer is 'No').
- Correct calculations but wrong conclusion. States the model meets requirements when it clearly fails recall requirement (69% < 80%). Both metrics are needed; failing one means overall failure.
- Wrong calculations. Recall = 180/260 = 69%, not 90%. Precision = 180/200 = 90%, not 82%. Numbers are swapped and both incorrect. Shows misunderstanding of confusion matrix arithmetic.

</details>

---

