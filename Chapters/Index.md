# AI Systems Performance Engineering

*Uploaded by Pedro Valera - AI Specialist*

---


# Index

### Symbols

- #pragma unroll, Exposing Instruction-Level Parallelism, Loop Unrolling, Interleaving, and Compiler Hintingunrolling too many iterations, Profiling and Mitigating Register Pressure
- 100-trillion parameter models, Toward 100-Trillion-Parameter ModelsNVIDIA supercomputers in a rack, NVIDIA’s “AI Supercomputer in a Rack”-NVIDIA’s “AI Supercomputer in a Rack”reason for discussing, Toward 100-Trillion-Parameter Modelsscaling toward, Scaling Toward Multimillion GPU Clusters and 100-Trillion-Parameter Models-Scaling Toward Multimillion GPU Clusters and 100-Trillion-Parameter Models
- 3FS (see Fire-Flyer File System (3FS; DeepSeek))
- <<< >>> (chevron syntax)CUDA-specific launch parameters, CUDA Programming Refresherhow many threads, CUDA Programming Refresher
- __global__, CUDA Programming Refresher
- __launch_bounds__, Tuning Occupancy with Launch Bounds-Tuning Occupancy with Launch Bounds, Techniques for Occupancy Tuningoccupancy optimization, Compiler Hints to Optimize Occupancy
- __ldg(), Read-Only Data Caches
- __shared__ regions, Distributed Shared Memory
- __shfl_sync, Warp Shuffle Intrinsics: Avoid Shared Memory and Explicit Synchronization-Warp Shuffle Intrinsics: Avoid Shared Memory and Explicit Synchronization
- __syncthreads(), Cooperative Groups
- ~680-billion parameter models, DeepSeek Scales to ~680-Billion Parameter Models Despite US Export Hardware Restrictions in China-DeepSeek Scales to ~680-Billion Parameter Models Despite US Export Hardware Restrictions in China

- unrolling too many iterations, Profiling and Mitigating Register Pressure

- NVIDIA supercomputers in a rack, NVIDIA’s “AI Supercomputer in a Rack”-NVIDIA’s “AI Supercomputer in a Rack”
- reason for discussing, Toward 100-Trillion-Parameter Models
- scaling toward, Scaling Toward Multimillion GPU Clusters and 100-Trillion-Parameter Models-Scaling Toward Multimillion GPU Clusters and 100-Trillion-Parameter Models

- CUDA-specific launch parameters, CUDA Programming Refresher
- how many threads, CUDA Programming Refresher

- occupancy optimization, Compiler Hints to Optimize Occupancy

### A

- achieved occupancy, Maintaining High Occupancy and GPU Utilization, Inspecting Achieved Occupancy and GPU Utilizationinspecting, Inspecting Achieved Occupancy and GPU Utilization-Optimizing the Kernel, Dynamic Scheduling with Atomic Work Queueskernel compute throughput versus GPU FLOPS, Kernel Compute Throughput Versus Peak GPU FLOPSkernel memory throughput versus HBM, Kernel Memory Throughput Versus Peak HBM Memory Bandwidthparallelism to increase, Maintaining High Occupancy and GPU Utilization
- activation checkpointing, Activation Checkpointing for Memory SavingsFSDP automatic checkpointing, FSDP Automatic Checkpointing and Offloading-FSDP Automatic Checkpointing and Offloading
- adaptive batching, Adaptive Batching and Chunked Prefill Scheduling-Adaptive Batching and Chunked Prefill Scheduling
- adaptive routing, Adaptive Expert Routing and Real-Time Monitoring, Multinode and Multirack Communication with GPUDirect RDMA
- admission control (see early rejection)
- AI agents that self-improve, Self-Improving AI Agents (AI Futures Project)-Self-Improving AI Agents (AI Futures Project)
- AI factories via NVL72 building blocks, Co-Packaged Optics: Future of Networking Hardware
- AI Futures Project, Self-Improving AI Agents (AI Futures Project)-Self-Improving AI Agents (AI Futures Project)
- AI optimizing AI systems, AI-Assisted Real-Time System Optimizations and Cluster Operations-AI-Assisted Real-Time System Optimizations and Cluster Operations
- AI supercomputers in a rack (NVIDIA), NVIDIA’s “AI Supercomputer in a Rack”-NVIDIA’s “AI Supercomputer in a Rack”about, AI System Hardware Overviewcache coherency, The CPU and GPU SuperchipCPU + GPU superchip, The CPU and GPU Superchip-Streaming Multiprocessor, Threads, and WarpsGPU Tensor Cores and Transformer Engine, NVIDIA GPU Tensor Cores and Transformer Engine-NVIDIA GPU Tensor Cores and Transformer Enginemany GPUs as one, Ultrascale Networking Treating Many GPUs as One-Co-Packaged Optics: Future of Networking Hardwaremulti-GPU programming, Multi-GPU Programming-Multi-GPU Programmingmultirack and storage communication, Multirack and Storage CommunicationNVIDIA Blackwell dual-die GPU, NVIDIA Blackwell “Dual-Die” GPU-NVIDIA Blackwell “Dual-Die” GPUNVIDIA Grace CPU, NVIDIA Grace CPUNVL72, Ultrascale Networking Treating Many GPUs as OneNVLink and NVSwitch, NVLink and NVSwitch-NVLink and NVSwitchSHARP for in-network aggregations, In-Network Aggregations with NVIDIA SHARP-In-Network Aggregations with NVIDIA SHARPstreaming multiprocessors, threads, warps, Streaming Multiprocessor, Threads, and Warps-Streaming Multiprocessor, Threads, and Warps
- AI systems performance engineers, The AI Systems Performance Engineerbenchmarking and profiling, Benchmarking and Profilingcross-team collaboration, Cross-Team Collaborationfuture capabilities of AI, Scaling Toward Multimillion GPU Clusters and 100-Trillion-Parameter Modelsgenerating returns well above cost, Measuring “Goodput” Useful Throughputmanaging resources efficiently, Managing Resources Efficientlyoverview, The AI Systems Performance Engineerperformance monitoring, Performance Monitoring and Utilization in PracticeGPUs near 100% utilized, Performance Monitoring and Utilization in PracticeNVLink usage, Performance Monitoring and Utilization in Practicescaling distributed training and inference, Scaling Distributed Training and Inferencetransparency and reproducibility, Transparency and Reproducibility
- air cooling versus liquid cooling, Liquid Cooling Versus Air Cooling-Liquid Cooling Versus Air Cooling
- all-to-all communication, Expert Communication Optimizationmixture-of-experts models, Expert Communication Optimization
- AlphaTensor (DeepMind), AlphaTensor AI-Discovered Algorithms Boosting GPU Performance (Google DeepMind)
- Amazon FSx for Lustre, Distributed, Parallel Filesystems and Object Storesperformance improvement verification, Distributed, Parallel Filesystems and Object Stores
- Amazon S3, Distributed, Parallel Filesystems and Object Stores
- AMP (see automatic mixed precision (AMP; PyTorch))
- AOT Autograd, AOT Autograd Fusion for Forward and Backward PassesFX Graph, TorchDynamo for Bytecode Capture and Graph Extraction, AOT Autograd Fusion for Forward and Backward PassesPyTorch compiler pipeline, PyTorch Compiler Deep Dive
- AOTInductor (PyTorch), TorchInductor Backend Code Generation
- application performance management (APM), Monitoring System Metrics and Counters
- application-level inference optimizations, Application-Level Optimizations-Token Output Limits and Timeoutsdebouncing and request coalescing, Debouncing and Request Coalescingmodel cascading and tiered deployment, Model Cascading and Tiered Model Deployment-Model Cascading and Tiered Model Deploymentprefix caching, Prefix Caching-Prefix Cachingprompt cleansing, Prompt Cleansingprompt compression, Prompt Compressionstreaming responses, Streaming Responses-Streaming Responsestoken output limits and timeouts, Token Output Limits and Timeouts
- Apptainer, Reduce Image Size for Faster Container Startup
- architecture of GPUs, Understanding GPU Architecture-CUDA GPU Backward and Forward Compatibility Modelmaximum threads per SM and block, Tuning Occupancy with Launch Boundsmemory hierarchy, Understanding GPU Memory Hierarchy-Understanding GPU Memory Hierarchyparallelism, Understanding GPU ArchitecturecuTile, C++ and Python CUDA LibrariesData Parallel versus Distributed Data Parallel, Distributed Data Parallel Strategies-Distributed Data Parallel Strategiesfully sharded data parallel, Distributed Data Parallel StrategiesPTX, Inline PTX and SASS Tuning for MicrooptimizationsSpecial Function Unit, Understanding GPU Architecturestreaming multiprocessors, Understanding GPU Architecturewarps, Streaming Multiprocessor, Threads, and Warps, Understanding GPU Architecture-Understanding GPU Architecturethread block clusters, Threads, Warps, Blocks, and Gridsthread blocks, Threads, Warps, Blocks, and Grids-Threads, Warps, Blocks, and Gridsthread block size, Choosing Threads-per-Block and Blocks-per-Grid Sizesthreads, Threads, Warps, Blocks, and Grids-Threads, Warps, Blocks, and Gridsthreads-per-block and blocks-per-grid, Choosing Threads-per-Block and Blocks-per-Grid Sizes-Choosing Threads-per-Block and Blocks-per-Grid Sizes
- arithmetic intensity, Roofline Model: Compute-Bound or Memory-Bound Workloads, Increasing CUDA Kernel Efficiency and Arithmetic Intensitybatching increasing, Structured SparsityCUTLASS for optimal, Using CUTLASS for Optimal Arithmetic Intensity and Tensor Core Performance-Using CUTLASS for Optimal Arithmetic Intensity and Tensor Core Performancedata reuse increasing, Multilevel Microtiling and Software Prefetchingdistributed shared memory increasing, Distributed Shared Memoryincreasing, Increasing CUDA Kernel Efficiency and Arithmetic Intensity, Kernel Fusion-Kernel Fusionkernel fusion increasing, Kernel Fusion-Kernel Fusionlow versus high arithmetic intensity, Increasing CUDA Kernel Efficiency and Arithmetic Intensityprefill and decode phase differences, Adaptive Batching and Chunked Prefill Schedulingpruning increasing, Structured SparsityPyTorch, PyTorch and Arithmetic Intensityrecomputation of cheap expressions increasing, Recomputation Versus Memory Trade-OffRoofline performance model, Increasing CUDA Kernel Efficiency and Arithmetic Intensity-Increasing CUDA Kernel Efficiency and Arithmetic Intensity, Roofline-Guided Scheduling and Orchestration Decisionslower precision with Tensor Cores, Mixed Precision and Utilizing Tensor Coressoftware prefetching, Multilevel Microtiling and Software Prefetchingtiling increasing, Tiling and Data Reuse Using Shared Memory, Multilevel Microtiling and Software Prefetchingmultilevel tiling, Multilevel Microtiling and Software Prefetchingsoftware prefetching, Multilevel Microtiling and Software Prefetching
- asynchronous execution with streams, Asynchronous Execution with Streams
- asynchronous memory transfers with CUTLASS, Using CUTLASS for Optimal Arithmetic Intensity and Tensor Core Performance
- AsyncTP (asynchronous tensor parallel), TorchTitan, AsyncTP, AutoParallel, and SimpleFSDP
- aten operationsFX Graph, TorchDynamo for Bytecode Capture and Graph ExtractionPrimTorch IR decomposing, PrimTorch IR (Prims) Simplified Operator SetPyTorch compiler pipeline, PyTorch Compiler Deep Dive
- atomic work queues, Dynamic Scheduling with Atomic Work Queues-Atomic Queuesatomic counters, Atomic CountersNsight Compute, Atomic Counters-Atomic Countersatomic queues, Atomic Queues-Atomic Queuesdynamic work allocation, Atomic CountersCUDA Graphs device initiated, Atomic Queues and Device-Initiated CUDA Graphs for In-Kernel Persistent Scheduling
- attentionAI-produced CUDA kernel for attention, Automated GPU Kernel Optimizations with DeepSeek-R1 (NVIDIA)-Automated GPU Kernel Optimizations with DeepSeek-R1 (NVIDIA)bottleneck fixed by FlashAttention, Mechanical Sympathy: Hardware-Software Codesigncontext parallelism communicating across partitions, Context (Sequence) ParallelismPyTorch optimized attention mechanisms, PyTorch Optimized Attention Mechanisms
- automated performance tests, Benchmarking and ProfilingAI optimizing AI systems, AI-Assisted Real-Time System Optimizations and Cluster Operations-AI-Assisted Real-Time System Optimizations and Cluster Operations
- automatic mixed precision (AMP; PyTorch), TF32 and Automatic Mixed Precision (PyTorch)-TF32 and Automatic Mixed Precision (PyTorch)
- AutoParallel (PyTorch), TorchTitan, AsyncTP, AutoParallel, and SimpleFSDP
- autotuningAI-assisted autotuning, AI-Assisted Real-Time System Optimizations and Cluster Operations-AI-Assisted Real-Time System Optimizations and Cluster Operationsstatic kernels transformed into adaptive, Kernel Autotuning for Transformer Self-Attention and MLP PathsTorchInductor, Using the PyTorch Compiler, Autotuning with TorchInductor-Autotuning with TorchInductortransformer self-attention and MLP paths, Kernel Autotuning for Transformer Self-Attention and MLP Paths-Kernel Autotuning for Transformer Self-Attention and MLP PathsTriton kernels, Autotuning Triton Kernels

- inspecting, Inspecting Achieved Occupancy and GPU Utilization-Optimizing the Kernel, Dynamic Scheduling with Atomic Work Queueskernel compute throughput versus GPU FLOPS, Kernel Compute Throughput Versus Peak GPU FLOPSkernel memory throughput versus HBM, Kernel Memory Throughput Versus Peak HBM Memory Bandwidth
- parallelism to increase, Maintaining High Occupancy and GPU Utilization

- kernel compute throughput versus GPU FLOPS, Kernel Compute Throughput Versus Peak GPU FLOPS
- kernel memory throughput versus HBM, Kernel Memory Throughput Versus Peak HBM Memory Bandwidth

- FSDP automatic checkpointing, FSDP Automatic Checkpointing and Offloading-FSDP Automatic Checkpointing and Offloading

- about, AI System Hardware Overview
- cache coherency, The CPU and GPU Superchip
- CPU + GPU superchip, The CPU and GPU Superchip-Streaming Multiprocessor, Threads, and Warps
- GPU Tensor Cores and Transformer Engine, NVIDIA GPU Tensor Cores and Transformer Engine-NVIDIA GPU Tensor Cores and Transformer Engine
- many GPUs as one, Ultrascale Networking Treating Many GPUs as One-Co-Packaged Optics: Future of Networking Hardware
- multi-GPU programming, Multi-GPU Programming-Multi-GPU Programming
- multirack and storage communication, Multirack and Storage Communication
- NVIDIA Blackwell dual-die GPU, NVIDIA Blackwell “Dual-Die” GPU-NVIDIA Blackwell “Dual-Die” GPU
- NVIDIA Grace CPU, NVIDIA Grace CPU
- NVL72, Ultrascale Networking Treating Many GPUs as One
- NVLink and NVSwitch, NVLink and NVSwitch-NVLink and NVSwitch
- SHARP for in-network aggregations, In-Network Aggregations with NVIDIA SHARP-In-Network Aggregations with NVIDIA SHARP
- streaming multiprocessors, threads, warps, Streaming Multiprocessor, Threads, and Warps-Streaming Multiprocessor, Threads, and Warps

- benchmarking and profiling, Benchmarking and Profiling
- cross-team collaboration, Cross-Team Collaboration
- future capabilities of AI, Scaling Toward Multimillion GPU Clusters and 100-Trillion-Parameter Models
- generating returns well above cost, Measuring “Goodput” Useful Throughput
- managing resources efficiently, Managing Resources Efficiently
- overview, The AI Systems Performance Engineer
- performance monitoring, Performance Monitoring and Utilization in PracticeGPUs near 100% utilized, Performance Monitoring and Utilization in PracticeNVLink usage, Performance Monitoring and Utilization in Practice
- scaling distributed training and inference, Scaling Distributed Training and Inference
- transparency and reproducibility, Transparency and Reproducibility

- GPUs near 100% utilized, Performance Monitoring and Utilization in Practice
- NVLink usage, Performance Monitoring and Utilization in Practice

- mixture-of-experts models, Expert Communication Optimization

- performance improvement verification, Distributed, Parallel Filesystems and Object Stores

- FX Graph, TorchDynamo for Bytecode Capture and Graph Extraction, AOT Autograd Fusion for Forward and Backward Passes
- PyTorch compiler pipeline, PyTorch Compiler Deep Dive

- debouncing and request coalescing, Debouncing and Request Coalescing
- model cascading and tiered deployment, Model Cascading and Tiered Model Deployment-Model Cascading and Tiered Model Deployment
- prefix caching, Prefix Caching-Prefix Caching
- prompt cleansing, Prompt Cleansing
- prompt compression, Prompt Compression
- streaming responses, Streaming Responses-Streaming Responses
- token output limits and timeouts, Token Output Limits and Timeouts

- maximum threads per SM and block, Tuning Occupancy with Launch Bounds
- memory hierarchy, Understanding GPU Memory Hierarchy-Understanding GPU Memory Hierarchy
- parallelism, Understanding GPU ArchitecturecuTile, C++ and Python CUDA LibrariesData Parallel versus Distributed Data Parallel, Distributed Data Parallel Strategies-Distributed Data Parallel Strategiesfully sharded data parallel, Distributed Data Parallel StrategiesPTX, Inline PTX and SASS Tuning for Microoptimizations
- Special Function Unit, Understanding GPU Architecture
- streaming multiprocessors, Understanding GPU Architecturewarps, Streaming Multiprocessor, Threads, and Warps, Understanding GPU Architecture-Understanding GPU Architecture
- thread block clusters, Threads, Warps, Blocks, and Grids
- thread blocks, Threads, Warps, Blocks, and Grids-Threads, Warps, Blocks, and Gridsthread block size, Choosing Threads-per-Block and Blocks-per-Grid Sizes
- threads, Threads, Warps, Blocks, and Grids-Threads, Warps, Blocks, and Grids
- threads-per-block and blocks-per-grid, Choosing Threads-per-Block and Blocks-per-Grid Sizes-Choosing Threads-per-Block and Blocks-per-Grid Sizes

- cuTile, C++ and Python CUDA Libraries
- Data Parallel versus Distributed Data Parallel, Distributed Data Parallel Strategies-Distributed Data Parallel Strategies
- fully sharded data parallel, Distributed Data Parallel Strategies
- PTX, Inline PTX and SASS Tuning for Microoptimizations

- warps, Streaming Multiprocessor, Threads, and Warps, Understanding GPU Architecture-Understanding GPU Architecture

- thread block size, Choosing Threads-per-Block and Blocks-per-Grid Sizes

- batching increasing, Structured Sparsity
- CUTLASS for optimal, Using CUTLASS for Optimal Arithmetic Intensity and Tensor Core Performance-Using CUTLASS for Optimal Arithmetic Intensity and Tensor Core Performance
- data reuse increasing, Multilevel Microtiling and Software Prefetching
- distributed shared memory increasing, Distributed Shared Memory
- increasing, Increasing CUDA Kernel Efficiency and Arithmetic Intensity, Kernel Fusion-Kernel Fusion
- kernel fusion increasing, Kernel Fusion-Kernel Fusion
- low versus high arithmetic intensity, Increasing CUDA Kernel Efficiency and Arithmetic Intensity
- prefill and decode phase differences, Adaptive Batching and Chunked Prefill Scheduling
- pruning increasing, Structured Sparsity
- PyTorch, PyTorch and Arithmetic Intensity
- recomputation of cheap expressions increasing, Recomputation Versus Memory Trade-Off
- Roofline performance model, Increasing CUDA Kernel Efficiency and Arithmetic Intensity-Increasing CUDA Kernel Efficiency and Arithmetic Intensity, Roofline-Guided Scheduling and Orchestration Decisionslower precision with Tensor Cores, Mixed Precision and Utilizing Tensor Cores
- software prefetching, Multilevel Microtiling and Software Prefetching
- tiling increasing, Tiling and Data Reuse Using Shared Memory, Multilevel Microtiling and Software Prefetchingmultilevel tiling, Multilevel Microtiling and Software Prefetchingsoftware prefetching, Multilevel Microtiling and Software Prefetching

- lower precision with Tensor Cores, Mixed Precision and Utilizing Tensor Cores

- multilevel tiling, Multilevel Microtiling and Software Prefetching
- software prefetching, Multilevel Microtiling and Software Prefetching

- FX Graph, TorchDynamo for Bytecode Capture and Graph Extraction
- PrimTorch IR decomposing, PrimTorch IR (Prims) Simplified Operator Set
- PyTorch compiler pipeline, PyTorch Compiler Deep Dive

- atomic counters, Atomic CountersNsight Compute, Atomic Counters-Atomic Counters
- atomic queues, Atomic Queues-Atomic Queuesdynamic work allocation, Atomic Counters
- CUDA Graphs device initiated, Atomic Queues and Device-Initiated CUDA Graphs for In-Kernel Persistent Scheduling

- Nsight Compute, Atomic Counters-Atomic Counters

- dynamic work allocation, Atomic Counters

- AI-produced CUDA kernel for attention, Automated GPU Kernel Optimizations with DeepSeek-R1 (NVIDIA)-Automated GPU Kernel Optimizations with DeepSeek-R1 (NVIDIA)
- bottleneck fixed by FlashAttention, Mechanical Sympathy: Hardware-Software Codesign
- context parallelism communicating across partitions, Context (Sequence) Parallelism
- PyTorch optimized attention mechanisms, PyTorch Optimized Attention Mechanisms

- AI optimizing AI systems, AI-Assisted Real-Time System Optimizations and Cluster Operations-AI-Assisted Real-Time System Optimizations and Cluster Operations

- AI-assisted autotuning, AI-Assisted Real-Time System Optimizations and Cluster Operations-AI-Assisted Real-Time System Optimizations and Cluster Operations
- static kernels transformed into adaptive, Kernel Autotuning for Transformer Self-Attention and MLP Paths
- TorchInductor, Using the PyTorch Compiler, Autotuning with TorchInductor-Autotuning with TorchInductor
- transformer self-attention and MLP paths, Kernel Autotuning for Transformer Self-Attention and MLP Paths-Kernel Autotuning for Transformer Self-Attention and MLP Paths
- Triton kernels, Autotuning Triton Kernels

### B

- B200 (NVIDIA Blackwell), Introduction and AI System Overview, NVIDIA Blackwell “Dual-Die” GPU, Choosing Threads-per-Block and Blocks-per-Grid Sizes, Understanding GPU Memory Hierarchy
- B300 (NVIDIA Blackwell Ultra), Toward 100-Trillion-Parameter Models, Blackwell Ultra and Grace Blackwell Ultra
- bandwidthTest of CPU-GPU memory data transfer, NUMA-Friendly Memory Allocation and Memory Pinning
- batchingadaptive batching, Adaptive Batching and Chunked Prefill Scheduling-Adaptive Batching and Chunked Prefill Schedulingarithmetic intensity increased, Structured Sparsitycomparing static, dynamic, continuous, Continuous Batchingcontinuous batching, Continuous Batchingdynamic batching, Dynamic Batching-Dynamic Batchingstatic batching, Dynamic Batching
- benchmarksAI systems performance engineer, Benchmarking and ProfilingbandwidthTest, NUMA-Friendly Memory Allocation and Memory PinningCUDA Graphs impact, Capturing a CUDA Graph with a CUDA Streamdata pipelines, Multimodal Data Processing with NVIDIA DALII/O continuous profiling and tuning, Continuous Profiling and Tuning Workflow-Continuous Profiling and Tuning Workflowdynamic parallelism, Dynamic ParallelismGPU clock and, GPU Clock Speeds and ECCkernels generated by TorchInductor, Profiling and Debugging Compiler Performance IssuesMLPerf, Transparency and Reproducibilityperformance benchmarking, Continuous Integration and Performance Benchmarking-Performance Benchmarks and MLPerf LoggingPyTorch performance heads-up display, Continuous Integration and Performance Benchmarking-PyTorch HUD Performance Dashboard
- best-fit with coalescing (BFC), Dynamic Memory-Allocation Switching (Slab Versus Caching Versus Stream-Ordered)
- BF16, BF16/FP16, FP8, and FP4 Reduced Precision
- Blackwell GPU (NVIDIA)100-trillion parameter models, Toward 100-Trillion-Parameter ModelsarchitectureB200, Choosing Threads-per-Block and Blocks-per-Grid Sizes, Understanding GPU Memory HierarchyB300, Blackwell Ultra and Grace Blackwell UltraFP32 and INT32 pipelines merged, Warp Scheduling and Dual Issue InstructionsBlackwell B200 GPU, NVIDIA Blackwell “Dual-Die” GPU, Choosing Threads-per-Block and Blocks-per-Grid Sizes, Understanding GPU Memory Hierarchyappearing as one GPU, NVIDIA Blackwell “Dual-Die” GPUchiplet approach, NVIDIA Blackwell “Dual-Die” GPUData Center GPU Manager, Performance Monitoring and Utilization in PracticeI/O monitoring, Monitoring Storage I/Oinference monitoring, Monitoring System Metrics and Countersdual-die GPU, NVIDIA Blackwell “Dual-Die” GPU-NVIDIA Blackwell “Dual-Die” GPUappearing as one GPU, NVIDIA Blackwell “Dual-Die” GPUchiplet approach, NVIDIA Blackwell “Dual-Die” GPUremote memory access latency, NUMA Awareness and CPU PinningFP32 80 TFLOP theoretical, Tiling and Data Reuse Using Shared Memoryglobal memory access, Vectorized Memory AccessILP configuration, ILP and Occupancymany GPUs as one, Ultrascale Networking Treating Many GPUs as One-Co-Packaged Optics: Future of Networking Hardwaremulti-GPU programming, Multi-GPU Programming-Multi-GPU Programming(see also multiple GPUs)multirack and storage communication, Multirack and Storage CommunicationSHARP for in-network aggregations, In-Network Aggregations with NVIDIA SHARP-In-Network Aggregations with NVIDIA SHARPmemory, The CPU and GPU SuperchipNVL72, Ultrascale Networking Treating Many GPUs as OneNVLink and NVSwitch, NVLink and NVSwitch-NVLink and NVSwitchroofline model, Roofline Model: Compute-Bound or Memory-Bound Workloads-Roofline Model: Compute-Bound or Memory-Bound Workloadsshared memory of 32 banks, Avoid Shared-Memory Bank Conflictsstreaming multiprocessors, threads, warps, Streaming Multiprocessor, Threads, and Warps-Streaming Multiprocessor, Threads, and WarpsTFLOPS, Tiling and Data Reuse Using Shared Memory
- Blackwell HBM3e, NVIDIA Blackwell “Dual-Die” GPU
- Blackwell NVL72 (see NVL72)
- Blackwell Ultra B300100-trillion parameter models, Toward 100-Trillion-Parameter ModelsNVL72 drop-in upgrade, Blackwell Ultra and Grace Blackwell Ultra
- blocksPerGrid, CUDA Programming Refresher, CUDA Programming Refresherlaunch parameter, Configuring Launch Parameters: Blocks per Grid and Threads per Block-Configuring Launch Parameters: Blocks per Grid and Threads per Block
- BlueField-3 DPU (NVIDIA), Multirack and Storage Communication
- book supplemental material online, Using Code Examples
- book web page, How to Contact Us
- bottlenecksattention mechanisms optimized by PyTorch, PyTorch Optimized Attention Mechanismscommunication versus compute, Diagnosing Communication- Versus Compute-Bound Workloadsexperts overloaded in MoE, Expert ParallelismGPU bottlenecks, Profiling and Diagnosing GPU Bottlenecks-Profiler-Guided Analysisabout, Profiling and Diagnosing GPU Bottlenecksdata pipeline profiling and tuning, Profiling and Tuning the Data Pipelineiteratively profiling, Iteratively Profiling and Determining the Kernel Bottleneck-Iteratively Profiling and Determining the Kernel Bottleneckkernel optimization, Optimizing the Kernel-Optimizing the KernelNsight Compute and Roofline analysis, Nsight Compute and Roofline AnalysisNsight Systems timeline view, Profiling and Diagnosing GPU BottlenecksPyTorch profiler via Kineto, PyTorch Profiler and Visualization Tools-Profiler-Guided Analysiswarp stall reasons, Analyzing Warp Stall Reasons with Nsight Compute-Other Stall Reasonsidentifying bottlenecks, Benchmarking and ProfilingI/O continuous profiling and tuning, Continuous Profiling and Tuning Workflow-Continuous Profiling and Tuning WorkflowNFS servers, Distributed, Parallel Filesystems and Object Storestuning parameters, Distributed, Parallel Filesystems and Object StoresNVIDIA mechanical sympathy, Mechanical Sympathy: Hardware-Software Codesignprofilingexample MoE model, Profiling PyTorch to Identify Bottlenecks-CPU and GPU Profiling with Linux perfLinux perf on CPU and GPU, CPU and GPU Profiling with Linux perf-CPU and GPU Profiling with Linux perfNsight Systems and NVTX Timelines, System Profiling with Nsight Systems and NVTX Timelines-System Profiling with Nsight Systems and NVTX TimelinesNVTX markers in profiling tools, NVTX Markers and Profiling Tools-NVTX Markers and Profiling Toolstools for profiling, NVTX Markers and Profiling Tools-NVTX Markers and Profiling Toolsshared filesystems, Distributed, Parallel Filesystems and Object StoresSHARP and adaptive routing minimizing, Multinode and Multirack Communication with GPUDirect RDMA
- bucketing (PyTorch), Reducing Communication Frequency and Volume
- butterfly schedules, Expert Communication Optimization

- adaptive batching, Adaptive Batching and Chunked Prefill Scheduling-Adaptive Batching and Chunked Prefill Scheduling
- arithmetic intensity increased, Structured Sparsity
- comparing static, dynamic, continuous, Continuous Batching
- continuous batching, Continuous Batching
- dynamic batching, Dynamic Batching-Dynamic Batching
- static batching, Dynamic Batching

- AI systems performance engineer, Benchmarking and Profiling
- bandwidthTest, NUMA-Friendly Memory Allocation and Memory Pinning
- CUDA Graphs impact, Capturing a CUDA Graph with a CUDA Stream
- data pipelines, Multimodal Data Processing with NVIDIA DALII/O continuous profiling and tuning, Continuous Profiling and Tuning Workflow-Continuous Profiling and Tuning Workflow
- dynamic parallelism, Dynamic Parallelism
- GPU clock and, GPU Clock Speeds and ECC
- kernels generated by TorchInductor, Profiling and Debugging Compiler Performance Issues
- MLPerf, Transparency and Reproducibility
- performance benchmarking, Continuous Integration and Performance Benchmarking-Performance Benchmarks and MLPerf LoggingPyTorch performance heads-up display, Continuous Integration and Performance Benchmarking-PyTorch HUD Performance Dashboard

- I/O continuous profiling and tuning, Continuous Profiling and Tuning Workflow-Continuous Profiling and Tuning Workflow

- PyTorch performance heads-up display, Continuous Integration and Performance Benchmarking-PyTorch HUD Performance Dashboard

- 100-trillion parameter models, Toward 100-Trillion-Parameter Models
- architectureB200, Choosing Threads-per-Block and Blocks-per-Grid Sizes, Understanding GPU Memory HierarchyB300, Blackwell Ultra and Grace Blackwell UltraFP32 and INT32 pipelines merged, Warp Scheduling and Dual Issue Instructions
- Blackwell B200 GPU, NVIDIA Blackwell “Dual-Die” GPU, Choosing Threads-per-Block and Blocks-per-Grid Sizes, Understanding GPU Memory Hierarchyappearing as one GPU, NVIDIA Blackwell “Dual-Die” GPUchiplet approach, NVIDIA Blackwell “Dual-Die” GPU
- Data Center GPU Manager, Performance Monitoring and Utilization in PracticeI/O monitoring, Monitoring Storage I/Oinference monitoring, Monitoring System Metrics and Counters
- dual-die GPU, NVIDIA Blackwell “Dual-Die” GPU-NVIDIA Blackwell “Dual-Die” GPUappearing as one GPU, NVIDIA Blackwell “Dual-Die” GPUchiplet approach, NVIDIA Blackwell “Dual-Die” GPUremote memory access latency, NUMA Awareness and CPU Pinning
- FP32 80 TFLOP theoretical, Tiling and Data Reuse Using Shared Memory
- global memory access, Vectorized Memory Access
- ILP configuration, ILP and Occupancy
- many GPUs as one, Ultrascale Networking Treating Many GPUs as One-Co-Packaged Optics: Future of Networking Hardwaremulti-GPU programming, Multi-GPU Programming-Multi-GPU Programming(see also multiple GPUs)multirack and storage communication, Multirack and Storage CommunicationSHARP for in-network aggregations, In-Network Aggregations with NVIDIA SHARP-In-Network Aggregations with NVIDIA SHARP
- memory, The CPU and GPU Superchip
- NVL72, Ultrascale Networking Treating Many GPUs as OneNVLink and NVSwitch, NVLink and NVSwitch-NVLink and NVSwitch
- roofline model, Roofline Model: Compute-Bound or Memory-Bound Workloads-Roofline Model: Compute-Bound or Memory-Bound Workloads
- shared memory of 32 banks, Avoid Shared-Memory Bank Conflicts
- streaming multiprocessors, threads, warps, Streaming Multiprocessor, Threads, and Warps-Streaming Multiprocessor, Threads, and Warps
- TFLOPS, Tiling and Data Reuse Using Shared Memory

- B200, Choosing Threads-per-Block and Blocks-per-Grid Sizes, Understanding GPU Memory Hierarchy
- B300, Blackwell Ultra and Grace Blackwell Ultra
- FP32 and INT32 pipelines merged, Warp Scheduling and Dual Issue Instructions

- appearing as one GPU, NVIDIA Blackwell “Dual-Die” GPU
- chiplet approach, NVIDIA Blackwell “Dual-Die” GPU

- I/O monitoring, Monitoring Storage I/O
- inference monitoring, Monitoring System Metrics and Counters

- appearing as one GPU, NVIDIA Blackwell “Dual-Die” GPU
- chiplet approach, NVIDIA Blackwell “Dual-Die” GPU
- remote memory access latency, NUMA Awareness and CPU Pinning

- multi-GPU programming, Multi-GPU Programming-Multi-GPU Programming(see also multiple GPUs)
- multirack and storage communication, Multirack and Storage Communication
- SHARP for in-network aggregations, In-Network Aggregations with NVIDIA SHARP-In-Network Aggregations with NVIDIA SHARP

- (see also multiple GPUs)

- NVLink and NVSwitch, NVLink and NVSwitch-NVLink and NVSwitch

- 100-trillion parameter models, Toward 100-Trillion-Parameter Models
- NVL72 drop-in upgrade, Blackwell Ultra and Grace Blackwell Ultra

- launch parameter, Configuring Launch Parameters: Blocks per Grid and Threads per Block-Configuring Launch Parameters: Blocks per Grid and Threads per Block

- attention mechanisms optimized by PyTorch, PyTorch Optimized Attention Mechanisms
- communication versus compute, Diagnosing Communication- Versus Compute-Bound Workloads
- experts overloaded in MoE, Expert Parallelism
- GPU bottlenecks, Profiling and Diagnosing GPU Bottlenecks-Profiler-Guided Analysisabout, Profiling and Diagnosing GPU Bottlenecksdata pipeline profiling and tuning, Profiling and Tuning the Data Pipelineiteratively profiling, Iteratively Profiling and Determining the Kernel Bottleneck-Iteratively Profiling and Determining the Kernel Bottleneckkernel optimization, Optimizing the Kernel-Optimizing the KernelNsight Compute and Roofline analysis, Nsight Compute and Roofline AnalysisNsight Systems timeline view, Profiling and Diagnosing GPU BottlenecksPyTorch profiler via Kineto, PyTorch Profiler and Visualization Tools-Profiler-Guided Analysiswarp stall reasons, Analyzing Warp Stall Reasons with Nsight Compute-Other Stall Reasons
- identifying bottlenecks, Benchmarking and Profiling
- I/O continuous profiling and tuning, Continuous Profiling and Tuning Workflow-Continuous Profiling and Tuning Workflow
- NFS servers, Distributed, Parallel Filesystems and Object Storestuning parameters, Distributed, Parallel Filesystems and Object Stores
- NVIDIA mechanical sympathy, Mechanical Sympathy: Hardware-Software Codesign
- profilingexample MoE model, Profiling PyTorch to Identify Bottlenecks-CPU and GPU Profiling with Linux perfLinux perf on CPU and GPU, CPU and GPU Profiling with Linux perf-CPU and GPU Profiling with Linux perfNsight Systems and NVTX Timelines, System Profiling with Nsight Systems and NVTX Timelines-System Profiling with Nsight Systems and NVTX TimelinesNVTX markers in profiling tools, NVTX Markers and Profiling Tools-NVTX Markers and Profiling Toolstools for profiling, NVTX Markers and Profiling Tools-NVTX Markers and Profiling Tools
- shared filesystems, Distributed, Parallel Filesystems and Object Stores
- SHARP and adaptive routing minimizing, Multinode and Multirack Communication with GPUDirect RDMA

- about, Profiling and Diagnosing GPU Bottlenecks
- data pipeline profiling and tuning, Profiling and Tuning the Data Pipeline
- iteratively profiling, Iteratively Profiling and Determining the Kernel Bottleneck-Iteratively Profiling and Determining the Kernel Bottleneck
- kernel optimization, Optimizing the Kernel-Optimizing the Kernel
- Nsight Compute and Roofline analysis, Nsight Compute and Roofline Analysis
- Nsight Systems timeline view, Profiling and Diagnosing GPU Bottlenecks
- PyTorch profiler via Kineto, PyTorch Profiler and Visualization Tools-Profiler-Guided Analysis
- warp stall reasons, Analyzing Warp Stall Reasons with Nsight Compute-Other Stall Reasons

- tuning parameters, Distributed, Parallel Filesystems and Object Stores

- example MoE model, Profiling PyTorch to Identify Bottlenecks-CPU and GPU Profiling with Linux perf
- Linux perf on CPU and GPU, CPU and GPU Profiling with Linux perf-CPU and GPU Profiling with Linux perf
- Nsight Systems and NVTX Timelines, System Profiling with Nsight Systems and NVTX Timelines-System Profiling with Nsight Systems and NVTX Timelines
- NVTX markers in profiling tools, NVTX Markers and Profiling Tools-NVTX Markers and Profiling Tools
- tools for profiling, NVTX Markers and Profiling Tools-NVTX Markers and Profiling Tools

### C

- C++CUDA C++ Pipeline API, Asynchronous Memory Prefetching and Tensor Memory Accelerator-Asynchronous Memory Prefetching and Tensor Memory Acceleratorhardware-software codesign, Asynchronous Memory Prefetching and Tensor Memory Acceleratorlibraries in CUDA Toolkit, C++ and Python CUDA Libraries
- C-states and CPU frequency, CPU Frequency and C-states
- cache coherency, The CPU and GPU SuperchipNVLink-C2C, The CPU and GPU Superchip, Multi-GPU Programming
- caching mechanismscaching CUDA kernels, Continuous Prewarming of CUDA Graphs and Caches Using Time-Series Predictionmetrics for cache hits and misses, Profiling, Debugging, and Tuning Inference Performanceprefix caching, Prefix Caching-Prefix Cachingprewarming caches, Continuous Prewarming of CUDA Graphs and Caches Using Time-Series Prediction-Continuous Prewarming of CUDA Graphs and Caches Using Time-Series Predictionread-only data caches, Read-Only Data Caches-Read-Only Data Cachesconst __restrict__, Read-Only Data Caches, Read-Only Data Caches, Read-Only Data Cachespitfall of not stating read-only, Read-Only Data CachesvLLM LMCache, Profiling, Debugging, and Tuning Inference Performance, Zero-Copy GPU-to-GPU Transfer
- callback function between GPU and CPU, Fine-Grained Synchronization with Events and Callbacks
- CC (compute capability) and CUDA Toolkit version, CUDA Toolkit and Runtime
- CDU (Coolant Distribution Unit), Liquid Cooling Versus Air Cooling
- CFQ (completely fair queueing; Linux) obsolete, Tuning NVMe and Filesystem for Throughput
- CFS (Completely Fair Scheduler; Linux), Scheduler and Interrupt Affinity
- ChatGPT (OpenAI) stateful conversation, Prefix Caching
- checklist for AI systems performanceadvanced tuning strategies, Advanced Tuning Strategies and Algorithmic Tricks-Advanced Tuning Strategies and Algorithmic Tricksalgorithmic tricks, Advanced Tuning Strategies and Algorithmic Tricks-Advanced Tuning Strategies and Algorithmic Tricksarithmetic optimizations, Arithmetic Optimizations and Reduced/Mixed Precision-Arithmetic Optimizations and Reduced/Mixed Precisioncost optimization, Performance Tuning and Cost Optimization Mindset-Performance Tuning and Cost Optimization MindsetCUDA tuning optimizations, GPU Programming and CUDA Tuning Optimizations-GPU Programming and CUDA Tuning Optimizationsdata processing pipelines, Data Processing Pipelines-Data Processing PipelinesGPU programming, GPU Programming and CUDA Tuning Optimizations-GPU Programming and CUDA Tuning OptimizationsGPU resource management and scheduling, GPU Resource Management and Scheduling-GPU Resource Management and SchedulingI/O optimization, I/O Optimization-I/O Optimizationinference and serving, Efficient Inference and Serving-Efficient Inference and Servingmultinode, Multinode Inference and Serving-Multinode Inference and Servingkernel scheduling and execution optimizations, Kernel Scheduling and Execution Optimizations-Kernel Scheduling and Execution Optimizationsmulti-GPU scaling and interconnect optimizations, Multi-GPU Scaling and Interconnect Optimizations-Multi-GPU Scaling and Interconnect Optimizationsnetwork optimization, Distributed Training and Network Optimization-Distributed Training and Network Optimizationoperating system and driver optimizations, Operating System and Driver Optimizations-Operating System and Driver Optimizationsperformance profiling, debugging, and monitoring, Performance Profiling, Debugging, and Monitoring-Performance Profiling, Debugging, and Monitoringpower management, Power and Thermal Management-Power and Thermal Managementreduced/mixed precision, Arithmetic Optimizations and Reduced/Mixed Precision-Arithmetic Optimizations and Reduced/Mixed Precisionreproducibility and documentation, Reproducibility and Documentation Best Practices-Reproducibility and Documentation Best Practicessystem architecture and hardware planning, System Architecture and Hardware Planning-System Architecture and Hardware Planningthermal management, Power and Thermal Management-Power and Thermal Managementunified CPU-GPU “Superchip” architecture, Unified CPU-GPU “Superchip” Architecture-Unified CPU-GPU “Superchip” Architecture
- checkpoints written to disk, Filesystem Caching and Write-Back
- chevron syntax (<<< >>>)CUDA-specific launch parameters, CUDA Programming Refresherhow many threads, CUDA Programming Refresher
- child-kernel launch limit (CUDA), Dynamic Parallelism
- chiplet approach of dual-die GPUs, NVIDIA Blackwell “Dual-Die” GPUappearing as one GPU, NVIDIA Blackwell “Dual-Die” GPU
- Chrome tracing with Perfetto viewer, NVTX Markers and Profiling Tools, NVTX Markers and Profiling Tools
- chunked prefill, Stall-Free Scheduling (Chunked Prefill), Adaptive Batching and Chunked Prefill Scheduling-Adaptive Batching and Chunked Prefill Scheduling
- CI (see continuous integration (CI))
- cloud storage caches, Distributed, Parallel Filesystems and Object Storesperformance improvement verification, Distributed, Parallel Filesystems and Object Stores
- cluster pools, Disaggregated Prefill and Decode Cluster Pools-Memory management for the KV cache
- cluster scheduler, Sharing and Scheduling
- cluster size maximum, portable, Distributed Shared Memory
- co-packaged optics (CPO), Co-Packaged Optics: Future of Networking Hardware
- cold start training strategy, DeepSeek Scales to ~680-Billion Parameter Models Despite US Export Hardware Restrictions in China
- CollNet communication algorithm, NCCL Communication Algorithms
- communication (see distributed networking communication)
- compilerscommonly used compilers, NVIDIA Software Stackcompiling versus writing custom kernels, Compiling Versus Writing Custom Kernelsfloat8 not provided by CUDA, Vectorized Memory Access__global__, CUDA Programming Refresherinstruction-level parallelism exposed, Exposing Instruction-Level Parallelism-Profiling and Mitigating Register Pressurecompiler hints, Exposing Instruction-Level Parallelism__launch_bounds__, Tuning Occupancy with Launch Bounds-Tuning Occupancy with Launch Bounds, Techniques for Occupancy Tuningoccupancy optimization, Compiler Hints to Optimize Occupancynvcc CUDA compiler, CUDA Toolkit and Runtime(see also nvcc CUDA compiler)predication, Techniques to Avoid Warp DivergencePTX, CUDA GPU Backward and Forward Compatibility Model, TorchInductor Backend Code GenerationCUDA compatibility across GPU generations, CUDA Forward and Backward Compatibility Across GPU Hardware Generations, CUDA GPU Backward and Forward Compatibility ModelDeepSeek memory allocation optimization, DeepSeek’s Use of Inline PTX for Memory Allocation Optimizationinline PTX code, Inline PTX and SASS Tuning for Microoptimizations-Inline PTX and SASS Tuning for MicrooptimizationsJIT compiling, CUDA Forward and Backward Compatibility Across GPU Hardware Generationsstreaming assembler, Inline PTX and SASS Tuning for Microoptimizations-Inline PTX and SASS Tuning for Microoptimizationsstreaming assembler changing over generations, Inline PTX and SASS Tuning for MicrooptimizationsTorchInductor, TorchInductor Backend Code GenerationTriton, Triton Programming ModelPyTorch compiler, PyTorch and Higher-Level AI Frameworks, PyTorch Compiler (torch.compile)(see also PyTorch compiler (torch.compile))smart compilers, Smart Compilers and Automated Code Optimizations-Smart Compilers and Automated Code OptimizationsTriton, C++ and Python CUDA Libraries(see also Triton (OpenAI))vectorized memory access, Vectorized Memory Access-Vectorized Memory Accessvectorized memory operations, Vectorized Memory Access
- completely fair queueing (CFQ; Linux) obsolete, Tuning NVMe and Filesystem for Throughput
- Completely Fair Scheduler (CFS; Linux), Scheduler and Interrupt Affinity
- compressing data, Tuning, Replicating, and Compressing Data
- compute capability (CC) and CUDA Toolkit version, CUDA Toolkit and Runtime
- Compute Sanitizer (NVIDIA), Debugging Functional Correctness with NVIDIA Compute Sanitizercompute-sanitizer CLI, Debugging Functional Correctness with NVIDIA Compute SanitizerNVIDIA Tools Extension support, Debugging Functional Correctness with NVIDIA Compute Sanitizerconcurrency and synchronization, Using CUDA Streams with MoE Modelsfour primary tools, Debugging Functional Correctness with NVIDIA Compute Sanitizerinference correctness, Debugging Correctness Issues
- compute-bound GPU stalls, Iteratively Profiling and Determining the Kernel Bottleneck-Iteratively Profiling and Determining the Kernel Bottleneckkernel optimization, Optimizing the Kernel-Optimizing the Kernel
- compute-sanitizer CLI, Debugging Functional Correctness with NVIDIA Compute SanitizerNVIDIA Tools Extension support, Debugging Functional Correctness with NVIDIA Compute Sanitizer
- concurrency with CUDA streams, Concurrency with CUDA Streams-Using CUDA Streams with MoE Modelscompute overlapping with data transfers, Using Streams to Overlap Compute with Data Transfers-Using Streams to Overlap Compute with Data Transfersconcurrency, Overlapping Communication and Computation-Overlapping Communication and Computationlaunching on intended stream, Using CUDA Streams with MoE Modelsmixture-of-experts models, Using CUDA Streams with MoE ModelsNVIDIA Compute Sanitizer, Using CUDA Streams with MoE Modelsprofiling when introducing concurrency, Using CUDA Streams with MoE Modelsstream synchronization with events, Stream Synchronization with Events-Stream Synchronization with Events
- const __restrict__, Read-Only Data Caches, Read-Only Data Caches, Read-Only Data Caches
- “Contact” (Phish), Acknowledgments
- containerd with NVIDIA Container Toolkit, NVIDIA Container Runtime
- containers, Container Runtime Optimizations for GPUsNVIDIA Container Toolkit, Container Runtime Optimizations for GPUsCUDA compatibility, NVIDIA Container Toolkit and CUDA CompatibilityGPU performance, Container Runtime Optimizations for GPUsruntime optimizations for GPUs, Container Runtime Optimizations for GPUs-Reduce Image Size for Faster Container Startupcontainer overlay filesystem overhead, Avoiding Container Overlay Filesystem OverheadNVIDIA container runtime injecting libraries, NVIDIA Container RuntimeNVIDIA Container Toolkit and CUDA compatibility, NVIDIA Container Toolkit and CUDA Compatibilityreducing image size, Reduce Image Size for Faster Container Startup
- context parallelism (CP), Parallelism Strategies for Serving Massive MoE Models, Context (Sequence) ParallelismContext Parallel (PyTorch), PyTorch Optimized Attention Mechanismshybrid parallelism, Hybrid Parallelism
- continuous batching, Continuous Batching
- continuous integration (CI)Computer Sanitizer integrated, Debugging Functional Correctness with NVIDIA Compute Sanitizerperformance benchmarking, Continuous Integration and Performance Benchmarking-Performance Benchmarks and MLPerf Loggingforcing an error when full graph not captured, Graph Breaks and TorchDynamo explain()PyTorch performance heads-up display, Continuous Integration and Performance Benchmarking-PyTorch HUD Performance Dashboard
- continuous prewarming, Continuous Prewarming of CUDA Graphs and Caches Using Time-Series Prediction-Continuous Prewarming of CUDA Graphs and Caches Using Time-Series Prediction
- continuous scheduling, Continuous Scheduling
- Coolant Distribution Unit (CDU), Liquid Cooling Versus Air Cooling
- cooling via air versus liquid, Liquid Cooling Versus Air Cooling-Liquid Cooling Versus Air Cooling
- cooperative groups (CGs), Cooperative Groups-When to Combine Persistent Kernels and Cooperative GroupsCooperative Groups API, Warp Shuffle Intrinsics: Avoid Shared Memory and Explicit Synchronization, Cooperative Groupscoordinating thread block clusters, Coordinating Thread Block Clusters with Cooperative Groups API-Coordinating Thread Block Clusters with Cooperative Groups APIdescribed, Thread Block Clusters and Distributed Shared Memorylaunching a kernel in cooperative mode, Cooperative Groupsgrid size for launch, Cooperative Groupsreserving all SMs in the grid, When to Combine Persistent Kernels and Cooperative Groupspersistent kernels via grid sync, Cooperative Grid Synchronization and Persistent Kernels-Cooperative Grid Synchronization and Persistent Kernelswhen to combine, When to Combine Persistent Kernels and Cooperative Groupsthread block clusters contrasted with, Thread Block Clusters and Distributed Shared Memory
- Cooperative Groups API (CUDA), Warp Shuffle Intrinsics: Avoid Shared Memory and Explicit Synchronization, Cooperative Groupscoordinating thread block clusters, Coordinating Thread Block Clusters with Cooperative Groups API-Coordinating Thread Block Clusters with Cooperative Groups API
- cooperative thread array (CTA), Threads, Warps, Blocks, and Grids(see also thread blocks)runtime operation binding, Optimized KV Cache Memory Layout
- correctness issues in inference, Debugging Correctness Issues-Debugging Correctness Issues
- costsAI systems performance engineers generating returns, Measuring “Goodput” Useful Throughputcost optimization overview, Performance Tuning and Cost Optimization Mindset-Performance Tuning and Cost Optimization Mindsetmodel parameter-count explosion, Introduction and AI System Overview100-trillion parameter models, Toward 100-Trillion-Parameter Modelstrainingcold start strategy, DeepSeek Scales to ~680-Billion Parameter Models Despite US Export Hardware Restrictions in ChinaDeepSeek-R1, Introduction and AI System OverviewDeepSeek-V3, DeepSeek Scales to ~680-Billion Parameter Models Despite US Export Hardware Restrictions in ChinaGoogle Gemini Ultra, Introduction and AI System OverviewOpenAI GPT-4, Introduction and AI System Overview
- CP (see context parallelism (CP))
- CPO (co-packaged optics), Co-Packaged Optics: Future of Networking Hardware
- CPU + GPU superchip, The CPU and GPU Superchip-Streaming Multiprocessor, Threads, and WarpsNVIDIA Blackwell dual-die GPU, NVIDIA Blackwell “Dual-Die” GPU-NVIDIA Blackwell “Dual-Die” GPUappearing as one GPU, NVIDIA Blackwell “Dual-Die” GPUNVIDIA GPU Tensor Cores and Transformer Engine, NVIDIA GPU Tensor Cores and Transformer Engine-NVIDIA GPU Tensor Cores and Transformer EngineNVIDIA Grace CPU, NVIDIA Grace CPUNVLink-C2C, The CPU and GPU Superchipstreaming multiprocessors, threads, warps, Streaming Multiprocessor, Threads, and Warps-Streaming Multiprocessor, Threads, and Warps
- CPU and OS configuration, Configuring the CPUs and OS for GPU Environments-Tune Host CPU Memory Allocatorabout, Configuring the CPUs and OS for GPU EnvironmentsCPU frequency and C-states, CPU Frequency and C-statesfilesystem caching and write-back, Filesystem Caching and Write-BackNUMA awareness and CPU pinning, NUMA Awareness and CPU Pinning-NUMA Awareness and CPU PinningCPU + GPU superchip CPU-to-GPU data transfers, NUMA Awareness and CPU PinningCPU pinning, NUMA Awareness and CPU Pinning-NUMA Awareness and CPU PinningNUMA nodes, NUMA Awareness and CPU PinningNUMA-friendly memory allocation and pinning, NUMA-Friendly Memory Allocation and Memory Pinning-NUMA-Friendly Memory Allocation and Memory Pinningefficiency of pinned memory, NUMA-Friendly Memory Allocation and Memory Pinningmax locked memory setting, Transparent HugepagesOS limit on pinned memory, NUMA-Friendly Memory Allocation and Memory Pinningpinned memory for data loaders, NUMA-Friendly Memory Allocation and Memory Pinningscheduler and interrupt affinity, Scheduler and Interrupt Affinityirqbalance daemon, Scheduler and Interrupt Affinitytransparent hugepages, Transparent Hugepagestuning host CPU memory allocator, Tune Host CPU Memory Allocatorvirtual memory and swapping, Virtual Memory and Swapping
- CPU DRAMCPU pinning, NUMA Awareness and CPU Pinning-NUMA Awareness and CPU PinningCPU-GPU memory data-transfer bandwidth, NUMA-Friendly Memory Allocation and Memory Pinning, GPU Memory Oversubscription, Fragmentation, and Out-of-Memory HandlingNIXL GPU-to-GPU direct transfer, Separate Prefill and Decode Inference StagesCUDA programming flow, Understanding GPU ArchitectureCUDA Unified Memory, GPU Memory Oversubscription, Fragmentation, and Out-of-Memory Handlingprogramming CUDA, Unified Memory-Unified MemoryUnified CPU-GPU Memory, The CPU and GPU SuperchipGDS, NUMA-Friendly Memory Allocation and Memory Pinning, Using NVIDIA GDS-Measuring GDS with gdsio, Offloading Parameters to CPU and NVMehost CPU memory allocator tuned, Tune Host CPU Memory Allocatorlarge slower extension of GPU memory, The CPU and GPU Superchipsuperchips’ high-speed extension of GPU memory, Topology Awareness in NCCLNIXL offloading KV cache to, NVIDIA’s NIXL and Disaggregated Inference, KV Cache Offloading with NIXL-KV Cache Offloading with NIXLparameters offloaded to, Offloading Parameters to CPU and NVMepeer-to-peer DMA avoiding, Enabling Peer-to-Peer DMA and UCXswapping, GPU Memory Oversubscription, Fragmentation, and Out-of-Memory Handling
- CPU host and GPU device communication via CUDA events, Fine-Grained Synchronization with Events and Callbackscallback function, Fine-Grained Synchronization with Events and Callbacks
- CPython Frame Evaluation API, TorchDynamo for Bytecode Capture and Graph Extraction
- CTA (cooperative thread array), Threads, Warps, Blocks, and Grids(see also thread blocks)
- CTRL approach to prompt trimming, Prompt Cleansing
- CUBIN, CUDA Forward and Backward Compatibility Across GPU Hardware Generations
- cuBLASCUDA Toolkit, CUDA Toolkit and Runtime, PyTorch and Higher-Level AI FrameworksPyTorch delegating tasks to, PyTorch and Higher-Level AI Frameworks
- CUDAchild-kernel launch limit, Dynamic Parallelismcompatibility across GPU generations, CUDA Forward and Backward Compatibility Across GPU Hardware Generations, CUDA GPU Backward and Forward Compatibility ModelCUDA Graph launched, Capturing a CUDA Graph with a CUDA Stream-Capturing a CUDA Graph with a CUDA Streamfloat8 not provided, Vectorized Memory Accesskernels created in Python, NVIDIA Software Stackkernels executing asynchronously, CUDA Programming Refresherglobal fault flag for illegal operations, CUDA Programming Refresherprogrammingasynchronous memory allocation, Asynchronous Memory Allocation and Memory Pools-Asynchronous Memory Allocation and Memory PoolsCUDA streams, Asynchronous Memory Allocation and Memory Poolsflow between CPU and GPU, Understanding GPU Architecture__global__, CUDA Programming RefresherGPU memory hierarchy, Understanding GPU Memory Hierarchy-Understanding GPU Memory Hierarchyhigh occupancy and GPU utilization, Maintaining High Occupancy and GPU Utilization-Maintaining High Occupancy and GPU Utilizationhighest-level and most-recent APIs available, Asynchronous Memory Prefetching and Tensor Memory Acceleratorkernel errors, CUDA Programming Refresherkernel inputs in 1D, 2D and 3D Kernel Inputskernel inputs in 2D and 3D, 2D and 3D Kernel Inputskernels for parallel work, CUDA Programming Refresher-CUDA Programming Refresherlaunch parameters, Configuring Launch Parameters: Blocks per Grid and Threads per Block-Configuring Launch Parameters: Blocks per Grid and Threads per Blockmaximum threadsPerBlock compile time parameter, Tuning Occupancy with Launch Bounds-Tuning Occupancy with Launch Boundsmemory pools, Asynchronous Memory Allocation and Memory Pools-Asynchronous Memory Allocation and Memory Poolsminimum thread blocks resident on each SM, Tuning Occupancy with Launch Bounds-Tuning Occupancy with Launch BoundsNVIDIA Compute Sanitizer, Debugging Functional Correctness with NVIDIA Compute Sanitizeroccupancy tuning with __launch_bounds__, Tuning Occupancy with Launch Bounds-Tuning Occupancy with Launch Boundsroofline model, Roofline Model: Compute-Bound or Memory-Bound Workloads-Roofline Model: Compute-Bound or Memory-Bound WorkloadsUnified Memory, Unified Memory-Unified MemoryPython-based frameworks, PyTorch and Higher-Level AI FrameworksPython-centric libraries, Writing Custom Kernels with OpenAI TritonUnified Memory, GPU Memory Oversubscription, Fragmentation, and Out-of-Memory Handlingprogramming CUDA, Unified Memory-Unified MemoryUnified CPU-GPU Memory, The CPU and GPU Superchipversions of container libraries and host driver, NVIDIA Container Toolkit and CUDA CompatibilityNVIDIA Container Toolkit, NVIDIA Container Toolkit and CUDA Compatibility
- CUDA C++ Pipeline API, Asynchronous Memory Prefetching and Tensor Memory Accelerator-Asynchronous Memory Prefetching and Tensor Memory Acceleratorhardware-software codesign, Asynchronous Memory Prefetching and Tensor Memory Accelerator
- CUDA compiler (see nvcc CUDA compiler)
- CUDA eventscommunication from GPU device to CPU host, Fine-Grained Synchronization with Events and CallbacksCUDA stream synchronization, Fine-Grained Synchronization with Events and Callbacks, Stream Synchronization with Events-Stream Synchronization with Eventscross-stream synchronization, Using CUDA Events for Cross-Stream Synchronizationmulti-GPU overlap of compute and data, Multi-GPU Compute and Data Transfer Overlap with CUDA Streams-Multi-GPU Compute and Data Transfer Overlap with CUDA Streamsprofiling with, Using CUDA Events for Cross-Stream Synchronization
- CUDA Graphs, CUDA Graphs-Conditional Graph Nodesabout CUDA Graphs, Techniques for Occupancy Tuning, CUDA Graphsbenchmarking, Capturing a CUDA Graph with a CUDA Streambest practices, Best Practices for CUDA Graphs-Best Practices for CUDA Graphscapturing a CUDA Graph, Capturing a CUDA Graph with a CUDA Stream-Capturing a CUDA Graph with a CUDA Stream, Capturing a CUDA Graph and Preallocating Memory-Capturing a CUDA Graph and Preallocating Memorycapturing only a portion of pipeline, Capturing a CUDA Graph with a CUDA Streamdevice-initiated launch, Device-Initiated CUDA Graph Launch-Device-Initiated CUDA Graph Launchdynamic graph update, Dynamic Graph Updatewarm-up pass necessary, Capturing a CUDA Graph with a CUDA Streamcompiler modes that trigger, Autotuning with TorchInductorstatic shapes required, Autotuning with TorchInductor, Dynamic Shapes and Variable Sequence Lengthsconditional graph nodes, Conditional Graph Nodes-Conditional Graph Nodesnested, Conditional Graph NodesCUDA streams, Multi-GPU Compute and Data Transfer Overlap with CUDA Streamscapturing a CUDA Graph, Capturing a CUDA Graph with a CUDA Stream-Capturing a CUDA Graph with a CUDA StreamPyTorch, Multi-GPU Compute and Data Transfer Overlap with CUDA Streamsdevice-initiated launch, Device-Initiated CUDA Graph Launch-Device-Initiated CUDA Graph Launchdependency management, Device-Initiated CUDA Graph Launchexample of use, Device-Initiated CUDA Graph Launchpersistent scheduling, Atomic Queues and Device-Initiated CUDA Graphs for In-Kernel Persistent Schedulingdynamic graph update, Dynamic Graph Updateinference engines, PyTorch, Inference Engines, and CUDA Graphskernel launch overhead reduced, Reducing Kernel Launch Overhead with CUDA Graphs-CUDA Graph Trees (PyTorch Compiler Internal)kernels that use NVSHMEM, Fine-Grained GPU-to-GPU Memory Sharing with NVSHMEMlaunching a CUDA Graph, Capturing a CUDA Graph with a CUDA Stream-Capturing a CUDA Graph with a CUDA Streamdevice-initiated launch, Device-Initiated CUDA Graph Launch-Device-Initiated CUDA Graph Launchmemory management, Memory Pools for CUDA Graphsmemory pools, Replaying the Graphmultiple-GPU collectives captured, Capturing Multi-GPU Collectives with NCCL and CUDA Graphs-Capturing Multi-GPU Collectives with NCCL and CUDA Graphspitfalls, Capturing a CUDA Graph with a CUDA Streamprewarming, Continuous Prewarming of CUDA Graphs and Caches Using Time-Series Prediction-Continuous Prewarming of CUDA Graphs and Caches Using Time-Series PredictionPyTorch, PyTorch, Inference Engines, and CUDA Graphs, Capturing a CUDA Graph with a CUDA StreamCUDA Graph Trees, CUDA Graph Trees (PyTorch Compiler Internal)static memory pools, Memory Pools for CUDA Graphsreplaying the graph, Replaying the Graph-Replaying the Graphany stream for replay, Capturing a CUDA Graph and Preallocating MemoryTorchInductor, Autotuning with TorchInductorwhen to use dynamic parallelism instead, Dynamic Parallelism
- CUDA Managed Memory (see CUDA Unified Memory)
- CUDA memory allocator, Tuning the CUDA Memory Allocator
- CUDA Memory Inspector, Profiling and Tuning Memory in PyTorch
- CUDA Ninjas, Automated GPU Kernel Optimizations with DeepSeek-R1 (NVIDIA)
- CUDA Occupancy API, Techniques for Occupancy Tuning, Techniques for Occupancy Tuning, Determine Optimal Launch Configuration with the Occupancy APIcooperative grids, Cooperative Grid Synchronization and Persistent Kernels
- CUDA Pipeline APIdouble-buffering, Cooperative Tiling and Double-Buffering with the CUDA Pipeline API-Cooperative Tiling and Double-Buffering with the CUDA Pipeline APITMA plus, Asynchronous Memory Prefetching and Tensor Memory Acceleratorwarp specialization, Using CUDA Pipeline API for Warp Specialization-Using CUDA Pipeline API for Warp SpecializationPyTorch, PyTorch, CUDA Pipeline API, and Warp Specialization
- CUDA Profiling Tools Interface (CUPTI), PyTorch Profiler and Visualization Tools
- CUDA Program Counter Sampling (Nsight Compute), Profiling with Nsight Systems and Nsight Compute
- CUDA Python library (NVIDIA), NVIDIA Software Stack
- CUDA Runtime API callback function, Fine-Grained Synchronization with Events and Callbacks
- CUDA Runtime library, CUDA Toolkit and Runtime
- CUDA streams, Overlapping Kernel Execution with CUDA Streamsasynchronous execution, Asynchronous Execution with Streamscompute overlapping with data transfers, Using Streams to Overlap Compute with Data Transfers-Using Streams to Overlap Compute with Data Transfersconcurrency, Overlapping Communication and Computation-Overlapping Communication and Computationmultiple GPUs, Multi-GPU Compute and Data Transfer Overlap with CUDA Streams-Multi-GPU Compute and Data Transfer Overlap with CUDA Streamsconcurrency, Concurrency with CUDA Streams-Using CUDA Streams with MoE Modelslaunching on intended stream, Using CUDA Streams with MoE Modelsprofiling when introducing concurrency, Using CUDA Streams with MoE Modelscreating in PyTorch, Concurrency with CUDA Streamsprofiling when adding streams, Overlapping Communication and ComputationCUDA Graph captured, Capturing a CUDA Graph with a CUDA Stream-Capturing a CUDA Graph with a CUDA Streamdefault streamsdefault versus explicit streams, Default Versus Explicit (Nondefault) Streamslegacy default stream, Legacy Default Streamper-thread default streams, Modern Per-Thread Default Streamusing default streams, Best Practices for Default Stream Usage-Best Practices for Default Stream Usagekernel execution overlapping with, Overlapping Kernel Execution with CUDA Streams-Overlapping Kernel Execution with CUDA Streamslaunching five kernels on two streams, Overlapping Kernel Execution with CUDA Streams-Overlapping Kernel Execution with CUDA Streamswarp specialization replaced with CUDA streams, Overlapping Kernel Execution with CUDA Streams-Overlapping Kernel Execution with CUDA Streamsmixture-of-experts models, Using CUDA Streams with MoE Modelsmulti-GPU overlap of compute and data, Multi-GPU Compute and Data Transfer Overlap with CUDA Streams-Multi-GPU Compute and Data Transfer Overlap with CUDA Streamsmultiple streams per GPU, Multi-GPU Compute and Data Transfer Overlap with CUDA StreamsProgrammatic Dependent Launch, Programmatic Dependent Launch-Programmatic Dependent Launchstream-ordered memory allocator, Stream-Ordered Memory Allocator-Stream-Ordered Memory Allocatormultiple GPUs, Multi-GPU Compute and Data Transfer Overlap with CUDA Streams-Multi-GPU Compute and Data Transfer Overlap with CUDA Streamsusing, Using CUDA Streams and Stream-Ordered Memory Allocator with LLMs-Using CUDA Streams and Stream-Ordered Memory Allocator with LLMssynchronization with events and callbacks, Fine-Grained Synchronization with Events and Callbacks, Stream Synchronization with Events-Stream Synchronization with Eventscross-stream synchronization, Using CUDA Events for Cross-Stream Synchronizationwarp specialization, Pipelining with Warp Specialization (Intra-Kernel) and CUDA Streams (Inter-Kernel)-Pipelining with Warp Specialization (Intra-Kernel) and CUDA Streams (Inter-Kernel)thread block clusters, Warp Specialization with Thread Block Clusters and CUDA Streams
- CUDA Toolkitnvcc CUDA compiler, CUDA Toolkit and RuntimeNVIDIA Compute Sanitizer, Debugging Functional Correctness with NVIDIA Compute SanitizerNVIDIA software stack, CUDA Toolkit and Runtimeoptimized libraries included, CUDA Toolkit and RuntimeC++ and Python libraries, C++ and Python CUDA Libraries
- CUDA Unified Memory, GPU Memory Oversubscription, Fragmentation, and Out-of-Memory Handlingprogramming CUDA, Unified Memory-Unified MemoryUnified CPU-GPU Memory, The CPU and GPU Superchip
- cudaMemPool, Monitoring System Metrics and Counters
- cudart (CUDA Runtime library), CUDA Toolkit and Runtime
- CUDA_DEVICE_ORDER, Pitfall #4: Verify CPU-GPU NUMA-node affinity for NCCL threads
- CUDA_VISIBLE_DEVICES, Pitfall #4: Verify CPU-GPU NUMA-node affinity for NCCL threads
- cuDNN, CUDA Toolkit and Runtime
- CUPTI (CUDA Profiling Tools Interface), PyTorch Profiler and Visualization Tools
- CuPy, C++ and Python CUDA Libraries, Adaptive Batching and Chunked Prefill Scheduling
- cuPyNumeric, C++ and Python CUDA Libraries
- current used at typical data center, Compute Density and Power Requirementsliquid cooling, Liquid Cooling Versus Air Cooling
- custom kernels with OpenAI Triton, Writing Custom Kernels with OpenAI Triton-Profiling with Triton Proton ProfilerCUDA C++ custom kernels, Registering Custom Kernels with PyTorchregistering with PyTorch, Registering Custom Kernels with PyTorch-Registering Custom Kernels with PyTorchtuning kernel launch parameters, Tuning Kernel-Launch Parameters
- cuTile, C++ and Python CUDA LibrariesCUDA kernels in Python, NVIDIA Software StackCUDA Toolkit, C++ and Python CUDA Libraries
- CUTLASS (NVIDIA)arithmetic intensity and Tensor Core performance, Using CUTLASS for Optimal Arithmetic Intensity and Tensor Core Performance-Using CUTLASS for Optimal Arithmetic Intensity and Tensor Core PerformanceCUDA kernels in Python, NVIDIA Software StackCUDA Toolkit, C++ and Python CUDA Librariesthread block pairs, Thread Block PairTMEM, Feeding Tensor Cores with TMEM and TMAwarp specialization, Using CUTLASS for Optimal Arithmetic Intensity and Tensor Core Performance

- CUDA C++ Pipeline API, Asynchronous Memory Prefetching and Tensor Memory Accelerator-Asynchronous Memory Prefetching and Tensor Memory Acceleratorhardware-software codesign, Asynchronous Memory Prefetching and Tensor Memory Accelerator
- libraries in CUDA Toolkit, C++ and Python CUDA Libraries

- hardware-software codesign, Asynchronous Memory Prefetching and Tensor Memory Accelerator

- NVLink-C2C, The CPU and GPU Superchip, Multi-GPU Programming

- caching CUDA kernels, Continuous Prewarming of CUDA Graphs and Caches Using Time-Series Prediction
- metrics for cache hits and misses, Profiling, Debugging, and Tuning Inference Performance
- prefix caching, Prefix Caching-Prefix Caching
- prewarming caches, Continuous Prewarming of CUDA Graphs and Caches Using Time-Series Prediction-Continuous Prewarming of CUDA Graphs and Caches Using Time-Series Prediction
- read-only data caches, Read-Only Data Caches-Read-Only Data Cachesconst __restrict__, Read-Only Data Caches, Read-Only Data Caches, Read-Only Data Cachespitfall of not stating read-only, Read-Only Data Caches
- vLLM LMCache, Profiling, Debugging, and Tuning Inference Performance, Zero-Copy GPU-to-GPU Transfer

- const __restrict__, Read-Only Data Caches, Read-Only Data Caches, Read-Only Data Caches
- pitfall of not stating read-only, Read-Only Data Caches

- advanced tuning strategies, Advanced Tuning Strategies and Algorithmic Tricks-Advanced Tuning Strategies and Algorithmic Tricks
- algorithmic tricks, Advanced Tuning Strategies and Algorithmic Tricks-Advanced Tuning Strategies and Algorithmic Tricks
- arithmetic optimizations, Arithmetic Optimizations and Reduced/Mixed Precision-Arithmetic Optimizations and Reduced/Mixed Precision
- cost optimization, Performance Tuning and Cost Optimization Mindset-Performance Tuning and Cost Optimization Mindset
- CUDA tuning optimizations, GPU Programming and CUDA Tuning Optimizations-GPU Programming and CUDA Tuning Optimizations
- data processing pipelines, Data Processing Pipelines-Data Processing Pipelines
- GPU programming, GPU Programming and CUDA Tuning Optimizations-GPU Programming and CUDA Tuning Optimizations
- GPU resource management and scheduling, GPU Resource Management and Scheduling-GPU Resource Management and Scheduling
- I/O optimization, I/O Optimization-I/O Optimization
- inference and serving, Efficient Inference and Serving-Efficient Inference and Servingmultinode, Multinode Inference and Serving-Multinode Inference and Serving
- kernel scheduling and execution optimizations, Kernel Scheduling and Execution Optimizations-Kernel Scheduling and Execution Optimizations
- multi-GPU scaling and interconnect optimizations, Multi-GPU Scaling and Interconnect Optimizations-Multi-GPU Scaling and Interconnect Optimizations
- network optimization, Distributed Training and Network Optimization-Distributed Training and Network Optimization
- operating system and driver optimizations, Operating System and Driver Optimizations-Operating System and Driver Optimizations
- performance profiling, debugging, and monitoring, Performance Profiling, Debugging, and Monitoring-Performance Profiling, Debugging, and Monitoring
- power management, Power and Thermal Management-Power and Thermal Management
- reduced/mixed precision, Arithmetic Optimizations and Reduced/Mixed Precision-Arithmetic Optimizations and Reduced/Mixed Precision
- reproducibility and documentation, Reproducibility and Documentation Best Practices-Reproducibility and Documentation Best Practices
- system architecture and hardware planning, System Architecture and Hardware Planning-System Architecture and Hardware Planning
- thermal management, Power and Thermal Management-Power and Thermal Management
- unified CPU-GPU “Superchip” architecture, Unified CPU-GPU “Superchip” Architecture-Unified CPU-GPU “Superchip” Architecture

- multinode, Multinode Inference and Serving-Multinode Inference and Serving

- CUDA-specific launch parameters, CUDA Programming Refresher
- how many threads, CUDA Programming Refresher

- appearing as one GPU, NVIDIA Blackwell “Dual-Die” GPU

- performance improvement verification, Distributed, Parallel Filesystems and Object Stores

- commonly used compilers, NVIDIA Software Stack
- compiling versus writing custom kernels, Compiling Versus Writing Custom Kernels
- float8 not provided by CUDA, Vectorized Memory Access
- __global__, CUDA Programming Refresher
- instruction-level parallelism exposed, Exposing Instruction-Level Parallelism-Profiling and Mitigating Register Pressurecompiler hints, Exposing Instruction-Level Parallelism
- __launch_bounds__, Tuning Occupancy with Launch Bounds-Tuning Occupancy with Launch Bounds, Techniques for Occupancy Tuningoccupancy optimization, Compiler Hints to Optimize Occupancy
- nvcc CUDA compiler, CUDA Toolkit and Runtime(see also nvcc CUDA compiler)
- predication, Techniques to Avoid Warp Divergence
- PTX, CUDA GPU Backward and Forward Compatibility Model, TorchInductor Backend Code GenerationCUDA compatibility across GPU generations, CUDA Forward and Backward Compatibility Across GPU Hardware Generations, CUDA GPU Backward and Forward Compatibility ModelDeepSeek memory allocation optimization, DeepSeek’s Use of Inline PTX for Memory Allocation Optimizationinline PTX code, Inline PTX and SASS Tuning for Microoptimizations-Inline PTX and SASS Tuning for MicrooptimizationsJIT compiling, CUDA Forward and Backward Compatibility Across GPU Hardware Generationsstreaming assembler, Inline PTX and SASS Tuning for Microoptimizations-Inline PTX and SASS Tuning for Microoptimizationsstreaming assembler changing over generations, Inline PTX and SASS Tuning for MicrooptimizationsTorchInductor, TorchInductor Backend Code GenerationTriton, Triton Programming Model
- PyTorch compiler, PyTorch and Higher-Level AI Frameworks, PyTorch Compiler (torch.compile)(see also PyTorch compiler (torch.compile))
- smart compilers, Smart Compilers and Automated Code Optimizations-Smart Compilers and Automated Code Optimizations
- Triton, C++ and Python CUDA Libraries(see also Triton (OpenAI))
- vectorized memory access, Vectorized Memory Access-Vectorized Memory Accessvectorized memory operations, Vectorized Memory Access

- compiler hints, Exposing Instruction-Level Parallelism

- occupancy optimization, Compiler Hints to Optimize Occupancy

- (see also nvcc CUDA compiler)

- CUDA compatibility across GPU generations, CUDA Forward and Backward Compatibility Across GPU Hardware Generations, CUDA GPU Backward and Forward Compatibility Model
- DeepSeek memory allocation optimization, DeepSeek’s Use of Inline PTX for Memory Allocation Optimization
- inline PTX code, Inline PTX and SASS Tuning for Microoptimizations-Inline PTX and SASS Tuning for Microoptimizations
- JIT compiling, CUDA Forward and Backward Compatibility Across GPU Hardware Generations
- streaming assembler, Inline PTX and SASS Tuning for Microoptimizations-Inline PTX and SASS Tuning for Microoptimizations
- streaming assembler changing over generations, Inline PTX and SASS Tuning for Microoptimizations
- TorchInductor, TorchInductor Backend Code Generation
- Triton, Triton Programming Model

- (see also PyTorch compiler (torch.compile))

- (see also Triton (OpenAI))

- vectorized memory operations, Vectorized Memory Access

- compute-sanitizer CLI, Debugging Functional Correctness with NVIDIA Compute SanitizerNVIDIA Tools Extension support, Debugging Functional Correctness with NVIDIA Compute Sanitizer
- concurrency and synchronization, Using CUDA Streams with MoE Models
- four primary tools, Debugging Functional Correctness with NVIDIA Compute Sanitizer
- inference correctness, Debugging Correctness Issues

- NVIDIA Tools Extension support, Debugging Functional Correctness with NVIDIA Compute Sanitizer

- kernel optimization, Optimizing the Kernel-Optimizing the Kernel

- NVIDIA Tools Extension support, Debugging Functional Correctness with NVIDIA Compute Sanitizer

- compute overlapping with data transfers, Using Streams to Overlap Compute with Data Transfers-Using Streams to Overlap Compute with Data Transfersconcurrency, Overlapping Communication and Computation-Overlapping Communication and Computation
- launching on intended stream, Using CUDA Streams with MoE Models
- mixture-of-experts models, Using CUDA Streams with MoE Models
- NVIDIA Compute Sanitizer, Using CUDA Streams with MoE Models
- profiling when introducing concurrency, Using CUDA Streams with MoE Models
- stream synchronization with events, Stream Synchronization with Events-Stream Synchronization with Events

- concurrency, Overlapping Communication and Computation-Overlapping Communication and Computation

- NVIDIA Container Toolkit, Container Runtime Optimizations for GPUsCUDA compatibility, NVIDIA Container Toolkit and CUDA CompatibilityGPU performance, Container Runtime Optimizations for GPUs
- runtime optimizations for GPUs, Container Runtime Optimizations for GPUs-Reduce Image Size for Faster Container Startupcontainer overlay filesystem overhead, Avoiding Container Overlay Filesystem OverheadNVIDIA container runtime injecting libraries, NVIDIA Container RuntimeNVIDIA Container Toolkit and CUDA compatibility, NVIDIA Container Toolkit and CUDA Compatibilityreducing image size, Reduce Image Size for Faster Container Startup

- CUDA compatibility, NVIDIA Container Toolkit and CUDA Compatibility
- GPU performance, Container Runtime Optimizations for GPUs

- container overlay filesystem overhead, Avoiding Container Overlay Filesystem Overhead
- NVIDIA container runtime injecting libraries, NVIDIA Container Runtime
- NVIDIA Container Toolkit and CUDA compatibility, NVIDIA Container Toolkit and CUDA Compatibility
- reducing image size, Reduce Image Size for Faster Container Startup

- Context Parallel (PyTorch), PyTorch Optimized Attention Mechanisms
- hybrid parallelism, Hybrid Parallelism

- Computer Sanitizer integrated, Debugging Functional Correctness with NVIDIA Compute Sanitizer
- performance benchmarking, Continuous Integration and Performance Benchmarking-Performance Benchmarks and MLPerf Loggingforcing an error when full graph not captured, Graph Breaks and TorchDynamo explain()PyTorch performance heads-up display, Continuous Integration and Performance Benchmarking-PyTorch HUD Performance Dashboard

- forcing an error when full graph not captured, Graph Breaks and TorchDynamo explain()
- PyTorch performance heads-up display, Continuous Integration and Performance Benchmarking-PyTorch HUD Performance Dashboard

- Cooperative Groups API, Warp Shuffle Intrinsics: Avoid Shared Memory and Explicit Synchronization, Cooperative Groupscoordinating thread block clusters, Coordinating Thread Block Clusters with Cooperative Groups API-Coordinating Thread Block Clusters with Cooperative Groups API
- described, Thread Block Clusters and Distributed Shared Memory
- launching a kernel in cooperative mode, Cooperative Groupsgrid size for launch, Cooperative Groupsreserving all SMs in the grid, When to Combine Persistent Kernels and Cooperative Groups
- persistent kernels via grid sync, Cooperative Grid Synchronization and Persistent Kernels-Cooperative Grid Synchronization and Persistent Kernelswhen to combine, When to Combine Persistent Kernels and Cooperative Groups
- thread block clusters contrasted with, Thread Block Clusters and Distributed Shared Memory

- coordinating thread block clusters, Coordinating Thread Block Clusters with Cooperative Groups API-Coordinating Thread Block Clusters with Cooperative Groups API

- grid size for launch, Cooperative Groups
- reserving all SMs in the grid, When to Combine Persistent Kernels and Cooperative Groups

- when to combine, When to Combine Persistent Kernels and Cooperative Groups

- coordinating thread block clusters, Coordinating Thread Block Clusters with Cooperative Groups API-Coordinating Thread Block Clusters with Cooperative Groups API

- (see also thread blocks)
- runtime operation binding, Optimized KV Cache Memory Layout

- AI systems performance engineers generating returns, Measuring “Goodput” Useful Throughput
- cost optimization overview, Performance Tuning and Cost Optimization Mindset-Performance Tuning and Cost Optimization Mindset
- model parameter-count explosion, Introduction and AI System Overview100-trillion parameter models, Toward 100-Trillion-Parameter Models
- trainingcold start strategy, DeepSeek Scales to ~680-Billion Parameter Models Despite US Export Hardware Restrictions in ChinaDeepSeek-R1, Introduction and AI System OverviewDeepSeek-V3, DeepSeek Scales to ~680-Billion Parameter Models Despite US Export Hardware Restrictions in ChinaGoogle Gemini Ultra, Introduction and AI System OverviewOpenAI GPT-4, Introduction and AI System Overview

- 100-trillion parameter models, Toward 100-Trillion-Parameter Models

- cold start strategy, DeepSeek Scales to ~680-Billion Parameter Models Despite US Export Hardware Restrictions in China
- DeepSeek-R1, Introduction and AI System Overview
- DeepSeek-V3, DeepSeek Scales to ~680-Billion Parameter Models Despite US Export Hardware Restrictions in China
- Google Gemini Ultra, Introduction and AI System Overview
- OpenAI GPT-4, Introduction and AI System Overview

- NVIDIA Blackwell dual-die GPU, NVIDIA Blackwell “Dual-Die” GPU-NVIDIA Blackwell “Dual-Die” GPUappearing as one GPU, NVIDIA Blackwell “Dual-Die” GPU
- NVIDIA GPU Tensor Cores and Transformer Engine, NVIDIA GPU Tensor Cores and Transformer Engine-NVIDIA GPU Tensor Cores and Transformer Engine
- NVIDIA Grace CPU, NVIDIA Grace CPU
- NVLink-C2C, The CPU and GPU Superchip
- streaming multiprocessors, threads, warps, Streaming Multiprocessor, Threads, and Warps-Streaming Multiprocessor, Threads, and Warps

- appearing as one GPU, NVIDIA Blackwell “Dual-Die” GPU

- about, Configuring the CPUs and OS for GPU Environments
- CPU frequency and C-states, CPU Frequency and C-states
- filesystem caching and write-back, Filesystem Caching and Write-Back
- NUMA awareness and CPU pinning, NUMA Awareness and CPU Pinning-NUMA Awareness and CPU PinningCPU + GPU superchip CPU-to-GPU data transfers, NUMA Awareness and CPU PinningCPU pinning, NUMA Awareness and CPU Pinning-NUMA Awareness and CPU PinningNUMA nodes, NUMA Awareness and CPU Pinning
- NUMA-friendly memory allocation and pinning, NUMA-Friendly Memory Allocation and Memory Pinning-NUMA-Friendly Memory Allocation and Memory Pinningefficiency of pinned memory, NUMA-Friendly Memory Allocation and Memory Pinningmax locked memory setting, Transparent HugepagesOS limit on pinned memory, NUMA-Friendly Memory Allocation and Memory Pinningpinned memory for data loaders, NUMA-Friendly Memory Allocation and Memory Pinning
- scheduler and interrupt affinity, Scheduler and Interrupt Affinityirqbalance daemon, Scheduler and Interrupt Affinity
- transparent hugepages, Transparent Hugepages
- tuning host CPU memory allocator, Tune Host CPU Memory Allocator
- virtual memory and swapping, Virtual Memory and Swapping

- CPU + GPU superchip CPU-to-GPU data transfers, NUMA Awareness and CPU Pinning
- CPU pinning, NUMA Awareness and CPU Pinning-NUMA Awareness and CPU Pinning
- NUMA nodes, NUMA Awareness and CPU Pinning

- efficiency of pinned memory, NUMA-Friendly Memory Allocation and Memory Pinning
- max locked memory setting, Transparent Hugepages
- OS limit on pinned memory, NUMA-Friendly Memory Allocation and Memory Pinning
- pinned memory for data loaders, NUMA-Friendly Memory Allocation and Memory Pinning

- irqbalance daemon, Scheduler and Interrupt Affinity

- CPU pinning, NUMA Awareness and CPU Pinning-NUMA Awareness and CPU Pinning
- CPU-GPU memory data-transfer bandwidth, NUMA-Friendly Memory Allocation and Memory Pinning, GPU Memory Oversubscription, Fragmentation, and Out-of-Memory HandlingNIXL GPU-to-GPU direct transfer, Separate Prefill and Decode Inference Stages
- CUDA programming flow, Understanding GPU Architecture
- CUDA Unified Memory, GPU Memory Oversubscription, Fragmentation, and Out-of-Memory Handlingprogramming CUDA, Unified Memory-Unified MemoryUnified CPU-GPU Memory, The CPU and GPU Superchip
- GDS, NUMA-Friendly Memory Allocation and Memory Pinning, Using NVIDIA GDS-Measuring GDS with gdsio, Offloading Parameters to CPU and NVMe
- host CPU memory allocator tuned, Tune Host CPU Memory Allocator
- large slower extension of GPU memory, The CPU and GPU Superchipsuperchips’ high-speed extension of GPU memory, Topology Awareness in NCCL
- NIXL offloading KV cache to, NVIDIA’s NIXL and Disaggregated Inference, KV Cache Offloading with NIXL-KV Cache Offloading with NIXL
- parameters offloaded to, Offloading Parameters to CPU and NVMe
- peer-to-peer DMA avoiding, Enabling Peer-to-Peer DMA and UCX
- swapping, GPU Memory Oversubscription, Fragmentation, and Out-of-Memory Handling

- NIXL GPU-to-GPU direct transfer, Separate Prefill and Decode Inference Stages

- programming CUDA, Unified Memory-Unified Memory
- Unified CPU-GPU Memory, The CPU and GPU Superchip

- superchips’ high-speed extension of GPU memory, Topology Awareness in NCCL

- callback function, Fine-Grained Synchronization with Events and Callbacks

- (see also thread blocks)

- CUDA Toolkit, CUDA Toolkit and Runtime, PyTorch and Higher-Level AI Frameworks
- PyTorch delegating tasks to, PyTorch and Higher-Level AI Frameworks

- child-kernel launch limit, Dynamic Parallelism
- compatibility across GPU generations, CUDA Forward and Backward Compatibility Across GPU Hardware Generations, CUDA GPU Backward and Forward Compatibility Model
- CUDA Graph launched, Capturing a CUDA Graph with a CUDA Stream-Capturing a CUDA Graph with a CUDA Stream
- float8 not provided, Vectorized Memory Access
- kernels created in Python, NVIDIA Software Stack
- kernels executing asynchronously, CUDA Programming Refresherglobal fault flag for illegal operations, CUDA Programming Refresher
- programmingasynchronous memory allocation, Asynchronous Memory Allocation and Memory Pools-Asynchronous Memory Allocation and Memory PoolsCUDA streams, Asynchronous Memory Allocation and Memory Poolsflow between CPU and GPU, Understanding GPU Architecture__global__, CUDA Programming RefresherGPU memory hierarchy, Understanding GPU Memory Hierarchy-Understanding GPU Memory Hierarchyhigh occupancy and GPU utilization, Maintaining High Occupancy and GPU Utilization-Maintaining High Occupancy and GPU Utilizationhighest-level and most-recent APIs available, Asynchronous Memory Prefetching and Tensor Memory Acceleratorkernel errors, CUDA Programming Refresherkernel inputs in 1D, 2D and 3D Kernel Inputskernel inputs in 2D and 3D, 2D and 3D Kernel Inputskernels for parallel work, CUDA Programming Refresher-CUDA Programming Refresherlaunch parameters, Configuring Launch Parameters: Blocks per Grid and Threads per Block-Configuring Launch Parameters: Blocks per Grid and Threads per Blockmaximum threadsPerBlock compile time parameter, Tuning Occupancy with Launch Bounds-Tuning Occupancy with Launch Boundsmemory pools, Asynchronous Memory Allocation and Memory Pools-Asynchronous Memory Allocation and Memory Poolsminimum thread blocks resident on each SM, Tuning Occupancy with Launch Bounds-Tuning Occupancy with Launch BoundsNVIDIA Compute Sanitizer, Debugging Functional Correctness with NVIDIA Compute Sanitizeroccupancy tuning with __launch_bounds__, Tuning Occupancy with Launch Bounds-Tuning Occupancy with Launch Boundsroofline model, Roofline Model: Compute-Bound or Memory-Bound Workloads-Roofline Model: Compute-Bound or Memory-Bound WorkloadsUnified Memory, Unified Memory-Unified Memory
- Python-based frameworks, PyTorch and Higher-Level AI Frameworks
- Python-centric libraries, Writing Custom Kernels with OpenAI Triton
- Unified Memory, GPU Memory Oversubscription, Fragmentation, and Out-of-Memory Handlingprogramming CUDA, Unified Memory-Unified MemoryUnified CPU-GPU Memory, The CPU and GPU Superchip
- versions of container libraries and host driver, NVIDIA Container Toolkit and CUDA CompatibilityNVIDIA Container Toolkit, NVIDIA Container Toolkit and CUDA Compatibility

- global fault flag for illegal operations, CUDA Programming Refresher

- asynchronous memory allocation, Asynchronous Memory Allocation and Memory Pools-Asynchronous Memory Allocation and Memory Pools
- CUDA streams, Asynchronous Memory Allocation and Memory Pools
- flow between CPU and GPU, Understanding GPU Architecture
- __global__, CUDA Programming Refresher
- GPU memory hierarchy, Understanding GPU Memory Hierarchy-Understanding GPU Memory Hierarchy
- high occupancy and GPU utilization, Maintaining High Occupancy and GPU Utilization-Maintaining High Occupancy and GPU Utilization
- highest-level and most-recent APIs available, Asynchronous Memory Prefetching and Tensor Memory Accelerator
- kernel errors, CUDA Programming Refresher
- kernel inputs in 1D, 2D and 3D Kernel Inputs
- kernel inputs in 2D and 3D, 2D and 3D Kernel Inputs
- kernels for parallel work, CUDA Programming Refresher-CUDA Programming Refresher
- launch parameters, Configuring Launch Parameters: Blocks per Grid and Threads per Block-Configuring Launch Parameters: Blocks per Grid and Threads per Block
- maximum threadsPerBlock compile time parameter, Tuning Occupancy with Launch Bounds-Tuning Occupancy with Launch Bounds
- memory pools, Asynchronous Memory Allocation and Memory Pools-Asynchronous Memory Allocation and Memory Pools
- minimum thread blocks resident on each SM, Tuning Occupancy with Launch Bounds-Tuning Occupancy with Launch Bounds
- NVIDIA Compute Sanitizer, Debugging Functional Correctness with NVIDIA Compute Sanitizer
- occupancy tuning with __launch_bounds__, Tuning Occupancy with Launch Bounds-Tuning Occupancy with Launch Bounds
- roofline model, Roofline Model: Compute-Bound or Memory-Bound Workloads-Roofline Model: Compute-Bound or Memory-Bound Workloads
- Unified Memory, Unified Memory-Unified Memory

- programming CUDA, Unified Memory-Unified Memory
- Unified CPU-GPU Memory, The CPU and GPU Superchip

- NVIDIA Container Toolkit, NVIDIA Container Toolkit and CUDA Compatibility

- hardware-software codesign, Asynchronous Memory Prefetching and Tensor Memory Accelerator

- communication from GPU device to CPU host, Fine-Grained Synchronization with Events and Callbacks
- CUDA stream synchronization, Fine-Grained Synchronization with Events and Callbacks, Stream Synchronization with Events-Stream Synchronization with Eventscross-stream synchronization, Using CUDA Events for Cross-Stream Synchronization
- multi-GPU overlap of compute and data, Multi-GPU Compute and Data Transfer Overlap with CUDA Streams-Multi-GPU Compute and Data Transfer Overlap with CUDA Streams
- profiling with, Using CUDA Events for Cross-Stream Synchronization

- cross-stream synchronization, Using CUDA Events for Cross-Stream Synchronization

- about CUDA Graphs, Techniques for Occupancy Tuning, CUDA Graphs
- benchmarking, Capturing a CUDA Graph with a CUDA Stream
- best practices, Best Practices for CUDA Graphs-Best Practices for CUDA Graphs
- capturing a CUDA Graph, Capturing a CUDA Graph with a CUDA Stream-Capturing a CUDA Graph with a CUDA Stream, Capturing a CUDA Graph and Preallocating Memory-Capturing a CUDA Graph and Preallocating Memorycapturing only a portion of pipeline, Capturing a CUDA Graph with a CUDA Streamdevice-initiated launch, Device-Initiated CUDA Graph Launch-Device-Initiated CUDA Graph Launchdynamic graph update, Dynamic Graph Updatewarm-up pass necessary, Capturing a CUDA Graph with a CUDA Stream
- compiler modes that trigger, Autotuning with TorchInductorstatic shapes required, Autotuning with TorchInductor, Dynamic Shapes and Variable Sequence Lengths
- conditional graph nodes, Conditional Graph Nodes-Conditional Graph Nodesnested, Conditional Graph Nodes
- CUDA streams, Multi-GPU Compute and Data Transfer Overlap with CUDA Streamscapturing a CUDA Graph, Capturing a CUDA Graph with a CUDA Stream-Capturing a CUDA Graph with a CUDA StreamPyTorch, Multi-GPU Compute and Data Transfer Overlap with CUDA Streams
- device-initiated launch, Device-Initiated CUDA Graph Launch-Device-Initiated CUDA Graph Launchdependency management, Device-Initiated CUDA Graph Launchexample of use, Device-Initiated CUDA Graph Launchpersistent scheduling, Atomic Queues and Device-Initiated CUDA Graphs for In-Kernel Persistent Scheduling
- dynamic graph update, Dynamic Graph Update
- inference engines, PyTorch, Inference Engines, and CUDA Graphs
- kernel launch overhead reduced, Reducing Kernel Launch Overhead with CUDA Graphs-CUDA Graph Trees (PyTorch Compiler Internal)
- kernels that use NVSHMEM, Fine-Grained GPU-to-GPU Memory Sharing with NVSHMEM
- launching a CUDA Graph, Capturing a CUDA Graph with a CUDA Stream-Capturing a CUDA Graph with a CUDA Streamdevice-initiated launch, Device-Initiated CUDA Graph Launch-Device-Initiated CUDA Graph Launch
- memory management, Memory Pools for CUDA Graphs
- memory pools, Replaying the Graph
- multiple-GPU collectives captured, Capturing Multi-GPU Collectives with NCCL and CUDA Graphs-Capturing Multi-GPU Collectives with NCCL and CUDA Graphs
- pitfalls, Capturing a CUDA Graph with a CUDA Stream
- prewarming, Continuous Prewarming of CUDA Graphs and Caches Using Time-Series Prediction-Continuous Prewarming of CUDA Graphs and Caches Using Time-Series Prediction
- PyTorch, PyTorch, Inference Engines, and CUDA Graphs, Capturing a CUDA Graph with a CUDA StreamCUDA Graph Trees, CUDA Graph Trees (PyTorch Compiler Internal)static memory pools, Memory Pools for CUDA Graphs
- replaying the graph, Replaying the Graph-Replaying the Graphany stream for replay, Capturing a CUDA Graph and Preallocating Memory
- TorchInductor, Autotuning with TorchInductor
- when to use dynamic parallelism instead, Dynamic Parallelism

- capturing only a portion of pipeline, Capturing a CUDA Graph with a CUDA Stream
- device-initiated launch, Device-Initiated CUDA Graph Launch-Device-Initiated CUDA Graph Launch
- dynamic graph update, Dynamic Graph Update
- warm-up pass necessary, Capturing a CUDA Graph with a CUDA Stream

- static shapes required, Autotuning with TorchInductor, Dynamic Shapes and Variable Sequence Lengths

- nested, Conditional Graph Nodes

- capturing a CUDA Graph, Capturing a CUDA Graph with a CUDA Stream-Capturing a CUDA Graph with a CUDA Stream
- PyTorch, Multi-GPU Compute and Data Transfer Overlap with CUDA Streams

- dependency management, Device-Initiated CUDA Graph Launch
- example of use, Device-Initiated CUDA Graph Launch
- persistent scheduling, Atomic Queues and Device-Initiated CUDA Graphs for In-Kernel Persistent Scheduling

- device-initiated launch, Device-Initiated CUDA Graph Launch-Device-Initiated CUDA Graph Launch

- CUDA Graph Trees, CUDA Graph Trees (PyTorch Compiler Internal)
- static memory pools, Memory Pools for CUDA Graphs

- any stream for replay, Capturing a CUDA Graph and Preallocating Memory

- cooperative grids, Cooperative Grid Synchronization and Persistent Kernels

- double-buffering, Cooperative Tiling and Double-Buffering with the CUDA Pipeline API-Cooperative Tiling and Double-Buffering with the CUDA Pipeline API
- TMA plus, Asynchronous Memory Prefetching and Tensor Memory Accelerator
- warp specialization, Using CUDA Pipeline API for Warp Specialization-Using CUDA Pipeline API for Warp SpecializationPyTorch, PyTorch, CUDA Pipeline API, and Warp Specialization

- PyTorch, PyTorch, CUDA Pipeline API, and Warp Specialization

- asynchronous execution, Asynchronous Execution with Streams
- compute overlapping with data transfers, Using Streams to Overlap Compute with Data Transfers-Using Streams to Overlap Compute with Data Transfersconcurrency, Overlapping Communication and Computation-Overlapping Communication and Computationmultiple GPUs, Multi-GPU Compute and Data Transfer Overlap with CUDA Streams-Multi-GPU Compute and Data Transfer Overlap with CUDA Streams
- concurrency, Concurrency with CUDA Streams-Using CUDA Streams with MoE Modelslaunching on intended stream, Using CUDA Streams with MoE Modelsprofiling when introducing concurrency, Using CUDA Streams with MoE Models
- creating in PyTorch, Concurrency with CUDA Streamsprofiling when adding streams, Overlapping Communication and Computation
- CUDA Graph captured, Capturing a CUDA Graph with a CUDA Stream-Capturing a CUDA Graph with a CUDA Stream
- default streamsdefault versus explicit streams, Default Versus Explicit (Nondefault) Streamslegacy default stream, Legacy Default Streamper-thread default streams, Modern Per-Thread Default Streamusing default streams, Best Practices for Default Stream Usage-Best Practices for Default Stream Usage
- kernel execution overlapping with, Overlapping Kernel Execution with CUDA Streams-Overlapping Kernel Execution with CUDA Streamslaunching five kernels on two streams, Overlapping Kernel Execution with CUDA Streams-Overlapping Kernel Execution with CUDA Streamswarp specialization replaced with CUDA streams, Overlapping Kernel Execution with CUDA Streams-Overlapping Kernel Execution with CUDA Streams
- mixture-of-experts models, Using CUDA Streams with MoE Models
- multi-GPU overlap of compute and data, Multi-GPU Compute and Data Transfer Overlap with CUDA Streams-Multi-GPU Compute and Data Transfer Overlap with CUDA Streamsmultiple streams per GPU, Multi-GPU Compute and Data Transfer Overlap with CUDA Streams
- Programmatic Dependent Launch, Programmatic Dependent Launch-Programmatic Dependent Launch
- stream-ordered memory allocator, Stream-Ordered Memory Allocator-Stream-Ordered Memory Allocatormultiple GPUs, Multi-GPU Compute and Data Transfer Overlap with CUDA Streams-Multi-GPU Compute and Data Transfer Overlap with CUDA Streamsusing, Using CUDA Streams and Stream-Ordered Memory Allocator with LLMs-Using CUDA Streams and Stream-Ordered Memory Allocator with LLMs
- synchronization with events and callbacks, Fine-Grained Synchronization with Events and Callbacks, Stream Synchronization with Events-Stream Synchronization with Eventscross-stream synchronization, Using CUDA Events for Cross-Stream Synchronization
- warp specialization, Pipelining with Warp Specialization (Intra-Kernel) and CUDA Streams (Inter-Kernel)-Pipelining with Warp Specialization (Intra-Kernel) and CUDA Streams (Inter-Kernel)thread block clusters, Warp Specialization with Thread Block Clusters and CUDA Streams

- concurrency, Overlapping Communication and Computation-Overlapping Communication and Computation
- multiple GPUs, Multi-GPU Compute and Data Transfer Overlap with CUDA Streams-Multi-GPU Compute and Data Transfer Overlap with CUDA Streams

- launching on intended stream, Using CUDA Streams with MoE Models
- profiling when introducing concurrency, Using CUDA Streams with MoE Models

- profiling when adding streams, Overlapping Communication and Computation

- default versus explicit streams, Default Versus Explicit (Nondefault) Streams
- legacy default stream, Legacy Default Stream
- per-thread default streams, Modern Per-Thread Default Stream
- using default streams, Best Practices for Default Stream Usage-Best Practices for Default Stream Usage

- launching five kernels on two streams, Overlapping Kernel Execution with CUDA Streams-Overlapping Kernel Execution with CUDA Streams
- warp specialization replaced with CUDA streams, Overlapping Kernel Execution with CUDA Streams-Overlapping Kernel Execution with CUDA Streams

- multiple streams per GPU, Multi-GPU Compute and Data Transfer Overlap with CUDA Streams

- multiple GPUs, Multi-GPU Compute and Data Transfer Overlap with CUDA Streams-Multi-GPU Compute and Data Transfer Overlap with CUDA Streams
- using, Using CUDA Streams and Stream-Ordered Memory Allocator with LLMs-Using CUDA Streams and Stream-Ordered Memory Allocator with LLMs

- cross-stream synchronization, Using CUDA Events for Cross-Stream Synchronization

- thread block clusters, Warp Specialization with Thread Block Clusters and CUDA Streams

- nvcc CUDA compiler, CUDA Toolkit and Runtime
- NVIDIA Compute Sanitizer, Debugging Functional Correctness with NVIDIA Compute Sanitizer
- NVIDIA software stack, CUDA Toolkit and Runtime
- optimized libraries included, CUDA Toolkit and RuntimeC++ and Python libraries, C++ and Python CUDA Libraries

- C++ and Python libraries, C++ and Python CUDA Libraries

- programming CUDA, Unified Memory-Unified Memory
- Unified CPU-GPU Memory, The CPU and GPU Superchip

- liquid cooling, Liquid Cooling Versus Air Cooling

- CUDA C++ custom kernels, Registering Custom Kernels with PyTorch
- registering with PyTorch, Registering Custom Kernels with PyTorch-Registering Custom Kernels with PyTorch
- tuning kernel launch parameters, Tuning Kernel-Launch Parameters

- CUDA kernels in Python, NVIDIA Software Stack
- CUDA Toolkit, C++ and Python CUDA Libraries

- arithmetic intensity and Tensor Core performance, Using CUTLASS for Optimal Arithmetic Intensity and Tensor Core Performance-Using CUTLASS for Optimal Arithmetic Intensity and Tensor Core Performance
- CUDA kernels in Python, NVIDIA Software Stack
- CUDA Toolkit, C++ and Python CUDA Libraries
- thread block pairs, Thread Block Pair
- TMEM, Feeding Tensor Cores with TMEM and TMA
- warp specialization, Using CUTLASS for Optimal Arithmetic Intensity and Tensor Core Performance

### D

- DALI (Data Loading Library; NVIDIA), Multimodal Data Processing with NVIDIA DALI, Optimizing the Data Input Pipeline
- DASH framework, Input-Aware Layer Skipping (DASH)
- Data Center GPU Manager (DCGM; NVIDIA), Performance Monitoring and Utilization in PracticeI/O monitoring, Monitoring Storage I/Oinference monitoring, Monitoring System Metrics and Countersmonitoring straggler nodes or processes, Pitfall #5: Straggler nodes or processes
- data centerscurrent used at typical data center, Compute Density and Power Requirementsliquid cooling, Liquid Cooling Versus Air Cooling
- data compression, Tuning, Replicating, and Compressing Data
- data input pipeline optimization, Optimizing the Data Input Pipeline-Optimizing the Data Input Pipeline
- data parallelism (DP), Parallelism Strategies for Serving Massive MoE Models, Data Parallelismdefinition of data parallelism, Distributed Data Parallel Strategies, Data ParallelismDistributed Data Parallel versus Data Parallel, Distributed Data Parallel Strategies-Distributed Data Parallel Strategiesdefinition of Distributed Data Parallel, Distributed Data Parallel Strategiesmodel weights and data split over GPUs, Parallelism Strategies for Serving Massive MoE Modelsversus TP, PP, and hybrid, Adaptive Parallelism Strategies (TP Versus PP Versus Hybrid)-Adaptive Parallelism Strategies (TP Versus PP Versus Hybrid)
- data pipelines (see pipelines)
- Data Processing Unit (DPU) in NVL72, Multirack and Storage Communication
- data reuse via shared memory, Tiling and Data Reuse Using Shared Memory-Tiling and Data Reuse Using Shared Memoryarithmetic intensity increased by, Multilevel Microtiling and Software Prefetching
- DataLoader (PyTorch)CPU affinity for each worker process, NUMA Awareness and CPU Pinning-NUMA Awareness and CPU Pinningdata input pipeline optimization, Optimizing the Data Input Pipeline-Optimizing the Data Input Pipelinedata loading and preprocessing, Efficient Data Loading and Preprocessing-Efficient Data Loading and Preprocessingmultiple workers, Tuning NVMe and Filesystem for Throughput, Efficient Data Loading and Preprocessing, Scaling Out Workers as You Scale Out Number of GPUs, Optimizing the Data Input Pipelinepersistent workers, Efficient Data Loading and Preprocessingpinned memory flag, NUMA-Friendly Memory Allocation and Memory Pinning, Using Streams to Overlap Compute with Data Transfers, Optimizing the Data Input Pipelineprefetch factor, Optimizing the Data Input Pipeline
- datasets for training via NeMo Curator, Creating High-Quality LLM Datasets with NVIDIA NeMo Curator
- DCGM (see Data Center GPU Manager (DCGM; NVIDIA))
- DDP (see Distributed Data Parallel (DDP; PyTorch))
- debouncing, Debouncing and Request Coalescing
- debuggingcompiler performance issues, Profiling and Debugging Compiler Performance IssuesDWARF, CPU and GPU Profiling with Linux perfgraph breaks, Graph Breaks and TorchDynamo explain(), Debugging Compiler Phases, Graph Breaks, and Performancemonitoring graph recompilations, TorchDynamo for Bytecode Capture and Graph Extraction, Minimize Graph Recompilationsinference correctness issues, Debugging Correctness Issues-Debugging Correctness IssuesNCCL, Profiling and Debugging NCCLdebugging enabled, Profiling and Debugging NCCL, Debugging Correctness IssuesNCCL logs, In-Network SHARP Aggregation, Debugging Correctness IssuesNCCL_DEBUG, Topology Awareness in NCCL, Pitfall #3: Avoid overtuning or disabling NCCL features with environment variables, Profiling and Debugging NCCL, Debugging Correctness Issuestest suite, Debugging Correctness Issuesnumerical correctness and accuracy, Debugging Numerical Correctness and Accuracy-Debugging Numerical Correctness and AccuracyNVIDIA Compute Sanitizer, Debugging Functional Correctness with NVIDIA Compute SanitizerPyTorch compiler, Performance Hints and Debugging Generated Code
- Debugging With Attributed Record Formats (DWARF), CPU and GPU Profiling with Linux perf
- decodeabout decode phase, Scaling Disaggregated Prefill and Decode for Inferenceconstrained decoding, Constrained Decoding Performance Implicationsdisaggregated from prefillabout disaggregation, Multinode Inference, Parallelism, Decoding, and Routing Optimizations, Prefill-Decode Interference, Scaling Disaggregated Prefill and Decode for Inferencearchitecture, Disaggregated Prefill and Decode Architecturebatch size optimizations, Adaptive Batching and Chunked Prefill Schedulingcluster pools, Disaggregated Prefill and Decode Cluster Pools-Memory management for the KV cachedeploying with Kubernetes, Deploying Disaggregated Prefill and Decode with Kubernetes-Deploying Disaggregated Prefill and Decode with Kubernetes, Full-Stack Inference Optimizationslatency and throughput, Impact on Latency (TTFT) and Throughput (TPOT)load balancing, Dynamic Scheduling and Load Balancing-Dynamic resource scalingprefill to decode data transfers, KV Cache Data Transfer and NIXL, Fast KV Cache Transfer Between Prefill and Decode-Zero-Copy GPU-to-GPU Transferscaling, Scaling Prefill and Worker Nodes Independently, Scalability of Disaggregated Prefill and Decodeseparate prefill and decode stages, Separate Prefill and Decode Inference Stages-Separate Prefill and Decode Inference Stages, Disaggregated Prefill and Decode Architecturedisaggregated routing policies, Disaggregated Routing and Scheduling Policies-QoS and early rejection policiescapacity-aware routing, Capacity-aware routing-Capacity-aware routingexample dynamic routing policy configuration, Example dynamic routing policy configurationexample dynamic routing policy in code, Example dynamic routing policy in codelatency-aware routing, Latency-aware routing-Latency-aware routingmultipath inference (racing), Multipath inference (racing)QoS and early rejection policies, QoS and early rejection policies-QoS and early rejection policiesrouting factors, Routing factors-Routing factorsspeculative decoding, Multibranch, parallel speculative decoding across workershardware and parallelism, Heterogeneous Hardware and Parallelism Strategies for Prefill and Decode-Different precision for prefill and decodeKV cache transfer from prefill, Fast KV Cache Transfer Between Prefill and Decode-Zero-Copy GPU-to-GPU Transferconnector and data path design, Connector and Data Path Design-Connector and Data Path DesignKV cache size, KV Cache Sizezero-copy GPU-to-GPU, Zero-Copy GPU-to-GPU Transfer-Zero-Copy GPU-to-GPU TransferLLM decode phase as memory bound, Maintaining High Occupancy and GPU Utilizationoptimized decode kernels, Optimized Decode Kernels-FlexDecoding (PyTorch)FlashMLA, FlashMLA (DeepSeek)FlexDecoding, FlexDecoding (PyTorch)-FlexDecoding (PyTorch)ThunderMLA, ThunderMLA (Stanford)scale-out of workers, Continuous Prewarming of CUDA Graphs and Caches Using Time-Series Predictionspeculative decoding, Disaggregated Prefill and Decode Architecture, Speculative Decoding and Parallel Token Generation Techniques-Combining Decoding Techniques and Evaluating Complexitycombining techniques, Combining Decoding Techniques and Evaluating ComplexityEAGLE algorithm, Two-Model, Draft-Based Speculative Decoding and EAGLEinterleaving decode steps, Interleaving Decode Steps from Multiple RequestsMedusa multiple heads, Speculative Decoding and Parallel Token Generation TechniquesMedusa multiple heads multitoken, Multitoken Decoding with Medusa’s Multiple Heads-Multitoken Decoding with Medusa’s Multiple Headssingle-model self-speculative decoding, Single-Model Self-Speculative Decodingtwo-model draft-based, Two-Model, Draft-Based Speculative Decoding and EAGLE-Two-Model, Draft-Based Speculative Decoding and EAGLEwhy disaggregation, Why Prefill-Decode Disaggregation?-Scalability of Disaggregated Prefill and Decodeadvantages of disaggregation, Advantages of Disaggregation-Phase-specific optimizationscluster pools, Disaggregated Prefill and Decode Cluster Pools-Memory management for the KV cacherouting and scheduling policies, Disaggregated Routing and Scheduling Policies-QoS and early rejection policiesscalability, Scaling Prefill and Worker Nodes Independently, Scalability of Disaggregated Prefill and Decode
- DeepEP (Deep Experts Parallelism; DeepSeek), Transparency and Reproducibilityinline PTX code, DeepSeek’s Use of Inline PTX for Memory Allocation Optimization
- DeepGEMM (DeepSeek), Transparency and Reproducibility
- DeepMind AlphaTensor, AlphaTensor AI-Discovered Algorithms Boosting GPU Performance (Google DeepMind)
- DeepSeek Fire-Flyer File System (3FS), Transparency and Reproducibility, DeepSeek’s Fire-Flyer File System-DeepSeek’s Fire-Flyer File System
- DeepSeek Multi-Head Latent Attention (MLA), Mechanical Sympathy: Hardware-Software Codesign
- DeepSeek-R1 (NVIDIA)automated kernel optimizations, Automated GPU Kernel Optimizations with DeepSeek-R1 (NVIDIA)-Automated GPU Kernel Optimizations with DeepSeek-R1 (NVIDIA)chain-of-thought reasoning, DeepSeek Scales to ~680-Billion Parameter Models Despite US Export Hardware Restrictions in Chinacustom kernels, advanced optimization, less-capable GPUs, Introduction and AI System Overviewefficiency gains, Introduction and AI System Overview, Introduction and AI System OverviewMoE efficiency, Toward 100-Trillion-Parameter Modelstraining costs, Introduction and AI System Overview
- DeepSeek-V3, DeepSeek Scales to ~680-Billion Parameter Models Despite US Export Hardware Restrictions in China-DeepSeek Scales to ~680-Billion Parameter Models Despite US Export Hardware Restrictions in ChinaMoE efficiency, Toward 100-Trillion-Parameter Models
- DeepSeek.AIexport restrictions, Introduction and AI System Overview, DeepSeek Scales to ~680-Billion Parameter Models Despite US Export Hardware Restrictions in China-DeepSeek Scales to ~680-Billion Parameter Models Despite US Export Hardware Restrictions in Chinaopen source GitHub repositories, Transparency and ReproducibilityOpen-Source Week, Transparency and Reproducibility, Transparency and Reproducibility
- deploymentdisaggregated inference system, Deploying Disaggregated Prefill and Decode with Kubernetes-Deploying Disaggregated Prefill and Decode with Kubernetesinference optimizations, Full-Stack Inference OptimizationsNVL72 as preintegrated rack appliance, Preintegrated Rack ApplianceKubernetes, Preintegrated Rack ApplianceNVIDIA Base Command Manager, Preintegrated Rack ApplianceSimple Linux Utility for Resource Management, Preintegrated Rack Appliancetiered deployment, Model Cascading and Tiered Model Deployment
- direct memory access (DMA)NVIDIA GDS, Using NVIDIA GDSO_DIRECT, Using NVIDIA GDSGDS, Using NVIDIA GDSpeer-to-peer DMA, Enabling Peer-to-Peer DMA and UCX
- disaggregated prefill and decodeabout, Multinode Inference, Parallelism, Decoding, and Routing Optimizations, Prefill-Decode Interference, Scaling Disaggregated Prefill and Decode for Inferencearchitecture, Disaggregated Prefill and Decode Architectureseparate prefill and decode, Separate Prefill and Decode Inference Stages-Separate Prefill and Decode Inference Stagesseparate prefill and decode stages, Disaggregated Prefill and Decode Architecturespeculative decoding, Disaggregated Prefill and Decode Architecturebatch size optimizations, Adaptive Batching and Chunked Prefill Schedulingdeploying with Kubernetes, Deploying Disaggregated Prefill and Decode with Kubernetes-Deploying Disaggregated Prefill and Decode with Kubernetes, Full-Stack Inference Optimizationshardware and parallelism, Heterogeneous Hardware and Parallelism Strategies for Prefill and Decode-Different precision for prefill and decodeKV cache transfer from prefill to decode, Fast KV Cache Transfer Between Prefill and Decode-Zero-Copy GPU-to-GPU Transferlatency and throughput, Impact on Latency (TTFT) and Throughput (TPOT)load balancing, Dynamic Scheduling and Load BalancingNIXL, NVIDIA’s NIXL and Disaggregated Inference-NCCL Versus NIXLabout NIXL, NVIDIA’s NIXL and Disaggregated Inference-NVIDIA’s NIXL and Disaggregated Inferenceasynchronous API with callbacks, NIXL Asynchronous API with Callbacks-NIXL Asynchronous API with CallbacksDynamo throughput improved, NIXL and High-Performance Inference Systems Like NVIDIA Dynamointelligent interconnect routing, Intelligent Interconnect Routing for KV Cache Transfers, KV Cache Data Transfer and NIXLKV cache offloading with NIXL, KV Cache Offloading with NIXL-KV Cache Offloading with NIXLprefill to decode data transfers, KV Cache Data Transfer and NIXLseparate prefill and decode inference stages, Separate Prefill and Decode Inference Stages-Separate Prefill and Decode Inference Stagesprefill to decode data transfers, KV Cache Data Transfer and NIXLconnector and data path design, Connector and Data Path Design-Connector and Data Path DesignKV cache, Fast KV Cache Transfer Between Prefill and Decode-Zero-Copy GPU-to-GPU TransferKV cache size, KV Cache Sizezero-copy GPU-to-GPU, Zero-Copy GPU-to-GPU Transfer-Zero-Copy GPU-to-GPU Transferprefill-decode interference, Prefill-Decode Interferencerouting and scheduling policies, Disaggregated Routing and Scheduling Policies-QoS and early rejection policiescapacity-aware routing, Capacity-aware routing-Capacity-aware routingexample dynamic routing policy configuration, Example dynamic routing policy configurationexample dynamic routing policy in code, Example dynamic routing policy in codelatency-aware routing, Latency-aware routing-Latency-aware routingmultipath inference (racing), Multipath inference (racing)QoS and early rejection policies, QoS and early rejection policies-QoS and early rejection policiesrouting factors, Routing factors-Routing factorsspeculative decoding, Multibranch, parallel speculative decoding across workersscaling, Scaling Prefill and Worker Nodes Independentlyscale-out of workers, Continuous Prewarming of CUDA Graphs and Caches Using Time-Series Predictionwhy disaggregation, Why Prefill-Decode Disaggregation?-Scalability of Disaggregated Prefill and Decodeadvantages of disaggregation, Advantages of Disaggregation-Phase-specific optimizations, Heterogeneous Hardware and Parallelism Strategies for Prefill and Decodecluster pools, Disaggregated Prefill and Decode Cluster Pools-Memory management for the KV cacherouting and scheduling policies, Disaggregated Routing and Scheduling Policies-QoS and early rejection policiesscalability, Scalability of Disaggregated Prefill and Decode
- Distributed Data Parallel (DDP; PyTorch)bucketing, Reducing Communication Frequency and VolumeData Parallel versus, Distributed Data Parallel Strategies-Distributed Data Parallel Strategiesdata parallelism as parallelism strategy, Parallelism Strategies for Serving Massive MoE Models, Data Parallelismdefinition of Data Parallel, Distributed Data Parallel Strategies, Data Parallelismdefinition of Distributed Data Parallel, Distributed Data Parallel Strategiesscaling via DDP with torch.compile, DDP with torch.compile
- distributed networking communicationmulti-GPU communication, NCCL for Distributed Multi-GPU Communication-In-Network SHARP Aggregationcongestion- and topology-aware scheduling, Congestion-Aware and Topology-Aware Scheduling with Multiple GPUs-Coordinating NVSwitch Transfers with Fine-Tuned Schedulingdefinition of Data Parallel, Distributed Data Parallel Strategiesdefinition of Distributed Data Parallel, Distributed Data Parallel StrategiesDistributed Data Parallel strategies, Distributed Data Parallel Strategies-Distributed Data Parallel StrategiesNCCL communication algorithms, NCCL Communication Algorithms-NCCL Communication AlgorithmsNCCL topology awareness, Topology Awareness in NCCL-Topology Awareness in NCCLprofiling and debugging NCCL, Profiling and Debugging NCCLmultinode communication pitfalls, Multinode Communication Pitfalls-Pitfall #6: GPU memory fragmentation under UCX/RDMAGLOO backend instead of NCCL, Pitfall #1: Using a CPU-bound Gloo backend instead of NCCL-Pitfall #1: Using a CPU-bound Gloo backend instead of NCCLGPU memory fragmentation under UCX/RDMA, Pitfall #6: GPU memory fragmentation under UCX/RDMA-Pitfall #6: GPU memory fragmentation under UCX/RDMAinsufficient bandwidth or misconfigured NICs, Pitfall #4: Insufficient network bandwidth or misconfigured NICsmismatched NCCL versions, Pitfall #2: Mismatched NCCL versionsstraggler nodes or processes, Pitfall #5: Straggler nodes or processessummary, Pitfall #6: GPU memory fragmentation under UCX/RDMATCP port exhaustion during NCCL bootstrap, Pitfall #3: TCP port exhaustion during NCCL bootstrapmultinode connectivity tuned, Tuning Multinode Connectivity-Tuning Multinode ConnectivityNCCL pitfalls and gotchas, NCCL Communicator Lifecycle and Environment Gotchas-Pitfall #6: NCCL communicator hangs, errors, or shuts down completelycommunicator hanging, giving errors, shutting down, Pitfall #6: NCCL communicator hangs, errors, or shuts down completely-Pitfall #6: NCCL communicator hangs, errors, or shuts down completelyCPU-GPU NUMA-node for NCCL threads, Pitfall #4: Verify CPU-GPU NUMA-node affinity for NCCL threadscreating and destroying communicators, Pitfall #2: Do not create and destroy NCCL communicators on every iterationcreating communicators too often, Pitfall #1: Creating NCCL communicators too often-Pitfall #1: Creating NCCL communicators too oftenignoring warnings and errors, Pitfall #5: Resist the temptation to ignore NCCL warnings and errorsoverturning or disabling features via environment variables, Pitfall #3: Avoid overtuning or disabling NCCL features with environment variables-NCCL_SHARP_DISABLENVIDIA Magnum IO, NVIDIA Magnum IO Optimization Stackoptimizing for Kubernetes, Optimizing Network Communication for Kubernetesoverlapping communication and computation, Overlapping Communication and Computation (Pipelining)-Achieving Maximal Overlap in Practicecomparison of no overlap with overlap, Achieving Maximal Overlap in Practice-Achieving Maximal Overlap in PracticeDistributed Data Parallel strategies, Distributed Data Parallel Strategies-Distributed Data Parallel Strategiesreducing frequency and volume, Reducing Communication Frequency and Volumestreams for asynchronous execution, Asynchronous Execution with StreamsRDMA, High-Speed, Low-Overhead Data Transfers with RDMA-Pitfall #6: GPU memory fragmentation under UCX/RDMA
- distributed shared memory (DSMEM), Tiling with Thread Block Clusterssharing state between thread blocks, Distributed Shared Memorythread block clusters, Thread Block Clusters and Distributed Shared Memory-Thread Block Pairalgorithms for parallelizing workloads, Designing Efficient Algorithms with Thread Block Clusters-Designing Efficient Algorithms with Thread Block Clusterscoordinating thread block clusters, Coordinating Thread Block Clusters with Cooperative Groups API-Coordinating Thread Block Clusters with Cooperative Groups APIdescribed, Intra-Kernel Pipelining, Warp Specialization, and Cooperative Thread Block Clustersglobal memory traffic reduced, Reducing Global Memory Traffic with Thread Block Clusters-Reducing Global Memory Traffic with Thread Block Clusterslaunching a thread block cluster, Launching a Thread Block Cluster, Designing Efficient Algorithms with Thread Block Clustersscratch memory, Scratch Memorythread block pair, Thread Block Pair-Thread Block Pairthread block swizzling, Thread Block Swizzlingthreads accessing each other’s shared memory, Threads, Warps, Blocks, and Grids, Intra-Kernel Pipelining, Warp Specialization, and Cooperative Thread Block Clusters, Thread Block Clusters and Distributed Shared Memorywarp specialization with thread block clusters, Warp Specialization with Thread Block Clusters-Warp Specialization with Thread Block Clusters
- DistServe, Scaling Disaggregated Prefill and Decode for Inference
- DMA (see direct memory access (DMA))
- Dockerno swap and enforcing NUMA affinity, Virtual Memory and SwappingNVIDIA Container Toolkit, Container Runtime Optimizations for GPUsRDMA, High-Speed, Low-Overhead Data Transfers with RDMAruntime optimizationsabout containers, Container Runtime Optimizations for GPUscontainer overlay filesystem overhead, Avoiding Container Overlay Filesystem OverheadNVIDIA container runtime injecting libraries, NVIDIA Container RuntimeNVIDIA Container Toolkit and CUDA compatibility, NVIDIA Container Toolkit and CUDA Compatibilityreducing image size, Reduce Image Size for Faster Container Startupruntime optimizations for GPUs, Container Runtime Optimizations for GPUs-Reduce Image Size for Faster Container Startup
- double buffering with CUTLASS, Using CUTLASS for Optimal Arithmetic Intensity and Tensor Core Performance
- DP (see data parallelism (DP); dynamic parallelism (DP))
- DPU (Data Processing Unit) in NVL72, Multirack and Storage Communication
- DRAM as shared memory, Tiling and Data Reuse Using Shared Memory(see also CPU DRAM)
- DSMEM (see distributed shared memory (DSMEM))
- DTensor (PyTorch), TorchTitan, AsyncTP, AutoParallel, and SimpleFSDP
- dual-die GPUs, NVIDIA Blackwell “Dual-Die” GPUappearing as one GPU, NVIDIA Blackwell “Dual-Die” GPUGPC “partitions”, Thread Block Clusters and Distributed Shared Memoryprogramming, Understanding GPU Memory Hierarchyremote memory access latency, NUMA Awareness and CPU Pinning
- DualPipe (DeepSeek), Transparency and Reproducibilitymixture-of-experts model DeepSeek-V3, DeepSeek Scales to ~680-Billion Parameter Models Despite US Export Hardware Restrictions in China
- DWARF (Debugging With Attributed Record Formats), CPU and GPU Profiling with Linux perf
- dynamic activation quantization, Dynamic Quantization and Activation Range Adjustment
- dynamic batching, Dynamic Batching-Dynamic Batchingadaptive batching delay, Dynamic Batchingstatic batching versus, Dynamic Batching
- dynamic CUDA Graph update, Dynamic Graph Update
- dynamic early exit networks, Dynamic Early-Exit Networks
- dynamic parallelism (DP), Dynamic Parallelism-Dynamic Parallelismbenchmarks, Dynamic Parallelismprofiling before and after a change, Dynamic Parallelismchild-kernel launch limit, Dynamic Parallelismdevice memory usage monitored, Dynamic Parallelism“stack overflow” errors avoided, Dynamic Parallelismwhen to capture CUDA Graph instead, Dynamic Parallelism
- dynamic routinglatency-aware scheduling, Latency-Aware Scheduling and Dynamic Routing-Latency-Aware Scheduling and Dynamic RoutingMoE inference strategies, Dynamic Routing Strategies for MoE Inference-Adaptive Expert Routing and Real-Time Monitoringexpert communication optimization, Expert Communication Optimization-Expert Communication Optimizationexpert replication, Load Balancing, Capacity Factor, and Expert Replicationload balancing and capacity factors, Load Balancing, Capacity Factor, and Expert Replication
- dynamic schedulingatomic work queues, Dynamic Scheduling with Atomic Work Queues-Atomic Queuesatomic counters, Atomic Countersatomic queues, Atomic Queues-Atomic Queuesdynamic work allocation, Atomic CountersNsight Compute, Atomic Counters-Atomic Countersdynamic congestion-aware scheduling, Dynamic Congestion-Aware Scheduling-Dynamic Congestion-Aware Schedulingload balancing, Dynamic Scheduling and Load Balancing-Dynamic resource scalingadaptive resource scheduling and hotspot prevention, Adaptive Resource Scheduling and Hotspot Prevention-Dynamic resource scalingArrow’s adaptive instance scaling, Arrow’s adaptive instance scaling-Arrow’s adaptive instance scalingdynamic resource scaling, Dynamic resource scalingMooncake adaptive strategies, Mooncake adaptive strategiesTetriInfer’s two-level scheduler, TetriInfer’s two-level scheduler
- dynamic shapes (PyTorch)TorchDynamo, Profiling and Debugging Compiler Performance Issues, TorchDynamo for Bytecode Capture and Graph Extractionvariable-length sequences via, Dynamic Shapes and Variable Sequence Lengths-Dynamic Shapes and Variable Sequence Lengthsprofiling dynamic shapes versus padding, Dynamic Shapes and Variable Sequence LengthsXLA, PyTorch XLA Backend
- dynamic shared-memory allocation, Dynamic Shared-Memory Allocation and Occupancy-Aware Kernel Selection
- Dynamo (see NVIDIA Dynamo; TorchDynamo (PyTorch))

- I/O monitoring, Monitoring Storage I/O
- inference monitoring, Monitoring System Metrics and Counters
- monitoring straggler nodes or processes, Pitfall #5: Straggler nodes or processes

- current used at typical data center, Compute Density and Power Requirements
- liquid cooling, Liquid Cooling Versus Air Cooling

- definition of data parallelism, Distributed Data Parallel Strategies, Data Parallelism
- Distributed Data Parallel versus Data Parallel, Distributed Data Parallel Strategies-Distributed Data Parallel Strategiesdefinition of Distributed Data Parallel, Distributed Data Parallel Strategies
- model weights and data split over GPUs, Parallelism Strategies for Serving Massive MoE Models
- versus TP, PP, and hybrid, Adaptive Parallelism Strategies (TP Versus PP Versus Hybrid)-Adaptive Parallelism Strategies (TP Versus PP Versus Hybrid)

- definition of Distributed Data Parallel, Distributed Data Parallel Strategies

- arithmetic intensity increased by, Multilevel Microtiling and Software Prefetching

- CPU affinity for each worker process, NUMA Awareness and CPU Pinning-NUMA Awareness and CPU Pinning
- data input pipeline optimization, Optimizing the Data Input Pipeline-Optimizing the Data Input Pipeline
- data loading and preprocessing, Efficient Data Loading and Preprocessing-Efficient Data Loading and Preprocessing
- multiple workers, Tuning NVMe and Filesystem for Throughput, Efficient Data Loading and Preprocessing, Scaling Out Workers as You Scale Out Number of GPUs, Optimizing the Data Input Pipeline
- persistent workers, Efficient Data Loading and Preprocessing
- pinned memory flag, NUMA-Friendly Memory Allocation and Memory Pinning, Using Streams to Overlap Compute with Data Transfers, Optimizing the Data Input Pipeline
- prefetch factor, Optimizing the Data Input Pipeline

- compiler performance issues, Profiling and Debugging Compiler Performance Issues
- DWARF, CPU and GPU Profiling with Linux perf
- graph breaks, Graph Breaks and TorchDynamo explain(), Debugging Compiler Phases, Graph Breaks, and Performancemonitoring graph recompilations, TorchDynamo for Bytecode Capture and Graph Extraction, Minimize Graph Recompilations
- inference correctness issues, Debugging Correctness Issues-Debugging Correctness Issues
- NCCL, Profiling and Debugging NCCLdebugging enabled, Profiling and Debugging NCCL, Debugging Correctness IssuesNCCL logs, In-Network SHARP Aggregation, Debugging Correctness IssuesNCCL_DEBUG, Topology Awareness in NCCL, Pitfall #3: Avoid overtuning or disabling NCCL features with environment variables, Profiling and Debugging NCCL, Debugging Correctness Issuestest suite, Debugging Correctness Issues
- numerical correctness and accuracy, Debugging Numerical Correctness and Accuracy-Debugging Numerical Correctness and Accuracy
- NVIDIA Compute Sanitizer, Debugging Functional Correctness with NVIDIA Compute Sanitizer
- PyTorch compiler, Performance Hints and Debugging Generated Code

- monitoring graph recompilations, TorchDynamo for Bytecode Capture and Graph Extraction, Minimize Graph Recompilations

- debugging enabled, Profiling and Debugging NCCL, Debugging Correctness Issues
- NCCL logs, In-Network SHARP Aggregation, Debugging Correctness Issues
- NCCL_DEBUG, Topology Awareness in NCCL, Pitfall #3: Avoid overtuning or disabling NCCL features with environment variables, Profiling and Debugging NCCL, Debugging Correctness Issues
- test suite, Debugging Correctness Issues

- about decode phase, Scaling Disaggregated Prefill and Decode for Inference
- constrained decoding, Constrained Decoding Performance Implications
- disaggregated from prefillabout disaggregation, Multinode Inference, Parallelism, Decoding, and Routing Optimizations, Prefill-Decode Interference, Scaling Disaggregated Prefill and Decode for Inferencearchitecture, Disaggregated Prefill and Decode Architecturebatch size optimizations, Adaptive Batching and Chunked Prefill Schedulingcluster pools, Disaggregated Prefill and Decode Cluster Pools-Memory management for the KV cachedeploying with Kubernetes, Deploying Disaggregated Prefill and Decode with Kubernetes-Deploying Disaggregated Prefill and Decode with Kubernetes, Full-Stack Inference Optimizationslatency and throughput, Impact on Latency (TTFT) and Throughput (TPOT)load balancing, Dynamic Scheduling and Load Balancing-Dynamic resource scalingprefill to decode data transfers, KV Cache Data Transfer and NIXL, Fast KV Cache Transfer Between Prefill and Decode-Zero-Copy GPU-to-GPU Transferscaling, Scaling Prefill and Worker Nodes Independently, Scalability of Disaggregated Prefill and Decodeseparate prefill and decode stages, Separate Prefill and Decode Inference Stages-Separate Prefill and Decode Inference Stages, Disaggregated Prefill and Decode Architecture
- disaggregated routing policies, Disaggregated Routing and Scheduling Policies-QoS and early rejection policiescapacity-aware routing, Capacity-aware routing-Capacity-aware routingexample dynamic routing policy configuration, Example dynamic routing policy configurationexample dynamic routing policy in code, Example dynamic routing policy in codelatency-aware routing, Latency-aware routing-Latency-aware routingmultipath inference (racing), Multipath inference (racing)QoS and early rejection policies, QoS and early rejection policies-QoS and early rejection policiesrouting factors, Routing factors-Routing factorsspeculative decoding, Multibranch, parallel speculative decoding across workers
- hardware and parallelism, Heterogeneous Hardware and Parallelism Strategies for Prefill and Decode-Different precision for prefill and decode
- KV cache transfer from prefill, Fast KV Cache Transfer Between Prefill and Decode-Zero-Copy GPU-to-GPU Transferconnector and data path design, Connector and Data Path Design-Connector and Data Path DesignKV cache size, KV Cache Sizezero-copy GPU-to-GPU, Zero-Copy GPU-to-GPU Transfer-Zero-Copy GPU-to-GPU Transfer
- LLM decode phase as memory bound, Maintaining High Occupancy and GPU Utilization
- optimized decode kernels, Optimized Decode Kernels-FlexDecoding (PyTorch)FlashMLA, FlashMLA (DeepSeek)FlexDecoding, FlexDecoding (PyTorch)-FlexDecoding (PyTorch)ThunderMLA, ThunderMLA (Stanford)
- scale-out of workers, Continuous Prewarming of CUDA Graphs and Caches Using Time-Series Prediction
- speculative decoding, Disaggregated Prefill and Decode Architecture, Speculative Decoding and Parallel Token Generation Techniques-Combining Decoding Techniques and Evaluating Complexitycombining techniques, Combining Decoding Techniques and Evaluating ComplexityEAGLE algorithm, Two-Model, Draft-Based Speculative Decoding and EAGLEinterleaving decode steps, Interleaving Decode Steps from Multiple RequestsMedusa multiple heads, Speculative Decoding and Parallel Token Generation TechniquesMedusa multiple heads multitoken, Multitoken Decoding with Medusa’s Multiple Heads-Multitoken Decoding with Medusa’s Multiple Headssingle-model self-speculative decoding, Single-Model Self-Speculative Decodingtwo-model draft-based, Two-Model, Draft-Based Speculative Decoding and EAGLE-Two-Model, Draft-Based Speculative Decoding and EAGLE
- why disaggregation, Why Prefill-Decode Disaggregation?-Scalability of Disaggregated Prefill and Decodeadvantages of disaggregation, Advantages of Disaggregation-Phase-specific optimizationscluster pools, Disaggregated Prefill and Decode Cluster Pools-Memory management for the KV cacherouting and scheduling policies, Disaggregated Routing and Scheduling Policies-QoS and early rejection policiesscalability, Scaling Prefill and Worker Nodes Independently, Scalability of Disaggregated Prefill and Decode

- about disaggregation, Multinode Inference, Parallelism, Decoding, and Routing Optimizations, Prefill-Decode Interference, Scaling Disaggregated Prefill and Decode for Inference
- architecture, Disaggregated Prefill and Decode Architecture
- batch size optimizations, Adaptive Batching and Chunked Prefill Scheduling
- cluster pools, Disaggregated Prefill and Decode Cluster Pools-Memory management for the KV cache
- deploying with Kubernetes, Deploying Disaggregated Prefill and Decode with Kubernetes-Deploying Disaggregated Prefill and Decode with Kubernetes, Full-Stack Inference Optimizations
- latency and throughput, Impact on Latency (TTFT) and Throughput (TPOT)
- load balancing, Dynamic Scheduling and Load Balancing-Dynamic resource scaling
- prefill to decode data transfers, KV Cache Data Transfer and NIXL, Fast KV Cache Transfer Between Prefill and Decode-Zero-Copy GPU-to-GPU Transfer
- scaling, Scaling Prefill and Worker Nodes Independently, Scalability of Disaggregated Prefill and Decode
- separate prefill and decode stages, Separate Prefill and Decode Inference Stages-Separate Prefill and Decode Inference Stages, Disaggregated Prefill and Decode Architecture

- capacity-aware routing, Capacity-aware routing-Capacity-aware routing
- example dynamic routing policy configuration, Example dynamic routing policy configuration
- example dynamic routing policy in code, Example dynamic routing policy in code
- latency-aware routing, Latency-aware routing-Latency-aware routing
- multipath inference (racing), Multipath inference (racing)
- QoS and early rejection policies, QoS and early rejection policies-QoS and early rejection policies
- routing factors, Routing factors-Routing factors
- speculative decoding, Multibranch, parallel speculative decoding across workers

- connector and data path design, Connector and Data Path Design-Connector and Data Path Design
- KV cache size, KV Cache Size
- zero-copy GPU-to-GPU, Zero-Copy GPU-to-GPU Transfer-Zero-Copy GPU-to-GPU Transfer

- FlashMLA, FlashMLA (DeepSeek)
- FlexDecoding, FlexDecoding (PyTorch)-FlexDecoding (PyTorch)
- ThunderMLA, ThunderMLA (Stanford)

- combining techniques, Combining Decoding Techniques and Evaluating Complexity
- EAGLE algorithm, Two-Model, Draft-Based Speculative Decoding and EAGLE
- interleaving decode steps, Interleaving Decode Steps from Multiple Requests
- Medusa multiple heads, Speculative Decoding and Parallel Token Generation Techniques
- Medusa multiple heads multitoken, Multitoken Decoding with Medusa’s Multiple Heads-Multitoken Decoding with Medusa’s Multiple Heads
- single-model self-speculative decoding, Single-Model Self-Speculative Decoding
- two-model draft-based, Two-Model, Draft-Based Speculative Decoding and EAGLE-Two-Model, Draft-Based Speculative Decoding and EAGLE

- advantages of disaggregation, Advantages of Disaggregation-Phase-specific optimizations
- cluster pools, Disaggregated Prefill and Decode Cluster Pools-Memory management for the KV cache
- routing and scheduling policies, Disaggregated Routing and Scheduling Policies-QoS and early rejection policies
- scalability, Scaling Prefill and Worker Nodes Independently, Scalability of Disaggregated Prefill and Decode

- inline PTX code, DeepSeek’s Use of Inline PTX for Memory Allocation Optimization

- automated kernel optimizations, Automated GPU Kernel Optimizations with DeepSeek-R1 (NVIDIA)-Automated GPU Kernel Optimizations with DeepSeek-R1 (NVIDIA)
- chain-of-thought reasoning, DeepSeek Scales to ~680-Billion Parameter Models Despite US Export Hardware Restrictions in China
- custom kernels, advanced optimization, less-capable GPUs, Introduction and AI System Overview
- efficiency gains, Introduction and AI System Overview, Introduction and AI System Overview
- MoE efficiency, Toward 100-Trillion-Parameter Models
- training costs, Introduction and AI System Overview

- MoE efficiency, Toward 100-Trillion-Parameter Models

- export restrictions, Introduction and AI System Overview, DeepSeek Scales to ~680-Billion Parameter Models Despite US Export Hardware Restrictions in China-DeepSeek Scales to ~680-Billion Parameter Models Despite US Export Hardware Restrictions in China
- open source GitHub repositories, Transparency and ReproducibilityOpen-Source Week, Transparency and Reproducibility, Transparency and Reproducibility

- Open-Source Week, Transparency and Reproducibility, Transparency and Reproducibility

- disaggregated inference system, Deploying Disaggregated Prefill and Decode with Kubernetes-Deploying Disaggregated Prefill and Decode with Kubernetes
- inference optimizations, Full-Stack Inference Optimizations
- NVL72 as preintegrated rack appliance, Preintegrated Rack ApplianceKubernetes, Preintegrated Rack ApplianceNVIDIA Base Command Manager, Preintegrated Rack ApplianceSimple Linux Utility for Resource Management, Preintegrated Rack Appliance
- tiered deployment, Model Cascading and Tiered Model Deployment

- Kubernetes, Preintegrated Rack Appliance
- NVIDIA Base Command Manager, Preintegrated Rack Appliance
- Simple Linux Utility for Resource Management, Preintegrated Rack Appliance

- NVIDIA GDS, Using NVIDIA GDS
- O_DIRECT, Using NVIDIA GDSGDS, Using NVIDIA GDS
- peer-to-peer DMA, Enabling Peer-to-Peer DMA and UCX

- GDS, Using NVIDIA GDS

- about, Multinode Inference, Parallelism, Decoding, and Routing Optimizations, Prefill-Decode Interference, Scaling Disaggregated Prefill and Decode for Inference
- architecture, Disaggregated Prefill and Decode Architectureseparate prefill and decode, Separate Prefill and Decode Inference Stages-Separate Prefill and Decode Inference Stagesseparate prefill and decode stages, Disaggregated Prefill and Decode Architecturespeculative decoding, Disaggregated Prefill and Decode Architecture
- batch size optimizations, Adaptive Batching and Chunked Prefill Scheduling
- deploying with Kubernetes, Deploying Disaggregated Prefill and Decode with Kubernetes-Deploying Disaggregated Prefill and Decode with Kubernetes, Full-Stack Inference Optimizations
- hardware and parallelism, Heterogeneous Hardware and Parallelism Strategies for Prefill and Decode-Different precision for prefill and decode
- KV cache transfer from prefill to decode, Fast KV Cache Transfer Between Prefill and Decode-Zero-Copy GPU-to-GPU Transfer
- latency and throughput, Impact on Latency (TTFT) and Throughput (TPOT)
- load balancing, Dynamic Scheduling and Load Balancing
- NIXL, NVIDIA’s NIXL and Disaggregated Inference-NCCL Versus NIXLabout NIXL, NVIDIA’s NIXL and Disaggregated Inference-NVIDIA’s NIXL and Disaggregated Inferenceasynchronous API with callbacks, NIXL Asynchronous API with Callbacks-NIXL Asynchronous API with CallbacksDynamo throughput improved, NIXL and High-Performance Inference Systems Like NVIDIA Dynamointelligent interconnect routing, Intelligent Interconnect Routing for KV Cache Transfers, KV Cache Data Transfer and NIXLKV cache offloading with NIXL, KV Cache Offloading with NIXL-KV Cache Offloading with NIXLprefill to decode data transfers, KV Cache Data Transfer and NIXLseparate prefill and decode inference stages, Separate Prefill and Decode Inference Stages-Separate Prefill and Decode Inference Stages
- prefill to decode data transfers, KV Cache Data Transfer and NIXLconnector and data path design, Connector and Data Path Design-Connector and Data Path DesignKV cache, Fast KV Cache Transfer Between Prefill and Decode-Zero-Copy GPU-to-GPU TransferKV cache size, KV Cache Sizezero-copy GPU-to-GPU, Zero-Copy GPU-to-GPU Transfer-Zero-Copy GPU-to-GPU Transfer
- prefill-decode interference, Prefill-Decode Interference
- routing and scheduling policies, Disaggregated Routing and Scheduling Policies-QoS and early rejection policiescapacity-aware routing, Capacity-aware routing-Capacity-aware routingexample dynamic routing policy configuration, Example dynamic routing policy configurationexample dynamic routing policy in code, Example dynamic routing policy in codelatency-aware routing, Latency-aware routing-Latency-aware routingmultipath inference (racing), Multipath inference (racing)QoS and early rejection policies, QoS and early rejection policies-QoS and early rejection policiesrouting factors, Routing factors-Routing factorsspeculative decoding, Multibranch, parallel speculative decoding across workers
- scaling, Scaling Prefill and Worker Nodes Independentlyscale-out of workers, Continuous Prewarming of CUDA Graphs and Caches Using Time-Series Prediction
- why disaggregation, Why Prefill-Decode Disaggregation?-Scalability of Disaggregated Prefill and Decodeadvantages of disaggregation, Advantages of Disaggregation-Phase-specific optimizations, Heterogeneous Hardware and Parallelism Strategies for Prefill and Decodecluster pools, Disaggregated Prefill and Decode Cluster Pools-Memory management for the KV cacherouting and scheduling policies, Disaggregated Routing and Scheduling Policies-QoS and early rejection policiesscalability, Scalability of Disaggregated Prefill and Decode

- separate prefill and decode, Separate Prefill and Decode Inference Stages-Separate Prefill and Decode Inference Stages
- separate prefill and decode stages, Disaggregated Prefill and Decode Architecture
- speculative decoding, Disaggregated Prefill and Decode Architecture

- about NIXL, NVIDIA’s NIXL and Disaggregated Inference-NVIDIA’s NIXL and Disaggregated Inference
- asynchronous API with callbacks, NIXL Asynchronous API with Callbacks-NIXL Asynchronous API with Callbacks
- Dynamo throughput improved, NIXL and High-Performance Inference Systems Like NVIDIA Dynamo
- intelligent interconnect routing, Intelligent Interconnect Routing for KV Cache Transfers, KV Cache Data Transfer and NIXL
- KV cache offloading with NIXL, KV Cache Offloading with NIXL-KV Cache Offloading with NIXL
- prefill to decode data transfers, KV Cache Data Transfer and NIXL
- separate prefill and decode inference stages, Separate Prefill and Decode Inference Stages-Separate Prefill and Decode Inference Stages

- connector and data path design, Connector and Data Path Design-Connector and Data Path Design
- KV cache, Fast KV Cache Transfer Between Prefill and Decode-Zero-Copy GPU-to-GPU Transfer
- KV cache size, KV Cache Size
- zero-copy GPU-to-GPU, Zero-Copy GPU-to-GPU Transfer-Zero-Copy GPU-to-GPU Transfer

- capacity-aware routing, Capacity-aware routing-Capacity-aware routing
- example dynamic routing policy configuration, Example dynamic routing policy configuration
- example dynamic routing policy in code, Example dynamic routing policy in code
- latency-aware routing, Latency-aware routing-Latency-aware routing
- multipath inference (racing), Multipath inference (racing)
- QoS and early rejection policies, QoS and early rejection policies-QoS and early rejection policies
- routing factors, Routing factors-Routing factors
- speculative decoding, Multibranch, parallel speculative decoding across workers

- scale-out of workers, Continuous Prewarming of CUDA Graphs and Caches Using Time-Series Prediction

- advantages of disaggregation, Advantages of Disaggregation-Phase-specific optimizations, Heterogeneous Hardware and Parallelism Strategies for Prefill and Decode
- cluster pools, Disaggregated Prefill and Decode Cluster Pools-Memory management for the KV cache
- routing and scheduling policies, Disaggregated Routing and Scheduling Policies-QoS and early rejection policies
- scalability, Scalability of Disaggregated Prefill and Decode

- bucketing, Reducing Communication Frequency and Volume
- Data Parallel versus, Distributed Data Parallel Strategies-Distributed Data Parallel Strategiesdata parallelism as parallelism strategy, Parallelism Strategies for Serving Massive MoE Models, Data Parallelismdefinition of Data Parallel, Distributed Data Parallel Strategies, Data Parallelismdefinition of Distributed Data Parallel, Distributed Data Parallel Strategies
- scaling via DDP with torch.compile, DDP with torch.compile

- data parallelism as parallelism strategy, Parallelism Strategies for Serving Massive MoE Models, Data Parallelism
- definition of Data Parallel, Distributed Data Parallel Strategies, Data Parallelism
- definition of Distributed Data Parallel, Distributed Data Parallel Strategies

- multi-GPU communication, NCCL for Distributed Multi-GPU Communication-In-Network SHARP Aggregationcongestion- and topology-aware scheduling, Congestion-Aware and Topology-Aware Scheduling with Multiple GPUs-Coordinating NVSwitch Transfers with Fine-Tuned Schedulingdefinition of Data Parallel, Distributed Data Parallel Strategiesdefinition of Distributed Data Parallel, Distributed Data Parallel StrategiesDistributed Data Parallel strategies, Distributed Data Parallel Strategies-Distributed Data Parallel StrategiesNCCL communication algorithms, NCCL Communication Algorithms-NCCL Communication AlgorithmsNCCL topology awareness, Topology Awareness in NCCL-Topology Awareness in NCCLprofiling and debugging NCCL, Profiling and Debugging NCCL
- multinode communication pitfalls, Multinode Communication Pitfalls-Pitfall #6: GPU memory fragmentation under UCX/RDMAGLOO backend instead of NCCL, Pitfall #1: Using a CPU-bound Gloo backend instead of NCCL-Pitfall #1: Using a CPU-bound Gloo backend instead of NCCLGPU memory fragmentation under UCX/RDMA, Pitfall #6: GPU memory fragmentation under UCX/RDMA-Pitfall #6: GPU memory fragmentation under UCX/RDMAinsufficient bandwidth or misconfigured NICs, Pitfall #4: Insufficient network bandwidth or misconfigured NICsmismatched NCCL versions, Pitfall #2: Mismatched NCCL versionsstraggler nodes or processes, Pitfall #5: Straggler nodes or processessummary, Pitfall #6: GPU memory fragmentation under UCX/RDMATCP port exhaustion during NCCL bootstrap, Pitfall #3: TCP port exhaustion during NCCL bootstrap
- multinode connectivity tuned, Tuning Multinode Connectivity-Tuning Multinode Connectivity
- NCCL pitfalls and gotchas, NCCL Communicator Lifecycle and Environment Gotchas-Pitfall #6: NCCL communicator hangs, errors, or shuts down completelycommunicator hanging, giving errors, shutting down, Pitfall #6: NCCL communicator hangs, errors, or shuts down completely-Pitfall #6: NCCL communicator hangs, errors, or shuts down completelyCPU-GPU NUMA-node for NCCL threads, Pitfall #4: Verify CPU-GPU NUMA-node affinity for NCCL threadscreating and destroying communicators, Pitfall #2: Do not create and destroy NCCL communicators on every iterationcreating communicators too often, Pitfall #1: Creating NCCL communicators too often-Pitfall #1: Creating NCCL communicators too oftenignoring warnings and errors, Pitfall #5: Resist the temptation to ignore NCCL warnings and errorsoverturning or disabling features via environment variables, Pitfall #3: Avoid overtuning or disabling NCCL features with environment variables-NCCL_SHARP_DISABLE
- NVIDIA Magnum IO, NVIDIA Magnum IO Optimization Stack
- optimizing for Kubernetes, Optimizing Network Communication for Kubernetes
- overlapping communication and computation, Overlapping Communication and Computation (Pipelining)-Achieving Maximal Overlap in Practicecomparison of no overlap with overlap, Achieving Maximal Overlap in Practice-Achieving Maximal Overlap in PracticeDistributed Data Parallel strategies, Distributed Data Parallel Strategies-Distributed Data Parallel Strategiesreducing frequency and volume, Reducing Communication Frequency and Volumestreams for asynchronous execution, Asynchronous Execution with Streams
- RDMA, High-Speed, Low-Overhead Data Transfers with RDMA-Pitfall #6: GPU memory fragmentation under UCX/RDMA

- congestion- and topology-aware scheduling, Congestion-Aware and Topology-Aware Scheduling with Multiple GPUs-Coordinating NVSwitch Transfers with Fine-Tuned Scheduling
- definition of Data Parallel, Distributed Data Parallel Strategies
- definition of Distributed Data Parallel, Distributed Data Parallel Strategies
- Distributed Data Parallel strategies, Distributed Data Parallel Strategies-Distributed Data Parallel Strategies
- NCCL communication algorithms, NCCL Communication Algorithms-NCCL Communication Algorithms
- NCCL topology awareness, Topology Awareness in NCCL-Topology Awareness in NCCL
- profiling and debugging NCCL, Profiling and Debugging NCCL

- GLOO backend instead of NCCL, Pitfall #1: Using a CPU-bound Gloo backend instead of NCCL-Pitfall #1: Using a CPU-bound Gloo backend instead of NCCL
- GPU memory fragmentation under UCX/RDMA, Pitfall #6: GPU memory fragmentation under UCX/RDMA-Pitfall #6: GPU memory fragmentation under UCX/RDMA
- insufficient bandwidth or misconfigured NICs, Pitfall #4: Insufficient network bandwidth or misconfigured NICs
- mismatched NCCL versions, Pitfall #2: Mismatched NCCL versions
- straggler nodes or processes, Pitfall #5: Straggler nodes or processes
- summary, Pitfall #6: GPU memory fragmentation under UCX/RDMA
- TCP port exhaustion during NCCL bootstrap, Pitfall #3: TCP port exhaustion during NCCL bootstrap

- communicator hanging, giving errors, shutting down, Pitfall #6: NCCL communicator hangs, errors, or shuts down completely-Pitfall #6: NCCL communicator hangs, errors, or shuts down completely
- CPU-GPU NUMA-node for NCCL threads, Pitfall #4: Verify CPU-GPU NUMA-node affinity for NCCL threads
- creating and destroying communicators, Pitfall #2: Do not create and destroy NCCL communicators on every iteration
- creating communicators too often, Pitfall #1: Creating NCCL communicators too often-Pitfall #1: Creating NCCL communicators too often
- ignoring warnings and errors, Pitfall #5: Resist the temptation to ignore NCCL warnings and errors
- overturning or disabling features via environment variables, Pitfall #3: Avoid overtuning or disabling NCCL features with environment variables-NCCL_SHARP_DISABLE

- comparison of no overlap with overlap, Achieving Maximal Overlap in Practice-Achieving Maximal Overlap in Practice
- Distributed Data Parallel strategies, Distributed Data Parallel Strategies-Distributed Data Parallel Strategies
- reducing frequency and volume, Reducing Communication Frequency and Volume
- streams for asynchronous execution, Asynchronous Execution with Streams

- sharing state between thread blocks, Distributed Shared Memory
- thread block clusters, Thread Block Clusters and Distributed Shared Memory-Thread Block Pairalgorithms for parallelizing workloads, Designing Efficient Algorithms with Thread Block Clusters-Designing Efficient Algorithms with Thread Block Clusterscoordinating thread block clusters, Coordinating Thread Block Clusters with Cooperative Groups API-Coordinating Thread Block Clusters with Cooperative Groups APIdescribed, Intra-Kernel Pipelining, Warp Specialization, and Cooperative Thread Block Clustersglobal memory traffic reduced, Reducing Global Memory Traffic with Thread Block Clusters-Reducing Global Memory Traffic with Thread Block Clusterslaunching a thread block cluster, Launching a Thread Block Cluster, Designing Efficient Algorithms with Thread Block Clustersscratch memory, Scratch Memorythread block pair, Thread Block Pair-Thread Block Pairthread block swizzling, Thread Block Swizzlingthreads accessing each other’s shared memory, Threads, Warps, Blocks, and Grids, Intra-Kernel Pipelining, Warp Specialization, and Cooperative Thread Block Clusters, Thread Block Clusters and Distributed Shared Memorywarp specialization with thread block clusters, Warp Specialization with Thread Block Clusters-Warp Specialization with Thread Block Clusters

- algorithms for parallelizing workloads, Designing Efficient Algorithms with Thread Block Clusters-Designing Efficient Algorithms with Thread Block Clusters
- coordinating thread block clusters, Coordinating Thread Block Clusters with Cooperative Groups API-Coordinating Thread Block Clusters with Cooperative Groups API
- described, Intra-Kernel Pipelining, Warp Specialization, and Cooperative Thread Block Clusters
- global memory traffic reduced, Reducing Global Memory Traffic with Thread Block Clusters-Reducing Global Memory Traffic with Thread Block Clusters
- launching a thread block cluster, Launching a Thread Block Cluster, Designing Efficient Algorithms with Thread Block Clusters
- scratch memory, Scratch Memory
- thread block pair, Thread Block Pair-Thread Block Pair
- thread block swizzling, Thread Block Swizzling
- threads accessing each other’s shared memory, Threads, Warps, Blocks, and Grids, Intra-Kernel Pipelining, Warp Specialization, and Cooperative Thread Block Clusters, Thread Block Clusters and Distributed Shared Memory
- warp specialization with thread block clusters, Warp Specialization with Thread Block Clusters-Warp Specialization with Thread Block Clusters

- no swap and enforcing NUMA affinity, Virtual Memory and Swapping
- NVIDIA Container Toolkit, Container Runtime Optimizations for GPUs
- RDMA, High-Speed, Low-Overhead Data Transfers with RDMA
- runtime optimizationsabout containers, Container Runtime Optimizations for GPUscontainer overlay filesystem overhead, Avoiding Container Overlay Filesystem OverheadNVIDIA container runtime injecting libraries, NVIDIA Container RuntimeNVIDIA Container Toolkit and CUDA compatibility, NVIDIA Container Toolkit and CUDA Compatibilityreducing image size, Reduce Image Size for Faster Container Startup
- runtime optimizations for GPUs, Container Runtime Optimizations for GPUs-Reduce Image Size for Faster Container Startup

- about containers, Container Runtime Optimizations for GPUs
- container overlay filesystem overhead, Avoiding Container Overlay Filesystem Overhead
- NVIDIA container runtime injecting libraries, NVIDIA Container Runtime
- NVIDIA Container Toolkit and CUDA compatibility, NVIDIA Container Toolkit and CUDA Compatibility
- reducing image size, Reduce Image Size for Faster Container Startup

- (see also CPU DRAM)

- appearing as one GPU, NVIDIA Blackwell “Dual-Die” GPU
- GPC “partitions”, Thread Block Clusters and Distributed Shared Memory
- programming, Understanding GPU Memory Hierarchy
- remote memory access latency, NUMA Awareness and CPU Pinning

- mixture-of-experts model DeepSeek-V3, DeepSeek Scales to ~680-Billion Parameter Models Despite US Export Hardware Restrictions in China

- adaptive batching delay, Dynamic Batching
- static batching versus, Dynamic Batching

- benchmarks, Dynamic Parallelismprofiling before and after a change, Dynamic Parallelism
- child-kernel launch limit, Dynamic Parallelism
- device memory usage monitored, Dynamic Parallelism
- “stack overflow” errors avoided, Dynamic Parallelism
- when to capture CUDA Graph instead, Dynamic Parallelism

- profiling before and after a change, Dynamic Parallelism

- latency-aware scheduling, Latency-Aware Scheduling and Dynamic Routing-Latency-Aware Scheduling and Dynamic Routing
- MoE inference strategies, Dynamic Routing Strategies for MoE Inference-Adaptive Expert Routing and Real-Time Monitoringexpert communication optimization, Expert Communication Optimization-Expert Communication Optimizationexpert replication, Load Balancing, Capacity Factor, and Expert Replicationload balancing and capacity factors, Load Balancing, Capacity Factor, and Expert Replication

- expert communication optimization, Expert Communication Optimization-Expert Communication Optimization
- expert replication, Load Balancing, Capacity Factor, and Expert Replication
- load balancing and capacity factors, Load Balancing, Capacity Factor, and Expert Replication

- atomic work queues, Dynamic Scheduling with Atomic Work Queues-Atomic Queuesatomic counters, Atomic Countersatomic queues, Atomic Queues-Atomic Queuesdynamic work allocation, Atomic CountersNsight Compute, Atomic Counters-Atomic Counters
- dynamic congestion-aware scheduling, Dynamic Congestion-Aware Scheduling-Dynamic Congestion-Aware Scheduling
- load balancing, Dynamic Scheduling and Load Balancing-Dynamic resource scalingadaptive resource scheduling and hotspot prevention, Adaptive Resource Scheduling and Hotspot Prevention-Dynamic resource scalingArrow’s adaptive instance scaling, Arrow’s adaptive instance scaling-Arrow’s adaptive instance scalingdynamic resource scaling, Dynamic resource scalingMooncake adaptive strategies, Mooncake adaptive strategiesTetriInfer’s two-level scheduler, TetriInfer’s two-level scheduler

- atomic counters, Atomic Counters
- atomic queues, Atomic Queues-Atomic Queues
- dynamic work allocation, Atomic Counters
- Nsight Compute, Atomic Counters-Atomic Counters

- adaptive resource scheduling and hotspot prevention, Adaptive Resource Scheduling and Hotspot Prevention-Dynamic resource scaling
- Arrow’s adaptive instance scaling, Arrow’s adaptive instance scaling-Arrow’s adaptive instance scaling
- dynamic resource scaling, Dynamic resource scaling
- Mooncake adaptive strategies, Mooncake adaptive strategies
- TetriInfer’s two-level scheduler, TetriInfer’s two-level scheduler

- TorchDynamo, Profiling and Debugging Compiler Performance Issues, TorchDynamo for Bytecode Capture and Graph Extraction
- variable-length sequences via, Dynamic Shapes and Variable Sequence Lengths-Dynamic Shapes and Variable Sequence Lengthsprofiling dynamic shapes versus padding, Dynamic Shapes and Variable Sequence Lengths
- XLA, PyTorch XLA Backend

- profiling dynamic shapes versus padding, Dynamic Shapes and Variable Sequence Lengths

### E

- early rejection, Early Rejection (Admission Control)-Early Rejection (Admission Control)QoS, QoS and early rejection policies-QoS and early rejection policies
- ECC mode for GPU memory, GPU Clock Speeds and ECC
- EGM (Extended GPU Memory), The CPU and GPU Superchip
- environment variablesbeing explicit instead of default values, Pitfall #6: NCCL communicator hangs, errors, or shuts down completelyCUDA_DEVICE_ORDER, Pitfall #4: Verify CPU-GPU NUMA-node affinity for NCCL threadsCUDA_VISIBLE_DEVICES, Pitfall #4: Verify CPU-GPU NUMA-node affinity for NCCL threadsdebugging compiler phases, Debugging Compiler Phases, Graph Breaks, and PerformanceMALLOC_CONF, Tune Host CPU Memory AllocatorNCCL features overturned or disabled, Pitfall #3: Avoid overtuning or disabling NCCL features with environment variables-NCCL_SHARP_DISABLENCCL information online, NCCL Communicator Lifecycle and Environment GotchasNCCL_ALGO, Topology Awareness in NCCL, NCCL Communication Algorithms, Ring versus tree all-reduceNCCL_ASYNC_ERROR_HANDLING, Pitfall #6: NCCL communicator hangs, errors, or shuts down completely, Profiling and Debugging NCCLNCCL_BUFFSIZE, Pitfall #3: Avoid overtuning or disabling NCCL features with environment variablesNCCL_CROSS_NIC, Multinode and Multirack Communication with GPUDirect RDMANCCL_DEBUG, Topology Awareness in NCCL, Pitfall #3: Avoid overtuning or disabling NCCL features with environment variables, Profiling and Debugging NCCL, Debugging Correctness IssuesNCCL_IGNORE_CPU_AFFINITY, Pitfall #4: Verify CPU-GPU NUMA-node affinity for NCCL threadsNCCL_MAX_NCHANNELS, NCCL_MIN_NCHANNELS and NCCL_MAX_NCHANNELSNCCL_MIN_NCHANNELS, NCCL_MIN_NCHANNELS and NCCL_MAX_NCHANNELSNCCL_NSOCKS_PERTHREAD, Tuning Multinode Connectivity, Pitfall #4: Insufficient network bandwidth or misconfigured NICs, NCCL_NSOCKS_PERTHREAD and NCCL_SOCKET_NTHREADSNCCL_P2P_DISABLE, Pitfall #3: Avoid overtuning or disabling NCCL features with environment variablesNCCL_PORT_RANGE, Optimizing Network Communication for KubernetesNCCL_PROFILER_PLUGIN, Profiling and Debugging NCCLNCCL_SHARP_DISABLE, NCCL_SHARP_DISABLENCCL_SHM_DISABLE, Pitfall #3: Avoid overtuning or disabling NCCL features with environment variablesNCCL_SOCKET_IFNAME, Optimizing Network Communication for Kubernetes, Pitfall #5: Resist the temptation to ignore NCCL warnings and errorsNCCL_SOCKET_NTHREADS, Tuning Multinode Connectivity, Pitfall #4: Insufficient network bandwidth or misconfigured NICs, NCCL_NSOCKS_PERTHREAD and NCCL_SOCKET_NTHREADSNCCL_TOPO_DUMP_FILE, Topology Awareness in NCCLNCCL_TOPO_FILE, NCCL_TOPO_FILEPYTORCH_CUDA_ALLOC_CONF, GPU Memory Oversubscription, Fragmentation, and Out-of-Memory Handling, Stream-Ordered Memory AllocatorTCMALLOC_MAX_TOTAL_THREAD_CACHE_BYTES, Tune Host CPU Memory AllocatorTCMALLOC_RELEASE_RATE, Tune Host CPU Memory AllocatorTORCHDYNAMO_REPRO_*, Debugging Compiler Phases, Graph Breaks, and PerformanceTORCHINDUCTOR_BENCHMARK_KERNEL, Profiling and Debugging Compiler Performance IssuesTORCHINDUCTOR_UNIQUE_KERNEL_NAMES, Profiling and Debugging Compiler Performance IssuesTORCH_COMPILE_DEBUG, Debugging Compiler Phases, Graph Breaks, and PerformanceTORCH_LOGS, Profiling and Debugging Compiler Performance Issuescompiler debugging, Debugging Compiler Phases, Graph Breaks, and Performancelogging options summary, Debugging Compiler Phases, Graph Breaks, and Performanceperformance hints, Performance Hints and Debugging Generated Codeupdating debugging values to production, Pitfall #3: Avoid overtuning or disabling NCCL features with environment variables
- EP (see expert parallelism (EP))
- EPLB (expert parallelism load balancer; DeepSeek), Transparency and Reproducibility
- error handlingCUDA kernels, CUDA Programming Refresherinference system, Error HandlingNCCL, Profiling and Debugging NCCLNCCL_ASYNC_ERROR_HANDLING, Pitfall #6: NCCL communicator hangs, errors, or shuts down completely
- ExecuTorch, NVTX Markers and Profiling Tools, NVTX Markers and Profiling Tools
- expert collocation, Expert Communication Optimization
- expert parallelism (EP), Parallelism Strategies for Serving Massive MoE Models, Expert Parallelism-Expert Parallelismhybrid parallelism, Hybrid Parallelismload balancing, Expert Parallelismmodel weights and data split over GPUs, Parallelism Strategies for Serving Massive MoE Modelsversus TP, PP, and hybrid, Adaptive Parallelism Strategies (TP Versus PP Versus Hybrid)-Adaptive Parallelism Strategies (TP Versus PP Versus Hybrid)
- expert parallelism load balancer (EPLB; DeepSeek), Transparency and Reproducibility
- expert replication, Load Balancing, Capacity Factor, and Expert Replication
- exponential computation unit, Mechanical Sympathy: Hardware-Software Codesign
- Extended GPU Memory (EGM), The CPU and GPU Superchip

- QoS, QoS and early rejection policies-QoS and early rejection policies

- being explicit instead of default values, Pitfall #6: NCCL communicator hangs, errors, or shuts down completely
- CUDA_DEVICE_ORDER, Pitfall #4: Verify CPU-GPU NUMA-node affinity for NCCL threads
- CUDA_VISIBLE_DEVICES, Pitfall #4: Verify CPU-GPU NUMA-node affinity for NCCL threads
- debugging compiler phases, Debugging Compiler Phases, Graph Breaks, and Performance
- MALLOC_CONF, Tune Host CPU Memory Allocator
- NCCL features overturned or disabled, Pitfall #3: Avoid overtuning or disabling NCCL features with environment variables-NCCL_SHARP_DISABLE
- NCCL information online, NCCL Communicator Lifecycle and Environment Gotchas
- NCCL_ALGO, Topology Awareness in NCCL, NCCL Communication Algorithms, Ring versus tree all-reduce
- NCCL_ASYNC_ERROR_HANDLING, Pitfall #6: NCCL communicator hangs, errors, or shuts down completely, Profiling and Debugging NCCL
- NCCL_BUFFSIZE, Pitfall #3: Avoid overtuning or disabling NCCL features with environment variables
- NCCL_CROSS_NIC, Multinode and Multirack Communication with GPUDirect RDMA
- NCCL_DEBUG, Topology Awareness in NCCL, Pitfall #3: Avoid overtuning or disabling NCCL features with environment variables, Profiling and Debugging NCCL, Debugging Correctness Issues
- NCCL_IGNORE_CPU_AFFINITY, Pitfall #4: Verify CPU-GPU NUMA-node affinity for NCCL threads
- NCCL_MAX_NCHANNELS, NCCL_MIN_NCHANNELS and NCCL_MAX_NCHANNELS
- NCCL_MIN_NCHANNELS, NCCL_MIN_NCHANNELS and NCCL_MAX_NCHANNELS
- NCCL_NSOCKS_PERTHREAD, Tuning Multinode Connectivity, Pitfall #4: Insufficient network bandwidth or misconfigured NICs, NCCL_NSOCKS_PERTHREAD and NCCL_SOCKET_NTHREADS
- NCCL_P2P_DISABLE, Pitfall #3: Avoid overtuning or disabling NCCL features with environment variables
- NCCL_PORT_RANGE, Optimizing Network Communication for Kubernetes
- NCCL_PROFILER_PLUGIN, Profiling and Debugging NCCL
- NCCL_SHARP_DISABLE, NCCL_SHARP_DISABLE
- NCCL_SHM_DISABLE, Pitfall #3: Avoid overtuning or disabling NCCL features with environment variables
- NCCL_SOCKET_IFNAME, Optimizing Network Communication for Kubernetes, Pitfall #5: Resist the temptation to ignore NCCL warnings and errors
- NCCL_SOCKET_NTHREADS, Tuning Multinode Connectivity, Pitfall #4: Insufficient network bandwidth or misconfigured NICs, NCCL_NSOCKS_PERTHREAD and NCCL_SOCKET_NTHREADS
- NCCL_TOPO_DUMP_FILE, Topology Awareness in NCCL
- NCCL_TOPO_FILE, NCCL_TOPO_FILE
- PYTORCH_CUDA_ALLOC_CONF, GPU Memory Oversubscription, Fragmentation, and Out-of-Memory Handling, Stream-Ordered Memory Allocator
- TCMALLOC_MAX_TOTAL_THREAD_CACHE_BYTES, Tune Host CPU Memory Allocator
- TCMALLOC_RELEASE_RATE, Tune Host CPU Memory Allocator
- TORCHDYNAMO_REPRO_*, Debugging Compiler Phases, Graph Breaks, and Performance
- TORCHINDUCTOR_BENCHMARK_KERNEL, Profiling and Debugging Compiler Performance Issues
- TORCHINDUCTOR_UNIQUE_KERNEL_NAMES, Profiling and Debugging Compiler Performance Issues
- TORCH_COMPILE_DEBUG, Debugging Compiler Phases, Graph Breaks, and Performance
- TORCH_LOGS, Profiling and Debugging Compiler Performance Issuescompiler debugging, Debugging Compiler Phases, Graph Breaks, and Performancelogging options summary, Debugging Compiler Phases, Graph Breaks, and Performanceperformance hints, Performance Hints and Debugging Generated Code
- updating debugging values to production, Pitfall #3: Avoid overtuning or disabling NCCL features with environment variables

- compiler debugging, Debugging Compiler Phases, Graph Breaks, and Performance
- logging options summary, Debugging Compiler Phases, Graph Breaks, and Performance
- performance hints, Performance Hints and Debugging Generated Code

- CUDA kernels, CUDA Programming Refresher
- inference system, Error Handling
- NCCL, Profiling and Debugging NCCL
- NCCL_ASYNC_ERROR_HANDLING, Pitfall #6: NCCL communicator hangs, errors, or shuts down completely

- hybrid parallelism, Hybrid Parallelism
- load balancing, Expert Parallelism
- model weights and data split over GPUs, Parallelism Strategies for Serving Massive MoE Models
- versus TP, PP, and hybrid, Adaptive Parallelism Strategies (TP Versus PP Versus Hybrid)-Adaptive Parallelism Strategies (TP Versus PP Versus Hybrid)

### F

- failurefilesystem caching and write-back, Filesystem Caching and Write-BackGPU failures at scale per Meta, Pitfall #6: NCCL communicator hangs, errors, or shuts down completelyNCCL communicator hanging, giving errors, shutting down, Pitfall #6: NCCL communicator hangs, errors, or shuts down completely-Pitfall #6: NCCL communicator hangs, errors, or shuts down completely
- fatbinary model, CUDA Forward and Backward Compatibility Across GPU Hardware Generations
- fault tolerance, Fault Tolerance
- Feedforward Neural Network (FFN) activation checkpointing, Activation Checkpointing for Memory Savings
- Feynman GPU, Feynman GPU (2028) and Doubling Something Every Year
- filesystemscaching and write-back, Filesystem Caching and Write-BackDeepSeek Fire-Flyer File System, DeepSeek’s Fire-Flyer File System-DeepSeek’s Fire-Flyer File SystemFUSE unable to GDS, DeepSeek’s Fire-Flyer File Systemmust be optimized for large concurrent I/O, Sequential Versus Random Read PatternsNFS, Distributed, Parallel Filesystems and Object Storestuning parameters, Distributed, Parallel Filesystems and Object Storesparallel filesystems, Distributed, Parallel Filesystems and Object Storesshared filesystems, Distributed, Parallel Filesystems and Object Storestuning, Tuning, Replicating, and Compressing Data
- Fire-Flyer File System (3FS; DeepSeek), Transparency and Reproducibility, DeepSeek’s Fire-Flyer File System-DeepSeek’s Fire-Flyer File System
- FireworksAI, Replaying the Graph
- first in, first out (FIFO) scheduler, Latency-Aware Scheduling and Dynamic Routing-Latency-Aware Scheduling and Dynamic Routing
- FlashAttention, Mechanical Sympathy: Hardware-Software Codesign
- FlashMLA (Flash Multi-Head Latent Attention; DeepSeek), Transparency and Reproducibility, FlashMLA (DeepSeek)hardware-software codesign, Mechanical Sympathy: Hardware-Software Codesign
- FlexAttention (PyTorch), PyTorch Optimized Attention Mechanisms, Autotuning with TorchInductor, FlexDecoding (PyTorch)nested jagged-layout tensors, FlexDecoding (PyTorch)PagedAttention, FlexDecoding (PyTorch)
- FlexDecoding (PyTorch), PyTorch Optimized Attention Mechanisms, FlexDecoding (PyTorch)-FlexDecoding (PyTorch)nested jagged-layout tensors, FlexDecoding (PyTorch)
- float8 not provided by CUDA, Vectorized Memory Access
- FP4, FP8, FP16, NVIDIA GPU Tensor Cores and Transformer Engine, BF16/FP16, FP8, and FP4 Reduced Precisiondynamic precision changes, Dynamic Precision Changes-Dynamic Precision Changesinference quantization, Reducing Precision from FP16 to FP8 and FP4weight-only quantization, Weight-Only Quantization (GPTQ, AWQ)
- FSDP (Fully Sharded Data Parallel; PyTorch), Distributed Data Parallel Strategies, FSDP Automatic Checkpointing and Offloading-FSDP Automatic Checkpointing and Offloadingautomated checkpointing, Activation Checkpointing for Memory Savings, FSDP Automatic Checkpointing and Offloading-FSDP Automatic Checkpointing and Offloadingscaling via FSDP with torch.compile, FSDP with torch.compile-FSDP with torch.compileSimpleFSDP, TorchTitan, AsyncTP, AutoParallel, and SimpleFSDPTorchTitan, TorchTitan, AsyncTP, AutoParallel, and SimpleFSDP
- FUSE unable to GDS, DeepSeek’s Fire-Flyer File System
- fused kernels (see kernel fusion)
- future hardware from NVIDIA, A Glimpse into the Future: NVIDIA’s Roadmap-Feynman GPU (2028) and Doubling Something Every YearBlackwell Ultra and Grace Blackwell Ultra, Blackwell Ultra and Grace Blackwell Ultradoubling every year, Feynman GPU (2028) and Doubling Something Every YearFeynman GPU, Feynman GPU (2028) and Doubling Something Every YearGPU programming across generations, CUDA Forward and Backward Compatibility Across GPU Hardware Generations, CUDA GPU Backward and Forward Compatibility ModelRubin Ultra and Vera Rubin Ultra, Rubin Ultra and Vera Rubin Ultra (2027)Vera Rubin Superchip, Vera Rubin Superchip (2026)
- FX GraphAOT Autograd, TorchDynamo for Bytecode Capture and Graph Extraction, AOT Autograd Fusion for Forward and Backward PassesAutoParallel, TorchTitan, AsyncTP, AutoParallel, and SimpleFSDPPrimTorch IR, PrimTorch IR (Prims) Simplified Operator SetTorchDynamo, TorchDynamo for Bytecode Capture and Graph Extractiondumping graph after each stage, Debugging Compiler Phases, Graph Breaks, and PerformanceTorchInductor, AOT Autograd Fusion for Forward and Backward Passes

- filesystem caching and write-back, Filesystem Caching and Write-Back
- GPU failures at scale per Meta, Pitfall #6: NCCL communicator hangs, errors, or shuts down completely
- NCCL communicator hanging, giving errors, shutting down, Pitfall #6: NCCL communicator hangs, errors, or shuts down completely-Pitfall #6: NCCL communicator hangs, errors, or shuts down completely

- caching and write-back, Filesystem Caching and Write-Back
- DeepSeek Fire-Flyer File System, DeepSeek’s Fire-Flyer File System-DeepSeek’s Fire-Flyer File System
- FUSE unable to GDS, DeepSeek’s Fire-Flyer File System
- must be optimized for large concurrent I/O, Sequential Versus Random Read Patterns
- NFS, Distributed, Parallel Filesystems and Object Storestuning parameters, Distributed, Parallel Filesystems and Object Stores
- parallel filesystems, Distributed, Parallel Filesystems and Object Stores
- shared filesystems, Distributed, Parallel Filesystems and Object Stores
- tuning, Tuning, Replicating, and Compressing Data

- tuning parameters, Distributed, Parallel Filesystems and Object Stores

- hardware-software codesign, Mechanical Sympathy: Hardware-Software Codesign

- nested jagged-layout tensors, FlexDecoding (PyTorch)
- PagedAttention, FlexDecoding (PyTorch)

- nested jagged-layout tensors, FlexDecoding (PyTorch)

- dynamic precision changes, Dynamic Precision Changes-Dynamic Precision Changes
- inference quantization, Reducing Precision from FP16 to FP8 and FP4weight-only quantization, Weight-Only Quantization (GPTQ, AWQ)

- weight-only quantization, Weight-Only Quantization (GPTQ, AWQ)

- automated checkpointing, Activation Checkpointing for Memory Savings, FSDP Automatic Checkpointing and Offloading-FSDP Automatic Checkpointing and Offloading
- scaling via FSDP with torch.compile, FSDP with torch.compile-FSDP with torch.compile
- SimpleFSDP, TorchTitan, AsyncTP, AutoParallel, and SimpleFSDP
- TorchTitan, TorchTitan, AsyncTP, AutoParallel, and SimpleFSDP

- Blackwell Ultra and Grace Blackwell Ultra, Blackwell Ultra and Grace Blackwell Ultra
- doubling every year, Feynman GPU (2028) and Doubling Something Every Year
- Feynman GPU, Feynman GPU (2028) and Doubling Something Every Year
- GPU programming across generations, CUDA Forward and Backward Compatibility Across GPU Hardware Generations, CUDA GPU Backward and Forward Compatibility Model
- Rubin Ultra and Vera Rubin Ultra, Rubin Ultra and Vera Rubin Ultra (2027)
- Vera Rubin Superchip, Vera Rubin Superchip (2026)

- AOT Autograd, TorchDynamo for Bytecode Capture and Graph Extraction, AOT Autograd Fusion for Forward and Backward Passes
- AutoParallel, TorchTitan, AsyncTP, AutoParallel, and SimpleFSDP
- PrimTorch IR, PrimTorch IR (Prims) Simplified Operator Set
- TorchDynamo, TorchDynamo for Bytecode Capture and Graph Extractiondumping graph after each stage, Debugging Compiler Phases, Graph Breaks, and Performance
- TorchInductor, AOT Autograd Fusion for Forward and Backward Passes

- dumping graph after each stage, Debugging Compiler Phases, Graph Breaks, and Performance

### G

- GB200 (see NVIDIA GB200 NVL72)
- GB300 (see NVIDIA GB300 NVL72)
- GDS (GPUDirect Storage; NVIDIA), NUMA-Friendly Memory Allocation and Memory Pinning, Using NVIDIA GDS-Measuring GDS with gdsio, Offloading Parameters to CPU and NVMe
- Gemini Ultra (Google) training costs, Introduction and AI System Overview
- GEMM (general matrix multiply)AI-discovered algorithms, AlphaTensor AI-Discovered Algorithms Boosting GPU Performance (Google DeepMind)-AlphaTensor AI-Discovered Algorithms Boosting GPU Performance (Google DeepMind)GEMM kernels, Structured SparsityCUTLASS, Using CUTLASS for Optimal Arithmetic Intensity and Tensor Core PerformanceGEMM pipeline, Combining PDL and Thread Block Clusters with Warp Specialization-Combining PDL and Thread Block Clusters with Warp SpecializationPyTorch compiler TorchInductor, Using the PyTorch Compilerroofline analysis, Kernel Roofline Analysis for General Matrix Multiply (GEMM)-Kernel Roofline Analysis for General Matrix Multiply (GEMM)thread block clusters, Designing Efficient Algorithms with Thread Block Clusters
- GitHubDeepSeek repositories, Transparency and Reproducibilityreporting an issue, Debugging Numerical Correctness and AccuracyTorchBench-based performance regression, Continuous Integration and Performance Benchmarking
- GLaM (Google), Expert Parallelism
- global memorycaching loads, Read-Only Data Cachestraffic reduced, Reducing Global Memory Traffic with Thread Block Clusters-Reducing Global Memory Traffic with Thread Block Clusters
- global memoryarithmetic intensity, Roofline Model: Compute-Bound or Memory-Bound Workloadscoalesced versus uncoalesced access, Coalesced Versus Uncoalesced Global Memory Access-Coalesced Versus Uncoalesced Global Memory AccessGPU memory hierarchy, Understanding GPU Memory Hierarchy-Understanding GPU Memory HierarchyInitcheck of NVIDIA Compute Sanitizer, Debugging Functional Correctness with NVIDIA Compute Sanitizermatching loads to native transaction size, Vectorized Memory Accesspitfall of repeatedly reading same data, Tiling and Data Reuse Using Shared Memory
- __global__, CUDA Programming Refresher
- Gloo backend should be NCCL, Pitfall #1: Using a CPU-bound Gloo backend instead of NCCL-Pitfall #1: Using a CPU-bound Gloo backend instead of NCCL
- goodput as throughput metric, Measuring “Goodput” Useful Throughput
- Google Cloud TPU and XLA, TorchInductor Backend Code Generation, PyTorch XLA Backend
- Google Gemini Ultra training costs, Introduction and AI System Overview
- Google GLaM, Expert Parallelism
- Google Switch Transformer MoE efficiency, Toward 100-Trillion-Parameter Models
- GPCs (GPU processing clusters), Thread Block Clusters and Distributed Shared Memory
- GPT-4 (OpenAI) training costs, Introduction and AI System Overview
- GPU Boost (NVIDIA), GPU Clock Speeds and ECCsetting power limit below TDP, GPU Clock Speeds and ECCunderclocking to reduce heat, GPU Clock Speeds and ECC
- GPU driverkeeping GPU driver loaded, GPU Memory Oversubscription, Fragmentation, and Out-of-Memory Handlingnvidia-smi, GPU Driverpersistence mode for performance, GPU Persistence Modesoftware stack, GPU Driver
- GPU processing clusters (GPCs), Thread Block Clusters and Distributed Shared Memory
- GPUDirect RDMA, Multi-GPU Programming, High-Speed, Low-Overhead Data Transfers with RDMAmultinode, multirack communication, Multinode and Multirack Communication with GPUDirect RDMA-Multinode and Multirack Communication with GPUDirect RDMANCCL’s native allocator in PyTorch MemPool, Pluggable Memory Allocators and Cross-GPU Data Transfersverify true GPUDirectRDMA, High-Speed, Low-Overhead Data Transfers with RDMA
- GPUDirect Storage (see GDS (GPUDirect Storage; NVIDIA))
- GPUsarchitecture, Understanding GPU Architecture-CUDA GPU Backward and Forward Compatibility Modelmaximum threads per SM and block, Tuning Occupancy with Launch Boundsmemory hierarchy, Understanding GPU Memory Hierarchy-Understanding GPU Memory HierarchySpecial Function Unit, Understanding GPU Architecturestreaming multiprocessors, Understanding GPU Architecturethread block clusters, Threads, Warps, Blocks, and Gridsthread block size, Choosing Threads-per-Block and Blocks-per-Grid Sizesthread blocks, Threads, Warps, Blocks, and Grids-Threads, Warps, Blocks, and Gridsthreads, Threads, Warps, Blocks, and Grids-Threads, Warps, Blocks, and Gridsthreads-per-block and blocks-per-grid, Choosing Threads-per-Block and Blocks-per-Grid Sizes-Choosing Threads-per-Block and Blocks-per-Grid SizesUnified CPU-GPU Memory, The CPU and GPU Superchipwarps, Streaming Multiprocessor, Threads, and Warps, Understanding GPU Architecture-Understanding GPU Architecturebottlenecks profiled and diagnosed, Profiling and Diagnosing GPU Bottlenecks-Profiler-Guided Analysisabout, Profiling and Diagnosing GPU Bottlenecksachieved occupancy and GPU utilization, Inspecting Achieved Occupancy and GPU Utilization-Optimizing the Kerneldata pipeline profiling and tuning, Profiling and Tuning the Data Pipelineiteratively profiling, Iteratively Profiling and Determining the Kernel Bottleneck-Iteratively Profiling and Determining the Kernel Bottleneckkernel optimization, Optimizing the Kernel-Optimizing the Kernelmonitoring in production, Monitoring System Metrics and CountersNsight Compute and Roofline analysis, Nsight Compute and Roofline AnalysisNsight Systems timeline view, Profiling and Diagnosing GPU Bottlenecks, Nsight Compute and Roofline AnalysisPyTorch profiler via Kineto, PyTorch Profiler and Visualization Tools-Profiler-Guided Analysiswarp stall reasons, Analyzing Warp Stall Reasons with Nsight Compute-Other Stall Reasonscompressed data, Tuning, Replicating, and Compressing Dataconcurrent kernels across all SMs maximum, Stream-Ordered Memory Allocatorcontainer runtime optimizations, Container Runtime Optimizations for GPUs-Reduce Image Size for Faster Container Startupabout containers, Container Runtime Optimizations for GPUscontainer overlay filesystem overhead, Avoiding Container Overlay Filesystem OverheadNVIDIA container runtime injecting libraries, NVIDIA Container RuntimeNVIDIA Container Toolkit, Container Runtime Optimizations for GPUsNVIDIA Container Toolkit and CUDA compatibility, NVIDIA Container Toolkit and CUDA Compatibilityreducing image size, Reduce Image Size for Faster Container StartupCPU-GPU memory data-transfer bandwidth, NUMA-Friendly Memory Allocation and Memory Pinning, GPU Memory Oversubscription, Fragmentation, and Out-of-Memory HandlingCUDA events for communication with CPU host, Fine-Grained Synchronization with Events and Callbackscallback function, Fine-Grained Synchronization with Events and Callbackscustom kernels with OpenAI Triton, Writing Custom Kernels with OpenAI Triton-Profiling with Triton Proton Profilerregistering with PyTorch, Registering Custom Kernels with PyTorch-Registering Custom Kernels with PyTorchtuning kernel launch parameters, Tuning Kernel-Launch Parametersdistributed multi-GPU communication, NCCL for Distributed Multi-GPU Communication-In-Network SHARP AggregationFLOPs outpacing memory bandwidth, Maintaining High Occupancy and GPU UtilizationGPU failures at scale per Meta, Pitfall #6: NCCL communicator hangs, errors, or shuts down completelyGPU-NIC affinity forced, Multinode and Multirack Communication with GPUDirect RDMAGPU-to-GPU memory sharing with NVSHMEM, Fine-Grained GPU-to-GPU Memory Sharing with NVSHMEM-Fine-Grained GPU-to-GPU Memory Sharing with NVSHMEMidle time measurement, Monitoring Storage I/OKubernetes scheduler via NVIDIA device plugin, Kubernetes for Topology-Aware Container Orchestration and Networkingmemory access patternsavoiding shared memory, Warp Shuffle Intrinsics: Avoid Shared Memory and Explicit Synchronization-Warp Shuffle Intrinsics: Avoid Shared Memory and Explicit Synchronizationavoiding shared-memory bank conflicts, Avoid Shared-Memory Bank Conflicts-Avoid Shared-Memory Bank Conflictscoalesced versus uncoalesced global memory access, Coalesced Versus Uncoalesced Global Memory Access-Coalesced Versus Uncoalesced Global Memory Accessread-only data caches, Read-Only Data Caches-Read-Only Data Cachesread-only data caches pitfall, Read-Only Data Cachessymmetric memory, PyTorch Symmetric MemoryTensor Memory Accelerator for tile fetch, Asynchronous Memory Prefetching and Tensor Memory Accelerator-Asynchronous Memory Prefetching and Tensor Memory Acceleratortiling and data reuse via shared memory, Tiling and Data Reuse Using Shared Memory-Tiling and Data Reuse Using Shared Memoryvectorized memory access, Vectorized Memory Access-Vectorized Memory Accesswarp shuffle intrinsics, Warp Shuffle Intrinsics: Avoid Shared Memory and Explicit Synchronizationmemory fragmentation under UCX/RDMA, Pitfall #6: GPU memory fragmentation under UCX/RDMA-Pitfall #6: GPU memory fragmentation under UCX/RDMAmultiple GPU orchestration, Orchestrate Across Multiple GPUs and Cluster Nodes (NVSHMEM)-Pattern for N-GPU Scaling(see also multiple GPUs)GPU-to-GPU memory sharing, Fine-Grained GPU-to-GPU Memory Sharing with NVSHMEM-Fine-Grained GPU-to-GPU Memory Sharing with NVSHMEMNVIDIA SHMEM library, Fine-Grained GPU-to-GPU Memory Sharing with NVSHMEM-Fine-Grained GPU-to-GPU Memory Sharing with NVSHMEMparallelismadding two vectors sequentially and in parallel, Maintaining High Occupancy and GPU Utilization-Maintaining High Occupancy and GPU Utilizationarchitecture of GPU, Understanding GPU Architecturechild-kernel launch limit, Dynamic ParallelismcuTile, C++ and Python CUDA LibrariesData Parallel versus Distributed Data Parallel, Distributed Data Parallel Strategies-Distributed Data Parallel Strategiesdynamic parallelism, Dynamic Parallelism-Dynamic Parallelismfully sharded data parallel, Distributed Data Parallel StrategiesPTX, Inline PTX and SASS Tuning for Microoptimizationsthread block cluster algorithms, Designing Efficient Algorithms with Thread Block Clusters-Designing Efficient Algorithms with Thread Block Clusterspower limits, Performance Monitoring and Utilization in Practice, Power and Thermal Constraintsprocess-GPU mapping, Adaptive Process-GPU Mapping-Adaptive Process-GPU Mappingprogramming CUDA, Maintaining High Occupancy and GPU Utilization-Maintaining High Occupancy and GPU Utilizationmemory hierarchy, Understanding GPU Memory Hierarchy-Understanding GPU Memory Hierarchyreading data directly from storage devices, Using NVIDIA GDS-Measuring GDS with gdsioruntime settings for performance, GPU Driver and Runtime Settings for Performance-GPU Memory Oversubscription, Fragmentation, and Out-of-Memory HandlingECC mode for memory, GPU Clock Speeds and ECCGPU clock speeds, GPU Clock Speeds and ECCGPU memory fragmentation and oversubscription, GPU Memory Oversubscription, Fragmentation, and Out-of-Memory Handling-GPU Memory Oversubscription, Fragmentation, and Out-of-Memory HandlingGPU persistence mode, GPU Persistence Modekeeping GPU driver loaded, GPU Memory Oversubscription, Fragmentation, and Out-of-Memory HandlingMIG, MIG-MIGMPS, MPS-MPSutilization versus latency, Maximizing GPU Utilization and Throughput Versus Latency Trade-Offsvirtualization via Multi-Instance GPU, Managing Resources Efficiently, Slicing a GPU with MIG
- Grace Blackwell focus but not limit, NVIDIA’s “AI Supercomputer in a Rack”CPU and GPU linked by NVLink-C2C, NUMA Awareness and CPU Pinning, NUMA Awareness and CPU Pinning
- Grace Blackwell GB200 NVL72, GPU and CPU-GPU Superchip Improvements, NVLink/NVSwitch Topology and Bandwidth Constraints
- Grace Blackwell GB200 Superchip, The CPU and GPU Superchip
- Grace Blackwell Superchip, NVIDIA’s “AI Supercomputer in a Rack”memory, The CPU and GPU Superchip, Topology Awareness in NCCLNIXL offloading KV cache to CPU memory, KV Cache Offloading with NIXL-KV Cache Offloading with NIXLtricks using CPU-GPU unified memory, Continuous Prewarming of CUDA Graphs and Caches Using Time-Series Prediction
- Grace Blackwell Ultra GB300 NVL72, Blackwell Ultra and Grace Blackwell Ultra, NVLink/NVSwitch Topology and Bandwidth Constraints
- Grace CPU, NVIDIA Grace CPUGrace Blackwell Superchip, NVIDIA’s “AI Supercomputer in a Rack”memory, The CPU and GPU SuperchipNVL72, Ultrascale Networking Treating Many GPUs as One
- Grace Hopper GH200, The CPU and GPU Superchip
- Grace Hopper Superchip NIXL offloading KV cache, KV Cache Offloading with NIXL-KV Cache Offloading with NIXL
- gradient checkpointing, Activation Checkpointing for Memory Savings(see also activation checkpointing)
- Grafana dashboardsinference monitoring, Monitoring System Metrics and Counters, Monitoring System Metrics and CountersNVLink/NVSwitch monitoring, Real-Time Link Telemetry and Monitoring
- graph breaks, Explaining and Minimizing Graph Breaks-Tips for Handling Graph Breaksavoiding whenever possible, TorchDynamo for Bytecode Capture and Graph Extraction, Graph Breaks and TorchDynamo explain()conditionals as tensor operations, Graph Breaks and TorchDynamo explain()data-dependent branches, Graph Breaks and TorchDynamo explain()marking functions and code blocks as safe, Mark Functions and Code Blocks as Safe with allow_in_graphminimizing graph recompilations, Minimize Graph Recompilationscauses, Tips for Handling Graph Breaks-Tips for Handling Graph Breakssequence not captured into a single graph, Graph Breaks and TorchDynamo explain()shape mismatches, Profiling and Debugging Compiler Performance Issuestips for handling common causes, Tips for Handling Graph Breaks-Tips for Handling Graph Breaksunsupported operations in model, Profiling and Debugging Compiler Performance Issuescomplex graphs requiring graph breaks, Graph Breaks and TorchDynamo explain()debugging, Graph Breaks and TorchDynamo explain(), Debugging Compiler Phases, Graph Breaks, and Performancemonitoring graph recompilations, TorchDynamo for Bytecode Capture and Graph Extraction, Minimize Graph Recompilationsforcing an error when full graph not captured, Graph Breaks and TorchDynamo explain()PyTorch compiler trace, Using PyTorch ProfilerTorchDynamo, TorchDynamo for Bytecode Capture and Graph Extractionexplain() output, Profiling and Debugging Compiler Performance Issues, FSDP with torch.compileforcing error to be raised at compile, TorchDynamo for Bytecode Capture and Graph Extraction
- grids, Threads, Warps, Blocks, and Grids-Threads, Warps, Blocks, and Gridsblocks-per-grid sizes, Choosing Threads-per-Block and Blocks-per-Grid Sizes-Choosing Threads-per-Block and Blocks-per-Grid SizesblocksPerGrid parameter, CUDA Programming Refresher, Configuring Launch Parameters: Blocks per Grid and Threads per Block-Configuring Launch Parameters: Blocks per Grid and Threads per Blockcooperative groupscooperative kernel launch grid size, Cooperative Groups, Cooperative Grid Synchronization and Persistent Kernelspersistent kernels via grid sync, Cooperative Grid Synchronization and Persistent Kernels-Cooperative Grid Synchronization and Persistent Kernelsmaximum dimensions, Choosing Threads-per-Block and Blocks-per-Grid Sizes

- AI-discovered algorithms, AlphaTensor AI-Discovered Algorithms Boosting GPU Performance (Google DeepMind)-AlphaTensor AI-Discovered Algorithms Boosting GPU Performance (Google DeepMind)
- GEMM kernels, Structured SparsityCUTLASS, Using CUTLASS for Optimal Arithmetic Intensity and Tensor Core PerformanceGEMM pipeline, Combining PDL and Thread Block Clusters with Warp Specialization-Combining PDL and Thread Block Clusters with Warp SpecializationPyTorch compiler TorchInductor, Using the PyTorch Compilerroofline analysis, Kernel Roofline Analysis for General Matrix Multiply (GEMM)-Kernel Roofline Analysis for General Matrix Multiply (GEMM)thread block clusters, Designing Efficient Algorithms with Thread Block Clusters

- CUTLASS, Using CUTLASS for Optimal Arithmetic Intensity and Tensor Core Performance
- GEMM pipeline, Combining PDL and Thread Block Clusters with Warp Specialization-Combining PDL and Thread Block Clusters with Warp Specialization
- PyTorch compiler TorchInductor, Using the PyTorch Compiler
- roofline analysis, Kernel Roofline Analysis for General Matrix Multiply (GEMM)-Kernel Roofline Analysis for General Matrix Multiply (GEMM)
- thread block clusters, Designing Efficient Algorithms with Thread Block Clusters

- DeepSeek repositories, Transparency and Reproducibility
- reporting an issue, Debugging Numerical Correctness and Accuracy
- TorchBench-based performance regression, Continuous Integration and Performance Benchmarking

- caching loads, Read-Only Data Caches
- traffic reduced, Reducing Global Memory Traffic with Thread Block Clusters-Reducing Global Memory Traffic with Thread Block Clusters

- arithmetic intensity, Roofline Model: Compute-Bound or Memory-Bound Workloads
- coalesced versus uncoalesced access, Coalesced Versus Uncoalesced Global Memory Access-Coalesced Versus Uncoalesced Global Memory Access
- GPU memory hierarchy, Understanding GPU Memory Hierarchy-Understanding GPU Memory Hierarchy
- Initcheck of NVIDIA Compute Sanitizer, Debugging Functional Correctness with NVIDIA Compute Sanitizer
- matching loads to native transaction size, Vectorized Memory Access
- pitfall of repeatedly reading same data, Tiling and Data Reuse Using Shared Memory

- setting power limit below TDP, GPU Clock Speeds and ECC
- underclocking to reduce heat, GPU Clock Speeds and ECC

- keeping GPU driver loaded, GPU Memory Oversubscription, Fragmentation, and Out-of-Memory Handling
- nvidia-smi, GPU Driver
- persistence mode for performance, GPU Persistence Mode
- software stack, GPU Driver

- multinode, multirack communication, Multinode and Multirack Communication with GPUDirect RDMA-Multinode and Multirack Communication with GPUDirect RDMA
- NCCL’s native allocator in PyTorch MemPool, Pluggable Memory Allocators and Cross-GPU Data Transfers
- verify true GPUDirectRDMA, High-Speed, Low-Overhead Data Transfers with RDMA

- architecture, Understanding GPU Architecture-CUDA GPU Backward and Forward Compatibility Modelmaximum threads per SM and block, Tuning Occupancy with Launch Boundsmemory hierarchy, Understanding GPU Memory Hierarchy-Understanding GPU Memory HierarchySpecial Function Unit, Understanding GPU Architecturestreaming multiprocessors, Understanding GPU Architecturethread block clusters, Threads, Warps, Blocks, and Gridsthread block size, Choosing Threads-per-Block and Blocks-per-Grid Sizesthread blocks, Threads, Warps, Blocks, and Grids-Threads, Warps, Blocks, and Gridsthreads, Threads, Warps, Blocks, and Grids-Threads, Warps, Blocks, and Gridsthreads-per-block and blocks-per-grid, Choosing Threads-per-Block and Blocks-per-Grid Sizes-Choosing Threads-per-Block and Blocks-per-Grid SizesUnified CPU-GPU Memory, The CPU and GPU Superchipwarps, Streaming Multiprocessor, Threads, and Warps, Understanding GPU Architecture-Understanding GPU Architecture
- bottlenecks profiled and diagnosed, Profiling and Diagnosing GPU Bottlenecks-Profiler-Guided Analysisabout, Profiling and Diagnosing GPU Bottlenecksachieved occupancy and GPU utilization, Inspecting Achieved Occupancy and GPU Utilization-Optimizing the Kerneldata pipeline profiling and tuning, Profiling and Tuning the Data Pipelineiteratively profiling, Iteratively Profiling and Determining the Kernel Bottleneck-Iteratively Profiling and Determining the Kernel Bottleneckkernel optimization, Optimizing the Kernel-Optimizing the Kernelmonitoring in production, Monitoring System Metrics and CountersNsight Compute and Roofline analysis, Nsight Compute and Roofline AnalysisNsight Systems timeline view, Profiling and Diagnosing GPU Bottlenecks, Nsight Compute and Roofline AnalysisPyTorch profiler via Kineto, PyTorch Profiler and Visualization Tools-Profiler-Guided Analysiswarp stall reasons, Analyzing Warp Stall Reasons with Nsight Compute-Other Stall Reasons
- compressed data, Tuning, Replicating, and Compressing Data
- concurrent kernels across all SMs maximum, Stream-Ordered Memory Allocator
- container runtime optimizations, Container Runtime Optimizations for GPUs-Reduce Image Size for Faster Container Startupabout containers, Container Runtime Optimizations for GPUscontainer overlay filesystem overhead, Avoiding Container Overlay Filesystem OverheadNVIDIA container runtime injecting libraries, NVIDIA Container RuntimeNVIDIA Container Toolkit, Container Runtime Optimizations for GPUsNVIDIA Container Toolkit and CUDA compatibility, NVIDIA Container Toolkit and CUDA Compatibilityreducing image size, Reduce Image Size for Faster Container Startup
- CPU-GPU memory data-transfer bandwidth, NUMA-Friendly Memory Allocation and Memory Pinning, GPU Memory Oversubscription, Fragmentation, and Out-of-Memory Handling
- CUDA events for communication with CPU host, Fine-Grained Synchronization with Events and Callbackscallback function, Fine-Grained Synchronization with Events and Callbacks
- custom kernels with OpenAI Triton, Writing Custom Kernels with OpenAI Triton-Profiling with Triton Proton Profilerregistering with PyTorch, Registering Custom Kernels with PyTorch-Registering Custom Kernels with PyTorchtuning kernel launch parameters, Tuning Kernel-Launch Parameters
- distributed multi-GPU communication, NCCL for Distributed Multi-GPU Communication-In-Network SHARP Aggregation
- FLOPs outpacing memory bandwidth, Maintaining High Occupancy and GPU Utilization
- GPU failures at scale per Meta, Pitfall #6: NCCL communicator hangs, errors, or shuts down completely
- GPU-NIC affinity forced, Multinode and Multirack Communication with GPUDirect RDMA
- GPU-to-GPU memory sharing with NVSHMEM, Fine-Grained GPU-to-GPU Memory Sharing with NVSHMEM-Fine-Grained GPU-to-GPU Memory Sharing with NVSHMEM
- idle time measurement, Monitoring Storage I/O
- Kubernetes scheduler via NVIDIA device plugin, Kubernetes for Topology-Aware Container Orchestration and Networking
- memory access patternsavoiding shared memory, Warp Shuffle Intrinsics: Avoid Shared Memory and Explicit Synchronization-Warp Shuffle Intrinsics: Avoid Shared Memory and Explicit Synchronizationavoiding shared-memory bank conflicts, Avoid Shared-Memory Bank Conflicts-Avoid Shared-Memory Bank Conflictscoalesced versus uncoalesced global memory access, Coalesced Versus Uncoalesced Global Memory Access-Coalesced Versus Uncoalesced Global Memory Accessread-only data caches, Read-Only Data Caches-Read-Only Data Cachesread-only data caches pitfall, Read-Only Data Cachessymmetric memory, PyTorch Symmetric MemoryTensor Memory Accelerator for tile fetch, Asynchronous Memory Prefetching and Tensor Memory Accelerator-Asynchronous Memory Prefetching and Tensor Memory Acceleratortiling and data reuse via shared memory, Tiling and Data Reuse Using Shared Memory-Tiling and Data Reuse Using Shared Memoryvectorized memory access, Vectorized Memory Access-Vectorized Memory Accesswarp shuffle intrinsics, Warp Shuffle Intrinsics: Avoid Shared Memory and Explicit Synchronization
- memory fragmentation under UCX/RDMA, Pitfall #6: GPU memory fragmentation under UCX/RDMA-Pitfall #6: GPU memory fragmentation under UCX/RDMA
- multiple GPU orchestration, Orchestrate Across Multiple GPUs and Cluster Nodes (NVSHMEM)-Pattern for N-GPU Scaling(see also multiple GPUs)GPU-to-GPU memory sharing, Fine-Grained GPU-to-GPU Memory Sharing with NVSHMEM-Fine-Grained GPU-to-GPU Memory Sharing with NVSHMEMNVIDIA SHMEM library, Fine-Grained GPU-to-GPU Memory Sharing with NVSHMEM-Fine-Grained GPU-to-GPU Memory Sharing with NVSHMEM
- parallelismadding two vectors sequentially and in parallel, Maintaining High Occupancy and GPU Utilization-Maintaining High Occupancy and GPU Utilizationarchitecture of GPU, Understanding GPU Architecturechild-kernel launch limit, Dynamic ParallelismcuTile, C++ and Python CUDA LibrariesData Parallel versus Distributed Data Parallel, Distributed Data Parallel Strategies-Distributed Data Parallel Strategiesdynamic parallelism, Dynamic Parallelism-Dynamic Parallelismfully sharded data parallel, Distributed Data Parallel StrategiesPTX, Inline PTX and SASS Tuning for Microoptimizationsthread block cluster algorithms, Designing Efficient Algorithms with Thread Block Clusters-Designing Efficient Algorithms with Thread Block Clusters
- power limits, Performance Monitoring and Utilization in Practice, Power and Thermal Constraints
- process-GPU mapping, Adaptive Process-GPU Mapping-Adaptive Process-GPU Mapping
- programming CUDA, Maintaining High Occupancy and GPU Utilization-Maintaining High Occupancy and GPU Utilizationmemory hierarchy, Understanding GPU Memory Hierarchy-Understanding GPU Memory Hierarchy
- reading data directly from storage devices, Using NVIDIA GDS-Measuring GDS with gdsio
- runtime settings for performance, GPU Driver and Runtime Settings for Performance-GPU Memory Oversubscription, Fragmentation, and Out-of-Memory HandlingECC mode for memory, GPU Clock Speeds and ECCGPU clock speeds, GPU Clock Speeds and ECCGPU memory fragmentation and oversubscription, GPU Memory Oversubscription, Fragmentation, and Out-of-Memory Handling-GPU Memory Oversubscription, Fragmentation, and Out-of-Memory HandlingGPU persistence mode, GPU Persistence Modekeeping GPU driver loaded, GPU Memory Oversubscription, Fragmentation, and Out-of-Memory HandlingMIG, MIG-MIGMPS, MPS-MPS
- utilization versus latency, Maximizing GPU Utilization and Throughput Versus Latency Trade-Offs
- virtualization via Multi-Instance GPU, Managing Resources Efficiently, Slicing a GPU with MIG

- maximum threads per SM and block, Tuning Occupancy with Launch Bounds
- memory hierarchy, Understanding GPU Memory Hierarchy-Understanding GPU Memory Hierarchy
- Special Function Unit, Understanding GPU Architecture
- streaming multiprocessors, Understanding GPU Architecture
- thread block clusters, Threads, Warps, Blocks, and Grids
- thread block size, Choosing Threads-per-Block and Blocks-per-Grid Sizes
- thread blocks, Threads, Warps, Blocks, and Grids-Threads, Warps, Blocks, and Grids
- threads, Threads, Warps, Blocks, and Grids-Threads, Warps, Blocks, and Grids
- threads-per-block and blocks-per-grid, Choosing Threads-per-Block and Blocks-per-Grid Sizes-Choosing Threads-per-Block and Blocks-per-Grid Sizes
- Unified CPU-GPU Memory, The CPU and GPU Superchip
- warps, Streaming Multiprocessor, Threads, and Warps, Understanding GPU Architecture-Understanding GPU Architecture

- about, Profiling and Diagnosing GPU Bottlenecks
- achieved occupancy and GPU utilization, Inspecting Achieved Occupancy and GPU Utilization-Optimizing the Kernel
- data pipeline profiling and tuning, Profiling and Tuning the Data Pipeline
- iteratively profiling, Iteratively Profiling and Determining the Kernel Bottleneck-Iteratively Profiling and Determining the Kernel Bottleneck
- kernel optimization, Optimizing the Kernel-Optimizing the Kernel
- monitoring in production, Monitoring System Metrics and Counters
- Nsight Compute and Roofline analysis, Nsight Compute and Roofline Analysis
- Nsight Systems timeline view, Profiling and Diagnosing GPU Bottlenecks, Nsight Compute and Roofline Analysis
- PyTorch profiler via Kineto, PyTorch Profiler and Visualization Tools-Profiler-Guided Analysis
- warp stall reasons, Analyzing Warp Stall Reasons with Nsight Compute-Other Stall Reasons

- about containers, Container Runtime Optimizations for GPUs
- container overlay filesystem overhead, Avoiding Container Overlay Filesystem Overhead
- NVIDIA container runtime injecting libraries, NVIDIA Container Runtime
- NVIDIA Container Toolkit, Container Runtime Optimizations for GPUs
- NVIDIA Container Toolkit and CUDA compatibility, NVIDIA Container Toolkit and CUDA Compatibility
- reducing image size, Reduce Image Size for Faster Container Startup

- callback function, Fine-Grained Synchronization with Events and Callbacks

- registering with PyTorch, Registering Custom Kernels with PyTorch-Registering Custom Kernels with PyTorch
- tuning kernel launch parameters, Tuning Kernel-Launch Parameters

- avoiding shared memory, Warp Shuffle Intrinsics: Avoid Shared Memory and Explicit Synchronization-Warp Shuffle Intrinsics: Avoid Shared Memory and Explicit Synchronization
- avoiding shared-memory bank conflicts, Avoid Shared-Memory Bank Conflicts-Avoid Shared-Memory Bank Conflicts
- coalesced versus uncoalesced global memory access, Coalesced Versus Uncoalesced Global Memory Access-Coalesced Versus Uncoalesced Global Memory Access
- read-only data caches, Read-Only Data Caches-Read-Only Data Caches
- read-only data caches pitfall, Read-Only Data Caches
- symmetric memory, PyTorch Symmetric Memory
- Tensor Memory Accelerator for tile fetch, Asynchronous Memory Prefetching and Tensor Memory Accelerator-Asynchronous Memory Prefetching and Tensor Memory Accelerator
- tiling and data reuse via shared memory, Tiling and Data Reuse Using Shared Memory-Tiling and Data Reuse Using Shared Memory
- vectorized memory access, Vectorized Memory Access-Vectorized Memory Access
- warp shuffle intrinsics, Warp Shuffle Intrinsics: Avoid Shared Memory and Explicit Synchronization

- (see also multiple GPUs)
- GPU-to-GPU memory sharing, Fine-Grained GPU-to-GPU Memory Sharing with NVSHMEM-Fine-Grained GPU-to-GPU Memory Sharing with NVSHMEM
- NVIDIA SHMEM library, Fine-Grained GPU-to-GPU Memory Sharing with NVSHMEM-Fine-Grained GPU-to-GPU Memory Sharing with NVSHMEM

- adding two vectors sequentially and in parallel, Maintaining High Occupancy and GPU Utilization-Maintaining High Occupancy and GPU Utilization
- architecture of GPU, Understanding GPU Architecture
- child-kernel launch limit, Dynamic Parallelism
- cuTile, C++ and Python CUDA Libraries
- Data Parallel versus Distributed Data Parallel, Distributed Data Parallel Strategies-Distributed Data Parallel Strategies
- dynamic parallelism, Dynamic Parallelism-Dynamic Parallelism
- fully sharded data parallel, Distributed Data Parallel Strategies
- PTX, Inline PTX and SASS Tuning for Microoptimizations
- thread block cluster algorithms, Designing Efficient Algorithms with Thread Block Clusters-Designing Efficient Algorithms with Thread Block Clusters

- memory hierarchy, Understanding GPU Memory Hierarchy-Understanding GPU Memory Hierarchy

- ECC mode for memory, GPU Clock Speeds and ECC
- GPU clock speeds, GPU Clock Speeds and ECC
- GPU memory fragmentation and oversubscription, GPU Memory Oversubscription, Fragmentation, and Out-of-Memory Handling-GPU Memory Oversubscription, Fragmentation, and Out-of-Memory Handling
- GPU persistence mode, GPU Persistence Mode
- keeping GPU driver loaded, GPU Memory Oversubscription, Fragmentation, and Out-of-Memory Handling
- MIG, MIG-MIG
- MPS, MPS-MPS

- CPU and GPU linked by NVLink-C2C, NUMA Awareness and CPU Pinning, NUMA Awareness and CPU Pinning

- memory, The CPU and GPU Superchip, Topology Awareness in NCCL
- NIXL offloading KV cache to CPU memory, KV Cache Offloading with NIXL-KV Cache Offloading with NIXL
- tricks using CPU-GPU unified memory, Continuous Prewarming of CUDA Graphs and Caches Using Time-Series Prediction

- Grace Blackwell Superchip, NVIDIA’s “AI Supercomputer in a Rack”
- memory, The CPU and GPU Superchip
- NVL72, Ultrascale Networking Treating Many GPUs as One

- (see also activation checkpointing)

- inference monitoring, Monitoring System Metrics and Counters, Monitoring System Metrics and Counters
- NVLink/NVSwitch monitoring, Real-Time Link Telemetry and Monitoring

- avoiding whenever possible, TorchDynamo for Bytecode Capture and Graph Extraction, Graph Breaks and TorchDynamo explain()conditionals as tensor operations, Graph Breaks and TorchDynamo explain()data-dependent branches, Graph Breaks and TorchDynamo explain()marking functions and code blocks as safe, Mark Functions and Code Blocks as Safe with allow_in_graphminimizing graph recompilations, Minimize Graph Recompilations
- causes, Tips for Handling Graph Breaks-Tips for Handling Graph Breakssequence not captured into a single graph, Graph Breaks and TorchDynamo explain()shape mismatches, Profiling and Debugging Compiler Performance Issuestips for handling common causes, Tips for Handling Graph Breaks-Tips for Handling Graph Breaksunsupported operations in model, Profiling and Debugging Compiler Performance Issues
- complex graphs requiring graph breaks, Graph Breaks and TorchDynamo explain()
- debugging, Graph Breaks and TorchDynamo explain(), Debugging Compiler Phases, Graph Breaks, and Performancemonitoring graph recompilations, TorchDynamo for Bytecode Capture and Graph Extraction, Minimize Graph Recompilations
- forcing an error when full graph not captured, Graph Breaks and TorchDynamo explain()
- PyTorch compiler trace, Using PyTorch Profiler
- TorchDynamo, TorchDynamo for Bytecode Capture and Graph Extractionexplain() output, Profiling and Debugging Compiler Performance Issues, FSDP with torch.compileforcing error to be raised at compile, TorchDynamo for Bytecode Capture and Graph Extraction

- conditionals as tensor operations, Graph Breaks and TorchDynamo explain()
- data-dependent branches, Graph Breaks and TorchDynamo explain()
- marking functions and code blocks as safe, Mark Functions and Code Blocks as Safe with allow_in_graph
- minimizing graph recompilations, Minimize Graph Recompilations

- sequence not captured into a single graph, Graph Breaks and TorchDynamo explain()
- shape mismatches, Profiling and Debugging Compiler Performance Issues
- tips for handling common causes, Tips for Handling Graph Breaks-Tips for Handling Graph Breaks
- unsupported operations in model, Profiling and Debugging Compiler Performance Issues

- monitoring graph recompilations, TorchDynamo for Bytecode Capture and Graph Extraction, Minimize Graph Recompilations

- explain() output, Profiling and Debugging Compiler Performance Issues, FSDP with torch.compile
- forcing error to be raised at compile, TorchDynamo for Bytecode Capture and Graph Extraction

- blocks-per-grid sizes, Choosing Threads-per-Block and Blocks-per-Grid Sizes-Choosing Threads-per-Block and Blocks-per-Grid Sizes
- blocksPerGrid parameter, CUDA Programming Refresher, Configuring Launch Parameters: Blocks per Grid and Threads per Block-Configuring Launch Parameters: Blocks per Grid and Threads per Block
- cooperative groupscooperative kernel launch grid size, Cooperative Groups, Cooperative Grid Synchronization and Persistent Kernelspersistent kernels via grid sync, Cooperative Grid Synchronization and Persistent Kernels-Cooperative Grid Synchronization and Persistent Kernels
- maximum dimensions, Choosing Threads-per-Block and Blocks-per-Grid Sizes

- cooperative kernel launch grid size, Cooperative Groups, Cooperative Grid Synchronization and Persistent Kernels
- persistent kernels via grid sync, Cooperative Grid Synchronization and Persistent Kernels-Cooperative Grid Synchronization and Persistent Kernels

### H

- H100 versus NVL72, Multi-GPU Programming
- half-precision, BF16/FP16, FP8, and FP4 Reduced Precision
- hardwareAI supercomputer-in-a-box overview, AI System Hardware Overviewcache coherency, The CPU and GPU SuperchipCPU + GPU superchip, The CPU and GPU Superchip-Streaming Multiprocessor, Threads, and WarpsGPU Tensor Cores and Transformer Engine, NVIDIA GPU Tensor Cores and Transformer Engine-NVIDIA GPU Tensor Cores and Transformer Enginemany GPUs as one, Ultrascale Networking Treating Many GPUs as One-Co-Packaged Optics: Future of Networking Hardwaremulti-GPU programming, Multi-GPU Programming-Multi-GPU Programmingmultirack and storage communication, Multirack and Storage CommunicationNVIDIA Blackwell dual-die GPU, NVIDIA Blackwell “Dual-Die” GPU-NVIDIA Blackwell “Dual-Die” GPUNVIDIA Grace CPU, NVIDIA Grace CPUNVL72, Ultrascale Networking Treating Many GPUs as One(see also NVL72)streaming multiprocessors, threads, warps, Streaming Multiprocessor, Threads, and Warps-Streaming Multiprocessor, Threads, and Warpscompute-optimized versus memory-optimized, Compute-Optimized Versus Memory-Optimized Hardware-Different precision for prefill and decodeDeepSeek.AI export restrictions, Introduction and AI System Overview, DeepSeek Scales to ~680-Billion Parameter Models Despite US Export Hardware Restrictions in China-DeepSeek Scales to ~680-Billion Parameter Models Despite US Export Hardware Restrictions in Chinadisaggregated prefill and decode, Heterogeneous Hardware and Parallelism Strategies for Prefill and Decode-Different precision for prefill and decodefuture evolution, Scaling Toward Multimillion GPU Clusters and 100-Trillion-Parameter Models-Scaling Toward Multimillion GPU Clusters and 100-Trillion-Parameter Modelsfuture hardware from NVIDIA, A Glimpse into the Future: NVIDIA’s Roadmap-Feynman GPU (2028) and Doubling Something Every YearBlackwell Ultra and Grace Blackwell Ultra, Blackwell Ultra and Grace Blackwell Ultradoubling every year, Feynman GPU (2028) and Doubling Something Every YearFeynman GPU, Feynman GPU (2028) and Doubling Something Every YearRubin Ultra and Vera Rubin Ultra, Rubin Ultra and Vera Rubin Ultra (2027)Vera Rubin Superchip, Vera Rubin Superchip (2026)GPU programming across generations, CUDA Forward and Backward Compatibility Across GPU Hardware GenerationsGPU to Kubernetes scheduler, Kubernetes for Topology-Aware Container Orchestration and Networkinghardware-software codesign, Mechanical Sympathy: Hardware-Software Codesign-Mechanical Sympathy: Hardware-Software CodesignCUDA Pipeline API plus TMA, Asynchronous Memory Prefetching and Tensor Memory AcceleratorDeepSeek MLA, Mechanical Sympathy: Hardware-Software Codesignhomogeneous to avoid straggler nodes or processes, Pitfall #5: Straggler nodes or processesoccupancy limited by, Choosing Threads-per-Block and Blocks-per-Grid Sizes, Choosing Threads-per-Block and Blocks-per-Grid Sizesoptimizations open source, Transparency and ReproducibilityPage Migration Engine, GPU Memory Oversubscription, Fragmentation, and Out-of-Memory HandlingROI of upgrading, ROI of Upgrading Your Hardwareroofline model, Roofline Model: Compute-Bound or Memory-Bound Workloads-Roofline Model: Compute-Bound or Memory-Bound Workloads
- HBM (high-bandwidth memory), Scaling Toward Multimillion GPU Clusters and 100-Trillion-Parameter Models
- hiding latency (see latency hiding)
- high occupancy, Threads, Warps, Blocks, and Gridsmaintaining high occupancy, Maintaining High Occupancy and GPU Utilization-Maintaining High Occupancy and GPU Utilizationimpact of occupancy, Maintaining High Occupancy and GPU Utilization-Maintaining High Occupancy and GPU Utilizationprogramming CUDA, Maintaining High Occupancy and GPU Utilization-Maintaining High Occupancy and GPU Utilization
- high-bandwidth memory (HBM), Scaling Toward Multimillion GPU Clusters and 100-Trillion-Parameter Models
- Holistic Trace Analysis (HTA; PyTorch), NVTX Markers and Profiling Tools, NVTX Markers and Profiling Toolsmulti-GPU profiling, Multi-GPU Profiling with HTA
- Hopper H100, Introduction and AI System Overview
- HTA (see Holistic Trace Analysis (HTA; PyTorch))
- hugepages, Transparent Hugepagestransparent hugepages, Transparent Hugepages
- Hugging Face Transformers libraryCTRL, Prompt CleansingKV cache offloading, Speculative KV Prefetching for Faster TTFTKV cache quantized, Real-Time KV Cache Compression and Policy SwitchingHalf-Quadratic Quantization implementation, Real-Time KV Cache Compression and Policy Switching
- Hugging Face Ultra-Scale Playbook, Smart Compilers and Automated Code Optimizations
- hybrid parallelism, Hybrid Parallelism-Hybrid Parallelismversus TP versus PP, Adaptive Parallelism Strategies (TP Versus PP Versus Hybrid)-Adaptive Parallelism Strategies (TP Versus PP Versus Hybrid)
- hyperthreading, NUMA Awareness and CPU Pinning

- AI supercomputer-in-a-box overview, AI System Hardware Overviewcache coherency, The CPU and GPU SuperchipCPU + GPU superchip, The CPU and GPU Superchip-Streaming Multiprocessor, Threads, and WarpsGPU Tensor Cores and Transformer Engine, NVIDIA GPU Tensor Cores and Transformer Engine-NVIDIA GPU Tensor Cores and Transformer Enginemany GPUs as one, Ultrascale Networking Treating Many GPUs as One-Co-Packaged Optics: Future of Networking Hardwaremulti-GPU programming, Multi-GPU Programming-Multi-GPU Programmingmultirack and storage communication, Multirack and Storage CommunicationNVIDIA Blackwell dual-die GPU, NVIDIA Blackwell “Dual-Die” GPU-NVIDIA Blackwell “Dual-Die” GPUNVIDIA Grace CPU, NVIDIA Grace CPUNVL72, Ultrascale Networking Treating Many GPUs as One(see also NVL72)streaming multiprocessors, threads, warps, Streaming Multiprocessor, Threads, and Warps-Streaming Multiprocessor, Threads, and Warps
- compute-optimized versus memory-optimized, Compute-Optimized Versus Memory-Optimized Hardware-Different precision for prefill and decode
- DeepSeek.AI export restrictions, Introduction and AI System Overview, DeepSeek Scales to ~680-Billion Parameter Models Despite US Export Hardware Restrictions in China-DeepSeek Scales to ~680-Billion Parameter Models Despite US Export Hardware Restrictions in China
- disaggregated prefill and decode, Heterogeneous Hardware and Parallelism Strategies for Prefill and Decode-Different precision for prefill and decode
- future evolution, Scaling Toward Multimillion GPU Clusters and 100-Trillion-Parameter Models-Scaling Toward Multimillion GPU Clusters and 100-Trillion-Parameter Models
- future hardware from NVIDIA, A Glimpse into the Future: NVIDIA’s Roadmap-Feynman GPU (2028) and Doubling Something Every YearBlackwell Ultra and Grace Blackwell Ultra, Blackwell Ultra and Grace Blackwell Ultradoubling every year, Feynman GPU (2028) and Doubling Something Every YearFeynman GPU, Feynman GPU (2028) and Doubling Something Every YearRubin Ultra and Vera Rubin Ultra, Rubin Ultra and Vera Rubin Ultra (2027)Vera Rubin Superchip, Vera Rubin Superchip (2026)
- GPU programming across generations, CUDA Forward and Backward Compatibility Across GPU Hardware Generations
- GPU to Kubernetes scheduler, Kubernetes for Topology-Aware Container Orchestration and Networking
- hardware-software codesign, Mechanical Sympathy: Hardware-Software Codesign-Mechanical Sympathy: Hardware-Software CodesignCUDA Pipeline API plus TMA, Asynchronous Memory Prefetching and Tensor Memory AcceleratorDeepSeek MLA, Mechanical Sympathy: Hardware-Software Codesign
- homogeneous to avoid straggler nodes or processes, Pitfall #5: Straggler nodes or processes
- occupancy limited by, Choosing Threads-per-Block and Blocks-per-Grid Sizes, Choosing Threads-per-Block and Blocks-per-Grid Sizes
- optimizations open source, Transparency and Reproducibility
- Page Migration Engine, GPU Memory Oversubscription, Fragmentation, and Out-of-Memory Handling
- ROI of upgrading, ROI of Upgrading Your Hardware
- roofline model, Roofline Model: Compute-Bound or Memory-Bound Workloads-Roofline Model: Compute-Bound or Memory-Bound Workloads

- cache coherency, The CPU and GPU Superchip
- CPU + GPU superchip, The CPU and GPU Superchip-Streaming Multiprocessor, Threads, and Warps
- GPU Tensor Cores and Transformer Engine, NVIDIA GPU Tensor Cores and Transformer Engine-NVIDIA GPU Tensor Cores and Transformer Engine
- many GPUs as one, Ultrascale Networking Treating Many GPUs as One-Co-Packaged Optics: Future of Networking Hardware
- multi-GPU programming, Multi-GPU Programming-Multi-GPU Programming
- multirack and storage communication, Multirack and Storage Communication
- NVIDIA Blackwell dual-die GPU, NVIDIA Blackwell “Dual-Die” GPU-NVIDIA Blackwell “Dual-Die” GPU
- NVIDIA Grace CPU, NVIDIA Grace CPU
- NVL72, Ultrascale Networking Treating Many GPUs as One(see also NVL72)
- streaming multiprocessors, threads, warps, Streaming Multiprocessor, Threads, and Warps-Streaming Multiprocessor, Threads, and Warps

- (see also NVL72)

- Blackwell Ultra and Grace Blackwell Ultra, Blackwell Ultra and Grace Blackwell Ultra
- doubling every year, Feynman GPU (2028) and Doubling Something Every Year
- Feynman GPU, Feynman GPU (2028) and Doubling Something Every Year
- Rubin Ultra and Vera Rubin Ultra, Rubin Ultra and Vera Rubin Ultra (2027)
- Vera Rubin Superchip, Vera Rubin Superchip (2026)

- CUDA Pipeline API plus TMA, Asynchronous Memory Prefetching and Tensor Memory Accelerator
- DeepSeek MLA, Mechanical Sympathy: Hardware-Software Codesign

- maintaining high occupancy, Maintaining High Occupancy and GPU Utilization-Maintaining High Occupancy and GPU Utilizationimpact of occupancy, Maintaining High Occupancy and GPU Utilization-Maintaining High Occupancy and GPU Utilization
- programming CUDA, Maintaining High Occupancy and GPU Utilization-Maintaining High Occupancy and GPU Utilization

- impact of occupancy, Maintaining High Occupancy and GPU Utilization-Maintaining High Occupancy and GPU Utilization

- multi-GPU profiling, Multi-GPU Profiling with HTA

- transparent hugepages, Transparent Hugepages

- CTRL, Prompt Cleansing
- KV cache offloading, Speculative KV Prefetching for Faster TTFT
- KV cache quantized, Real-Time KV Cache Compression and Policy SwitchingHalf-Quadratic Quantization implementation, Real-Time KV Cache Compression and Policy Switching

- Half-Quadratic Quantization implementation, Real-Time KV Cache Compression and Policy Switching

- versus TP versus PP, Adaptive Parallelism Strategies (TP Versus PP Versus Hybrid)-Adaptive Parallelism Strategies (TP Versus PP Versus Hybrid)

### I

- IB networking (see InfiniBand (IB) networking)
- IBTA (InfiniBand Trade Association), Multi-GPU Programming
- ILP (see instruction-level parallelism (ILP))
- in-flight batching, Continuous Batching
- Inductor (see TorchInductor (PyTorch))
- inferencebatchingarithmetic intensity increased, Structured Sparsitycomparing static, dynamic, continuous, Continuous Batchingcontinuous batching, Continuous Batchingdynamic batching, Dynamic Batching-Dynamic Batchinglatency-aware scheduling, Latency-Aware Scheduling and Dynamic Routing-Latency-Aware Scheduling and Dynamic Routingstall-free scheduling, Stall-Free Scheduling (Chunked Prefill)static batching, Dynamic Batchingcontinuous scheduling, Continuous SchedulingCUDA Graphs and inference engines, PyTorch, Inference Engines, and CUDA Graphsdisaggregated prefill and decode, Multinode Inference, Parallelism, Decoding, and Routing Optimizations(see also disaggregated prefill and decode)dynamic routing strategies for MoE models, Dynamic Routing Strategies for MoE Inference-Adaptive Expert Routing and Real-Time Monitoringexpert communication optimization, Expert Communication Optimization-Expert Communication Optimizationexpert replication, Load Balancing, Capacity Factor, and Expert Replicationload balancing and capacity factors, Load Balancing, Capacity Factor, and Expert ReplicationDynamo throughput improved by NIXL, NIXL and High-Performance Inference Systems Like NVIDIA Dynamomassive inference clusters, Data Parallelismmegakernels, Megakernels for Inferencemultinode inference, Multinode Inference, Parallelism, Decoding, and Routing Optimizationsdisaggregated inference (see disaggregated prefill and decode)Kubernetes for deploying, Deploying Disaggregated Prefill and Decode with Kubernetes-Deploying Disaggregated Prefill and Decode with Kubernetes, Full-Stack Inference OptimizationsNIXL disaggregated inference, NVIDIA’s NIXL and Disaggregated Inference-NCCL Versus NIXLabout NIXL, NCCL for Distributed Multi-GPU Communication, NVIDIA’s NIXL and Disaggregated Inference-NVIDIA’s NIXL and Disaggregated Inferenceasynchronous API with callbacks, NIXL Asynchronous API with Callbacks-NIXL Asynchronous API with Callbacksintelligent interconnect routing, Intelligent Interconnect Routing for KV Cache TransfersKV cache offloading with NIXL, KV Cache Offloading with NIXL-KV Cache Offloading with NIXLseparate prefill and decode inference stages, Separate Prefill and Decode Inference Stages-Separate Prefill and Decode Inference Stagesoptimizationsabout ultralarge language model inference, Dynamic and Adaptive Inference Engine Optimizationsadaptive batching, Adaptive Batching and Chunked Prefill Scheduling-Adaptive Batching and Chunked Prefill Schedulingadaptive parallelism strategies, Adaptive Parallelism Strategies (TP Versus PP Versus Hybrid)-Adaptive Parallelism Strategies (TP Versus PP Versus Hybrid)chunked prefill scheduling, Adaptive Batching and Chunked Prefill Scheduling-Adaptive Batching and Chunked Prefill Schedulingdynamic activation quantization, Dynamic Quantization and Activation Range Adjustmentdynamic early exit networks, Dynamic Early-Exit Networksdynamic memory-allocation switching, Dynamic Memory-Allocation Switching (Slab Versus Caching Versus Stream-Ordered)-Dynamic Memory-Allocation Switching (Slab Versus Caching Versus Stream-Ordered)dynamic precision changes, Dynamic Precision Changes-Dynamic Precision Changesdynamic shared memory allocation, Dynamic Shared-Memory Allocation and Occupancy-Aware Kernel Selection-Dynamic Shared-Memory Allocation and Occupancy-Aware Kernel Selectiondynamic token pruning, Dynamic Token Pruning with LazyLLMedge-oriented MoE memory budgeting, Edge-Oriented MoE Memory Budgetinginput-aware layer skipping, Input-Aware Layer Skipping (DASH)kernel autotuning, Kernel Autotuning for Transformer Self-Attention and MLP Paths-Kernel Autotuning for Transformer Self-Attention and MLP PathsKV cache compression and policy switching, Real-Time KV Cache Compression and Policy Switching-Real-Time KV Cache Compression and Policy SwitchingMoE speculative expert routing, Speculative MoE Expert Routing and Communication Reductionmulti-GPU congestion- and topology-aware scheduling, Congestion-Aware and Topology-Aware Scheduling with Multiple GPUs-Coordinating NVSwitch Transfers with Fine-Tuned Schedulingoccupancy-aware kernel selection, Dynamic Shared-Memory Allocation and Occupancy-Aware Kernel Selection-Dynamic Shared-Memory Allocation and Occupancy-Aware Kernel Selectionprewarming graphs and caches, Continuous Prewarming of CUDA Graphs and Caches Using Time-Series Prediction-Continuous Prewarming of CUDA Graphs and Caches Using Time-Series Predictionruntime kernel patching, Runtime Kernel Performance Improvements and Hot-Swappable Implementations-Runtime Kernel Performance Improvements and Hot-Swappable Implementationsspeculative KV prefetching, Speculative KV Prefetching for Faster TTFT-Speculative KV Prefetching for Faster TTFTtuning AI at runtime via RL agents, Reinforcement Learning Agents for Tuning AI Systems at Runtimeoptimizations at the application level, Application-Level Optimizations-Token Output Limits and Timeoutsdebouncing and request coalescing, Debouncing and Request Coalescingmodel cascading and tiered deployment, Model Cascading and Tiered Model Deployment-Model Cascading and Tiered Model Deploymentprefix caching, Prefix Caching-Prefix Cachingprompt cleansing, Prompt Cleansingprompt compression, Prompt Compressionstreaming responses, Streaming Responses-Streaming Responsestoken output limits and timeouts, Token Output Limits and Timeoutsoptimizations at the systems level, Systems-Level OptimizationsGPU utilization versus latency, Maximizing GPU Utilization and Throughput Versus Latency Trade-OffsKV cache, KV Cache Offloading and Memory Pool Allocationmemory, Memoryoverlapping communication and computation, Overlapping Communication and Computation-Overlapping Communication and Computationpower and thermal constraints, Power and Thermal Constraintsquantization for real-time inference, Quantization Approaches for Real-Time Inference-Fusing Quantization-Dequantization Steps into the Execution Graphabout quantization, Quantization Approaches for Real-Time Inferenceactivation quantization, Activation Quantizationcombining weight and activation quantization, Combining Weight and Activation Quantizationpost-training quantization workflow, Post-Training Quantization Workflowquant-dequant steps into execution graph, Fusing Quantization-Dequantization Steps into the Execution Graphreducing precision, Reducing Precision from FP16 to FP8 and FP4weight-only quantization, Weight-Only Quantization (GPTQ, AWQ)structured sparsity for inference workloads, Structured Sparsitytuning performancecorrectness issue debugging, Debugging Correctness Issues-Debugging Correctness Issuesfull-stack optimizations, Full-Stack Inference Optimizations-Full-Stack Inference Optimizationsmonitoring system metrics and counters, Monitoring System Metrics and Counters-Monitoring System Metrics and Countersobserve-hypothesize-tune loop, Profiling, Debugging, and Tuning Inference Performance-Profiling, Debugging, and Tuning Inference Performanceprofiling with Nsight, Profiling with Nsight Systems and Nsight Compute-Profiling with Nsight Systems and Nsight Computeruntime tuning via RL agents, Reinforcement Learning Agents for Tuning AI Systems at Runtimesystems-level optimizations, Systems-Level Optimizationstroubleshooting, Inference Troubleshooting Recipes
- InfiniBand (IB) networkingcontainer access to InfiniBand devices, High-Speed, Low-Overhead Data Transfers with RDMAcounters on nodes to spot stragglers, Pitfall #5: Straggler nodes or processesenabling GPUDirect RDMA, Optimizing Network Communication for Kuberneteshost channel adapter, Pitfall #1: Using a CPU-bound Gloo backend instead of NCCLRDMA support, High-Speed, Low-Overhead Data Transfers with RDMASHARP, In-Network SHARP Aggregation
- InfiniBand Trade Association (IBTA), Multi-GPU Programming
- inputvariable-length sequences, Dynamic Shapes and Variable Sequence Lengths-Dynamic Shapes and Variable Sequence Lengthsdynamic shapes, Dynamic Shapes and Variable Sequence Lengths-Dynamic Shapes and Variable Sequence Lengthspadding, Dynamic Shapes and Variable Sequence Lengthsprofiling dynamic shapes versus padding, Dynamic Shapes and Variable Sequence Lengthsvery long inputscontext parallelism, Context (Sequence) Parallelismprompt compression, Prompt Compressionstall-free scheduling, Stall-Free Scheduling (Chunked Prefill)
- instruction scheduling with PTX, Inline PTX and SASS Tuning for Microoptimizations
- instruction-level parallelism (ILP)exploitingcompiler hints, Exposing Instruction-Level Parallelism, Loop Unrolling, Interleaving, and Compiler Hintinginterleaving independent operations, Loop Unrolling, Interleaving, and Compiler Hintingunrolling loops, Loop Unrolling, Interleaving, and Compiler Hintingexposing, Exposing Instruction-Level Parallelism-Profiling and Mitigating Register Pressurecompiler hints, Exposing Instruction-Level Parallelism, Loop Unrolling, Interleaving, and Compiler Hintingoccupancy and ILP, Exposing Instruction-Level Parallelism, ILP and Occupancy, Loop Unrolling, Interleaving, and Compiler Hintingwarp schedulers and dual issue instructions, Warp Scheduling and Dual Issue Instructions-Warp Scheduling and Dual Issue Instructionsincreasing for latency-bound GPUs, Iteratively Profiling and Determining the Kernel Bottleneckincreasing ILP with PTX, Inline PTX and SASS Tuning for Microoptimizationsmisconception that more operations performed, Exposing Instruction-Level ParallelismPyTorch compiler not guaranteeing, Using Predication to Minimize Divergenceregister pressure, Profiling and Mitigating Register Pressurewarp stall reasons, Other Stall Reasons, Other Stall Reasons
- INT8, INT8 Reduced Precision and DP4A Instructions for Inference
- inter-kernel pipeliningabout inter-kernel pipelining, Inter-Kernel Pipelining, Synchronization, and CUDA Stream-Ordered Memory Allocationscompute overlapping with data transfers via streams, Using Streams to Overlap Compute with Data Transfers-Using Streams to Overlap Compute with Data Transfersconcurrency, Overlapping Communication and Computation-Overlapping Communication and Computationmultiple GPUs, Multi-GPU Compute and Data Transfer Overlap with CUDA Streams-Multi-GPU Compute and Data Transfer Overlap with CUDA StreamsCUDA stream ordered memory allocator, Stream-Ordered Memory Allocator-Stream-Ordered Memory Allocatorusing, Using CUDA Streams and Stream-Ordered Memory Allocator with LLMs-Using CUDA Streams and Stream-Ordered Memory Allocator with LLMsusing with multiple GPUs, Multi-GPU Compute and Data Transfer Overlap with CUDA Streams-Multi-GPU Compute and Data Transfer Overlap with CUDA Streamsdefault streamsdefault versus explicit streams, Default Versus Explicit (Nondefault) Streamslegacy default streams, Legacy Default Streamper-thread default streams, Modern Per-Thread Default Streamusing default streams, Best Practices for Default Stream Usage-Best Practices for Default Stream Usagekernel execution overlapped with CUDA streams, Overlapping Kernel Execution with CUDA Streams-Overlapping Kernel Execution with CUDA Streamslaunching five kernels on two streams, Overlapping Kernel Execution with CUDA Streams-Overlapping Kernel Execution with CUDA Streamswarp specialization replaced with CUDA streams, Overlapping Kernel Execution with CUDA Streams-Overlapping Kernel Execution with CUDA Streamsmulti-GPU overlap of compute and data, Multi-GPU Compute and Data Transfer Overlap with CUDA Streams-Multi-GPU Compute and Data Transfer Overlap with CUDA StreamsProgrammatic Dependent Launch, Programmatic Dependent Launch-Programmatic Dependent Launchthread block clusters and warp specialization, Combining PDL and Thread Block Clusters with Warp Specialization-Combining PDL and Thread Block Clusters with Warp Specializationsynchronization with events and callbacks, Fine-Grained Synchronization with Events and Callbackscross-stream synchronization, Using CUDA Events for Cross-Stream Synchronizationwarp specializationCUDA streams, Pipelining with Warp Specialization (Intra-Kernel) and CUDA Streams (Inter-Kernel)-Pipelining with Warp Specialization (Intra-Kernel) and CUDA Streams (Inter-Kernel)thread block clusters and CUDA streams, Warp Specialization with Thread Block Clusters and CUDA Streams
- interrupt requests (IRQs)irqbalance daemon, Scheduler and Interrupt Affinityscheduler and interrupt affinity, Scheduler and Interrupt Affinity
- intra-kernel pipelining, Intra-Kernel Pipelining Techniquesabout intra-kernel pipelining, Intra-Kernel Pipelining Techniquescomparing techniques, Intra-Kernel Pipelining Techniques, Using CUDA Pipeline API for Warp Specialization, Warp Specialization with Thread Block Clusters-Warp Specialization with Thread Block Clusterscooperative groups, Cooperative Groups-When to Combine Persistent Kernels and Cooperative GroupsCooperative Groups API, Warp Shuffle Intrinsics: Avoid Shared Memory and Explicit Synchronization, Cooperative Groups, Coordinating Thread Block Clusters with Cooperative Groups API-Coordinating Thread Block Clusters with Cooperative Groups APIdescribed, Thread Block Clusters and Distributed Shared Memorylaunching a kernel in cooperative mode, Cooperative Groups, When to Combine Persistent Kernels and Cooperative Groupspersistent kernels via grid sync, Cooperative Grid Synchronization and Persistent Kernels-Cooperative Grid Synchronization and Persistent Kernelsthread block clusters versus, Thread Block Clusters and Distributed Shared Memorywhen to combine with persistent kernels, When to Combine Persistent Kernels and Cooperative Groupsdouble-buffering with CUDA Pipeline API, Cooperative Tiling and Double-Buffering with the CUDA Pipeline API-Cooperative Tiling and Double-Buffering with the CUDA Pipeline APIpersistent kernels, Persistent Kernels and Megakernels-Persistent Kernels and Warp Specializationcommon workloads for, Common Workloads for Persistent Kernelscooperative groups via grid sync, Cooperative Grid Synchronization and Persistent Kernels-Cooperative Grid Synchronization and Persistent Kernelsmegakernels, Megakernels for Inferenceoccupancy, Persistent Kernels and Megakernelswarp specialization, Persistent Kernels and Warp Specializationwhen to combine with cooperative groups, When to Combine Persistent Kernels and Cooperative Groupsthread block clusters, Intra-Kernel Pipelining, Warp Specialization, and Cooperative Thread Block Clusters(see also thread block clusters)warp specialization, Warp Specialization and the Producer-Consumer Model-Warp Specialization and the Producer-Consumer ModelCUDA Pipeline API, Using CUDA Pipeline API for Warp Specialization-Using CUDA Pipeline API for Warp SpecializationCUDA Pipeline API in PyTorch, PyTorch, CUDA Pipeline API, and Warp SpecializationCUDA streams, Pipelining with Warp Specialization (Intra-Kernel) and CUDA Streams (Inter-Kernel)-Pipelining with Warp Specialization (Intra-Kernel) and CUDA Streams (Inter-Kernel)thread block clusters, Warp Specialization with Thread Block Clusters-Warp Specialization with Thread Block Clusterswhen to use warp specialization versus double-buffering, Warp Specialization and the Producer-Consumer Model
- I/Ocommunication versus compute bottlenecks, Diagnosing Communication- Versus Compute-Bound Workloadscontinuous profiling and tuning workflow, Continuous Profiling and Tuning Workflow-Continuous Profiling and Tuning Workflowdata pipeline tuning, Tuning the Data Pipelinedata loading and preprocessing, Efficient Data Loading and Preprocessing-Efficient Data Loading and PreprocessingNeMo Curator for training datasets, Creating High-Quality LLM Datasets with NVIDIA NeMo CuratorNVIDIA Data Loading Library, Multimodal Data Processing with NVIDIA DALIscaling out workers as GPUs scaled, Scaling Out Workers as You Scale Out Number of GPUsI/O isolation in Kubernetes, Dealing with I/O Isolationmonitoring, Monitoring Storage I/O-Monitoring Storage I/ONVIDIA Magnum IO, NVIDIA Magnum IO Optimization Stackoptimizing GPU-based storagecloud storage caches, Distributed, Parallel Filesystems and Object Storescompressing data, Tuning, Replicating, and Compressing DataDeepSeek Fire-Flyer File System, DeepSeek’s Fire-Flyer File System-DeepSeek’s Fire-Flyer File Systemfast storage and data locality, Fast Storage and Data Localityfilesystem optimized, Sequential Versus Random Read Patternsmultiqueue block I/O scheduler, Tuning NVMe and Filesystem for ThroughputNFS servers, Distributed, Parallel Filesystems and Object StoresNFS servers tuning parameters, Distributed, Parallel Filesystems and Object StoresNVIDIA GDS, Using NVIDIA GDS-Measuring GDS with gdsioNVMe and filesystem tuned for throughput, Tuning NVMe and Filesystem for Throughputparallel filesystems, Distributed, Parallel Filesystems and Object Storesread ahead, Tuning NVMe and Filesystem for Throughputread size tuned, Sequential Versus Random Read Patternssequential versus random reads, Sequential Versus Random Read Patterns-Sequential Versus Random Read Patternsshared filesystems, Distributed, Parallel Filesystems and Object Storesstriping files, Distributed, Parallel Filesystems and Object Storestuning filesystems, Tuning, Replicating, and Compressing Data
- irqbalance daemon, Scheduler and Interrupt Affinity
- iteration-level scheduling, Continuous Batching

- batchingarithmetic intensity increased, Structured Sparsitycomparing static, dynamic, continuous, Continuous Batchingcontinuous batching, Continuous Batchingdynamic batching, Dynamic Batching-Dynamic Batchinglatency-aware scheduling, Latency-Aware Scheduling and Dynamic Routing-Latency-Aware Scheduling and Dynamic Routingstall-free scheduling, Stall-Free Scheduling (Chunked Prefill)static batching, Dynamic Batching
- continuous scheduling, Continuous Scheduling
- CUDA Graphs and inference engines, PyTorch, Inference Engines, and CUDA Graphs
- disaggregated prefill and decode, Multinode Inference, Parallelism, Decoding, and Routing Optimizations(see also disaggregated prefill and decode)
- dynamic routing strategies for MoE models, Dynamic Routing Strategies for MoE Inference-Adaptive Expert Routing and Real-Time Monitoringexpert communication optimization, Expert Communication Optimization-Expert Communication Optimizationexpert replication, Load Balancing, Capacity Factor, and Expert Replicationload balancing and capacity factors, Load Balancing, Capacity Factor, and Expert Replication
- Dynamo throughput improved by NIXL, NIXL and High-Performance Inference Systems Like NVIDIA Dynamo
- massive inference clusters, Data Parallelism
- megakernels, Megakernels for Inference
- multinode inference, Multinode Inference, Parallelism, Decoding, and Routing Optimizationsdisaggregated inference (see disaggregated prefill and decode)Kubernetes for deploying, Deploying Disaggregated Prefill and Decode with Kubernetes-Deploying Disaggregated Prefill and Decode with Kubernetes, Full-Stack Inference Optimizations
- NIXL disaggregated inference, NVIDIA’s NIXL and Disaggregated Inference-NCCL Versus NIXLabout NIXL, NCCL for Distributed Multi-GPU Communication, NVIDIA’s NIXL and Disaggregated Inference-NVIDIA’s NIXL and Disaggregated Inferenceasynchronous API with callbacks, NIXL Asynchronous API with Callbacks-NIXL Asynchronous API with Callbacksintelligent interconnect routing, Intelligent Interconnect Routing for KV Cache TransfersKV cache offloading with NIXL, KV Cache Offloading with NIXL-KV Cache Offloading with NIXLseparate prefill and decode inference stages, Separate Prefill and Decode Inference Stages-Separate Prefill and Decode Inference Stages
- optimizationsabout ultralarge language model inference, Dynamic and Adaptive Inference Engine Optimizationsadaptive batching, Adaptive Batching and Chunked Prefill Scheduling-Adaptive Batching and Chunked Prefill Schedulingadaptive parallelism strategies, Adaptive Parallelism Strategies (TP Versus PP Versus Hybrid)-Adaptive Parallelism Strategies (TP Versus PP Versus Hybrid)chunked prefill scheduling, Adaptive Batching and Chunked Prefill Scheduling-Adaptive Batching and Chunked Prefill Schedulingdynamic activation quantization, Dynamic Quantization and Activation Range Adjustmentdynamic early exit networks, Dynamic Early-Exit Networksdynamic memory-allocation switching, Dynamic Memory-Allocation Switching (Slab Versus Caching Versus Stream-Ordered)-Dynamic Memory-Allocation Switching (Slab Versus Caching Versus Stream-Ordered)dynamic precision changes, Dynamic Precision Changes-Dynamic Precision Changesdynamic shared memory allocation, Dynamic Shared-Memory Allocation and Occupancy-Aware Kernel Selection-Dynamic Shared-Memory Allocation and Occupancy-Aware Kernel Selectiondynamic token pruning, Dynamic Token Pruning with LazyLLMedge-oriented MoE memory budgeting, Edge-Oriented MoE Memory Budgetinginput-aware layer skipping, Input-Aware Layer Skipping (DASH)kernel autotuning, Kernel Autotuning for Transformer Self-Attention and MLP Paths-Kernel Autotuning for Transformer Self-Attention and MLP PathsKV cache compression and policy switching, Real-Time KV Cache Compression and Policy Switching-Real-Time KV Cache Compression and Policy SwitchingMoE speculative expert routing, Speculative MoE Expert Routing and Communication Reductionmulti-GPU congestion- and topology-aware scheduling, Congestion-Aware and Topology-Aware Scheduling with Multiple GPUs-Coordinating NVSwitch Transfers with Fine-Tuned Schedulingoccupancy-aware kernel selection, Dynamic Shared-Memory Allocation and Occupancy-Aware Kernel Selection-Dynamic Shared-Memory Allocation and Occupancy-Aware Kernel Selectionprewarming graphs and caches, Continuous Prewarming of CUDA Graphs and Caches Using Time-Series Prediction-Continuous Prewarming of CUDA Graphs and Caches Using Time-Series Predictionruntime kernel patching, Runtime Kernel Performance Improvements and Hot-Swappable Implementations-Runtime Kernel Performance Improvements and Hot-Swappable Implementationsspeculative KV prefetching, Speculative KV Prefetching for Faster TTFT-Speculative KV Prefetching for Faster TTFTtuning AI at runtime via RL agents, Reinforcement Learning Agents for Tuning AI Systems at Runtime
- optimizations at the application level, Application-Level Optimizations-Token Output Limits and Timeoutsdebouncing and request coalescing, Debouncing and Request Coalescingmodel cascading and tiered deployment, Model Cascading and Tiered Model Deployment-Model Cascading and Tiered Model Deploymentprefix caching, Prefix Caching-Prefix Cachingprompt cleansing, Prompt Cleansingprompt compression, Prompt Compressionstreaming responses, Streaming Responses-Streaming Responsestoken output limits and timeouts, Token Output Limits and Timeouts
- optimizations at the systems level, Systems-Level OptimizationsGPU utilization versus latency, Maximizing GPU Utilization and Throughput Versus Latency Trade-OffsKV cache, KV Cache Offloading and Memory Pool Allocationmemory, Memoryoverlapping communication and computation, Overlapping Communication and Computation-Overlapping Communication and Computationpower and thermal constraints, Power and Thermal Constraints
- quantization for real-time inference, Quantization Approaches for Real-Time Inference-Fusing Quantization-Dequantization Steps into the Execution Graphabout quantization, Quantization Approaches for Real-Time Inferenceactivation quantization, Activation Quantizationcombining weight and activation quantization, Combining Weight and Activation Quantizationpost-training quantization workflow, Post-Training Quantization Workflowquant-dequant steps into execution graph, Fusing Quantization-Dequantization Steps into the Execution Graphreducing precision, Reducing Precision from FP16 to FP8 and FP4weight-only quantization, Weight-Only Quantization (GPTQ, AWQ)
- structured sparsity for inference workloads, Structured Sparsity
- tuning performancecorrectness issue debugging, Debugging Correctness Issues-Debugging Correctness Issuesfull-stack optimizations, Full-Stack Inference Optimizations-Full-Stack Inference Optimizationsmonitoring system metrics and counters, Monitoring System Metrics and Counters-Monitoring System Metrics and Countersobserve-hypothesize-tune loop, Profiling, Debugging, and Tuning Inference Performance-Profiling, Debugging, and Tuning Inference Performanceprofiling with Nsight, Profiling with Nsight Systems and Nsight Compute-Profiling with Nsight Systems and Nsight Computeruntime tuning via RL agents, Reinforcement Learning Agents for Tuning AI Systems at Runtimesystems-level optimizations, Systems-Level Optimizationstroubleshooting, Inference Troubleshooting Recipes

- arithmetic intensity increased, Structured Sparsity
- comparing static, dynamic, continuous, Continuous Batching
- continuous batching, Continuous Batching
- dynamic batching, Dynamic Batching-Dynamic Batching
- latency-aware scheduling, Latency-Aware Scheduling and Dynamic Routing-Latency-Aware Scheduling and Dynamic Routing
- stall-free scheduling, Stall-Free Scheduling (Chunked Prefill)
- static batching, Dynamic Batching

- (see also disaggregated prefill and decode)

- expert communication optimization, Expert Communication Optimization-Expert Communication Optimization
- expert replication, Load Balancing, Capacity Factor, and Expert Replication
- load balancing and capacity factors, Load Balancing, Capacity Factor, and Expert Replication

- disaggregated inference (see disaggregated prefill and decode)
- Kubernetes for deploying, Deploying Disaggregated Prefill and Decode with Kubernetes-Deploying Disaggregated Prefill and Decode with Kubernetes, Full-Stack Inference Optimizations

- about NIXL, NCCL for Distributed Multi-GPU Communication, NVIDIA’s NIXL and Disaggregated Inference-NVIDIA’s NIXL and Disaggregated Inference
- asynchronous API with callbacks, NIXL Asynchronous API with Callbacks-NIXL Asynchronous API with Callbacks
- intelligent interconnect routing, Intelligent Interconnect Routing for KV Cache Transfers
- KV cache offloading with NIXL, KV Cache Offloading with NIXL-KV Cache Offloading with NIXL
- separate prefill and decode inference stages, Separate Prefill and Decode Inference Stages-Separate Prefill and Decode Inference Stages

- about ultralarge language model inference, Dynamic and Adaptive Inference Engine Optimizations
- adaptive batching, Adaptive Batching and Chunked Prefill Scheduling-Adaptive Batching and Chunked Prefill Scheduling
- adaptive parallelism strategies, Adaptive Parallelism Strategies (TP Versus PP Versus Hybrid)-Adaptive Parallelism Strategies (TP Versus PP Versus Hybrid)
- chunked prefill scheduling, Adaptive Batching and Chunked Prefill Scheduling-Adaptive Batching and Chunked Prefill Scheduling
- dynamic activation quantization, Dynamic Quantization and Activation Range Adjustment
- dynamic early exit networks, Dynamic Early-Exit Networks
- dynamic memory-allocation switching, Dynamic Memory-Allocation Switching (Slab Versus Caching Versus Stream-Ordered)-Dynamic Memory-Allocation Switching (Slab Versus Caching Versus Stream-Ordered)
- dynamic precision changes, Dynamic Precision Changes-Dynamic Precision Changes
- dynamic shared memory allocation, Dynamic Shared-Memory Allocation and Occupancy-Aware Kernel Selection-Dynamic Shared-Memory Allocation and Occupancy-Aware Kernel Selection
- dynamic token pruning, Dynamic Token Pruning with LazyLLM
- edge-oriented MoE memory budgeting, Edge-Oriented MoE Memory Budgeting
- input-aware layer skipping, Input-Aware Layer Skipping (DASH)
- kernel autotuning, Kernel Autotuning for Transformer Self-Attention and MLP Paths-Kernel Autotuning for Transformer Self-Attention and MLP Paths
- KV cache compression and policy switching, Real-Time KV Cache Compression and Policy Switching-Real-Time KV Cache Compression and Policy Switching
- MoE speculative expert routing, Speculative MoE Expert Routing and Communication Reduction
- multi-GPU congestion- and topology-aware scheduling, Congestion-Aware and Topology-Aware Scheduling with Multiple GPUs-Coordinating NVSwitch Transfers with Fine-Tuned Scheduling
- occupancy-aware kernel selection, Dynamic Shared-Memory Allocation and Occupancy-Aware Kernel Selection-Dynamic Shared-Memory Allocation and Occupancy-Aware Kernel Selection
- prewarming graphs and caches, Continuous Prewarming of CUDA Graphs and Caches Using Time-Series Prediction-Continuous Prewarming of CUDA Graphs and Caches Using Time-Series Prediction
- runtime kernel patching, Runtime Kernel Performance Improvements and Hot-Swappable Implementations-Runtime Kernel Performance Improvements and Hot-Swappable Implementations
- speculative KV prefetching, Speculative KV Prefetching for Faster TTFT-Speculative KV Prefetching for Faster TTFT
- tuning AI at runtime via RL agents, Reinforcement Learning Agents for Tuning AI Systems at Runtime

- debouncing and request coalescing, Debouncing and Request Coalescing
- model cascading and tiered deployment, Model Cascading and Tiered Model Deployment-Model Cascading and Tiered Model Deployment
- prefix caching, Prefix Caching-Prefix Caching
- prompt cleansing, Prompt Cleansing
- prompt compression, Prompt Compression
- streaming responses, Streaming Responses-Streaming Responses
- token output limits and timeouts, Token Output Limits and Timeouts

- GPU utilization versus latency, Maximizing GPU Utilization and Throughput Versus Latency Trade-Offs
- KV cache, KV Cache Offloading and Memory Pool Allocation
- memory, Memory
- overlapping communication and computation, Overlapping Communication and Computation-Overlapping Communication and Computation
- power and thermal constraints, Power and Thermal Constraints

- about quantization, Quantization Approaches for Real-Time Inference
- activation quantization, Activation Quantization
- combining weight and activation quantization, Combining Weight and Activation Quantization
- post-training quantization workflow, Post-Training Quantization Workflow
- quant-dequant steps into execution graph, Fusing Quantization-Dequantization Steps into the Execution Graph
- reducing precision, Reducing Precision from FP16 to FP8 and FP4
- weight-only quantization, Weight-Only Quantization (GPTQ, AWQ)

- correctness issue debugging, Debugging Correctness Issues-Debugging Correctness Issues
- full-stack optimizations, Full-Stack Inference Optimizations-Full-Stack Inference Optimizations
- monitoring system metrics and counters, Monitoring System Metrics and Counters-Monitoring System Metrics and Counters
- observe-hypothesize-tune loop, Profiling, Debugging, and Tuning Inference Performance-Profiling, Debugging, and Tuning Inference Performance
- profiling with Nsight, Profiling with Nsight Systems and Nsight Compute-Profiling with Nsight Systems and Nsight Compute
- runtime tuning via RL agents, Reinforcement Learning Agents for Tuning AI Systems at Runtime
- systems-level optimizations, Systems-Level Optimizations
- troubleshooting, Inference Troubleshooting Recipes

- container access to InfiniBand devices, High-Speed, Low-Overhead Data Transfers with RDMA
- counters on nodes to spot stragglers, Pitfall #5: Straggler nodes or processes
- enabling GPUDirect RDMA, Optimizing Network Communication for Kubernetes
- host channel adapter, Pitfall #1: Using a CPU-bound Gloo backend instead of NCCL
- RDMA support, High-Speed, Low-Overhead Data Transfers with RDMA
- SHARP, In-Network SHARP Aggregation

- variable-length sequences, Dynamic Shapes and Variable Sequence Lengths-Dynamic Shapes and Variable Sequence Lengthsdynamic shapes, Dynamic Shapes and Variable Sequence Lengths-Dynamic Shapes and Variable Sequence Lengthspadding, Dynamic Shapes and Variable Sequence Lengthsprofiling dynamic shapes versus padding, Dynamic Shapes and Variable Sequence Lengths
- very long inputscontext parallelism, Context (Sequence) Parallelismprompt compression, Prompt Compressionstall-free scheduling, Stall-Free Scheduling (Chunked Prefill)

- dynamic shapes, Dynamic Shapes and Variable Sequence Lengths-Dynamic Shapes and Variable Sequence Lengths
- padding, Dynamic Shapes and Variable Sequence Lengths
- profiling dynamic shapes versus padding, Dynamic Shapes and Variable Sequence Lengths

- context parallelism, Context (Sequence) Parallelism
- prompt compression, Prompt Compression
- stall-free scheduling, Stall-Free Scheduling (Chunked Prefill)

- exploitingcompiler hints, Exposing Instruction-Level Parallelism, Loop Unrolling, Interleaving, and Compiler Hintinginterleaving independent operations, Loop Unrolling, Interleaving, and Compiler Hintingunrolling loops, Loop Unrolling, Interleaving, and Compiler Hinting
- exposing, Exposing Instruction-Level Parallelism-Profiling and Mitigating Register Pressurecompiler hints, Exposing Instruction-Level Parallelism, Loop Unrolling, Interleaving, and Compiler Hintingoccupancy and ILP, Exposing Instruction-Level Parallelism, ILP and Occupancy, Loop Unrolling, Interleaving, and Compiler Hintingwarp schedulers and dual issue instructions, Warp Scheduling and Dual Issue Instructions-Warp Scheduling and Dual Issue Instructions
- increasing for latency-bound GPUs, Iteratively Profiling and Determining the Kernel Bottleneck
- increasing ILP with PTX, Inline PTX and SASS Tuning for Microoptimizations
- misconception that more operations performed, Exposing Instruction-Level Parallelism
- PyTorch compiler not guaranteeing, Using Predication to Minimize Divergence
- register pressure, Profiling and Mitigating Register Pressure
- warp stall reasons, Other Stall Reasons, Other Stall Reasons

- compiler hints, Exposing Instruction-Level Parallelism, Loop Unrolling, Interleaving, and Compiler Hinting
- interleaving independent operations, Loop Unrolling, Interleaving, and Compiler Hinting
- unrolling loops, Loop Unrolling, Interleaving, and Compiler Hinting

- compiler hints, Exposing Instruction-Level Parallelism, Loop Unrolling, Interleaving, and Compiler Hinting
- occupancy and ILP, Exposing Instruction-Level Parallelism, ILP and Occupancy, Loop Unrolling, Interleaving, and Compiler Hinting
- warp schedulers and dual issue instructions, Warp Scheduling and Dual Issue Instructions-Warp Scheduling and Dual Issue Instructions

- about inter-kernel pipelining, Inter-Kernel Pipelining, Synchronization, and CUDA Stream-Ordered Memory Allocations
- compute overlapping with data transfers via streams, Using Streams to Overlap Compute with Data Transfers-Using Streams to Overlap Compute with Data Transfersconcurrency, Overlapping Communication and Computation-Overlapping Communication and Computationmultiple GPUs, Multi-GPU Compute and Data Transfer Overlap with CUDA Streams-Multi-GPU Compute and Data Transfer Overlap with CUDA Streams
- CUDA stream ordered memory allocator, Stream-Ordered Memory Allocator-Stream-Ordered Memory Allocatorusing, Using CUDA Streams and Stream-Ordered Memory Allocator with LLMs-Using CUDA Streams and Stream-Ordered Memory Allocator with LLMsusing with multiple GPUs, Multi-GPU Compute and Data Transfer Overlap with CUDA Streams-Multi-GPU Compute and Data Transfer Overlap with CUDA Streams
- default streamsdefault versus explicit streams, Default Versus Explicit (Nondefault) Streamslegacy default streams, Legacy Default Streamper-thread default streams, Modern Per-Thread Default Streamusing default streams, Best Practices for Default Stream Usage-Best Practices for Default Stream Usage
- kernel execution overlapped with CUDA streams, Overlapping Kernel Execution with CUDA Streams-Overlapping Kernel Execution with CUDA Streamslaunching five kernels on two streams, Overlapping Kernel Execution with CUDA Streams-Overlapping Kernel Execution with CUDA Streamswarp specialization replaced with CUDA streams, Overlapping Kernel Execution with CUDA Streams-Overlapping Kernel Execution with CUDA Streams
- multi-GPU overlap of compute and data, Multi-GPU Compute and Data Transfer Overlap with CUDA Streams-Multi-GPU Compute and Data Transfer Overlap with CUDA Streams
- Programmatic Dependent Launch, Programmatic Dependent Launch-Programmatic Dependent Launchthread block clusters and warp specialization, Combining PDL and Thread Block Clusters with Warp Specialization-Combining PDL and Thread Block Clusters with Warp Specialization
- synchronization with events and callbacks, Fine-Grained Synchronization with Events and Callbackscross-stream synchronization, Using CUDA Events for Cross-Stream Synchronization
- warp specializationCUDA streams, Pipelining with Warp Specialization (Intra-Kernel) and CUDA Streams (Inter-Kernel)-Pipelining with Warp Specialization (Intra-Kernel) and CUDA Streams (Inter-Kernel)thread block clusters and CUDA streams, Warp Specialization with Thread Block Clusters and CUDA Streams

- concurrency, Overlapping Communication and Computation-Overlapping Communication and Computation
- multiple GPUs, Multi-GPU Compute and Data Transfer Overlap with CUDA Streams-Multi-GPU Compute and Data Transfer Overlap with CUDA Streams

- using, Using CUDA Streams and Stream-Ordered Memory Allocator with LLMs-Using CUDA Streams and Stream-Ordered Memory Allocator with LLMs
- using with multiple GPUs, Multi-GPU Compute and Data Transfer Overlap with CUDA Streams-Multi-GPU Compute and Data Transfer Overlap with CUDA Streams

- default versus explicit streams, Default Versus Explicit (Nondefault) Streams
- legacy default streams, Legacy Default Stream
- per-thread default streams, Modern Per-Thread Default Stream
- using default streams, Best Practices for Default Stream Usage-Best Practices for Default Stream Usage

- launching five kernels on two streams, Overlapping Kernel Execution with CUDA Streams-Overlapping Kernel Execution with CUDA Streams
- warp specialization replaced with CUDA streams, Overlapping Kernel Execution with CUDA Streams-Overlapping Kernel Execution with CUDA Streams

- thread block clusters and warp specialization, Combining PDL and Thread Block Clusters with Warp Specialization-Combining PDL and Thread Block Clusters with Warp Specialization

- cross-stream synchronization, Using CUDA Events for Cross-Stream Synchronization

- CUDA streams, Pipelining with Warp Specialization (Intra-Kernel) and CUDA Streams (Inter-Kernel)-Pipelining with Warp Specialization (Intra-Kernel) and CUDA Streams (Inter-Kernel)
- thread block clusters and CUDA streams, Warp Specialization with Thread Block Clusters and CUDA Streams

- irqbalance daemon, Scheduler and Interrupt Affinity
- scheduler and interrupt affinity, Scheduler and Interrupt Affinity

- about intra-kernel pipelining, Intra-Kernel Pipelining Techniques
- comparing techniques, Intra-Kernel Pipelining Techniques, Using CUDA Pipeline API for Warp Specialization, Warp Specialization with Thread Block Clusters-Warp Specialization with Thread Block Clusters
- cooperative groups, Cooperative Groups-When to Combine Persistent Kernels and Cooperative GroupsCooperative Groups API, Warp Shuffle Intrinsics: Avoid Shared Memory and Explicit Synchronization, Cooperative Groups, Coordinating Thread Block Clusters with Cooperative Groups API-Coordinating Thread Block Clusters with Cooperative Groups APIdescribed, Thread Block Clusters and Distributed Shared Memorylaunching a kernel in cooperative mode, Cooperative Groups, When to Combine Persistent Kernels and Cooperative Groupspersistent kernels via grid sync, Cooperative Grid Synchronization and Persistent Kernels-Cooperative Grid Synchronization and Persistent Kernelsthread block clusters versus, Thread Block Clusters and Distributed Shared Memorywhen to combine with persistent kernels, When to Combine Persistent Kernels and Cooperative Groups
- double-buffering with CUDA Pipeline API, Cooperative Tiling and Double-Buffering with the CUDA Pipeline API-Cooperative Tiling and Double-Buffering with the CUDA Pipeline API
- persistent kernels, Persistent Kernels and Megakernels-Persistent Kernels and Warp Specializationcommon workloads for, Common Workloads for Persistent Kernelscooperative groups via grid sync, Cooperative Grid Synchronization and Persistent Kernels-Cooperative Grid Synchronization and Persistent Kernelsmegakernels, Megakernels for Inferenceoccupancy, Persistent Kernels and Megakernelswarp specialization, Persistent Kernels and Warp Specializationwhen to combine with cooperative groups, When to Combine Persistent Kernels and Cooperative Groups
- thread block clusters, Intra-Kernel Pipelining, Warp Specialization, and Cooperative Thread Block Clusters(see also thread block clusters)
- warp specialization, Warp Specialization and the Producer-Consumer Model-Warp Specialization and the Producer-Consumer ModelCUDA Pipeline API, Using CUDA Pipeline API for Warp Specialization-Using CUDA Pipeline API for Warp SpecializationCUDA Pipeline API in PyTorch, PyTorch, CUDA Pipeline API, and Warp SpecializationCUDA streams, Pipelining with Warp Specialization (Intra-Kernel) and CUDA Streams (Inter-Kernel)-Pipelining with Warp Specialization (Intra-Kernel) and CUDA Streams (Inter-Kernel)thread block clusters, Warp Specialization with Thread Block Clusters-Warp Specialization with Thread Block Clusters
- when to use warp specialization versus double-buffering, Warp Specialization and the Producer-Consumer Model

- Cooperative Groups API, Warp Shuffle Intrinsics: Avoid Shared Memory and Explicit Synchronization, Cooperative Groups, Coordinating Thread Block Clusters with Cooperative Groups API-Coordinating Thread Block Clusters with Cooperative Groups API
- described, Thread Block Clusters and Distributed Shared Memory
- launching a kernel in cooperative mode, Cooperative Groups, When to Combine Persistent Kernels and Cooperative Groups
- persistent kernels via grid sync, Cooperative Grid Synchronization and Persistent Kernels-Cooperative Grid Synchronization and Persistent Kernels
- thread block clusters versus, Thread Block Clusters and Distributed Shared Memory
- when to combine with persistent kernels, When to Combine Persistent Kernels and Cooperative Groups

- common workloads for, Common Workloads for Persistent Kernels
- cooperative groups via grid sync, Cooperative Grid Synchronization and Persistent Kernels-Cooperative Grid Synchronization and Persistent Kernels
- megakernels, Megakernels for Inference
- occupancy, Persistent Kernels and Megakernels
- warp specialization, Persistent Kernels and Warp Specialization
- when to combine with cooperative groups, When to Combine Persistent Kernels and Cooperative Groups

- (see also thread block clusters)

- CUDA Pipeline API, Using CUDA Pipeline API for Warp Specialization-Using CUDA Pipeline API for Warp Specialization
- CUDA Pipeline API in PyTorch, PyTorch, CUDA Pipeline API, and Warp Specialization
- CUDA streams, Pipelining with Warp Specialization (Intra-Kernel) and CUDA Streams (Inter-Kernel)-Pipelining with Warp Specialization (Intra-Kernel) and CUDA Streams (Inter-Kernel)
- thread block clusters, Warp Specialization with Thread Block Clusters-Warp Specialization with Thread Block Clusters

- communication versus compute bottlenecks, Diagnosing Communication- Versus Compute-Bound Workloads
- continuous profiling and tuning workflow, Continuous Profiling and Tuning Workflow-Continuous Profiling and Tuning Workflow
- data pipeline tuning, Tuning the Data Pipelinedata loading and preprocessing, Efficient Data Loading and Preprocessing-Efficient Data Loading and PreprocessingNeMo Curator for training datasets, Creating High-Quality LLM Datasets with NVIDIA NeMo CuratorNVIDIA Data Loading Library, Multimodal Data Processing with NVIDIA DALIscaling out workers as GPUs scaled, Scaling Out Workers as You Scale Out Number of GPUs
- I/O isolation in Kubernetes, Dealing with I/O Isolation
- monitoring, Monitoring Storage I/O-Monitoring Storage I/O
- NVIDIA Magnum IO, NVIDIA Magnum IO Optimization Stack
- optimizing GPU-based storagecloud storage caches, Distributed, Parallel Filesystems and Object Storescompressing data, Tuning, Replicating, and Compressing DataDeepSeek Fire-Flyer File System, DeepSeek’s Fire-Flyer File System-DeepSeek’s Fire-Flyer File Systemfast storage and data locality, Fast Storage and Data Localityfilesystem optimized, Sequential Versus Random Read Patternsmultiqueue block I/O scheduler, Tuning NVMe and Filesystem for ThroughputNFS servers, Distributed, Parallel Filesystems and Object StoresNFS servers tuning parameters, Distributed, Parallel Filesystems and Object StoresNVIDIA GDS, Using NVIDIA GDS-Measuring GDS with gdsioNVMe and filesystem tuned for throughput, Tuning NVMe and Filesystem for Throughputparallel filesystems, Distributed, Parallel Filesystems and Object Storesread ahead, Tuning NVMe and Filesystem for Throughputread size tuned, Sequential Versus Random Read Patternssequential versus random reads, Sequential Versus Random Read Patterns-Sequential Versus Random Read Patternsshared filesystems, Distributed, Parallel Filesystems and Object Storesstriping files, Distributed, Parallel Filesystems and Object Storestuning filesystems, Tuning, Replicating, and Compressing Data

- data loading and preprocessing, Efficient Data Loading and Preprocessing-Efficient Data Loading and Preprocessing
- NeMo Curator for training datasets, Creating High-Quality LLM Datasets with NVIDIA NeMo Curator
- NVIDIA Data Loading Library, Multimodal Data Processing with NVIDIA DALI
- scaling out workers as GPUs scaled, Scaling Out Workers as You Scale Out Number of GPUs

- cloud storage caches, Distributed, Parallel Filesystems and Object Stores
- compressing data, Tuning, Replicating, and Compressing Data
- DeepSeek Fire-Flyer File System, DeepSeek’s Fire-Flyer File System-DeepSeek’s Fire-Flyer File System
- fast storage and data locality, Fast Storage and Data Locality
- filesystem optimized, Sequential Versus Random Read Patterns
- multiqueue block I/O scheduler, Tuning NVMe and Filesystem for Throughput
- NFS servers, Distributed, Parallel Filesystems and Object Stores
- NFS servers tuning parameters, Distributed, Parallel Filesystems and Object Stores
- NVIDIA GDS, Using NVIDIA GDS-Measuring GDS with gdsio
- NVMe and filesystem tuned for throughput, Tuning NVMe and Filesystem for Throughput
- parallel filesystems, Distributed, Parallel Filesystems and Object Stores
- read ahead, Tuning NVMe and Filesystem for Throughput
- read size tuned, Sequential Versus Random Read Patterns
- sequential versus random reads, Sequential Versus Random Read Patterns-Sequential Versus Random Read Patterns
- shared filesystems, Distributed, Parallel Filesystems and Object Stores
- striping files, Distributed, Parallel Filesystems and Object Stores
- tuning filesystems, Tuning, Replicating, and Compressing Data

### J

- jemalloc, Tune Host CPU Memory Allocator
- job scheduling with Kubernetes and SLURM, Job Scheduling with Kubernetes and SLURM

### K

- K8s (see Kubernetes (K8s))
- kernel autotuning (see autotuning)
- kernel caching, Continuous Prewarming of CUDA Graphs and Caches Using Time-Series Prediction
- kernel fusionarithmetic intensity increased, Kernel Fusioncustom fusion less necessary with PyTorch, FlexDecoding (PyTorch)FlexAttention, Autotuning with TorchInductor, FlexDecoding (PyTorch)memory-bound workloads, Roofline-Guided Scheduling and Orchestration DecisionsTorchInductor, Tuning Occupancy with PyTorch, Kernel Fusion, TorchDynamo for Bytecode Capture and Graph Extractionfused attention kernels, PyTorch, CUDA Pipeline API, and Warp SpecializationGEMM kernel prologue and epilogue fusion, Using the PyTorch Compiler
- kernel launch overhead reduced with CUDA Graphs, Reducing Kernel Launch Overhead with CUDA Graphs-CUDA Graph Trees (PyTorch Compiler Internal)best practices for CUDA Graphs, Best Practices for CUDA Graphs-Best Practices for CUDA Graphscapturing a CUDA Graph, Capturing a CUDA Graph and Preallocating Memory-Capturing a CUDA Graph and Preallocating Memoryreplaying the graph, Replaying the Graph-Replaying the Graphany stream for replay, Capturing a CUDA Graph and Preallocating Memory
- kernel optimization, Optimizing the Kernel-Optimizing the Kernelautomated, Automated GPU Kernel Optimizations with DeepSeek-R1 (NVIDIA)-Automated GPU Kernel Optimizations with DeepSeek-R1 (NVIDIA)reinforcement learning approach, Reinforcement Learning Approach to Generating Optimized GPU Kernels (Predibase)-Reinforcement Learning Approach to Generating Optimized GPU Kernels (Predibase)runtime kernel patching, Runtime Kernel Performance Improvements and Hot-Swappable Implementations-Runtime Kernel Performance Improvements and Hot-Swappable Implementations
- Kimi K2 (Moonshot AI), Scaling Toward Multimillion GPU Clusters and 100-Trillion-Parameter Models
- Kineto (PyTorch profiler), PyTorch Profiler and Visualization Tools, NVTX Markers and Profiling Tools, NVTX Markers and Profiling ToolsNCCL profiler plugin, Profiling and Debugging NCCL
- Kubernetes (K8s)about, Kubernetes for Topology-Aware Container Orchestration and Networkingcluster scheduler, Sharing and Schedulingdeploying a disaggregated inference system, Deploying Disaggregated Prefill and Decode with Kubernetes-Deploying Disaggregated Prefill and Decode with Kubernetes, Full-Stack Inference Optimizationshost machine should be tuned, Dealing with I/O IsolationI/O isolation, Dealing with I/O Isolationimproving resource guarantees, Improving Resource Guaranteesjob scheduling, Job Scheduling with Kubernetes and SLURMmemory isolation, Memory Isolation and Avoiding the OOM Killermemory limit, Memory Isolation and Avoiding the OOM KillerMIG devices in device plugin, MIGMIG mode, Slicing a GPU with MIGno swap and enforcing NUMA affinity, Virtual Memory and SwappingNVIDIA Container Toolkit, Container Runtime Optimizations for GPUsNVIDIA device plugin for Kubernetes, Kubernetes for Topology-Aware Container Orchestration and NetworkingNVIDIA Kubernetes GPU Operator, Kubernetes for Topology-Aware Container Orchestration and NetworkingNVL72 deployment, Preintegrated Rack Applianceoptimizing network communication, Optimizing Network Communication for Kubernetesorchestrating containers with Topology Manager, Orchestrating Containers with Kubernetes Topology Managerorchestration jitter reduced, Reducing Kubernetes Orchestration JitterRDMA, High-Speed, Low-Overhead Data Transfers with RDMAruntime optimizationsabout containers, Container Runtime Optimizations for GPUscontainer overlay filesystem overhead, Avoiding Container Overlay Filesystem OverheadNVIDIA container runtime injecting libraries, NVIDIA Container RuntimeNVIDIA Container Toolkit and CUDA compatibility, NVIDIA Container Toolkit and CUDA Compatibilityreducing image size, Reduce Image Size for Faster Container Startupruntime optimizations for GPUs, Container Runtime Optimizations for GPUs-Reduce Image Size for Faster Container Startuptopology awareness, Kubernetes for Topology-Aware Container Orchestration and NetworkingTopology Manager, Orchestrating Containers with Kubernetes Topology Manager
- Kubernetes Topology Manager (NVIDIA), Orchestrating Containers with Kubernetes Topology Manager
- KV cache, NVIDIA’s NIXL and Disaggregated Inferencecompression and policy switching, Real-Time KV Cache Compression and Policy Switching-Real-Time KV Cache Compression and Policy Switchinginference tuning, KV Cache Offloading and Memory Pool AllocationNIXL streamlining transfers, NVIDIA’s NIXL and Disaggregated Inferenceintelligent interconnect for transfers, Intelligent Interconnect Routing for KV Cache Transfersoffloading to CPU memory, NVIDIA’s NIXL and Disaggregated Inference, KV Cache Offloading with NIXL-KV Cache Offloading with NIXLprefill and decode inference stages, Separate Prefill and Decode Inference Stages-Separate Prefill and Decode Inference Stagesprefill to decode data transfers, KV Cache Data Transfer and NIXL, Fast KV Cache Transfer Between Prefill and Decode-Zero-Copy GPU-to-GPU Transferprefill phase, Scaling Disaggregated Prefill and Decode for Inferenceprefix merge events, Profiling, Debugging, and Tuning Inference Performancespeculative KV prefetching, Speculative KV Prefetching for Faster TTFT-Speculative KV Prefetching for Faster TTFTKV cache offloading, Speculative KV Prefetching for Faster TTFTsuperchip memory bandwidth, GPU and CPU-GPU Superchip Improvementstransfer from prefill to decode, KV Cache Data Transfer and NIXL, Fast KV Cache Transfer Between Prefill and Decode-Zero-Copy GPU-to-GPU Transferconnector and data path design, Connector and Data Path Design-Connector and Data Path DesignKV cache size, KV Cache Sizezero-copy GPU-to-GPU, Zero-Copy GPU-to-GPU Transfer-Zero-Copy GPU-to-GPU Transfertuning, Tuning KV Cache Utilization and Management-GPU and CPU-GPU Superchip Improvementsdisaggregated KV cache pool, Disaggregated KV Cache Pool-Disaggregated KV Cache Poolmemory layout, Optimized KV Cache Memory Layoutreuse and prefix sharing, KV Cache Reuse and Prefix Sharing-KV Cache Reuse and Prefix SharingvLLM PagedAttention, Continuous Scheduling, KV Cache Offloading and Memory Pool Allocation
- KV Cache Manager (NVIDIA Dynamo), KV Cache Offloading with NIXL-KV Cache Offloading with NIXL

- arithmetic intensity increased, Kernel Fusion
- custom fusion less necessary with PyTorch, FlexDecoding (PyTorch)
- FlexAttention, Autotuning with TorchInductor, FlexDecoding (PyTorch)
- memory-bound workloads, Roofline-Guided Scheduling and Orchestration Decisions
- TorchInductor, Tuning Occupancy with PyTorch, Kernel Fusion, TorchDynamo for Bytecode Capture and Graph Extractionfused attention kernels, PyTorch, CUDA Pipeline API, and Warp SpecializationGEMM kernel prologue and epilogue fusion, Using the PyTorch Compiler

- fused attention kernels, PyTorch, CUDA Pipeline API, and Warp Specialization
- GEMM kernel prologue and epilogue fusion, Using the PyTorch Compiler

- best practices for CUDA Graphs, Best Practices for CUDA Graphs-Best Practices for CUDA Graphs
- capturing a CUDA Graph, Capturing a CUDA Graph and Preallocating Memory-Capturing a CUDA Graph and Preallocating Memory
- replaying the graph, Replaying the Graph-Replaying the Graphany stream for replay, Capturing a CUDA Graph and Preallocating Memory

- any stream for replay, Capturing a CUDA Graph and Preallocating Memory

- automated, Automated GPU Kernel Optimizations with DeepSeek-R1 (NVIDIA)-Automated GPU Kernel Optimizations with DeepSeek-R1 (NVIDIA)
- reinforcement learning approach, Reinforcement Learning Approach to Generating Optimized GPU Kernels (Predibase)-Reinforcement Learning Approach to Generating Optimized GPU Kernels (Predibase)
- runtime kernel patching, Runtime Kernel Performance Improvements and Hot-Swappable Implementations-Runtime Kernel Performance Improvements and Hot-Swappable Implementations

- NCCL profiler plugin, Profiling and Debugging NCCL

- about, Kubernetes for Topology-Aware Container Orchestration and Networking
- cluster scheduler, Sharing and Scheduling
- deploying a disaggregated inference system, Deploying Disaggregated Prefill and Decode with Kubernetes-Deploying Disaggregated Prefill and Decode with Kubernetes, Full-Stack Inference Optimizations
- host machine should be tuned, Dealing with I/O Isolation
- I/O isolation, Dealing with I/O Isolation
- improving resource guarantees, Improving Resource Guarantees
- job scheduling, Job Scheduling with Kubernetes and SLURM
- memory isolation, Memory Isolation and Avoiding the OOM Killer
- memory limit, Memory Isolation and Avoiding the OOM Killer
- MIG devices in device plugin, MIG
- MIG mode, Slicing a GPU with MIG
- no swap and enforcing NUMA affinity, Virtual Memory and Swapping
- NVIDIA Container Toolkit, Container Runtime Optimizations for GPUs
- NVIDIA device plugin for Kubernetes, Kubernetes for Topology-Aware Container Orchestration and Networking
- NVIDIA Kubernetes GPU Operator, Kubernetes for Topology-Aware Container Orchestration and Networking
- NVL72 deployment, Preintegrated Rack Appliance
- optimizing network communication, Optimizing Network Communication for Kubernetes
- orchestrating containers with Topology Manager, Orchestrating Containers with Kubernetes Topology Manager
- orchestration jitter reduced, Reducing Kubernetes Orchestration Jitter
- RDMA, High-Speed, Low-Overhead Data Transfers with RDMA
- runtime optimizationsabout containers, Container Runtime Optimizations for GPUscontainer overlay filesystem overhead, Avoiding Container Overlay Filesystem OverheadNVIDIA container runtime injecting libraries, NVIDIA Container RuntimeNVIDIA Container Toolkit and CUDA compatibility, NVIDIA Container Toolkit and CUDA Compatibilityreducing image size, Reduce Image Size for Faster Container Startup
- runtime optimizations for GPUs, Container Runtime Optimizations for GPUs-Reduce Image Size for Faster Container Startup
- topology awareness, Kubernetes for Topology-Aware Container Orchestration and Networking
- Topology Manager, Orchestrating Containers with Kubernetes Topology Manager

- about containers, Container Runtime Optimizations for GPUs
- container overlay filesystem overhead, Avoiding Container Overlay Filesystem Overhead
- NVIDIA container runtime injecting libraries, NVIDIA Container Runtime
- NVIDIA Container Toolkit and CUDA compatibility, NVIDIA Container Toolkit and CUDA Compatibility
- reducing image size, Reduce Image Size for Faster Container Startup

- compression and policy switching, Real-Time KV Cache Compression and Policy Switching-Real-Time KV Cache Compression and Policy Switching
- inference tuning, KV Cache Offloading and Memory Pool Allocation
- NIXL streamlining transfers, NVIDIA’s NIXL and Disaggregated Inferenceintelligent interconnect for transfers, Intelligent Interconnect Routing for KV Cache Transfersoffloading to CPU memory, NVIDIA’s NIXL and Disaggregated Inference, KV Cache Offloading with NIXL-KV Cache Offloading with NIXLprefill and decode inference stages, Separate Prefill and Decode Inference Stages-Separate Prefill and Decode Inference Stagesprefill to decode data transfers, KV Cache Data Transfer and NIXL, Fast KV Cache Transfer Between Prefill and Decode-Zero-Copy GPU-to-GPU Transfer
- prefill phase, Scaling Disaggregated Prefill and Decode for Inference
- prefix merge events, Profiling, Debugging, and Tuning Inference Performance
- speculative KV prefetching, Speculative KV Prefetching for Faster TTFT-Speculative KV Prefetching for Faster TTFTKV cache offloading, Speculative KV Prefetching for Faster TTFT
- superchip memory bandwidth, GPU and CPU-GPU Superchip Improvements
- transfer from prefill to decode, KV Cache Data Transfer and NIXL, Fast KV Cache Transfer Between Prefill and Decode-Zero-Copy GPU-to-GPU Transferconnector and data path design, Connector and Data Path Design-Connector and Data Path DesignKV cache size, KV Cache Sizezero-copy GPU-to-GPU, Zero-Copy GPU-to-GPU Transfer-Zero-Copy GPU-to-GPU Transfer
- tuning, Tuning KV Cache Utilization and Management-GPU and CPU-GPU Superchip Improvementsdisaggregated KV cache pool, Disaggregated KV Cache Pool-Disaggregated KV Cache Poolmemory layout, Optimized KV Cache Memory Layoutreuse and prefix sharing, KV Cache Reuse and Prefix Sharing-KV Cache Reuse and Prefix Sharing
- vLLM PagedAttention, Continuous Scheduling, KV Cache Offloading and Memory Pool Allocation

- intelligent interconnect for transfers, Intelligent Interconnect Routing for KV Cache Transfers
- offloading to CPU memory, NVIDIA’s NIXL and Disaggregated Inference, KV Cache Offloading with NIXL-KV Cache Offloading with NIXL
- prefill and decode inference stages, Separate Prefill and Decode Inference Stages-Separate Prefill and Decode Inference Stages
- prefill to decode data transfers, KV Cache Data Transfer and NIXL, Fast KV Cache Transfer Between Prefill and Decode-Zero-Copy GPU-to-GPU Transfer

- KV cache offloading, Speculative KV Prefetching for Faster TTFT

- connector and data path design, Connector and Data Path Design-Connector and Data Path Design
- KV cache size, KV Cache Size
- zero-copy GPU-to-GPU, Zero-Copy GPU-to-GPU Transfer-Zero-Copy GPU-to-GPU Transfer

- disaggregated KV cache pool, Disaggregated KV Cache Pool-Disaggregated KV Cache Pool
- memory layout, Optimized KV Cache Memory Layout
- reuse and prefix sharing, KV Cache Reuse and Prefix Sharing-KV Cache Reuse and Prefix Sharing

### L

- latencydisaggregation of prefill and decode, Impact on Latency (TTFT) and Throughput (TPOT)GPU utilization versus, Maximizing GPU Utilization and Throughput Versus Latency Trade-Offslatency-aware scheduling and dynamic routing, Latency-Aware Scheduling and Dynamic Routing-Latency-Aware Scheduling and Dynamic Routinglatency-bound GPU stalls, Iteratively Profiling and Determining the Kernel Bottleneck-Iteratively Profiling and Determining the Kernel Bottleneckkernel optimization, Optimizing the Kernel-Optimizing the Kernellatency-sensitive training workflows, Filesystem Caching and Write-Backmaximum latency for dynamic batching, Monitoring System Metrics and Countersprofiling when streaming enabled, Streaming Responses
- latency hiding, Streaming Multiprocessor, Threads, and Warps, Maintaining High Occupancy and GPU Utilizationbetween kernels or GPU and host (see inter-kernel pipelining)higher occupancy to better hide latency, Tuning Occupancyhigh occupancy plus high ILP, Loop Unrolling, Interleaving, and Compiler Hintinglow achieved occupancy, Inspecting Achieved Occupancy and GPU Utilizationwithin a kernel (see intra-kernel pipelining)
- __launch_bounds__, Tuning Occupancy with Launch Bounds-Tuning Occupancy with Launch Bounds, Techniques for Occupancy Tuningoccupancy optimization, Compiler Hints to Optimize Occupancy
- LazyLLM, Dynamic Token Pruning with LazyLLM
- __ldg(), Read-Only Data Caches
- LinuxCPU DRAM as CPU NUMA memory, NUMA Awareness and CPU PinningGPU driver, GPU DriverGPU servers typically Linux, Operating Systemmemory swapping avoided, Virtual Memory and SwappingNUMA balancing, NUMA Awareness and CPU PinningnumactlCPU pinning, NUMA Awareness and CPU Pinning, NUMA Awareness and CPU Pinningforcing memory allocation from specific NUMA node, NUMA Awareness and CPU Pinning, NUMA-Friendly Memory Allocation and Memory Pinningprocess CPU and memory policies applied on launch, NUMA-Friendly Memory Allocation and Memory PinningNVMeDeepSeek Fire-Flyer File System, DeepSeek’s Fire-Flyer File System-DeepSeek’s Fire-Flyer File Systemfast storage and data locality, Fast Storage and Data LocalityGPUDirect Storage, NUMA-Friendly Memory Allocation and Memory Pinning, Using NVIDIA GDS-Measuring GDS with gdsioparameters offloaded to, Offloading Parameters to CPU and NVMesequential versus random reads, Sequential Versus Random Read Patterns-Sequential Versus Random Read Patternstuning for throughput, Tuning NVMe and Filesystem for ThroughputXFS optimized, Sequential Versus Random Read Patternsperf, NVTX Markers and Profiling Tools, NVTX Markers and Profiling Toolsexample MoE model, CPU and GPU Profiling with Linux perf-CPU and GPU Profiling with Linux perfNVIDIA Performance Monitoring Unit, CPU and GPU Profiling with Linux perfschedulerscompletely fair queueing obsolete, Tuning NVMe and Filesystem for ThroughputCompletely Fair Scheduler, Scheduler and Interrupt Affinitymultiqueue block I/O scheduler, Tuning NVMe and Filesystem for Throughputnone scheduler, Tuning NVMe and Filesystem for Throughputsetting for, Tuning NVMe and Filesystem for ThroughputSimple Linux Utility for Resource Management, Preintegrated Rack Appliancecluster scheduler, Sharing and Scheduling
- liquid cooling versus air cooling, Liquid Cooling Versus Air Cooling-Liquid Cooling Versus Air Cooling
- Little Red Whiting Hood, Acknowledgments
- LLM request nonuniformity, Monitoring System Metrics and Counters
- LMCache (vLLM), Profiling, Debugging, and Tuning Inference Performancedisaggregated prefill and decode, Scaling Disaggregated Prefill and Decode for Inference, Zero-Copy GPU-to-GPU Transfer
- load balancingdynamic scheduling, Dynamic Scheduling and Load Balancing-Dynamic resource scalingadaptive resource scheduling and hotspot prevention, Adaptive Resource Scheduling and Hotspot Prevention-Dynamic resource scalingArrow’s adaptive instance scaling, Arrow’s adaptive instance scaling-Arrow’s adaptive instance scalingdynamic resource scaling, Dynamic resource scalingMooncake adaptive strategies, Mooncake adaptive strategiesTetriInfer’s two-level scheduler, TetriInfer’s two-level schedulerexpert parallelism, Expert ParallelismMoE expert rebalancing, MoE Expert Rebalancing and Regroupingtemporal load balancing, Dynamic Congestion-Aware Scheduling
- logginginference monitoring, Monitoring System Metrics and CountersMLPerf for performance benchmarks, Performance Benchmarks and MLPerf Logging-Performance Benchmarks and MLPerf LoggingNCCL logs, In-Network SHARP Aggregation, Debugging Correctness IssuesPyTorch compiler, Debugging Compiler Phases, Graph Breaks, and PerformanceTORCH_LOGS, Profiling and Debugging Compiler Performance Issues, Debugging Compiler Phases, Graph Breaks, and Performancelogging options summary, Debugging Compiler Phases, Graph Breaks, and Performanceperformance hints, Performance Hints and Debugging Generated Code
- loop unrolling, Loop Unrolling, Interleaving, and Compiler Hinting, Kernel Fusion

- disaggregation of prefill and decode, Impact on Latency (TTFT) and Throughput (TPOT)
- GPU utilization versus, Maximizing GPU Utilization and Throughput Versus Latency Trade-Offs
- latency-aware scheduling and dynamic routing, Latency-Aware Scheduling and Dynamic Routing-Latency-Aware Scheduling and Dynamic Routing
- latency-bound GPU stalls, Iteratively Profiling and Determining the Kernel Bottleneck-Iteratively Profiling and Determining the Kernel Bottleneckkernel optimization, Optimizing the Kernel-Optimizing the Kernel
- latency-sensitive training workflows, Filesystem Caching and Write-Back
- maximum latency for dynamic batching, Monitoring System Metrics and Counters
- profiling when streaming enabled, Streaming Responses

- kernel optimization, Optimizing the Kernel-Optimizing the Kernel

- between kernels or GPU and host (see inter-kernel pipelining)
- higher occupancy to better hide latency, Tuning Occupancyhigh occupancy plus high ILP, Loop Unrolling, Interleaving, and Compiler Hinting
- low achieved occupancy, Inspecting Achieved Occupancy and GPU Utilization
- within a kernel (see intra-kernel pipelining)

- high occupancy plus high ILP, Loop Unrolling, Interleaving, and Compiler Hinting

- occupancy optimization, Compiler Hints to Optimize Occupancy

- CPU DRAM as CPU NUMA memory, NUMA Awareness and CPU Pinning
- GPU driver, GPU Driver
- GPU servers typically Linux, Operating System
- memory swapping avoided, Virtual Memory and Swapping
- NUMA balancing, NUMA Awareness and CPU Pinning
- numactlCPU pinning, NUMA Awareness and CPU Pinning, NUMA Awareness and CPU Pinningforcing memory allocation from specific NUMA node, NUMA Awareness and CPU Pinning, NUMA-Friendly Memory Allocation and Memory Pinningprocess CPU and memory policies applied on launch, NUMA-Friendly Memory Allocation and Memory Pinning
- NVMeDeepSeek Fire-Flyer File System, DeepSeek’s Fire-Flyer File System-DeepSeek’s Fire-Flyer File Systemfast storage and data locality, Fast Storage and Data LocalityGPUDirect Storage, NUMA-Friendly Memory Allocation and Memory Pinning, Using NVIDIA GDS-Measuring GDS with gdsioparameters offloaded to, Offloading Parameters to CPU and NVMesequential versus random reads, Sequential Versus Random Read Patterns-Sequential Versus Random Read Patternstuning for throughput, Tuning NVMe and Filesystem for ThroughputXFS optimized, Sequential Versus Random Read Patterns
- perf, NVTX Markers and Profiling Tools, NVTX Markers and Profiling Toolsexample MoE model, CPU and GPU Profiling with Linux perf-CPU and GPU Profiling with Linux perfNVIDIA Performance Monitoring Unit, CPU and GPU Profiling with Linux perf
- schedulerscompletely fair queueing obsolete, Tuning NVMe and Filesystem for ThroughputCompletely Fair Scheduler, Scheduler and Interrupt Affinitymultiqueue block I/O scheduler, Tuning NVMe and Filesystem for Throughputnone scheduler, Tuning NVMe and Filesystem for Throughputsetting for, Tuning NVMe and Filesystem for Throughput
- Simple Linux Utility for Resource Management, Preintegrated Rack Appliancecluster scheduler, Sharing and Scheduling

- CPU pinning, NUMA Awareness and CPU Pinning, NUMA Awareness and CPU Pinning
- forcing memory allocation from specific NUMA node, NUMA Awareness and CPU Pinning, NUMA-Friendly Memory Allocation and Memory Pinning
- process CPU and memory policies applied on launch, NUMA-Friendly Memory Allocation and Memory Pinning

- DeepSeek Fire-Flyer File System, DeepSeek’s Fire-Flyer File System-DeepSeek’s Fire-Flyer File System
- fast storage and data locality, Fast Storage and Data Locality
- GPUDirect Storage, NUMA-Friendly Memory Allocation and Memory Pinning, Using NVIDIA GDS-Measuring GDS with gdsio
- parameters offloaded to, Offloading Parameters to CPU and NVMe
- sequential versus random reads, Sequential Versus Random Read Patterns-Sequential Versus Random Read Patterns
- tuning for throughput, Tuning NVMe and Filesystem for Throughput
- XFS optimized, Sequential Versus Random Read Patterns

- example MoE model, CPU and GPU Profiling with Linux perf-CPU and GPU Profiling with Linux perf
- NVIDIA Performance Monitoring Unit, CPU and GPU Profiling with Linux perf

- completely fair queueing obsolete, Tuning NVMe and Filesystem for Throughput
- Completely Fair Scheduler, Scheduler and Interrupt Affinity
- multiqueue block I/O scheduler, Tuning NVMe and Filesystem for Throughput
- none scheduler, Tuning NVMe and Filesystem for Throughput
- setting for, Tuning NVMe and Filesystem for Throughput

- cluster scheduler, Sharing and Scheduling

- disaggregated prefill and decode, Scaling Disaggregated Prefill and Decode for Inference, Zero-Copy GPU-to-GPU Transfer

- dynamic scheduling, Dynamic Scheduling and Load Balancing-Dynamic resource scalingadaptive resource scheduling and hotspot prevention, Adaptive Resource Scheduling and Hotspot Prevention-Dynamic resource scalingArrow’s adaptive instance scaling, Arrow’s adaptive instance scaling-Arrow’s adaptive instance scalingdynamic resource scaling, Dynamic resource scalingMooncake adaptive strategies, Mooncake adaptive strategiesTetriInfer’s two-level scheduler, TetriInfer’s two-level scheduler
- expert parallelism, Expert Parallelism
- MoE expert rebalancing, MoE Expert Rebalancing and Regrouping
- temporal load balancing, Dynamic Congestion-Aware Scheduling

- adaptive resource scheduling and hotspot prevention, Adaptive Resource Scheduling and Hotspot Prevention-Dynamic resource scaling
- Arrow’s adaptive instance scaling, Arrow’s adaptive instance scaling-Arrow’s adaptive instance scaling
- dynamic resource scaling, Dynamic resource scaling
- Mooncake adaptive strategies, Mooncake adaptive strategies
- TetriInfer’s two-level scheduler, TetriInfer’s two-level scheduler

- inference monitoring, Monitoring System Metrics and Counters
- MLPerf for performance benchmarks, Performance Benchmarks and MLPerf Logging-Performance Benchmarks and MLPerf Logging
- NCCL logs, In-Network SHARP Aggregation, Debugging Correctness Issues
- PyTorch compiler, Debugging Compiler Phases, Graph Breaks, and Performance
- TORCH_LOGS, Profiling and Debugging Compiler Performance Issues, Debugging Compiler Phases, Graph Breaks, and Performancelogging options summary, Debugging Compiler Phases, Graph Breaks, and Performanceperformance hints, Performance Hints and Debugging Generated Code

- logging options summary, Debugging Compiler Phases, Graph Breaks, and Performance
- performance hints, Performance Hints and Debugging Generated Code

### M

- Magnum IO (NVIDIA), NVIDIA Magnum IO Optimization Stack, CPU and GPU Profiling with Linux perf
- MALLOC_CONF, Tune Host CPU Memory Allocator
- massive modelsinference clusters, Data Parallelismparallelism strategies for serving, Parallelism Strategies for Serving Massive MoE Models-Hybrid Parallelismhybrid parallelism, Hybrid Parallelism-Hybrid Parallelism
- matmul (matrix multiplication)AsyncTP, TorchTitan, AsyncTP, AutoParallel, and SimpleFSDPDeepGEMM library FP8-optimized, Transparency and Reproducibilitypersistent Triton kernel, Tiled and Persistent GEMM Kernel (Triton)-Tiled and Persistent GEMM Kernel (Triton)
- mechanical sympathy, Mechanical Sympathy: Hardware-Software Codesign-Mechanical Sympathy: Hardware-Software Codesign
- Medusa multiple head speculative decoding, Speculative Decoding and Parallel Token Generation Techniquesmultitoken decoding, Multitoken Decoding with Medusa’s Multiple Heads-Multitoken Decoding with Medusa’s Multiple Heads
- megakernels, Megakernels for Inference
- Mellanox, In-Network Aggregations with NVIDIA SHARP
- memoryasynchronous memory transfers with CUTLASS, Using CUTLASS for Optimal Arithmetic Intensity and Tensor Core PerformanceBlackwell GPU, The CPU and GPU Superchipclose to CPU, close to GPU, NUMA-Friendly Memory Allocation and Memory PinningCPU-GPU interconnects on superchips, Topology Awareness in NCCLperformance, The CPU and GPU SuperchipCUDA stream ordered memory allocator, Stream-Ordered Memory Allocator-Stream-Ordered Memory Allocatormultiple GPUs, Multi-GPU Compute and Data Transfer Overlap with CUDA Streams-Multi-GPU Compute and Data Transfer Overlap with CUDA Streamsusing with CUDA streams, Using CUDA Streams and Stream-Ordered Memory Allocator with LLMs-Using CUDA Streams and Stream-Ordered Memory Allocator with LLMsCUDA Unified Memory, GPU Memory Oversubscription, Fragmentation, and Out-of-Memory Handlingprogramming CUDA, Unified Memory-Unified MemoryUnified CPU-GPU Memory, The CPU and GPU Superchipdistributed shared memory (see distributed shared memory (DSMEM))dynamic memory-allocation switching, Dynamic Memory-Allocation Switching (Slab Versus Caching Versus Stream-Ordered)-Dynamic Memory-Allocation Switching (Slab Versus Caching Versus Stream-Ordered)ECC mode for GPU memory, GPU Clock Speeds and ECCedge-oriented MoE memory budgeting, Edge-Oriented MoE Memory Budgetingfragmentation monitoring, Monitoring System Metrics and Countersdynamic memory-allocation switching, Dynamic Memory-Allocation Switching (Slab Versus Caching Versus Stream-Ordered)-Dynamic Memory-Allocation Switching (Slab Versus Caching Versus Stream-Ordered)GDS, NUMA-Friendly Memory Allocation and Memory Pinning, Using NVIDIA GDS-Measuring GDS with gdsio, Offloading Parameters to CPU and NVMeglobal memory traffic reduced, Reducing Global Memory Traffic with Thread Block Clusters-Reducing Global Memory Traffic with Thread Block Clusters(see also global memory)GPU FLOPs outpacing memory bandwidth, Maintaining High Occupancy and GPU UtilizationGPU memory access patternsavoiding shared memory, Warp Shuffle Intrinsics: Avoid Shared Memory and Explicit Synchronization-Warp Shuffle Intrinsics: Avoid Shared Memory and Explicit Synchronizationavoiding shared-memory bank conflicts, Avoid Shared-Memory Bank Conflicts-Avoid Shared-Memory Bank Conflictscoalesced versus uncoalesced global memory access, Coalesced Versus Uncoalesced Global Memory Access-Coalesced Versus Uncoalesced Global Memory Accessread-only data caches, Read-Only Data Caches-Read-Only Data Cachesread-only data caches pitfall, Read-Only Data Cachessymmetric memory, PyTorch Symmetric MemoryTensor Memory Accelerator for tile fetch, Asynchronous Memory Prefetching and Tensor Memory Accelerator-Asynchronous Memory Prefetching and Tensor Memory Acceleratortiling and data reuse via shared memory, Tiling and Data Reuse Using Shared Memory-Tiling and Data Reuse Using Shared Memoryvectorized memory access, Vectorized Memory Access-Vectorized Memory AccessGPU memory fragmentation and oversubscription, GPU Memory Oversubscription, Fragmentation, and Out-of-Memory Handling-GPU Memory Oversubscription, Fragmentation, and Out-of-Memory HandlingGPU memory fragmentation under UCX/RDMA, Pitfall #6: GPU memory fragmentation under UCX/RDMA-Pitfall #6: GPU memory fragmentation under UCX/RDMAGPU-to-GPU memory sharing with NVSHMEM, Fine-Grained GPU-to-GPU Memory Sharing with NVSHMEM-Fine-Grained GPU-to-GPU Memory Sharing with NVSHMEMGrace Blackwell Superchip, The CPU and GPU Superchip, Topology Awareness in NCCLGrace CPU, The CPU and GPU Superchiphigh-bandwidth memory, Scaling Toward Multimillion GPU Clusters and 100-Trillion-Parameter Modelshugepages, Transparent Hugepagestransparent hugepages, Transparent Hugepagesinference tuning, Memorykernel memory throughput versus HBM memory bandwidth, Kernel Memory Throughput Versus Peak HBM Memory BandwidthKubernetesmemory isolation, Memory Isolation and Avoiding the OOM Killermemory limit, Memory Isolation and Avoiding the OOM KillerKV cachememory layouts optimized, Optimized KV Cache Memory Layoutoffloaded to CPU memory, KV Cache Offloading with NIXL-KV Cache Offloading with NIXLmemory-bound workload example, Maintaining High Occupancy and GPU UtilizationNUMA-friendly allocation and pinning, NUMA-Friendly Memory Allocation and Memory Pinning-NUMA-Friendly Memory Allocation and Memory Pinningefficiency of pinned memory, NUMA-Friendly Memory Allocation and Memory Pinningmax locked memory setting, Transparent Hugepages, Virtual Memory and SwappingOS limit on pinned memory, NUMA-Friendly Memory Allocation and Memory Pinningpinned memory for data loaders, NUMA-Friendly Memory Allocation and Memory Pinningout of memory (OOM) errorsCPU + GPU architecture, The CPU and GPU Superchipmax locked memory setting, Transparent Hugepagesmemory fragmentation or excessive caching, GPU Memory Oversubscription, Fragmentation, and Out-of-Memory Handlingmonitoring GPU memory usage, GPU Memory Oversubscription, Fragmentation, and Out-of-Memory Handlingprofiling and tuning via PyTorch profiler, Profiling and Tuning Memory in PyTorch-Enabling Peer-to-Peer DMA and UCXactivation checkpointing, Activation Checkpointing for Memory SavingsCUDA memory allocator tuned, Tuning the CUDA Memory AllocatorFSDP automatic checkpointing, FSDP Automatic Checkpointing and Offloading-FSDP Automatic Checkpointing and OffloadingFSDP with tensor parallel and pipeline parallel, Combining FSDP with Tensor Parallel and Pipeline ParallelNCCL with UCX over multinode topologies, Enabling Peer-to-Peer DMA and UCXoffloading parameters to CPU and NVMe, Offloading Parameters to CPU and NVMepeer-to-peer DMA, Enabling Peer-to-Peer DMA and UCXpluggable memory allocators, Pluggable Memory Allocators and Cross-GPU Data Transfers-Pluggable Memory Allocators and Cross-GPU Data Transfersprogramming CUDAasynchronous memory allocation, Asynchronous Memory Allocation and Memory Pools-Asynchronous Memory Allocation and Memory PoolsCUDA streams, Asynchronous Memory Allocation and Memory PoolsGPU memory hierarchy, Understanding GPU Memory Hierarchy-Understanding GPU Memory Hierarchymemory pools, Asynchronous Memory Allocation and Memory Pools-Asynchronous Memory Allocation and Memory PoolsPyTorch memory allocator plugin, Pluggable Memory Allocators and Cross-GPU Data Transfers-Pluggable Memory Allocators and Cross-GPU Data Transfersrecomputation versus memory, Recomputation Versus Memory Trade-Offscratch memory, Scratch MemoryDSMEM, Scratch Memory(see also distributed shared memory (DSMEM))shared-memory exchanges disabled, Pitfall #3: Avoid overtuning or disabling NCCL features with environment variables(see also shared memory (SMEM))swapping avoided, Virtual Memory and Swappingmax locked memory setting, Virtual Memory and Swappingsymmetric memory, PyTorch Symmetric Memorytuning host CPU memory allocator, Tune Host CPU Memory AllocatorUnified CPU-GPU Memory, The CPU and GPU Superchip
- memory-bound GPU stalls, Iteratively Profiling and Determining the Kernel Bottleneck-Iteratively Profiling and Determining the Kernel Bottleneckkernel optimization, Optimizing the Kernel-Optimizing the Kernel
- MIG (Multi-Instance GPU; NVIDIA)GPU driver runtime settings, MIG-MIGGPU virtualization, Managing Resources EfficientlyKubernetes MIG mode, Slicing a GPU with MIGpersistence mode recommended for GPU, Slicing a GPU with MIGprofile naming, MIGsingle GPU split into smaller GPUs, Sharing and Schedulingwhen to use, MIG
- mixed precisionautomatic mixed precision, TF32 and Automatic Mixed Precision (PyTorch)-TF32 and Automatic Mixed Precision (PyTorch)reduced precision, BF16/FP16, FP8, and FP4 Reduced PrecisionTensor Cores, Mixed Precision and Utilizing Tensor Cores-Transformer Engine and TMEM in Depth
- mixture-of-experts (MoE) modelsabout MoE models, Profiling PyTorch to Identify Bottlenecksall experts active simultaneously, Expert Parallelismall-to-all communication, Expert Communication OptimizationCUDA streams, Using CUDA Streams with MoE ModelsDeepEP communication library, Transparency and ReproducibilityDeepSeek-V3, DeepSeek Scales to ~680-Billion Parameter Models Despite US Export Hardware Restrictions in China-DeepSeek Scales to ~680-Billion Parameter Models Despite US Export Hardware Restrictions in ChinaDualPipe parallelism algorithm, DeepSeek Scales to ~680-Billion Parameter Models Despite US Export Hardware Restrictions in Chinadynamic routing strategies for inference, Dynamic Routing Strategies for MoE Inference-Adaptive Expert Routing and Real-Time Monitoringexpert communication optimization, Expert Communication Optimization-Expert Communication Optimizationexpert replication, Load Balancing, Capacity Factor, and Expert Replicationload balancing and capacity factors, Load Balancing, Capacity Factor, and Expert Replicationedge-oriented memory budgeting, Edge-Oriented MoE Memory Budgetingexpert parallelism, Parallelism Strategies for Serving Massive MoE Models, Expert Parallelism-Expert Parallelismload balancing, Expert Parallelismmodel weights and data split over GPUs, Parallelism Strategies for Serving Massive MoE Modelsexpert rebalancing and regrouping, MoE Expert Rebalancing and Regroupinghybrid parallelism, Hybrid Parallelism-Hybrid ParallelismMoE efficiency, Toward 100-Trillion-Parameter Modelsopen MoE rivaling best closed models, DeepSeek Scales to ~680-Billion Parameter Models Despite US Export Hardware Restrictions in Chinamultinode inference needed, Multinode Inference, Parallelism, Decoding, and Routing Optimizationsparallelism strategies, Parallelism Strategies for Serving Massive MoE Models-Hybrid Parallelismprofiling, Profiling PyTorch to Identify Bottlenecks-CPU and GPU Profiling with Linux perfLinux perf on CPU and GPU, CPU and GPU Profiling with Linux perf-CPU and GPU Profiling with Linux perfNsight Systems and NVTX Timelines, System Profiling with Nsight Systems and NVTX Timelines-System Profiling with Nsight Systems and NVTX TimelinesPyTorch profiler, Using PyTorch Profiler-Using PyTorch Profilerroofline analysis for GEMM kernels, Kernel Roofline Analysis for General Matrix Multiply (GEMM)-Kernel Roofline Analysis for General Matrix Multiply (GEMM)sparse models reducing compute requirements, DeepSeek Scales to ~680-Billion Parameter Models Despite US Export Hardware Restrictions in China, Toward 100-Trillion-Parameter Modelsspeculative expert routing, Speculative MoE Expert Routing and Communication Reduction
- MLA (see DeepSeek Multi-Head Latent Attention (MLA))
- MLPerf (MLCommons), Transparency and Reproducibilitycaution on per-GPU results, Transparency and ReproducibilityLogging, Performance Benchmarks and MLPerf Logging-Performance Benchmarks and MLPerf LoggingMLPerf Inference, Transparency and ReproducibilityMLPerf Training, Transparency and Reproducibility
- MMA API (CUDA), Transformer Engine and TMEM in Depth-Transformer Engine and TMEM in Depth
- model cascading, Model Cascading and Tiered Model Deployment-Model Cascading and Tiered Model Deployment
- MoE (see mixture-of-experts (MoE) models)
- monitoringCPU utilization, Scaling Out Workers as You Scale Out Number of GPUsfilesystem I/O during training, Tuning, Replicating, and Compressing DataGPU memory usage, GPU Memory Oversubscription, Fragmentation, and Out-of-Memory Handlinggraph recompilations, TorchDynamo for Bytecode Capture and Graph Extraction, Minimize Graph Recompilationsinference system metrics and counters, Monitoring System Metrics and Counters-Monitoring System Metrics and Counterscorrectness issue debugging, Debugging Correctness Issues-Debugging Correctness Issuesproduction environment, Inference Troubleshooting Recipesmemory fragmentation, Monitoring System Metrics and Countersmodel rambling, Token Output Limits and Timeoutsmultiple-GPU NVLink utilization, Real-Time Link Telemetry and Monitoringnetwork throughput, Pitfall #4: Insufficient network bandwidth or misconfigured NICsperformance monitoring, Performance Monitoring and Utilization in Practicecommunication versus compute bottlenecks, Diagnosing Communication- Versus Compute-Bound WorkloadsGPUs near 100% utilized, Performance Monitoring and Utilization in PracticeI/O continuous profiling and tuning, Continuous Profiling and Tuning Workflow-Continuous Profiling and Tuning WorkflowNVLink usage, Performance Monitoring and Utilization in Practicepower monitoring, Performance Monitoring and Utilization in Practicestorage I/O, Monitoring Storage I/O-Monitoring Storage I/Ostraggler nodes or processes, Pitfall #5: Straggler nodes or processes
- Mooncakeadaptive strategies, Mooncake adaptive strategiesearly rejection, Early Rejection (Admission Control)
- Moonshot AI Kimi K2, Scaling Toward Multimillion GPU Clusters and 100-Trillion-Parameter Models
- MPS (NVIDIA), MPS-MPS
- multi-GPU (see multiple GPUs)
- Multi-Instance GPU (see MIG (Multi-Instance GPU; NVIDIA))
- multidie GPU GPC “partition”, Thread Block Clusters and Distributed Shared Memory(see also dual-die GPUs)
- multilevel tiling, Multilevel Microtiling and Software Prefetching
- multinode connectivity tuned, Tuning Multinode Connectivity-Tuning Multinode Connectivitymultinode communication pitfalls, Multinode Communication Pitfalls-Pitfall #6: GPU memory fragmentation under UCX/RDMAGLOO backend instead of NCCL, Pitfall #1: Using a CPU-bound Gloo backend instead of NCCL-Pitfall #1: Using a CPU-bound Gloo backend instead of NCCLGPU memory fragmentation under UCX/RDMA, Pitfall #6: GPU memory fragmentation under UCX/RDMA-Pitfall #6: GPU memory fragmentation under UCX/RDMAinsufficient bandwidth or misconfigured NICs, Pitfall #4: Insufficient network bandwidth or misconfigured NICsmismatched NCCL versions, Pitfall #2: Mismatched NCCL versionsstraggler nodes or processes, Pitfall #5: Straggler nodes or processessummary, Pitfall #6: GPU memory fragmentation under UCX/RDMATCP port exhaustion during NCCL bootstrap, Pitfall #3: TCP port exhaustion during NCCL bootstrapNCCL with UCX, Enabling Peer-to-Peer DMA and UCX
- multinode inference, Multinode Inference, Parallelism, Decoding, and Routing Optimizationsdisaggregated inference (see disaggregated prefill and decode)Kubernetes for deploying, Deploying Disaggregated Prefill and Decode with Kubernetes-Deploying Disaggregated Prefill and Decode with Kubernetes, Full-Stack Inference Optimizations
- multiple GPUscapturing collectives via NCCL and CUDA Graphs, Capturing Multi-GPU Collectives with NCCL and CUDA Graphs-Capturing Multi-GPU Collectives with NCCL and CUDA Graphscongestion- and topology-aware scheduling, Congestion-Aware and Topology-Aware Scheduling with Multiple GPUs-Coordinating NVSwitch Transfers with Fine-Tuned Schedulingadaptive process-GPU mapping, Adaptive Process-GPU Mapping-Adaptive Process-GPU Mappingdynamic congestion-aware scheduling, Dynamic Congestion-Aware Scheduling-Dynamic Congestion-Aware SchedulingGPUDirect RDMA, Multinode and Multirack Communication with GPUDirect RDMA-Multinode and Multirack Communication with GPUDirect RDMAMoE expert rebalancing and regrouping, MoE Expert Rebalancing and Regroupingmultinode and multirack communication, Multinode and Multirack Communication with GPUDirect RDMA-Multinode and Multirack Communication with GPUDirect RDMANCCL collective communication, Optimizing Collective Communication with NCCL-Wave scheduling of collectivesNVLink/NVSwitch topology and constraints, NVLink/NVSwitch Topology and Bandwidth Constraints-NVLink/NVSwitch Topology and Bandwidth ConstraintsNVSwitch and fine-tuned scheduling, Coordinating NVSwitch Transfers with Fine-Tuned Scheduling-Coordinating NVSwitch Transfers with Fine-Tuned Schedulingtelemetry and monitoring, Real-Time Link Telemetry and Monitoringmany GPUs as one, Ultrascale Networking Treating Many GPUs as One-Co-Packaged Optics: Future of Networking Hardwaremultimillion-GPU clusters, Scaling Toward Multimillion GPU Clusters and 100-Trillion-Parameter Models-Scaling Toward Multimillion GPU Clusters and 100-Trillion-Parameter Modelsn-GPU scaling pattern, Pattern for N-GPU ScalingNCCL for distributed communication, NCCL for Distributed Multi-GPU Communication-In-Network SHARP Aggregationorchestrating across, Orchestrate Across Multiple GPUs and Cluster Nodes (NVSHMEM)-Pattern for N-GPU ScalingGPU-to-GPU memory sharing, Fine-Grained GPU-to-GPU Memory Sharing with NVSHMEM-Fine-Grained GPU-to-GPU Memory Sharing with NVSHMEMNVIDIA SHMEM library, Fine-Grained GPU-to-GPU Memory Sharing with NVSHMEM-Fine-Grained GPU-to-GPU Memory Sharing with NVSHMEMsymmetric memory, PyTorch Symmetric Memoryoverlap of compute and data, Multi-GPU Compute and Data Transfer Overlap with CUDA Streams-Multi-GPU Compute and Data Transfer Overlap with CUDA Streamsmultiple streams per GPU, Multi-GPU Compute and Data Transfer Overlap with CUDA StreamsPyTorch, Multi-GPU Compute and Data Transfer Overlap with CUDA Streamspeer-to-peer DMA, Enabling Peer-to-Peer DMA and UCXprofiling with HTA, Multi-GPU Profiling with HTAprogramming multiple GPUs, Multi-GPU Programming-Multi-GPU ProgrammingPyTorch pluggable memory allocators, Pluggable Memory Allocators and Cross-GPU Data Transfers-Pluggable Memory Allocators and Cross-GPU Data Transfersstream-ordered memory allocator, Multi-GPU Compute and Data Transfer Overlap with CUDA Streams-Multi-GPU Compute and Data Transfer Overlap with CUDA Streams
- multirack and storage communication, Multirack and Storage Communication

- inference clusters, Data Parallelism
- parallelism strategies for serving, Parallelism Strategies for Serving Massive MoE Models-Hybrid Parallelismhybrid parallelism, Hybrid Parallelism-Hybrid Parallelism

- hybrid parallelism, Hybrid Parallelism-Hybrid Parallelism

- AsyncTP, TorchTitan, AsyncTP, AutoParallel, and SimpleFSDP
- DeepGEMM library FP8-optimized, Transparency and Reproducibility
- persistent Triton kernel, Tiled and Persistent GEMM Kernel (Triton)-Tiled and Persistent GEMM Kernel (Triton)

- multitoken decoding, Multitoken Decoding with Medusa’s Multiple Heads-Multitoken Decoding with Medusa’s Multiple Heads

- asynchronous memory transfers with CUTLASS, Using CUTLASS for Optimal Arithmetic Intensity and Tensor Core Performance
- Blackwell GPU, The CPU and GPU Superchip
- close to CPU, close to GPU, NUMA-Friendly Memory Allocation and Memory Pinning
- CPU-GPU interconnects on superchips, Topology Awareness in NCCLperformance, The CPU and GPU Superchip
- CUDA stream ordered memory allocator, Stream-Ordered Memory Allocator-Stream-Ordered Memory Allocatormultiple GPUs, Multi-GPU Compute and Data Transfer Overlap with CUDA Streams-Multi-GPU Compute and Data Transfer Overlap with CUDA Streamsusing with CUDA streams, Using CUDA Streams and Stream-Ordered Memory Allocator with LLMs-Using CUDA Streams and Stream-Ordered Memory Allocator with LLMs
- CUDA Unified Memory, GPU Memory Oversubscription, Fragmentation, and Out-of-Memory Handlingprogramming CUDA, Unified Memory-Unified MemoryUnified CPU-GPU Memory, The CPU and GPU Superchip
- distributed shared memory (see distributed shared memory (DSMEM))
- dynamic memory-allocation switching, Dynamic Memory-Allocation Switching (Slab Versus Caching Versus Stream-Ordered)-Dynamic Memory-Allocation Switching (Slab Versus Caching Versus Stream-Ordered)
- ECC mode for GPU memory, GPU Clock Speeds and ECC
- edge-oriented MoE memory budgeting, Edge-Oriented MoE Memory Budgeting
- fragmentation monitoring, Monitoring System Metrics and Countersdynamic memory-allocation switching, Dynamic Memory-Allocation Switching (Slab Versus Caching Versus Stream-Ordered)-Dynamic Memory-Allocation Switching (Slab Versus Caching Versus Stream-Ordered)
- GDS, NUMA-Friendly Memory Allocation and Memory Pinning, Using NVIDIA GDS-Measuring GDS with gdsio, Offloading Parameters to CPU and NVMe
- global memory traffic reduced, Reducing Global Memory Traffic with Thread Block Clusters-Reducing Global Memory Traffic with Thread Block Clusters(see also global memory)
- GPU FLOPs outpacing memory bandwidth, Maintaining High Occupancy and GPU Utilization
- GPU memory access patternsavoiding shared memory, Warp Shuffle Intrinsics: Avoid Shared Memory and Explicit Synchronization-Warp Shuffle Intrinsics: Avoid Shared Memory and Explicit Synchronizationavoiding shared-memory bank conflicts, Avoid Shared-Memory Bank Conflicts-Avoid Shared-Memory Bank Conflictscoalesced versus uncoalesced global memory access, Coalesced Versus Uncoalesced Global Memory Access-Coalesced Versus Uncoalesced Global Memory Accessread-only data caches, Read-Only Data Caches-Read-Only Data Cachesread-only data caches pitfall, Read-Only Data Cachessymmetric memory, PyTorch Symmetric MemoryTensor Memory Accelerator for tile fetch, Asynchronous Memory Prefetching and Tensor Memory Accelerator-Asynchronous Memory Prefetching and Tensor Memory Acceleratortiling and data reuse via shared memory, Tiling and Data Reuse Using Shared Memory-Tiling and Data Reuse Using Shared Memoryvectorized memory access, Vectorized Memory Access-Vectorized Memory Access
- GPU memory fragmentation and oversubscription, GPU Memory Oversubscription, Fragmentation, and Out-of-Memory Handling-GPU Memory Oversubscription, Fragmentation, and Out-of-Memory Handling
- GPU memory fragmentation under UCX/RDMA, Pitfall #6: GPU memory fragmentation under UCX/RDMA-Pitfall #6: GPU memory fragmentation under UCX/RDMA
- GPU-to-GPU memory sharing with NVSHMEM, Fine-Grained GPU-to-GPU Memory Sharing with NVSHMEM-Fine-Grained GPU-to-GPU Memory Sharing with NVSHMEM
- Grace Blackwell Superchip, The CPU and GPU Superchip, Topology Awareness in NCCL
- Grace CPU, The CPU and GPU Superchip
- high-bandwidth memory, Scaling Toward Multimillion GPU Clusters and 100-Trillion-Parameter Models
- hugepages, Transparent Hugepagestransparent hugepages, Transparent Hugepages
- inference tuning, Memory
- kernel memory throughput versus HBM memory bandwidth, Kernel Memory Throughput Versus Peak HBM Memory Bandwidth
- Kubernetesmemory isolation, Memory Isolation and Avoiding the OOM Killermemory limit, Memory Isolation and Avoiding the OOM Killer
- KV cachememory layouts optimized, Optimized KV Cache Memory Layoutoffloaded to CPU memory, KV Cache Offloading with NIXL-KV Cache Offloading with NIXL
- memory-bound workload example, Maintaining High Occupancy and GPU Utilization
- NUMA-friendly allocation and pinning, NUMA-Friendly Memory Allocation and Memory Pinning-NUMA-Friendly Memory Allocation and Memory Pinningefficiency of pinned memory, NUMA-Friendly Memory Allocation and Memory Pinningmax locked memory setting, Transparent Hugepages, Virtual Memory and SwappingOS limit on pinned memory, NUMA-Friendly Memory Allocation and Memory Pinningpinned memory for data loaders, NUMA-Friendly Memory Allocation and Memory Pinning
- out of memory (OOM) errorsCPU + GPU architecture, The CPU and GPU Superchipmax locked memory setting, Transparent Hugepagesmemory fragmentation or excessive caching, GPU Memory Oversubscription, Fragmentation, and Out-of-Memory Handlingmonitoring GPU memory usage, GPU Memory Oversubscription, Fragmentation, and Out-of-Memory Handling
- profiling and tuning via PyTorch profiler, Profiling and Tuning Memory in PyTorch-Enabling Peer-to-Peer DMA and UCXactivation checkpointing, Activation Checkpointing for Memory SavingsCUDA memory allocator tuned, Tuning the CUDA Memory AllocatorFSDP automatic checkpointing, FSDP Automatic Checkpointing and Offloading-FSDP Automatic Checkpointing and OffloadingFSDP with tensor parallel and pipeline parallel, Combining FSDP with Tensor Parallel and Pipeline ParallelNCCL with UCX over multinode topologies, Enabling Peer-to-Peer DMA and UCXoffloading parameters to CPU and NVMe, Offloading Parameters to CPU and NVMepeer-to-peer DMA, Enabling Peer-to-Peer DMA and UCXpluggable memory allocators, Pluggable Memory Allocators and Cross-GPU Data Transfers-Pluggable Memory Allocators and Cross-GPU Data Transfers
- programming CUDAasynchronous memory allocation, Asynchronous Memory Allocation and Memory Pools-Asynchronous Memory Allocation and Memory PoolsCUDA streams, Asynchronous Memory Allocation and Memory PoolsGPU memory hierarchy, Understanding GPU Memory Hierarchy-Understanding GPU Memory Hierarchymemory pools, Asynchronous Memory Allocation and Memory Pools-Asynchronous Memory Allocation and Memory Pools
- PyTorch memory allocator plugin, Pluggable Memory Allocators and Cross-GPU Data Transfers-Pluggable Memory Allocators and Cross-GPU Data Transfers
- recomputation versus memory, Recomputation Versus Memory Trade-Off
- scratch memory, Scratch MemoryDSMEM, Scratch Memory(see also distributed shared memory (DSMEM))
- shared-memory exchanges disabled, Pitfall #3: Avoid overtuning or disabling NCCL features with environment variables(see also shared memory (SMEM))
- swapping avoided, Virtual Memory and Swappingmax locked memory setting, Virtual Memory and Swapping
- symmetric memory, PyTorch Symmetric Memory
- tuning host CPU memory allocator, Tune Host CPU Memory Allocator
- Unified CPU-GPU Memory, The CPU and GPU Superchip

- performance, The CPU and GPU Superchip

- multiple GPUs, Multi-GPU Compute and Data Transfer Overlap with CUDA Streams-Multi-GPU Compute and Data Transfer Overlap with CUDA Streams
- using with CUDA streams, Using CUDA Streams and Stream-Ordered Memory Allocator with LLMs-Using CUDA Streams and Stream-Ordered Memory Allocator with LLMs

- programming CUDA, Unified Memory-Unified Memory
- Unified CPU-GPU Memory, The CPU and GPU Superchip

- dynamic memory-allocation switching, Dynamic Memory-Allocation Switching (Slab Versus Caching Versus Stream-Ordered)-Dynamic Memory-Allocation Switching (Slab Versus Caching Versus Stream-Ordered)

- (see also global memory)

- avoiding shared memory, Warp Shuffle Intrinsics: Avoid Shared Memory and Explicit Synchronization-Warp Shuffle Intrinsics: Avoid Shared Memory and Explicit Synchronization
- avoiding shared-memory bank conflicts, Avoid Shared-Memory Bank Conflicts-Avoid Shared-Memory Bank Conflicts
- coalesced versus uncoalesced global memory access, Coalesced Versus Uncoalesced Global Memory Access-Coalesced Versus Uncoalesced Global Memory Access
- read-only data caches, Read-Only Data Caches-Read-Only Data Caches
- read-only data caches pitfall, Read-Only Data Caches
- symmetric memory, PyTorch Symmetric Memory
- Tensor Memory Accelerator for tile fetch, Asynchronous Memory Prefetching and Tensor Memory Accelerator-Asynchronous Memory Prefetching and Tensor Memory Accelerator
- tiling and data reuse via shared memory, Tiling and Data Reuse Using Shared Memory-Tiling and Data Reuse Using Shared Memory
- vectorized memory access, Vectorized Memory Access-Vectorized Memory Access

- transparent hugepages, Transparent Hugepages

- memory isolation, Memory Isolation and Avoiding the OOM Killer
- memory limit, Memory Isolation and Avoiding the OOM Killer

- memory layouts optimized, Optimized KV Cache Memory Layout
- offloaded to CPU memory, KV Cache Offloading with NIXL-KV Cache Offloading with NIXL

- efficiency of pinned memory, NUMA-Friendly Memory Allocation and Memory Pinning
- max locked memory setting, Transparent Hugepages, Virtual Memory and Swapping
- OS limit on pinned memory, NUMA-Friendly Memory Allocation and Memory Pinning
- pinned memory for data loaders, NUMA-Friendly Memory Allocation and Memory Pinning

- CPU + GPU architecture, The CPU and GPU Superchip
- max locked memory setting, Transparent Hugepages
- memory fragmentation or excessive caching, GPU Memory Oversubscription, Fragmentation, and Out-of-Memory Handling
- monitoring GPU memory usage, GPU Memory Oversubscription, Fragmentation, and Out-of-Memory Handling

- activation checkpointing, Activation Checkpointing for Memory Savings
- CUDA memory allocator tuned, Tuning the CUDA Memory Allocator
- FSDP automatic checkpointing, FSDP Automatic Checkpointing and Offloading-FSDP Automatic Checkpointing and Offloading
- FSDP with tensor parallel and pipeline parallel, Combining FSDP with Tensor Parallel and Pipeline Parallel
- NCCL with UCX over multinode topologies, Enabling Peer-to-Peer DMA and UCX
- offloading parameters to CPU and NVMe, Offloading Parameters to CPU and NVMe
- peer-to-peer DMA, Enabling Peer-to-Peer DMA and UCX
- pluggable memory allocators, Pluggable Memory Allocators and Cross-GPU Data Transfers-Pluggable Memory Allocators and Cross-GPU Data Transfers

- asynchronous memory allocation, Asynchronous Memory Allocation and Memory Pools-Asynchronous Memory Allocation and Memory Pools
- CUDA streams, Asynchronous Memory Allocation and Memory Pools
- GPU memory hierarchy, Understanding GPU Memory Hierarchy-Understanding GPU Memory Hierarchy
- memory pools, Asynchronous Memory Allocation and Memory Pools-Asynchronous Memory Allocation and Memory Pools

- DSMEM, Scratch Memory(see also distributed shared memory (DSMEM))

- (see also distributed shared memory (DSMEM))

- (see also shared memory (SMEM))

- max locked memory setting, Virtual Memory and Swapping

- kernel optimization, Optimizing the Kernel-Optimizing the Kernel

- GPU driver runtime settings, MIG-MIG
- GPU virtualization, Managing Resources Efficiently
- Kubernetes MIG mode, Slicing a GPU with MIG
- persistence mode recommended for GPU, Slicing a GPU with MIG
- profile naming, MIG
- single GPU split into smaller GPUs, Sharing and Scheduling
- when to use, MIG

- automatic mixed precision, TF32 and Automatic Mixed Precision (PyTorch)-TF32 and Automatic Mixed Precision (PyTorch)
- reduced precision, BF16/FP16, FP8, and FP4 Reduced Precision
- Tensor Cores, Mixed Precision and Utilizing Tensor Cores-Transformer Engine and TMEM in Depth

- about MoE models, Profiling PyTorch to Identify Bottlenecks
- all experts active simultaneously, Expert Parallelism
- all-to-all communication, Expert Communication Optimization
- CUDA streams, Using CUDA Streams with MoE Models
- DeepEP communication library, Transparency and Reproducibility
- DeepSeek-V3, DeepSeek Scales to ~680-Billion Parameter Models Despite US Export Hardware Restrictions in China-DeepSeek Scales to ~680-Billion Parameter Models Despite US Export Hardware Restrictions in ChinaDualPipe parallelism algorithm, DeepSeek Scales to ~680-Billion Parameter Models Despite US Export Hardware Restrictions in China
- dynamic routing strategies for inference, Dynamic Routing Strategies for MoE Inference-Adaptive Expert Routing and Real-Time Monitoringexpert communication optimization, Expert Communication Optimization-Expert Communication Optimizationexpert replication, Load Balancing, Capacity Factor, and Expert Replicationload balancing and capacity factors, Load Balancing, Capacity Factor, and Expert Replication
- edge-oriented memory budgeting, Edge-Oriented MoE Memory Budgeting
- expert parallelism, Parallelism Strategies for Serving Massive MoE Models, Expert Parallelism-Expert Parallelismload balancing, Expert Parallelismmodel weights and data split over GPUs, Parallelism Strategies for Serving Massive MoE Models
- expert rebalancing and regrouping, MoE Expert Rebalancing and Regrouping
- hybrid parallelism, Hybrid Parallelism-Hybrid Parallelism
- MoE efficiency, Toward 100-Trillion-Parameter Modelsopen MoE rivaling best closed models, DeepSeek Scales to ~680-Billion Parameter Models Despite US Export Hardware Restrictions in China
- multinode inference needed, Multinode Inference, Parallelism, Decoding, and Routing Optimizations
- parallelism strategies, Parallelism Strategies for Serving Massive MoE Models-Hybrid Parallelism
- profiling, Profiling PyTorch to Identify Bottlenecks-CPU and GPU Profiling with Linux perfLinux perf on CPU and GPU, CPU and GPU Profiling with Linux perf-CPU and GPU Profiling with Linux perfNsight Systems and NVTX Timelines, System Profiling with Nsight Systems and NVTX Timelines-System Profiling with Nsight Systems and NVTX TimelinesPyTorch profiler, Using PyTorch Profiler-Using PyTorch Profilerroofline analysis for GEMM kernels, Kernel Roofline Analysis for General Matrix Multiply (GEMM)-Kernel Roofline Analysis for General Matrix Multiply (GEMM)
- sparse models reducing compute requirements, DeepSeek Scales to ~680-Billion Parameter Models Despite US Export Hardware Restrictions in China, Toward 100-Trillion-Parameter Models
- speculative expert routing, Speculative MoE Expert Routing and Communication Reduction

- DualPipe parallelism algorithm, DeepSeek Scales to ~680-Billion Parameter Models Despite US Export Hardware Restrictions in China

- expert communication optimization, Expert Communication Optimization-Expert Communication Optimization
- expert replication, Load Balancing, Capacity Factor, and Expert Replication
- load balancing and capacity factors, Load Balancing, Capacity Factor, and Expert Replication

- load balancing, Expert Parallelism
- model weights and data split over GPUs, Parallelism Strategies for Serving Massive MoE Models

- open MoE rivaling best closed models, DeepSeek Scales to ~680-Billion Parameter Models Despite US Export Hardware Restrictions in China

- Linux perf on CPU and GPU, CPU and GPU Profiling with Linux perf-CPU and GPU Profiling with Linux perf
- Nsight Systems and NVTX Timelines, System Profiling with Nsight Systems and NVTX Timelines-System Profiling with Nsight Systems and NVTX Timelines
- PyTorch profiler, Using PyTorch Profiler-Using PyTorch Profiler
- roofline analysis for GEMM kernels, Kernel Roofline Analysis for General Matrix Multiply (GEMM)-Kernel Roofline Analysis for General Matrix Multiply (GEMM)

- caution on per-GPU results, Transparency and Reproducibility
- Logging, Performance Benchmarks and MLPerf Logging-Performance Benchmarks and MLPerf Logging
- MLPerf Inference, Transparency and Reproducibility
- MLPerf Training, Transparency and Reproducibility

- CPU utilization, Scaling Out Workers as You Scale Out Number of GPUs
- filesystem I/O during training, Tuning, Replicating, and Compressing Data
- GPU memory usage, GPU Memory Oversubscription, Fragmentation, and Out-of-Memory Handling
- graph recompilations, TorchDynamo for Bytecode Capture and Graph Extraction, Minimize Graph Recompilations
- inference system metrics and counters, Monitoring System Metrics and Counters-Monitoring System Metrics and Counterscorrectness issue debugging, Debugging Correctness Issues-Debugging Correctness Issuesproduction environment, Inference Troubleshooting Recipes
- memory fragmentation, Monitoring System Metrics and Counters
- model rambling, Token Output Limits and Timeouts
- multiple-GPU NVLink utilization, Real-Time Link Telemetry and Monitoring
- network throughput, Pitfall #4: Insufficient network bandwidth or misconfigured NICs
- performance monitoring, Performance Monitoring and Utilization in Practicecommunication versus compute bottlenecks, Diagnosing Communication- Versus Compute-Bound WorkloadsGPUs near 100% utilized, Performance Monitoring and Utilization in PracticeI/O continuous profiling and tuning, Continuous Profiling and Tuning Workflow-Continuous Profiling and Tuning WorkflowNVLink usage, Performance Monitoring and Utilization in Practice
- power monitoring, Performance Monitoring and Utilization in Practice
- storage I/O, Monitoring Storage I/O-Monitoring Storage I/O
- straggler nodes or processes, Pitfall #5: Straggler nodes or processes

- correctness issue debugging, Debugging Correctness Issues-Debugging Correctness Issues
- production environment, Inference Troubleshooting Recipes

- communication versus compute bottlenecks, Diagnosing Communication- Versus Compute-Bound Workloads
- GPUs near 100% utilized, Performance Monitoring and Utilization in Practice
- I/O continuous profiling and tuning, Continuous Profiling and Tuning Workflow-Continuous Profiling and Tuning Workflow
- NVLink usage, Performance Monitoring and Utilization in Practice

- adaptive strategies, Mooncake adaptive strategies
- early rejection, Early Rejection (Admission Control)

- (see also dual-die GPUs)

- multinode communication pitfalls, Multinode Communication Pitfalls-Pitfall #6: GPU memory fragmentation under UCX/RDMAGLOO backend instead of NCCL, Pitfall #1: Using a CPU-bound Gloo backend instead of NCCL-Pitfall #1: Using a CPU-bound Gloo backend instead of NCCLGPU memory fragmentation under UCX/RDMA, Pitfall #6: GPU memory fragmentation under UCX/RDMA-Pitfall #6: GPU memory fragmentation under UCX/RDMAinsufficient bandwidth or misconfigured NICs, Pitfall #4: Insufficient network bandwidth or misconfigured NICsmismatched NCCL versions, Pitfall #2: Mismatched NCCL versionsstraggler nodes or processes, Pitfall #5: Straggler nodes or processessummary, Pitfall #6: GPU memory fragmentation under UCX/RDMATCP port exhaustion during NCCL bootstrap, Pitfall #3: TCP port exhaustion during NCCL bootstrap
- NCCL with UCX, Enabling Peer-to-Peer DMA and UCX

- GLOO backend instead of NCCL, Pitfall #1: Using a CPU-bound Gloo backend instead of NCCL-Pitfall #1: Using a CPU-bound Gloo backend instead of NCCL
- GPU memory fragmentation under UCX/RDMA, Pitfall #6: GPU memory fragmentation under UCX/RDMA-Pitfall #6: GPU memory fragmentation under UCX/RDMA
- insufficient bandwidth or misconfigured NICs, Pitfall #4: Insufficient network bandwidth or misconfigured NICs
- mismatched NCCL versions, Pitfall #2: Mismatched NCCL versions
- straggler nodes or processes, Pitfall #5: Straggler nodes or processes
- summary, Pitfall #6: GPU memory fragmentation under UCX/RDMA
- TCP port exhaustion during NCCL bootstrap, Pitfall #3: TCP port exhaustion during NCCL bootstrap

- disaggregated inference (see disaggregated prefill and decode)
- Kubernetes for deploying, Deploying Disaggregated Prefill and Decode with Kubernetes-Deploying Disaggregated Prefill and Decode with Kubernetes, Full-Stack Inference Optimizations

- capturing collectives via NCCL and CUDA Graphs, Capturing Multi-GPU Collectives with NCCL and CUDA Graphs-Capturing Multi-GPU Collectives with NCCL and CUDA Graphs
- congestion- and topology-aware scheduling, Congestion-Aware and Topology-Aware Scheduling with Multiple GPUs-Coordinating NVSwitch Transfers with Fine-Tuned Schedulingadaptive process-GPU mapping, Adaptive Process-GPU Mapping-Adaptive Process-GPU Mappingdynamic congestion-aware scheduling, Dynamic Congestion-Aware Scheduling-Dynamic Congestion-Aware SchedulingGPUDirect RDMA, Multinode and Multirack Communication with GPUDirect RDMA-Multinode and Multirack Communication with GPUDirect RDMAMoE expert rebalancing and regrouping, MoE Expert Rebalancing and Regroupingmultinode and multirack communication, Multinode and Multirack Communication with GPUDirect RDMA-Multinode and Multirack Communication with GPUDirect RDMANCCL collective communication, Optimizing Collective Communication with NCCL-Wave scheduling of collectivesNVLink/NVSwitch topology and constraints, NVLink/NVSwitch Topology and Bandwidth Constraints-NVLink/NVSwitch Topology and Bandwidth ConstraintsNVSwitch and fine-tuned scheduling, Coordinating NVSwitch Transfers with Fine-Tuned Scheduling-Coordinating NVSwitch Transfers with Fine-Tuned Schedulingtelemetry and monitoring, Real-Time Link Telemetry and Monitoring
- many GPUs as one, Ultrascale Networking Treating Many GPUs as One-Co-Packaged Optics: Future of Networking Hardware
- multimillion-GPU clusters, Scaling Toward Multimillion GPU Clusters and 100-Trillion-Parameter Models-Scaling Toward Multimillion GPU Clusters and 100-Trillion-Parameter Models
- n-GPU scaling pattern, Pattern for N-GPU Scaling
- NCCL for distributed communication, NCCL for Distributed Multi-GPU Communication-In-Network SHARP Aggregation
- orchestrating across, Orchestrate Across Multiple GPUs and Cluster Nodes (NVSHMEM)-Pattern for N-GPU ScalingGPU-to-GPU memory sharing, Fine-Grained GPU-to-GPU Memory Sharing with NVSHMEM-Fine-Grained GPU-to-GPU Memory Sharing with NVSHMEMNVIDIA SHMEM library, Fine-Grained GPU-to-GPU Memory Sharing with NVSHMEM-Fine-Grained GPU-to-GPU Memory Sharing with NVSHMEMsymmetric memory, PyTorch Symmetric Memory
- overlap of compute and data, Multi-GPU Compute and Data Transfer Overlap with CUDA Streams-Multi-GPU Compute and Data Transfer Overlap with CUDA Streamsmultiple streams per GPU, Multi-GPU Compute and Data Transfer Overlap with CUDA StreamsPyTorch, Multi-GPU Compute and Data Transfer Overlap with CUDA Streams
- peer-to-peer DMA, Enabling Peer-to-Peer DMA and UCX
- profiling with HTA, Multi-GPU Profiling with HTA
- programming multiple GPUs, Multi-GPU Programming-Multi-GPU Programming
- PyTorch pluggable memory allocators, Pluggable Memory Allocators and Cross-GPU Data Transfers-Pluggable Memory Allocators and Cross-GPU Data Transfers
- stream-ordered memory allocator, Multi-GPU Compute and Data Transfer Overlap with CUDA Streams-Multi-GPU Compute and Data Transfer Overlap with CUDA Streams

- adaptive process-GPU mapping, Adaptive Process-GPU Mapping-Adaptive Process-GPU Mapping
- dynamic congestion-aware scheduling, Dynamic Congestion-Aware Scheduling-Dynamic Congestion-Aware Scheduling
- GPUDirect RDMA, Multinode and Multirack Communication with GPUDirect RDMA-Multinode and Multirack Communication with GPUDirect RDMA
- MoE expert rebalancing and regrouping, MoE Expert Rebalancing and Regrouping
- multinode and multirack communication, Multinode and Multirack Communication with GPUDirect RDMA-Multinode and Multirack Communication with GPUDirect RDMA
- NCCL collective communication, Optimizing Collective Communication with NCCL-Wave scheduling of collectives
- NVLink/NVSwitch topology and constraints, NVLink/NVSwitch Topology and Bandwidth Constraints-NVLink/NVSwitch Topology and Bandwidth Constraints
- NVSwitch and fine-tuned scheduling, Coordinating NVSwitch Transfers with Fine-Tuned Scheduling-Coordinating NVSwitch Transfers with Fine-Tuned Scheduling
- telemetry and monitoring, Real-Time Link Telemetry and Monitoring

- GPU-to-GPU memory sharing, Fine-Grained GPU-to-GPU Memory Sharing with NVSHMEM-Fine-Grained GPU-to-GPU Memory Sharing with NVSHMEM
- NVIDIA SHMEM library, Fine-Grained GPU-to-GPU Memory Sharing with NVSHMEM-Fine-Grained GPU-to-GPU Memory Sharing with NVSHMEM
- symmetric memory, PyTorch Symmetric Memory

- multiple streams per GPU, Multi-GPU Compute and Data Transfer Overlap with CUDA Streams
- PyTorch, Multi-GPU Compute and Data Transfer Overlap with CUDA Streams

### N

- n-GPU scaling pattern, Pattern for N-GPU Scaling
- Nagle’s algorithm, Streaming Responses
- NCCL (NVIDIA Collective Communications Library)about NCCL, Scaling Distributed Training and Inference, NCCL for Distributed Multi-GPU Communication, Optimizing Collective Communication with NCCLbackend instead of GLOO, Pitfall #1: Using a CPU-bound Gloo backend instead of NCCL-Pitfall #1: Using a CPU-bound Gloo backend instead of NCCLbootstrap TCP port exhaustion, Pitfall #3: TCP port exhaustion during NCCL bootstrapCUDA Toolkit, CUDA Toolkit and Runtimedebugging, Profiling and Debugging NCCLinference correctness issue, Debugging Correctness IssuesNCCL debugging enabled, Debugging Correctness IssuesNCCL logs, In-Network SHARP Aggregation, Debugging Correctness Issuestest suite, Debugging Correctness Issuesdedicated high-priority streams for network transfers, Multi-GPU Compute and Data Transfer Overlap with CUDA Streamsring or tree algorithm, Multi-GPU Compute and Data Transfer Overlap with CUDA Streams-Multi-GPU Compute and Data Transfer Overlap with CUDA Streams, Ring versus tree all-reduce-Ring versus tree all-reducedistributed multi-GPU communication, NCCL for Distributed Multi-GPU Communication-In-Network SHARP Aggregationabout, NCCL for Distributed Multi-GPU Communicationall-reduces in parallelism, Distributed Data Parallel StrategiesCollNet communication algorithm, NCCL Communication Algorithmscommunication algorithms, NCCL Communication Algorithms-NCCL Communication Algorithmsmany-to-many collectives, NVIDIA’s NIXL and Disaggregated Inference, NVIDIA’s NIXL and Disaggregated Inference, NCCL Versus NIXLNCCL_ALGO environment variable, Topology Awareness in NCCL, NCCL Communication Algorithmsoptimizing, Optimizing Collective Communication with NCCL-Wave scheduling of collectivesparallel aggregated tree communication algorithm, NCCL Communication Algorithmsring communication algorithm, NCCL Communication Algorithmstopology awareness, Topology Awareness in NCCL-Topology Awareness in NCCLtree communication algorithm, NCCL Communication Algorithmswave scheduling of collectives, Wave scheduling of collectivesdistributed multi-GPU communicationsring or tree algorithm, Ring versus tree all-reduce-Ring versus tree all-reducerotating ring endpoints, Rotating ring endpointsGPU-NIC affinity forced, Multinode and Multirack Communication with GPUDirect RDMAmaximum TCP socket connections per communicator, NCCL_NSOCKS_PERTHREAD and NCCL_SOCKET_NTHREADSmemory allocator in PyTorch, Pluggable Memory Allocators and Cross-GPU Data Transfersmultiple NICs in parallel, Multinode and Multirack Communication with GPUDirect RDMAmultiple-GPU collectives captured, Capturing Multi-GPU Collectives with NCCL and CUDA Graphs-Capturing Multi-GPU Collectives with NCCL and CUDA GraphsNIXL versus, NCCL Versus NIXLmany-to-many collectives by NCCL, NVIDIA’s NIXL and Disaggregated Inference, NVIDIA’s NIXL and Disaggregated Inference, NCCL Versus NIXLpersistent user buffers, Persistent NCCL User Buffers and Zero-Copy Registrationpitfalls and gotchas, NCCL Communicator Lifecycle and Environment Gotchas-Pitfall #6: NCCL communicator hangs, errors, or shuts down completelycommunicator hanging, giving errors, shutting down, Pitfall #6: NCCL communicator hangs, errors, or shuts down completely-Pitfall #6: NCCL communicator hangs, errors, or shuts down completelyCPU-GPU NUMA-node for NCCL threads, Pitfall #4: Verify CPU-GPU NUMA-node affinity for NCCL threadscreating and destroying communicators, Pitfall #2: Do not create and destroy NCCL communicators on every iterationcreating communicators too often, Pitfall #1: Creating NCCL communicators too often-Pitfall #1: Creating NCCL communicators too oftenignoring warnings and errors, Pitfall #5: Resist the temptation to ignore NCCL warnings and errorsoverturning or disabling features via environment variables, Pitfall #3: Avoid overtuning or disabling NCCL features with environment variables-NCCL_SHARP_DISABLEversions mismatched, Pitfall #2: Mismatched NCCL versionsprofiling, Profiling and Debugging NCCLNCCL logs, In-Network SHARP AggregationRDMA container access to devices, High-Speed, Low-Overhead Data Transfers with RDMASHARP and, In-Network SHARP AggregationUCX for multinode topologies, Enabling Peer-to-Peer DMA and UCXuser buffer registration, Persistent NCCL User Buffers and Zero-Copy Registrationversions mismatched, Pitfall #2: Mismatched NCCL versions
- NCCL_ALGO, Topology Awareness in NCCL, NCCL Communication Algorithms, Ring versus tree all-reduce
- NCCL_ASYNC_ERROR_HANDLING, Pitfall #6: NCCL communicator hangs, errors, or shuts down completely, Profiling and Debugging NCCL
- NCCL_BUFFSIZE, Pitfall #3: Avoid overtuning or disabling NCCL features with environment variables
- NCCL_CROSS_NIC, Multinode and Multirack Communication with GPUDirect RDMA
- NCCL_DEBUG, Topology Awareness in NCCL, Pitfall #3: Avoid overtuning or disabling NCCL features with environment variables, Profiling and Debugging NCCL, Debugging Correctness IssuesNCCL logs, In-Network SHARP Aggregation, Debugging Correctness Issues
- NCCL_IGNORE_CPU_AFFINITY, Pitfall #4: Verify CPU-GPU NUMA-node affinity for NCCL threads
- NCCL_MAX_NCHANNELS, NCCL_MIN_NCHANNELS and NCCL_MAX_NCHANNELS
- NCCL_MIN_NCHANNELS, NCCL_MIN_NCHANNELS and NCCL_MAX_NCHANNELS
- NCCL_NSOCKS_PERTHREAD, Tuning Multinode Connectivity, Pitfall #4: Insufficient network bandwidth or misconfigured NICs, NCCL_NSOCKS_PERTHREAD and NCCL_SOCKET_NTHREADS
- NCCL_P2P_DISABLE, Pitfall #3: Avoid overtuning or disabling NCCL features with environment variables
- NCCL_PORT_RANGE, Optimizing Network Communication for Kubernetes
- NCCL_PROFILER_PLUGIN, Profiling and Debugging NCCL
- NCCL_SHARP_DISABLE, NCCL_SHARP_DISABLE
- NCCL_SHM_DISABLE, Pitfall #3: Avoid overtuning or disabling NCCL features with environment variables
- NCCL_SOCKET_IFNAME, Optimizing Network Communication for Kubernetes, Pitfall #5: Resist the temptation to ignore NCCL warnings and errors
- NCCL_SOCKET_NTHREADS, Tuning Multinode Connectivity, Pitfall #4: Insufficient network bandwidth or misconfigured NICs, NCCL_NSOCKS_PERTHREAD and NCCL_SOCKET_NTHREADS
- NCCL_TOPO_DUMP_FILE, Topology Awareness in NCCL
- NCCL_TOPO_FILE, NCCL_TOPO_FILE
- ncu for Nsight Computeprofiler, Profiling and Diagnosing GPU Bottlenecks, NVTX Markers and Profiling Tools, NVTX Markers and Profiling ToolsRoofline analysis, Nsight Compute and Roofline Analysis, Kernel Roofline Analysis for General Matrix Multiply (GEMM)-Kernel Roofline Analysis for General Matrix Multiply (GEMM)
- nested jagged-layout tensors (NJT), FlexDecoding (PyTorch)
- nested tensors (PyTorch), PyTorch and Arithmetic Intensity
- networking communication (see distributed networking communication)
- NFS servers, Distributed, Parallel Filesystems and Object Storestuning parameters, Distributed, Parallel Filesystems and Object Stores
- NIC-GPU affinity forced, Multinode and Multirack Communication with GPUDirect RDMA
- NIXL (NVIDIA Inference Xfer Library)about NIXL, Scaling Distributed Training and Inference, NCCL for Distributed Multi-GPU Communication, NVIDIA’s NIXL and Disaggregated Inference-NVIDIA’s NIXL and Disaggregated Inferencededicated high-priority streams for network transfers, Multi-GPU Compute and Data Transfer Overlap with CUDA Streamsdisaggregated inference, NVIDIA’s NIXL and Disaggregated Inference-NCCL Versus NIXLabout NIXL, NVIDIA’s NIXL and Disaggregated Inference-NVIDIA’s NIXL and Disaggregated Inferenceasynchronous API with callbacks, NIXL Asynchronous API with Callbacks-NIXL Asynchronous API with CallbacksDynamo throughput improved with NIXL, NIXL and High-Performance Inference Systems Like NVIDIA Dynamointelligent interconnect routing, Intelligent Interconnect Routing for KV Cache Transfers, KV Cache Data Transfer and NIXLKV cache offloading with NIXL, KV Cache Offloading with NIXL-KV Cache Offloading with NIXLprefill to decode data transfers, KV Cache Data Transfer and NIXLseparate prefill and decode inference stages, Separate Prefill and Decode Inference Stages-Separate Prefill and Decode Inference StagesNCCL versus, NCCL Versus NIXLone-to-one transfers by NIXL, Distributed Data Parallel Strategies, NVIDIA’s NIXL and Disaggregated Inference, NVIDIA’s NIXL and Disaggregated Inference, NCCL Versus NIXLNVIDIA Dynamo core component, NVIDIA’s NIXL and Disaggregated InferenceDynamo throughput improved, NIXL and High-Performance Inference Systems Like NVIDIA Dynamo
- Nsight Computeachieved occupancy, Inspecting Achieved Occupancy and GPU Utilization-Optimizing the Kernelkernel compute throughput versus GPU FLOPS, Kernel Compute Throughput Versus Peak GPU FLOPSkernel memory throughput versus HBM memory bandwidth, Kernel Memory Throughput Versus Peak HBM Memory Bandwidthunderutilization, Dynamic Scheduling with Atomic Work QueuesALU pipe busy, Execution Unit ContentionCLI ncuprofiler, Profiling and Diagnosing GPU Bottlenecks, NVTX Markers and Profiling Tools, NVTX Markers and Profiling ToolsRoofline analysis, Nsight Compute and Roofline Analysis, Kernel Roofline Analysis for General Matrix Multiply (GEMM)-Kernel Roofline Analysis for General Matrix Multiply (GEMM)CUDA Program Counter Sampling, Profiling with Nsight Systems and Nsight ComputeILP configuration, ILP and Occupancyimpact of parallel versus serial operation, Maintaining High Occupancy and GPU Utilizationinference profiling, Profiling with Nsight Systems and Nsight ComputeCUDA Program Counter Sampling, Profiling with Nsight Systems and Nsight Computekernel shift from memory to compute bound, Feeding Tensor Cores with TMEM and TMAlaunch statistics, Designing Efficient Algorithms with Thread Block ClustersMemory Workload Analysisatomic transactions counter, Atomic Counters-Atomic Counterscoalesced versus uncoalesced memory accesses, Coalesced Versus Uncoalesced Global Memory Accessmemory workload analysis, Kernel Autotuning for Transformer Self-Attention and MLP PathsOccupancyLimited by Registers, Profiling and Mitigating Register Pressurelow achieved FLOPS, Kernel Compute Throughput Versus Peak GPU FLOPSOccupancy Calculator, Techniques for Occupancy Tuningoccupancy limiters, Inspecting Achieved Occupancy and GPU Utilizationparallelism strategies profiled, Hybrid Parallelismper-kernel metrics, Read-Only Data CachesRegisters per Thread and Occupancy metrics, Tuning Occupancy with Launch BoundsRoofline analysis, Nsight Compute and Roofline Analysisexample MoE model, Kernel Roofline Analysis for General Matrix Multiply (GEMM)-Kernel Roofline Analysis for General Matrix Multiply (GEMM)Source Counterslow achieved FLOPS, Kernel Compute Throughput Versus Peak GPU FLOPSwarp divergence profiled, Profiling and Detecting Warp DivergenceSpeed of Light analysis, Mixed Precision and Utilizing Tensor CoresStallCompute Unit Busy, Execution Unit ContentionExec Dependency, Execution-Dependency Stalls, Profiling and Mitigating Register PressureIdle, Other Stall ReasonsLong Scoreboard, Memory-Related StallsMath Pipe Throttle, Execution Unit ContentionMemory Dependency, Other Stall ReasonsMemory Throttle, Memory-Related StallsNo Eligible, Other Stall ReasonsNot Selected, Memory-Related Stallsother reasons, Other Stall Reasons-Other Stall ReasonsShort Scoreboard, Memory-Related StallsTexture Throttle, Other Stall ReasonsTriton kernel support, Profiling with Triton Proton Profilerunderutilization metrics, Dynamic Scheduling with Atomic Work Queuesupdated to latest GPUs, Profiler-Guided AnalysisWarp State, Profiling and Detecting Warp Divergence
- Nsight Systemsabout, Profiling and Diagnosing GPU BottlenecksCLI nsysprofiler, Profiling and Diagnosing GPU Bottlenecks, NVTX Markers and Profiling Tools, NVTX Markers and Profiling Tools, System Profiling with Nsight Systems and NVTX Timelines-System Profiling with Nsight Systems and NVTX Timelinestimeline view, Nsight Systems Timeline View, System Profiling with Nsight Systems and NVTX Timelines-System Profiling with Nsight Systems and NVTX Timelinescommunication versus compute bottleneck, Diagnosing Communication- Versus Compute-Bound WorkloadsI/O monitoring, Monitoring Storage I/Oimpact of parallel versus serial operation, Maintaining High Occupancy and GPU Utilizationinference profiling, Profiling with Nsight Systems and Nsight Compute-Profiling with Nsight Systems and Nsight ComputeNVLink profiling, Topology Awareness in NCCL, Real-Time Link Telemetry and Monitoringparallelism strategies profiled, Hybrid ParallelismPython backtrace sampling, PyTorch Profiler and Visualization ToolsPyTorch focused mode, PyTorch Profiler and Visualization Toolstimeline view, Read-Only Data Caches, Nsight Systems Timeline View, Nsight Compute and Roofline Analysis, System Profiling with Nsight Systems and NVTX Timelines-System Profiling with Nsight Systems and NVTX Timelines, Kernel Autotuning for Transformer Self-Attention and MLP PathsNVTX markers, Nsight Systems Timeline View, System Profiling with Nsight Systems and NVTX Timelines-System Profiling with Nsight Systems and NVTX TimelinesTriton kernel support, Profiling with Triton Proton Profilerupdated to latest GPUs, Profiler-Guided Analysis
- nsys as CLI for Nsight Systemsprofiler, Profiling and Diagnosing GPU Bottlenecks, NVTX Markers and Profiling Tools, NVTX Markers and Profiling Tools, System Profiling with Nsight Systems and NVTX Timelines-System Profiling with Nsight Systems and NVTX Timelinestimeline view, Nsight Systems Timeline View, System Profiling with Nsight Systems and NVTX Timelines-System Profiling with Nsight Systems and NVTX Timelines
- NUMA nodes, NUMA Awareness and CPU PinningNUMA awareness and CPU pinning, NUMA Awareness and CPU Pinning-NUMA Awareness and CPU PinningCPU pinning, NUMA Awareness and CPU Pinning-NUMA Awareness and CPU PinningNUMA-friendly memory allocation and pinning, NUMA-Friendly Memory Allocation and Memory Pinning-NUMA-Friendly Memory Allocation and Memory Pinning
- numactlCPU pinning, NUMA Awareness and CPU Pinning, NUMA Awareness and CPU Pinningforcing memory allocation from specific NUMA node, NUMA Awareness and CPU Pinning, NUMA-Friendly Memory Allocation and Memory Pinningprocess CPU and memory policies applied on launch, NUMA-Friendly Memory Allocation and Memory Pinning
- numerical correctness and accuracy debugging, Debugging Numerical Correctness and Accuracy-Debugging Numerical Correctness and Accuracy
- numpy replacement cuPyNumeric, C++ and Python CUDA Libraries
- NV-HBI (High-Bandwidth Interface), NVIDIA Blackwell “Dual-Die” GPU
- nvcc CUDA compilercompatibility across GPU generations, CUDA Forward and Backward Compatibility Across GPU Hardware Generations, CUDA GPU Backward and Forward Compatibility Model-cubin option, CUDA Forward and Backward Compatibility Across GPU Hardware GenerationsCUDA Toolkit in software stack, CUDA Toolkit and Runtimefatbinary, CUDA Forward and Backward Compatibility Across GPU Hardware Generationsfloat8 not provided, Vectorized Memory Access__global__, CUDA Programming Refresher__launch_bounds__, Tuning Occupancy with Launch Bounds-Tuning Occupancy with Launch Boundsoptimizing for occupancy, Tuning Occupancy with Launch Bounds-Tuning Occupancy with Launch Boundsper-thread default stream enabled, Modern Per-Thread Default Streampredication, Techniques to Avoid Warp Divergence
- NVIDIAAI supercomputers in a rack, NVIDIA’s “AI Supercomputer in a Rack”-NVIDIA’s “AI Supercomputer in a Rack”CPU + GPU superchip, The CPU and GPU Superchip-Streaming Multiprocessor, Threads, and WarpsGPU Tensor Cores and Transformer Engine, NVIDIA GPU Tensor Cores and Transformer Engine-NVIDIA GPU Tensor Cores and Transformer Enginemany GPUs as one, Ultrascale Networking Treating Many GPUs as One-Co-Packaged Optics: Future of Networking Hardwaremulti-GPU programming, Multi-GPU Programming-Multi-GPU Programmingmultirack and storage communication, Multirack and Storage CommunicationNVIDIA Blackwell dual-die GPU, NVIDIA Blackwell “Dual-Die” GPU-NVIDIA Blackwell “Dual-Die” GPUNVIDIA Grace CPU, NVIDIA Grace CPUNVL72, Ultrascale Networking Treating Many GPUs as One(see also NVL72)NVLink and NVSwitch, NVLink and NVSwitch-NVLink and NVSwitchSHARP for in-network aggregations, In-Network Aggregations with NVIDIA SHARP-In-Network Aggregations with NVIDIA SHARPstreaming multiprocessors, threads, warps, Streaming Multiprocessor, Threads, and Warps-Streaming Multiprocessor, Threads, and Warpsfuture hardware, A Glimpse into the Future: NVIDIA’s Roadmap-Feynman GPU (2028) and Doubling Something Every YearBlackwell Ultra and Grace Blackwell Ultra, Blackwell Ultra and Grace Blackwell Ultradoubling every year, Feynman GPU (2028) and Doubling Something Every YearFeynman GPU, Feynman GPU (2028) and Doubling Something Every YearRubin Ultra and Vera Rubin Ultra, Rubin Ultra and Vera Rubin Ultra (2027)Vera Rubin Superchip, Vera Rubin Superchip (2026)stock drop on DeepSeek-R1 costs, Introduction and AI System Overview
- NVIDIA Base Command Manager, Preintegrated Rack Appliance
- NVIDIA Blackwell B200, Introduction and AI System Overview, NVIDIA Blackwell “Dual-Die” GPU, Choosing Threads-per-Block and Blocks-per-Grid Sizes, Understanding GPU Memory Hierarchy
- NVIDIA Blackwell B300 Ultra, Toward 100-Trillion-Parameter Models, Blackwell Ultra and Grace Blackwell Ultra
- NVIDIA Blackwell dual-die GPU, NVIDIA Blackwell “Dual-Die” GPU-NVIDIA Blackwell “Dual-Die” GPUremote memory access latency, NUMA Awareness and CPU Pinning
- NVIDIA Collective Communications Library (see NCCL (NVIDIA Collective Communications Library))
- NVIDIA Compute Sanitizer, Debugging Functional Correctness with NVIDIA Compute Sanitizercompute-sanitizer CLI, Debugging Functional Correctness with NVIDIA Compute SanitizerNVIDIA Tools Extension support, Debugging Functional Correctness with NVIDIA Compute Sanitizerconcurrency and synchronization, Using CUDA Streams with MoE Modelsfour primary tools, Debugging Functional Correctness with NVIDIA Compute Sanitizerinference correctness, Debugging Correctness Issues
- NVIDIA Container Toolkit, Container Runtime Optimizations for GPUs
- NVIDIA CPU + GPU superchip, The CPU and GPU Superchip-Streaming Multiprocessor, Threads, and WarpsGPU Tensor Cores and Transformer Engine, NVIDIA GPU Tensor Cores and Transformer Engine-NVIDIA GPU Tensor Cores and Transformer EngineNVIDIA Blackwell dual-die GPU, NVIDIA Blackwell “Dual-Die” GPU-NVIDIA Blackwell “Dual-Die” GPUappearing as one GPU, NVIDIA Blackwell “Dual-Die” GPUremote memory access latency, NUMA Awareness and CPU PinningNVIDIA Grace CPU, NVIDIA Grace CPUNVLink-C2C, The CPU and GPU Superchipstreaming multiprocessors, threads, warps, Streaming Multiprocessor, Threads, and Warps-Streaming Multiprocessor, Threads, and Warps
- NVIDIA Data Center GPU Manager (see Data Center GPU Manager (DCGM; NVIDIA))
- NVIDIA Data Loading Library (DALI), Multimodal Data Processing with NVIDIA DALI, Optimizing the Data Input Pipeline
- NVIDIA DCGM (see Data Center GPU Manager (DCGM; NVIDIA))
- NVIDIA Dynamo with core component NIXL, NVIDIA’s NIXL and Disaggregated Inferenceimproving throughput, NIXL and High-Performance Inference Systems Like NVIDIA Dynamo
- NVIDIA GB200 NVL72, NVIDIA’s “AI Supercomputer in a Rack”, Ultrascale Networking Treating Many GPUs as One, GPU and CPU-GPU Superchip ImprovementsP2P-capable NVLink, Enabling Peer-to-Peer DMA and UCXtraining throughput per GPU, Transparency and Reproducibility
- NVIDIA GB300 NVL72, NVIDIA’s “AI Supercomputer in a Rack”, Ultrascale Networking Treating Many GPUs as One, NVLink/NVSwitch Topology and Bandwidth ConstraintsGrace Blackwell Ultra Superchip, Blackwell Ultra and Grace Blackwell Ultra
- NVIDIA GDS (see GDS (GPUDirect Storage; NVIDIA))
- NVIDIA Grace Blackwell GB200 and GB300, NVIDIA’s “AI Supercomputer in a Rack”-NVIDIA’s “AI Supercomputer in a Rack”
- NVIDIA Grace CPU, NVIDIA Grace CPU
- NVIDIA H800 GPU, Introduction and AI System Overview, DeepSeek Scales to ~680-Billion Parameter Models Despite US Export Hardware Restrictions in China-DeepSeek Scales to ~680-Billion Parameter Models Despite US Export Hardware Restrictions in China
- NVIDIA Inference Xfer Library (see NIXL (NVIDIA Inference Xfer Library))
- NVIDIA Kubernetes GPU Operator, Kubernetes for Topology-Aware Container Orchestration and Networking
- NVIDIA Magnum IO, NVIDIA Magnum IO Optimization Stack, CPU and GPU Profiling with Linux perf
- NVIDIA Management Library (NVML)GPU-NIC affinity forced, Multinode and Multirack Communication with GPUDirect RDMANVLink telemetry and monitoring, Real-Time Link Telemetry and Monitoringpower limit monitoring, Kernel Compute Throughput Versus Peak GPU FLOPS
- NVIDIA NeMo, Creating High-Quality LLM Datasets with NVIDIA NeMo CuratorNeMo Curator, Creating High-Quality LLM Datasets with NVIDIA NeMo Curator
- NVIDIA NIXL (see NIXL (NVIDIA Inference Xfer Library))
- NVIDIA Performance Monitoring Unit (PMU), CPU and GPU Profiling with Linux perf
- NVIDIA SHMEM (NVSHMEM), Multi-GPU Programming, Fine-Grained GPU-to-GPU Memory Sharing with NVSHMEM-Fine-Grained GPU-to-GPU Memory Sharing with NVSHMEMCUDA Graphs, Fine-Grained GPU-to-GPU Memory Sharing with NVSHMEMwhen to use, Fine-Grained GPU-to-GPU Memory Sharing with NVSHMEM
- NVIDIA software stack, NVIDIA Software Stackabout, PyTorch and Higher-Level AI FrameworksCUDA kernels created in Python, NVIDIA Software StackCUDA Runtime and Toolkit, CUDA Toolkit and RuntimeToolkit C++ and Python libraries, C++ and Python CUDA LibrariesGPU driver, GPU Drivernvidia-smi, GPU DriverPyTorch introduction, PyTorch and Higher-Level AI Frameworks
- NVIDIA Tools Extension (see NVTX (NVIDIA Tools Extension))
- NVIDIA Topology-Aware GPU Selection (NVTAGS), Adaptive Process-GPU Mapping
- NVIDIA Transformer Engine (see Transformer Engine (TE; NVIDIA))
- NVIDIA WarpCUDA kernels in Python, NVIDIA Software StackCUDA Toolkit, C++ and Python CUDA Libraries
- nvidia-peermem driver, Multi-GPU Programming
- nvidia-persistenced daemon, GPU Persistence Mode
- nvidia-smi, GPU Driver
- nvJPEG, Tuning, Replicating, and Compressing Data
- NVL72, NVIDIA’s “AI Supercomputer in a Rack”, Ultrascale Networking Treating Many GPUs as OneAI factory building block, Co-Packaged Optics: Future of Networking HardwareBlackwell Ultra and Grace Blackwell Ultra upgrades, Blackwell Ultra and Grace Blackwell Ultradelivered as preintegrated rack appliance, Preintegrated Rack ApplianceKubernetes, Preintegrated Rack ApplianceNVIDIA Base Command Manager, Preintegrated Rack ApplianceSimple Linux Utility for Resource Management, Preintegrated Rack ApplianceGPU-to-GPU bandwidth, Multi-GPU Programming, Orchestrating Containers with Kubernetes Topology ManagerH100 comparison, Multi-GPU Programminginference throughput per GPU, Transparency and Reproducibilitymulti-GPU programming, Multi-GPU Programming-Multi-GPU Programmingmultirack and storage communication, Multirack and Storage CommunicationData Processing Units, Multirack and Storage CommunicationNVLink and NVSwitch, NVLink and NVSwitch-NVLink and NVSwitch, Co-Packaged Optics: Future of Networking Hardwareperformance monitoring, Performance Monitoring and Utilization in Practicepower requirements, Compute Density and Power Requirementsliquid cooling versus air cooling, Liquid Cooling Versus Air Cooling-Liquid Cooling Versus Air Coolingsharing and scheduling workloads, Sharing and Schedulingtopology and bandwidth constraints, NVLink/NVSwitch Topology and Bandwidth Constraints-NVLink/NVSwitch Topology and Bandwidth Constraintstelemetry and monitoring, Real-Time Link Telemetry and Monitoring
- NVLinkNCCL over dedicated streams, Multi-GPU Compute and Data Transfer Overlap with CUDA Streams-Multi-GPU Compute and Data Transfer Overlap with CUDA StreamsNVL in names, NVIDIA’s “AI Supercomputer in a Rack”NVL72, Ultrascale Networking Treating Many GPUs as One, Co-Packaged Optics: Future of Networking HardwareNVSwitch, NVLink and NVSwitch-NVLink and NVSwitchfine-tuned scheduling and, Coordinating NVSwitch Transfers with Fine-Tuned Scheduling-Coordinating NVSwitch Transfers with Fine-Tuned SchedulingMagnum IO support, NVIDIA Magnum IO Optimization StackNVIDIA SHARP, In-Network Aggregations with NVIDIA SHARP-In-Network Aggregations with NVIDIA SHARPNVIDIA supercomputer on a rack, NVIDIA’s “AI Supercomputer in a Rack”NVL72, Co-Packaged Optics: Future of Networking Hardwaretelemetry and monitoring, Real-Time Link Telemetry and Monitoringtopology and bandwidth constraints, NVLink/NVSwitch Topology and Bandwidth Constraints-NVLink/NVSwitch Topology and Bandwidth ConstraintsP2P-capable, Enabling Peer-to-Peer DMA and UCXsingle domain for data tensor pipeline, NVIDIA’s “AI Supercomputer in a Rack”topology and bandwidth constraints, NVLink/NVSwitch Topology and Bandwidth Constraints-NVLink/NVSwitch Topology and Bandwidth Constraintstelemetry and monitoring, Real-Time Link Telemetry and MonitoringTransformer Engine in modern GPUs, Mechanical Sympathy: Hardware-Software Codesign
- NVLink-C2C (chip-to-chip), The CPU and GPU Superchipcache coherency, The CPU and GPU Superchip, Multi-GPU ProgrammingGrace Blackwell CPU and GPU linked, NUMA Awareness and CPU Pinning, NUMA Awareness and CPU PinningLinux treatment of memory, NUMA Awareness and CPU Pinning
- NVMe (Linux)DeepSeek Fire-Flyer File System, DeepSeek’s Fire-Flyer File System-DeepSeek’s Fire-Flyer File Systemfast storage and data locality, Fast Storage and Data LocalityGPUDirect Storage, NUMA-Friendly Memory Allocation and Memory Pinning, Using NVIDIA GDS-Measuring GDS with gdsioparameters offloaded to, Offloading Parameters to CPU and NVMesequential versus random reads, Sequential Versus Random Read Patterns-Sequential Versus Random Read Patternstuning for throughput, Tuning NVMe and Filesystem for ThroughputXFS optimized, Sequential Versus Random Read Patterns
- NVMe over Fabrics (NVMe-oF), Multirack and Storage Communication
- NVML (see NVIDIA Management Library (NVML))
- NVSHMEM (see NVIDIA SHMEM (NVSHMEM))
- NVSwitch, NVLink and NVSwitch-NVLink and NVSwitchfine-tuned scheduling and, Coordinating NVSwitch Transfers with Fine-Tuned Scheduling-Coordinating NVSwitch Transfers with Fine-Tuned SchedulingMagnum IO support, NVIDIA Magnum IO Optimization StackNVIDIA SHARP, In-Network Aggregations with NVIDIA SHARP-In-Network Aggregations with NVIDIA SHARPNVIDIA supercomputer on a rack, NVIDIA’s “AI Supercomputer in a Rack”NVL72, Co-Packaged Optics: Future of Networking Hardwaretelemetry and monitoring, Real-Time Link Telemetry and Monitoringtopology and bandwidth constraints, NVLink/NVSwitch Topology and Bandwidth Constraints-NVLink/NVSwitch Topology and Bandwidth Constraints
- NVTAGS (NVIDIA Topology-Aware GPU Selection), Adaptive Process-GPU Mapping
- NVTX (NVIDIA Tools Extension)compute-sanitizer CLI supporting, Debugging Functional Correctness with NVIDIA Compute Sanitizermarkers, NVTX Markers and Profiling ToolsHolistic Trace Analysis, Multi-GPU Profiling with HTAprofiling tools that use markers, NVTX Markers and Profiling Tools-NVTX Markers and Profiling ToolsNsight Systemsexample MoE model, System Profiling with Nsight Systems and NVTX Timelines-System Profiling with Nsight Systems and NVTX Timelinestimeline view, Nsight Systems Timeline View, System Profiling with Nsight Systems and NVTX Timelines-System Profiling with Nsight Systems and NVTX Timelines

- about NCCL, Scaling Distributed Training and Inference, NCCL for Distributed Multi-GPU Communication, Optimizing Collective Communication with NCCL
- backend instead of GLOO, Pitfall #1: Using a CPU-bound Gloo backend instead of NCCL-Pitfall #1: Using a CPU-bound Gloo backend instead of NCCL
- bootstrap TCP port exhaustion, Pitfall #3: TCP port exhaustion during NCCL bootstrap
- CUDA Toolkit, CUDA Toolkit and Runtime
- debugging, Profiling and Debugging NCCLinference correctness issue, Debugging Correctness IssuesNCCL debugging enabled, Debugging Correctness IssuesNCCL logs, In-Network SHARP Aggregation, Debugging Correctness Issuestest suite, Debugging Correctness Issues
- dedicated high-priority streams for network transfers, Multi-GPU Compute and Data Transfer Overlap with CUDA Streamsring or tree algorithm, Multi-GPU Compute and Data Transfer Overlap with CUDA Streams-Multi-GPU Compute and Data Transfer Overlap with CUDA Streams, Ring versus tree all-reduce-Ring versus tree all-reduce
- distributed multi-GPU communication, NCCL for Distributed Multi-GPU Communication-In-Network SHARP Aggregationabout, NCCL for Distributed Multi-GPU Communicationall-reduces in parallelism, Distributed Data Parallel StrategiesCollNet communication algorithm, NCCL Communication Algorithmscommunication algorithms, NCCL Communication Algorithms-NCCL Communication Algorithmsmany-to-many collectives, NVIDIA’s NIXL and Disaggregated Inference, NVIDIA’s NIXL and Disaggregated Inference, NCCL Versus NIXLNCCL_ALGO environment variable, Topology Awareness in NCCL, NCCL Communication Algorithmsoptimizing, Optimizing Collective Communication with NCCL-Wave scheduling of collectivesparallel aggregated tree communication algorithm, NCCL Communication Algorithmsring communication algorithm, NCCL Communication Algorithmstopology awareness, Topology Awareness in NCCL-Topology Awareness in NCCLtree communication algorithm, NCCL Communication Algorithmswave scheduling of collectives, Wave scheduling of collectives
- distributed multi-GPU communicationsring or tree algorithm, Ring versus tree all-reduce-Ring versus tree all-reducerotating ring endpoints, Rotating ring endpoints
- GPU-NIC affinity forced, Multinode and Multirack Communication with GPUDirect RDMA
- maximum TCP socket connections per communicator, NCCL_NSOCKS_PERTHREAD and NCCL_SOCKET_NTHREADS
- memory allocator in PyTorch, Pluggable Memory Allocators and Cross-GPU Data Transfers
- multiple NICs in parallel, Multinode and Multirack Communication with GPUDirect RDMA
- multiple-GPU collectives captured, Capturing Multi-GPU Collectives with NCCL and CUDA Graphs-Capturing Multi-GPU Collectives with NCCL and CUDA Graphs
- NIXL versus, NCCL Versus NIXLmany-to-many collectives by NCCL, NVIDIA’s NIXL and Disaggregated Inference, NVIDIA’s NIXL and Disaggregated Inference, NCCL Versus NIXL
- persistent user buffers, Persistent NCCL User Buffers and Zero-Copy Registration
- pitfalls and gotchas, NCCL Communicator Lifecycle and Environment Gotchas-Pitfall #6: NCCL communicator hangs, errors, or shuts down completelycommunicator hanging, giving errors, shutting down, Pitfall #6: NCCL communicator hangs, errors, or shuts down completely-Pitfall #6: NCCL communicator hangs, errors, or shuts down completelyCPU-GPU NUMA-node for NCCL threads, Pitfall #4: Verify CPU-GPU NUMA-node affinity for NCCL threadscreating and destroying communicators, Pitfall #2: Do not create and destroy NCCL communicators on every iterationcreating communicators too often, Pitfall #1: Creating NCCL communicators too often-Pitfall #1: Creating NCCL communicators too oftenignoring warnings and errors, Pitfall #5: Resist the temptation to ignore NCCL warnings and errorsoverturning or disabling features via environment variables, Pitfall #3: Avoid overtuning or disabling NCCL features with environment variables-NCCL_SHARP_DISABLEversions mismatched, Pitfall #2: Mismatched NCCL versions
- profiling, Profiling and Debugging NCCLNCCL logs, In-Network SHARP Aggregation
- RDMA container access to devices, High-Speed, Low-Overhead Data Transfers with RDMA
- SHARP and, In-Network SHARP Aggregation
- UCX for multinode topologies, Enabling Peer-to-Peer DMA and UCX
- user buffer registration, Persistent NCCL User Buffers and Zero-Copy Registration
- versions mismatched, Pitfall #2: Mismatched NCCL versions

- inference correctness issue, Debugging Correctness Issues
- NCCL debugging enabled, Debugging Correctness Issues
- NCCL logs, In-Network SHARP Aggregation, Debugging Correctness Issues
- test suite, Debugging Correctness Issues

- ring or tree algorithm, Multi-GPU Compute and Data Transfer Overlap with CUDA Streams-Multi-GPU Compute and Data Transfer Overlap with CUDA Streams, Ring versus tree all-reduce-Ring versus tree all-reduce

- about, NCCL for Distributed Multi-GPU Communication
- all-reduces in parallelism, Distributed Data Parallel Strategies
- CollNet communication algorithm, NCCL Communication Algorithms
- communication algorithms, NCCL Communication Algorithms-NCCL Communication Algorithms
- many-to-many collectives, NVIDIA’s NIXL and Disaggregated Inference, NVIDIA’s NIXL and Disaggregated Inference, NCCL Versus NIXL
- NCCL_ALGO environment variable, Topology Awareness in NCCL, NCCL Communication Algorithms
- optimizing, Optimizing Collective Communication with NCCL-Wave scheduling of collectives
- parallel aggregated tree communication algorithm, NCCL Communication Algorithms
- ring communication algorithm, NCCL Communication Algorithms
- topology awareness, Topology Awareness in NCCL-Topology Awareness in NCCL
- tree communication algorithm, NCCL Communication Algorithms
- wave scheduling of collectives, Wave scheduling of collectives

- ring or tree algorithm, Ring versus tree all-reduce-Ring versus tree all-reduce
- rotating ring endpoints, Rotating ring endpoints

- many-to-many collectives by NCCL, NVIDIA’s NIXL and Disaggregated Inference, NVIDIA’s NIXL and Disaggregated Inference, NCCL Versus NIXL

- communicator hanging, giving errors, shutting down, Pitfall #6: NCCL communicator hangs, errors, or shuts down completely-Pitfall #6: NCCL communicator hangs, errors, or shuts down completely
- CPU-GPU NUMA-node for NCCL threads, Pitfall #4: Verify CPU-GPU NUMA-node affinity for NCCL threads
- creating and destroying communicators, Pitfall #2: Do not create and destroy NCCL communicators on every iteration
- creating communicators too often, Pitfall #1: Creating NCCL communicators too often-Pitfall #1: Creating NCCL communicators too often
- ignoring warnings and errors, Pitfall #5: Resist the temptation to ignore NCCL warnings and errors
- overturning or disabling features via environment variables, Pitfall #3: Avoid overtuning or disabling NCCL features with environment variables-NCCL_SHARP_DISABLE
- versions mismatched, Pitfall #2: Mismatched NCCL versions

- NCCL logs, In-Network SHARP Aggregation

- NCCL logs, In-Network SHARP Aggregation, Debugging Correctness Issues

- profiler, Profiling and Diagnosing GPU Bottlenecks, NVTX Markers and Profiling Tools, NVTX Markers and Profiling Tools
- Roofline analysis, Nsight Compute and Roofline Analysis, Kernel Roofline Analysis for General Matrix Multiply (GEMM)-Kernel Roofline Analysis for General Matrix Multiply (GEMM)

- tuning parameters, Distributed, Parallel Filesystems and Object Stores

- about NIXL, Scaling Distributed Training and Inference, NCCL for Distributed Multi-GPU Communication, NVIDIA’s NIXL and Disaggregated Inference-NVIDIA’s NIXL and Disaggregated Inference
- dedicated high-priority streams for network transfers, Multi-GPU Compute and Data Transfer Overlap with CUDA Streams
- disaggregated inference, NVIDIA’s NIXL and Disaggregated Inference-NCCL Versus NIXLabout NIXL, NVIDIA’s NIXL and Disaggregated Inference-NVIDIA’s NIXL and Disaggregated Inferenceasynchronous API with callbacks, NIXL Asynchronous API with Callbacks-NIXL Asynchronous API with CallbacksDynamo throughput improved with NIXL, NIXL and High-Performance Inference Systems Like NVIDIA Dynamointelligent interconnect routing, Intelligent Interconnect Routing for KV Cache Transfers, KV Cache Data Transfer and NIXLKV cache offloading with NIXL, KV Cache Offloading with NIXL-KV Cache Offloading with NIXLprefill to decode data transfers, KV Cache Data Transfer and NIXLseparate prefill and decode inference stages, Separate Prefill and Decode Inference Stages-Separate Prefill and Decode Inference Stages
- NCCL versus, NCCL Versus NIXLone-to-one transfers by NIXL, Distributed Data Parallel Strategies, NVIDIA’s NIXL and Disaggregated Inference, NVIDIA’s NIXL and Disaggregated Inference, NCCL Versus NIXL
- NVIDIA Dynamo core component, NVIDIA’s NIXL and Disaggregated InferenceDynamo throughput improved, NIXL and High-Performance Inference Systems Like NVIDIA Dynamo

- about NIXL, NVIDIA’s NIXL and Disaggregated Inference-NVIDIA’s NIXL and Disaggregated Inference
- asynchronous API with callbacks, NIXL Asynchronous API with Callbacks-NIXL Asynchronous API with Callbacks
- Dynamo throughput improved with NIXL, NIXL and High-Performance Inference Systems Like NVIDIA Dynamo
- intelligent interconnect routing, Intelligent Interconnect Routing for KV Cache Transfers, KV Cache Data Transfer and NIXL
- KV cache offloading with NIXL, KV Cache Offloading with NIXL-KV Cache Offloading with NIXL
- prefill to decode data transfers, KV Cache Data Transfer and NIXL
- separate prefill and decode inference stages, Separate Prefill and Decode Inference Stages-Separate Prefill and Decode Inference Stages

- one-to-one transfers by NIXL, Distributed Data Parallel Strategies, NVIDIA’s NIXL and Disaggregated Inference, NVIDIA’s NIXL and Disaggregated Inference, NCCL Versus NIXL

- Dynamo throughput improved, NIXL and High-Performance Inference Systems Like NVIDIA Dynamo

- achieved occupancy, Inspecting Achieved Occupancy and GPU Utilization-Optimizing the Kernelkernel compute throughput versus GPU FLOPS, Kernel Compute Throughput Versus Peak GPU FLOPSkernel memory throughput versus HBM memory bandwidth, Kernel Memory Throughput Versus Peak HBM Memory Bandwidthunderutilization, Dynamic Scheduling with Atomic Work Queues
- ALU pipe busy, Execution Unit Contention
- CLI ncuprofiler, Profiling and Diagnosing GPU Bottlenecks, NVTX Markers and Profiling Tools, NVTX Markers and Profiling ToolsRoofline analysis, Nsight Compute and Roofline Analysis, Kernel Roofline Analysis for General Matrix Multiply (GEMM)-Kernel Roofline Analysis for General Matrix Multiply (GEMM)
- CUDA Program Counter Sampling, Profiling with Nsight Systems and Nsight Compute
- ILP configuration, ILP and Occupancy
- impact of parallel versus serial operation, Maintaining High Occupancy and GPU Utilization
- inference profiling, Profiling with Nsight Systems and Nsight ComputeCUDA Program Counter Sampling, Profiling with Nsight Systems and Nsight Compute
- kernel shift from memory to compute bound, Feeding Tensor Cores with TMEM and TMA
- launch statistics, Designing Efficient Algorithms with Thread Block Clusters
- Memory Workload Analysisatomic transactions counter, Atomic Counters-Atomic Counterscoalesced versus uncoalesced memory accesses, Coalesced Versus Uncoalesced Global Memory Access
- memory workload analysis, Kernel Autotuning for Transformer Self-Attention and MLP Paths
- OccupancyLimited by Registers, Profiling and Mitigating Register Pressurelow achieved FLOPS, Kernel Compute Throughput Versus Peak GPU FLOPSOccupancy Calculator, Techniques for Occupancy Tuningoccupancy limiters, Inspecting Achieved Occupancy and GPU Utilization
- parallelism strategies profiled, Hybrid Parallelism
- per-kernel metrics, Read-Only Data Caches
- Registers per Thread and Occupancy metrics, Tuning Occupancy with Launch Bounds
- Roofline analysis, Nsight Compute and Roofline Analysisexample MoE model, Kernel Roofline Analysis for General Matrix Multiply (GEMM)-Kernel Roofline Analysis for General Matrix Multiply (GEMM)
- Source Counterslow achieved FLOPS, Kernel Compute Throughput Versus Peak GPU FLOPSwarp divergence profiled, Profiling and Detecting Warp Divergence
- Speed of Light analysis, Mixed Precision and Utilizing Tensor Cores
- StallCompute Unit Busy, Execution Unit ContentionExec Dependency, Execution-Dependency Stalls, Profiling and Mitigating Register PressureIdle, Other Stall ReasonsLong Scoreboard, Memory-Related StallsMath Pipe Throttle, Execution Unit ContentionMemory Dependency, Other Stall ReasonsMemory Throttle, Memory-Related StallsNo Eligible, Other Stall ReasonsNot Selected, Memory-Related Stallsother reasons, Other Stall Reasons-Other Stall ReasonsShort Scoreboard, Memory-Related StallsTexture Throttle, Other Stall Reasons
- Triton kernel support, Profiling with Triton Proton Profiler
- underutilization metrics, Dynamic Scheduling with Atomic Work Queues
- updated to latest GPUs, Profiler-Guided Analysis
- Warp State, Profiling and Detecting Warp Divergence

- kernel compute throughput versus GPU FLOPS, Kernel Compute Throughput Versus Peak GPU FLOPS
- kernel memory throughput versus HBM memory bandwidth, Kernel Memory Throughput Versus Peak HBM Memory Bandwidth
- underutilization, Dynamic Scheduling with Atomic Work Queues

- profiler, Profiling and Diagnosing GPU Bottlenecks, NVTX Markers and Profiling Tools, NVTX Markers and Profiling Tools
- Roofline analysis, Nsight Compute and Roofline Analysis, Kernel Roofline Analysis for General Matrix Multiply (GEMM)-Kernel Roofline Analysis for General Matrix Multiply (GEMM)

- CUDA Program Counter Sampling, Profiling with Nsight Systems and Nsight Compute

- atomic transactions counter, Atomic Counters-Atomic Counters
- coalesced versus uncoalesced memory accesses, Coalesced Versus Uncoalesced Global Memory Access

- Limited by Registers, Profiling and Mitigating Register Pressure
- low achieved FLOPS, Kernel Compute Throughput Versus Peak GPU FLOPS
- Occupancy Calculator, Techniques for Occupancy Tuning
- occupancy limiters, Inspecting Achieved Occupancy and GPU Utilization

- example MoE model, Kernel Roofline Analysis for General Matrix Multiply (GEMM)-Kernel Roofline Analysis for General Matrix Multiply (GEMM)

- low achieved FLOPS, Kernel Compute Throughput Versus Peak GPU FLOPS
- warp divergence profiled, Profiling and Detecting Warp Divergence

- Compute Unit Busy, Execution Unit Contention
- Exec Dependency, Execution-Dependency Stalls, Profiling and Mitigating Register Pressure
- Idle, Other Stall Reasons
- Long Scoreboard, Memory-Related Stalls
- Math Pipe Throttle, Execution Unit Contention
- Memory Dependency, Other Stall Reasons
- Memory Throttle, Memory-Related Stalls
- No Eligible, Other Stall Reasons
- Not Selected, Memory-Related Stalls
- other reasons, Other Stall Reasons-Other Stall Reasons
- Short Scoreboard, Memory-Related Stalls
- Texture Throttle, Other Stall Reasons

- about, Profiling and Diagnosing GPU Bottlenecks
- CLI nsysprofiler, Profiling and Diagnosing GPU Bottlenecks, NVTX Markers and Profiling Tools, NVTX Markers and Profiling Tools, System Profiling with Nsight Systems and NVTX Timelines-System Profiling with Nsight Systems and NVTX Timelinestimeline view, Nsight Systems Timeline View, System Profiling with Nsight Systems and NVTX Timelines-System Profiling with Nsight Systems and NVTX Timelines
- communication versus compute bottleneck, Diagnosing Communication- Versus Compute-Bound Workloads
- I/O monitoring, Monitoring Storage I/O
- impact of parallel versus serial operation, Maintaining High Occupancy and GPU Utilization
- inference profiling, Profiling with Nsight Systems and Nsight Compute-Profiling with Nsight Systems and Nsight Compute
- NVLink profiling, Topology Awareness in NCCL, Real-Time Link Telemetry and Monitoring
- parallelism strategies profiled, Hybrid Parallelism
- Python backtrace sampling, PyTorch Profiler and Visualization Tools
- PyTorch focused mode, PyTorch Profiler and Visualization Tools
- timeline view, Read-Only Data Caches, Nsight Systems Timeline View, Nsight Compute and Roofline Analysis, System Profiling with Nsight Systems and NVTX Timelines-System Profiling with Nsight Systems and NVTX Timelines, Kernel Autotuning for Transformer Self-Attention and MLP PathsNVTX markers, Nsight Systems Timeline View, System Profiling with Nsight Systems and NVTX Timelines-System Profiling with Nsight Systems and NVTX Timelines
- Triton kernel support, Profiling with Triton Proton Profiler
- updated to latest GPUs, Profiler-Guided Analysis

- profiler, Profiling and Diagnosing GPU Bottlenecks, NVTX Markers and Profiling Tools, NVTX Markers and Profiling Tools, System Profiling with Nsight Systems and NVTX Timelines-System Profiling with Nsight Systems and NVTX Timelines
- timeline view, Nsight Systems Timeline View, System Profiling with Nsight Systems and NVTX Timelines-System Profiling with Nsight Systems and NVTX Timelines

- NVTX markers, Nsight Systems Timeline View, System Profiling with Nsight Systems and NVTX Timelines-System Profiling with Nsight Systems and NVTX Timelines

- profiler, Profiling and Diagnosing GPU Bottlenecks, NVTX Markers and Profiling Tools, NVTX Markers and Profiling Tools, System Profiling with Nsight Systems and NVTX Timelines-System Profiling with Nsight Systems and NVTX Timelines
- timeline view, Nsight Systems Timeline View, System Profiling with Nsight Systems and NVTX Timelines-System Profiling with Nsight Systems and NVTX Timelines

- NUMA awareness and CPU pinning, NUMA Awareness and CPU Pinning-NUMA Awareness and CPU PinningCPU pinning, NUMA Awareness and CPU Pinning-NUMA Awareness and CPU Pinning
- NUMA-friendly memory allocation and pinning, NUMA-Friendly Memory Allocation and Memory Pinning-NUMA-Friendly Memory Allocation and Memory Pinning

- CPU pinning, NUMA Awareness and CPU Pinning-NUMA Awareness and CPU Pinning

- CPU pinning, NUMA Awareness and CPU Pinning, NUMA Awareness and CPU Pinning
- forcing memory allocation from specific NUMA node, NUMA Awareness and CPU Pinning, NUMA-Friendly Memory Allocation and Memory Pinning
- process CPU and memory policies applied on launch, NUMA-Friendly Memory Allocation and Memory Pinning

- compatibility across GPU generations, CUDA Forward and Backward Compatibility Across GPU Hardware Generations, CUDA GPU Backward and Forward Compatibility Model
- -cubin option, CUDA Forward and Backward Compatibility Across GPU Hardware Generations
- CUDA Toolkit in software stack, CUDA Toolkit and Runtime
- fatbinary, CUDA Forward and Backward Compatibility Across GPU Hardware Generations
- float8 not provided, Vectorized Memory Access
- __global__, CUDA Programming Refresher
- __launch_bounds__, Tuning Occupancy with Launch Bounds-Tuning Occupancy with Launch Bounds
- optimizing for occupancy, Tuning Occupancy with Launch Bounds-Tuning Occupancy with Launch Bounds
- per-thread default stream enabled, Modern Per-Thread Default Stream
- predication, Techniques to Avoid Warp Divergence

- AI supercomputers in a rack, NVIDIA’s “AI Supercomputer in a Rack”-NVIDIA’s “AI Supercomputer in a Rack”CPU + GPU superchip, The CPU and GPU Superchip-Streaming Multiprocessor, Threads, and WarpsGPU Tensor Cores and Transformer Engine, NVIDIA GPU Tensor Cores and Transformer Engine-NVIDIA GPU Tensor Cores and Transformer Enginemany GPUs as one, Ultrascale Networking Treating Many GPUs as One-Co-Packaged Optics: Future of Networking Hardwaremulti-GPU programming, Multi-GPU Programming-Multi-GPU Programmingmultirack and storage communication, Multirack and Storage CommunicationNVIDIA Blackwell dual-die GPU, NVIDIA Blackwell “Dual-Die” GPU-NVIDIA Blackwell “Dual-Die” GPUNVIDIA Grace CPU, NVIDIA Grace CPUNVL72, Ultrascale Networking Treating Many GPUs as One(see also NVL72)NVLink and NVSwitch, NVLink and NVSwitch-NVLink and NVSwitchSHARP for in-network aggregations, In-Network Aggregations with NVIDIA SHARP-In-Network Aggregations with NVIDIA SHARPstreaming multiprocessors, threads, warps, Streaming Multiprocessor, Threads, and Warps-Streaming Multiprocessor, Threads, and Warps
- future hardware, A Glimpse into the Future: NVIDIA’s Roadmap-Feynman GPU (2028) and Doubling Something Every YearBlackwell Ultra and Grace Blackwell Ultra, Blackwell Ultra and Grace Blackwell Ultradoubling every year, Feynman GPU (2028) and Doubling Something Every YearFeynman GPU, Feynman GPU (2028) and Doubling Something Every YearRubin Ultra and Vera Rubin Ultra, Rubin Ultra and Vera Rubin Ultra (2027)Vera Rubin Superchip, Vera Rubin Superchip (2026)
- stock drop on DeepSeek-R1 costs, Introduction and AI System Overview

- CPU + GPU superchip, The CPU and GPU Superchip-Streaming Multiprocessor, Threads, and Warps
- GPU Tensor Cores and Transformer Engine, NVIDIA GPU Tensor Cores and Transformer Engine-NVIDIA GPU Tensor Cores and Transformer Engine
- many GPUs as one, Ultrascale Networking Treating Many GPUs as One-Co-Packaged Optics: Future of Networking Hardware
- multi-GPU programming, Multi-GPU Programming-Multi-GPU Programming
- multirack and storage communication, Multirack and Storage Communication
- NVIDIA Blackwell dual-die GPU, NVIDIA Blackwell “Dual-Die” GPU-NVIDIA Blackwell “Dual-Die” GPU
- NVIDIA Grace CPU, NVIDIA Grace CPU
- NVL72, Ultrascale Networking Treating Many GPUs as One(see also NVL72)
- NVLink and NVSwitch, NVLink and NVSwitch-NVLink and NVSwitch
- SHARP for in-network aggregations, In-Network Aggregations with NVIDIA SHARP-In-Network Aggregations with NVIDIA SHARP
- streaming multiprocessors, threads, warps, Streaming Multiprocessor, Threads, and Warps-Streaming Multiprocessor, Threads, and Warps

- (see also NVL72)

- Blackwell Ultra and Grace Blackwell Ultra, Blackwell Ultra and Grace Blackwell Ultra
- doubling every year, Feynman GPU (2028) and Doubling Something Every Year
- Feynman GPU, Feynman GPU (2028) and Doubling Something Every Year
- Rubin Ultra and Vera Rubin Ultra, Rubin Ultra and Vera Rubin Ultra (2027)
- Vera Rubin Superchip, Vera Rubin Superchip (2026)

- remote memory access latency, NUMA Awareness and CPU Pinning

- compute-sanitizer CLI, Debugging Functional Correctness with NVIDIA Compute SanitizerNVIDIA Tools Extension support, Debugging Functional Correctness with NVIDIA Compute Sanitizer
- concurrency and synchronization, Using CUDA Streams with MoE Models
- four primary tools, Debugging Functional Correctness with NVIDIA Compute Sanitizer
- inference correctness, Debugging Correctness Issues

- NVIDIA Tools Extension support, Debugging Functional Correctness with NVIDIA Compute Sanitizer

- GPU Tensor Cores and Transformer Engine, NVIDIA GPU Tensor Cores and Transformer Engine-NVIDIA GPU Tensor Cores and Transformer Engine
- NVIDIA Blackwell dual-die GPU, NVIDIA Blackwell “Dual-Die” GPU-NVIDIA Blackwell “Dual-Die” GPUappearing as one GPU, NVIDIA Blackwell “Dual-Die” GPUremote memory access latency, NUMA Awareness and CPU Pinning
- NVIDIA Grace CPU, NVIDIA Grace CPU
- NVLink-C2C, The CPU and GPU Superchip
- streaming multiprocessors, threads, warps, Streaming Multiprocessor, Threads, and Warps-Streaming Multiprocessor, Threads, and Warps

- appearing as one GPU, NVIDIA Blackwell “Dual-Die” GPU
- remote memory access latency, NUMA Awareness and CPU Pinning

- improving throughput, NIXL and High-Performance Inference Systems Like NVIDIA Dynamo

- P2P-capable NVLink, Enabling Peer-to-Peer DMA and UCX
- training throughput per GPU, Transparency and Reproducibility

- Grace Blackwell Ultra Superchip, Blackwell Ultra and Grace Blackwell Ultra

- GPU-NIC affinity forced, Multinode and Multirack Communication with GPUDirect RDMA
- NVLink telemetry and monitoring, Real-Time Link Telemetry and Monitoring
- power limit monitoring, Kernel Compute Throughput Versus Peak GPU FLOPS

- NeMo Curator, Creating High-Quality LLM Datasets with NVIDIA NeMo Curator

- CUDA Graphs, Fine-Grained GPU-to-GPU Memory Sharing with NVSHMEM
- when to use, Fine-Grained GPU-to-GPU Memory Sharing with NVSHMEM

- about, PyTorch and Higher-Level AI Frameworks
- CUDA kernels created in Python, NVIDIA Software Stack
- CUDA Runtime and Toolkit, CUDA Toolkit and RuntimeToolkit C++ and Python libraries, C++ and Python CUDA Libraries
- GPU driver, GPU Drivernvidia-smi, GPU Driver
- PyTorch introduction, PyTorch and Higher-Level AI Frameworks

- Toolkit C++ and Python libraries, C++ and Python CUDA Libraries

- nvidia-smi, GPU Driver

- CUDA kernels in Python, NVIDIA Software Stack
- CUDA Toolkit, C++ and Python CUDA Libraries

- AI factory building block, Co-Packaged Optics: Future of Networking Hardware
- Blackwell Ultra and Grace Blackwell Ultra upgrades, Blackwell Ultra and Grace Blackwell Ultra
- delivered as preintegrated rack appliance, Preintegrated Rack ApplianceKubernetes, Preintegrated Rack ApplianceNVIDIA Base Command Manager, Preintegrated Rack ApplianceSimple Linux Utility for Resource Management, Preintegrated Rack Appliance
- GPU-to-GPU bandwidth, Multi-GPU Programming, Orchestrating Containers with Kubernetes Topology Manager
- H100 comparison, Multi-GPU Programming
- inference throughput per GPU, Transparency and Reproducibility
- multi-GPU programming, Multi-GPU Programming-Multi-GPU Programming
- multirack and storage communication, Multirack and Storage CommunicationData Processing Units, Multirack and Storage Communication
- NVLink and NVSwitch, NVLink and NVSwitch-NVLink and NVSwitch, Co-Packaged Optics: Future of Networking Hardware
- performance monitoring, Performance Monitoring and Utilization in Practice
- power requirements, Compute Density and Power Requirementsliquid cooling versus air cooling, Liquid Cooling Versus Air Cooling-Liquid Cooling Versus Air Cooling
- sharing and scheduling workloads, Sharing and Scheduling
- topology and bandwidth constraints, NVLink/NVSwitch Topology and Bandwidth Constraints-NVLink/NVSwitch Topology and Bandwidth Constraintstelemetry and monitoring, Real-Time Link Telemetry and Monitoring

- Kubernetes, Preintegrated Rack Appliance
- NVIDIA Base Command Manager, Preintegrated Rack Appliance
- Simple Linux Utility for Resource Management, Preintegrated Rack Appliance

- Data Processing Units, Multirack and Storage Communication

- liquid cooling versus air cooling, Liquid Cooling Versus Air Cooling-Liquid Cooling Versus Air Cooling

- telemetry and monitoring, Real-Time Link Telemetry and Monitoring

- NCCL over dedicated streams, Multi-GPU Compute and Data Transfer Overlap with CUDA Streams-Multi-GPU Compute and Data Transfer Overlap with CUDA Streams
- NVL in names, NVIDIA’s “AI Supercomputer in a Rack”
- NVL72, Ultrascale Networking Treating Many GPUs as One, Co-Packaged Optics: Future of Networking Hardware
- NVSwitch, NVLink and NVSwitch-NVLink and NVSwitchfine-tuned scheduling and, Coordinating NVSwitch Transfers with Fine-Tuned Scheduling-Coordinating NVSwitch Transfers with Fine-Tuned SchedulingMagnum IO support, NVIDIA Magnum IO Optimization StackNVIDIA SHARP, In-Network Aggregations with NVIDIA SHARP-In-Network Aggregations with NVIDIA SHARPNVIDIA supercomputer on a rack, NVIDIA’s “AI Supercomputer in a Rack”NVL72, Co-Packaged Optics: Future of Networking Hardwaretelemetry and monitoring, Real-Time Link Telemetry and Monitoringtopology and bandwidth constraints, NVLink/NVSwitch Topology and Bandwidth Constraints-NVLink/NVSwitch Topology and Bandwidth Constraints
- P2P-capable, Enabling Peer-to-Peer DMA and UCX
- single domain for data tensor pipeline, NVIDIA’s “AI Supercomputer in a Rack”
- topology and bandwidth constraints, NVLink/NVSwitch Topology and Bandwidth Constraints-NVLink/NVSwitch Topology and Bandwidth Constraintstelemetry and monitoring, Real-Time Link Telemetry and Monitoring
- Transformer Engine in modern GPUs, Mechanical Sympathy: Hardware-Software Codesign

- fine-tuned scheduling and, Coordinating NVSwitch Transfers with Fine-Tuned Scheduling-Coordinating NVSwitch Transfers with Fine-Tuned Scheduling
- Magnum IO support, NVIDIA Magnum IO Optimization Stack
- NVIDIA SHARP, In-Network Aggregations with NVIDIA SHARP-In-Network Aggregations with NVIDIA SHARP
- NVIDIA supercomputer on a rack, NVIDIA’s “AI Supercomputer in a Rack”
- NVL72, Co-Packaged Optics: Future of Networking Hardware
- telemetry and monitoring, Real-Time Link Telemetry and Monitoring
- topology and bandwidth constraints, NVLink/NVSwitch Topology and Bandwidth Constraints-NVLink/NVSwitch Topology and Bandwidth Constraints

- telemetry and monitoring, Real-Time Link Telemetry and Monitoring

- cache coherency, The CPU and GPU Superchip, Multi-GPU Programming
- Grace Blackwell CPU and GPU linked, NUMA Awareness and CPU Pinning, NUMA Awareness and CPU Pinning
- Linux treatment of memory, NUMA Awareness and CPU Pinning

- DeepSeek Fire-Flyer File System, DeepSeek’s Fire-Flyer File System-DeepSeek’s Fire-Flyer File System
- fast storage and data locality, Fast Storage and Data Locality
- GPUDirect Storage, NUMA-Friendly Memory Allocation and Memory Pinning, Using NVIDIA GDS-Measuring GDS with gdsio
- parameters offloaded to, Offloading Parameters to CPU and NVMe
- sequential versus random reads, Sequential Versus Random Read Patterns-Sequential Versus Random Read Patterns
- tuning for throughput, Tuning NVMe and Filesystem for Throughput
- XFS optimized, Sequential Versus Random Read Patterns

- fine-tuned scheduling and, Coordinating NVSwitch Transfers with Fine-Tuned Scheduling-Coordinating NVSwitch Transfers with Fine-Tuned Scheduling
- Magnum IO support, NVIDIA Magnum IO Optimization Stack
- NVIDIA SHARP, In-Network Aggregations with NVIDIA SHARP-In-Network Aggregations with NVIDIA SHARP
- NVIDIA supercomputer on a rack, NVIDIA’s “AI Supercomputer in a Rack”
- NVL72, Co-Packaged Optics: Future of Networking Hardware
- telemetry and monitoring, Real-Time Link Telemetry and Monitoring
- topology and bandwidth constraints, NVLink/NVSwitch Topology and Bandwidth Constraints-NVLink/NVSwitch Topology and Bandwidth Constraints

- compute-sanitizer CLI supporting, Debugging Functional Correctness with NVIDIA Compute Sanitizer
- markers, NVTX Markers and Profiling ToolsHolistic Trace Analysis, Multi-GPU Profiling with HTAprofiling tools that use markers, NVTX Markers and Profiling Tools-NVTX Markers and Profiling Tools
- Nsight Systemsexample MoE model, System Profiling with Nsight Systems and NVTX Timelines-System Profiling with Nsight Systems and NVTX Timelinestimeline view, Nsight Systems Timeline View, System Profiling with Nsight Systems and NVTX Timelines-System Profiling with Nsight Systems and NVTX Timelines

- Holistic Trace Analysis, Multi-GPU Profiling with HTA
- profiling tools that use markers, NVTX Markers and Profiling Tools-NVTX Markers and Profiling Tools

- example MoE model, System Profiling with Nsight Systems and NVTX Timelines-System Profiling with Nsight Systems and NVTX Timelines
- timeline view, Nsight Systems Timeline View, System Profiling with Nsight Systems and NVTX Timelines-System Profiling with Nsight Systems and NVTX Timelines

### O

- Object Storage Targets (OSTs), Distributed, Parallel Filesystems and Object Stores
- occupancyachieved occupancy and GPU utilization, Inspecting Achieved Occupancy and GPU Utilization-Optimizing the Kerneldefinition of achieved occupancy, Inspecting Achieved Occupancy and GPU Utilizationkernel compute throughput versus GPU FLOPS, Kernel Compute Throughput Versus Peak GPU FLOPSkernel memory throughput versus HBM memory bandwidth, Kernel Memory Throughput Versus Peak HBM Memory Bandwidthconfiguring launch parameters, Configuring Launch Parameters: Blocks per Grid and Threads per BlockCUDA Occupancy API, Techniques for Occupancy Tuning, Techniques for Occupancy Tuningoccupancy optimization, Determine Optimal Launch Configuration with the Occupancy APIdefinition of occupancy, Maintaining High Occupancy and GPU Utilization, Tuning Occupancyhardware limiting, Choosing Threads-per-Block and Blocks-per-Grid Sizes, Choosing Threads-per-Block and Blocks-per-Grid Sizeshigh occupancy, Threads, Warps, Blocks, and Gridshigh ILP with, Loop Unrolling, Interleaving, and Compiler Hintingmaintaining high occupancy, Maintaining High Occupancy and GPU Utilization-Maintaining High Occupancy and GPU UtilizationILP and, Exposing Instruction-Level Parallelism, ILP and Occupancy, Loop Unrolling, Interleaving, and Compiler Hintingimpact of occupancy, Maintaining High Occupancy and GPU Utilization-Maintaining High Occupancy and GPU Utilizationincreasing occupancy and per-thread resources, Tuning Occupancy with Launch Bounds, Tuning Occupancy, Compiler Hints to Optimize Occupancyoccupancy-aware kernel selection, Dynamic Shared-Memory Allocation and Occupancy-Aware Kernel Selection-Dynamic Shared-Memory Allocation and Occupancy-Aware Kernel Selectionpersistent kernels, Persistent Kernels and Megakernelsregister trade-off, Tuning Occupancy with Launch Boundstuning, Tuning Occupancyactive warps per scheduler, Find the Right Occupancy for Your WorkloadCUDA Occupancy API, Techniques for Occupancy Tuning, Determine Optimal Launch Configuration with the Occupancy APIeligible warps per cycle, Find the Right Occupancy for Your Workload__launch_bounds__, Tuning Occupancy with Launch Bounds-Tuning Occupancy with Launch Bounds, Techniques for Occupancy Tuning, Compiler Hints to Optimize Occupancyright occupancy for workload, Find the Right Occupancy for Your Workloadtechniques, Techniques for Occupancy Tuning
- online resources (see resources online)
- OOM (out of memory) errorsCPU + GPU architecture, The CPU and GPU Superchipmax locked memory setting, Transparent Hugepagesmemory fragmentation or excessive caching, GPU Memory Oversubscription, Fragmentation, and Out-of-Memory Handlingmonitoring GPU memory usage, GPU Memory Oversubscription, Fragmentation, and Out-of-Memory Handling
- OOM killer (Linux), Memory Isolation and Avoiding the OOM Killer
- Open Collective Foundation, Scaling Toward Multimillion GPU Clusters and 100-Trillion-Parameter Models
- Open Infra Index (DeepSeek), Transparency and Reproducibility
- Open-Source Week (DeepSeek), Transparency and Reproducibility, Transparency and Reproducibility
- OpenAIChatGPT stateful conversation, Prefix CachingGPT-4 training costs, Introduction and AI System OverviewTriton, Triton Programming Modelaccessing shared memory, Accessing Shared Memory in Tritonadvanced kernel implementations, Advanced Triton Kernel Implementations-Software Pipelining and Double Buffering with TritonCUDA kernels in Python, NVIDIA Software Stackcustom kernels, Writing Custom Kernels with OpenAI Triton-Profiling with Triton Proton Profilerdouble-buffering pipelining, Software Pipelining and Double Buffering with Triton-Software Pipelining and Double Buffering with Tritonlanguage and compiler, C++ and Python CUDA LibrariesNsight Compute and Systems support, Profiling with Triton Proton Profileropen source, C++ and Python CUDA Librariespersistent matmul kernel, Tiled and Persistent GEMM Kernel (Triton)-Tiled and Persistent GEMM Kernel (Triton)profiling with Triton Proton Profiler, Profiling with Triton Proton Profilerregistering custom kernels with PyTorch, TorchInductor Backend Code Generation, Registering Custom Kernels with PyTorch-Registering Custom Kernels with PyTorchsmart compilers, Smart Compilers and Automated Code Optimizations-Smart Compilers and Automated Code OptimizationsTorchInductor basis, PyTorch and Higher-Level AI Frameworks, Using the PyTorch Compilertuning kernel launch parameters, Tuning Kernel-Launch Parameterswarp specialization, Warp Specialization with TritonTriton Proton Profiler, Profiling with Triton Proton Profiler
- operating system, Operating System
- operational intensity (see arithmetic intensity)
- optical networking, Co-Packaged Optics: Future of Networking Hardware
- optimization of AI systems by AI, AI-Assisted Real-Time System Optimizations and Cluster Operations-AI-Assisted Real-Time System Optimizations and Cluster Operations
- OS and CPU configuration, Configuring the CPUs and OS for GPU Environments-Tune Host CPU Memory Allocatorabout, Configuring the CPUs and OS for GPU EnvironmentsCPU frequency and C-states, CPU Frequency and C-statesfilesystem caching and write-back, Filesystem Caching and Write-BackNUMA awareness and CPU pinning, NUMA Awareness and CPU Pinning-NUMA Awareness and CPU PinningCPU + GPU superchip CPU-to-GPU data transfers, NUMA Awareness and CPU PinningCPU pinning, NUMA Awareness and CPU Pinning-NUMA Awareness and CPU PinningNUMA nodes, NUMA Awareness and CPU PinningNUMA-friendly memory allocation and pinning, NUMA-Friendly Memory Allocation and Memory Pinning-NUMA-Friendly Memory Allocation and Memory Pinningefficiency of pinned memory, NUMA-Friendly Memory Allocation and Memory Pinningmax locked memory setting, Transparent HugepagesOS limit on pinned memory, NUMA-Friendly Memory Allocation and Memory Pinningpinned memory for data loaders, NUMA-Friendly Memory Allocation and Memory Pinningscheduler and interrupt affinity, Scheduler and Interrupt Affinityirqbalance daemon, Scheduler and Interrupt Affinitytransparent hugepages, Transparent Hugepagestuning host CPU memory allocator, Tune Host CPU Memory Allocatorvirtual memory and swapping, Virtual Memory and Swapping
- OSTs (Object Storage Targets), Distributed, Parallel Filesystems and Object Stores
- out of memory (OOM) errorsCPU + GPU architecture, The CPU and GPU Superchipmax locked memory setting, Transparent Hugepagesmemory fragmentation or excessive caching, GPU Memory Oversubscription, Fragmentation, and Out-of-Memory Handlingmonitoring GPU memory usage, GPU Memory Oversubscription, Fragmentation, and Out-of-Memory Handling
- outputstreaming responses, Streaming Responses-Streaming Responsesstructured outputs, Constrained Decoding Performance Implicationstoken output limits and timeouts, Token Output Limits and Timeouts
- overlapping communication and computationautomating, Smart Compilers and Automated Code OptimizationsCUDA streams, Using Streams to Overlap Compute with Data Transfers-Using Streams to Overlap Compute with Data Transfersconcurrency, Overlapping Communication and Computation-Overlapping Communication and Computationmultiple GPUs, Multi-GPU Compute and Data Transfer Overlap with CUDA Streams-Multi-GPU Compute and Data Transfer Overlap with CUDA Streamsdistributed networking communication, Overlapping Communication and Computation (Pipelining)-Achieving Maximal Overlap in PracticeDistributed Data Parallel strategies, Distributed Data Parallel Strategies-Distributed Data Parallel Strategiesinference tuning, Overlapping Communication and Computation-Overlapping Communication and ComputationNCCL communication and GEMM computations, Optimizing Collective Communication with NCCLoffloaded KV cache, Speculative KV Prefetching for Faster TTFTPyTorch pattern, Overlapping Communication and Computation
- O_DIRECTbypassing page cache, Filesystem Caching and Write-Back, Using NVIDIA GDSGDS, Using NVIDIA GDS

- achieved occupancy and GPU utilization, Inspecting Achieved Occupancy and GPU Utilization-Optimizing the Kerneldefinition of achieved occupancy, Inspecting Achieved Occupancy and GPU Utilizationkernel compute throughput versus GPU FLOPS, Kernel Compute Throughput Versus Peak GPU FLOPSkernel memory throughput versus HBM memory bandwidth, Kernel Memory Throughput Versus Peak HBM Memory Bandwidth
- configuring launch parameters, Configuring Launch Parameters: Blocks per Grid and Threads per Block
- CUDA Occupancy API, Techniques for Occupancy Tuning, Techniques for Occupancy Tuningoccupancy optimization, Determine Optimal Launch Configuration with the Occupancy API
- definition of occupancy, Maintaining High Occupancy and GPU Utilization, Tuning Occupancy
- hardware limiting, Choosing Threads-per-Block and Blocks-per-Grid Sizes, Choosing Threads-per-Block and Blocks-per-Grid Sizes
- high occupancy, Threads, Warps, Blocks, and Gridshigh ILP with, Loop Unrolling, Interleaving, and Compiler Hintingmaintaining high occupancy, Maintaining High Occupancy and GPU Utilization-Maintaining High Occupancy and GPU Utilization
- ILP and, Exposing Instruction-Level Parallelism, ILP and Occupancy, Loop Unrolling, Interleaving, and Compiler Hinting
- impact of occupancy, Maintaining High Occupancy and GPU Utilization-Maintaining High Occupancy and GPU Utilization
- increasing occupancy and per-thread resources, Tuning Occupancy with Launch Bounds, Tuning Occupancy, Compiler Hints to Optimize Occupancy
- occupancy-aware kernel selection, Dynamic Shared-Memory Allocation and Occupancy-Aware Kernel Selection-Dynamic Shared-Memory Allocation and Occupancy-Aware Kernel Selection
- persistent kernels, Persistent Kernels and Megakernels
- register trade-off, Tuning Occupancy with Launch Bounds
- tuning, Tuning Occupancyactive warps per scheduler, Find the Right Occupancy for Your WorkloadCUDA Occupancy API, Techniques for Occupancy Tuning, Determine Optimal Launch Configuration with the Occupancy APIeligible warps per cycle, Find the Right Occupancy for Your Workload__launch_bounds__, Tuning Occupancy with Launch Bounds-Tuning Occupancy with Launch Bounds, Techniques for Occupancy Tuning, Compiler Hints to Optimize Occupancyright occupancy for workload, Find the Right Occupancy for Your Workloadtechniques, Techniques for Occupancy Tuning

- definition of achieved occupancy, Inspecting Achieved Occupancy and GPU Utilization
- kernel compute throughput versus GPU FLOPS, Kernel Compute Throughput Versus Peak GPU FLOPS
- kernel memory throughput versus HBM memory bandwidth, Kernel Memory Throughput Versus Peak HBM Memory Bandwidth

- occupancy optimization, Determine Optimal Launch Configuration with the Occupancy API

- high ILP with, Loop Unrolling, Interleaving, and Compiler Hinting
- maintaining high occupancy, Maintaining High Occupancy and GPU Utilization-Maintaining High Occupancy and GPU Utilization

- active warps per scheduler, Find the Right Occupancy for Your Workload
- CUDA Occupancy API, Techniques for Occupancy Tuning, Determine Optimal Launch Configuration with the Occupancy API
- eligible warps per cycle, Find the Right Occupancy for Your Workload
- __launch_bounds__, Tuning Occupancy with Launch Bounds-Tuning Occupancy with Launch Bounds, Techniques for Occupancy Tuning, Compiler Hints to Optimize Occupancy
- right occupancy for workload, Find the Right Occupancy for Your Workload
- techniques, Techniques for Occupancy Tuning

- CPU + GPU architecture, The CPU and GPU Superchip
- max locked memory setting, Transparent Hugepages
- memory fragmentation or excessive caching, GPU Memory Oversubscription, Fragmentation, and Out-of-Memory Handling
- monitoring GPU memory usage, GPU Memory Oversubscription, Fragmentation, and Out-of-Memory Handling

- ChatGPT stateful conversation, Prefix Caching
- GPT-4 training costs, Introduction and AI System Overview
- Triton, Triton Programming Modelaccessing shared memory, Accessing Shared Memory in Tritonadvanced kernel implementations, Advanced Triton Kernel Implementations-Software Pipelining and Double Buffering with TritonCUDA kernels in Python, NVIDIA Software Stackcustom kernels, Writing Custom Kernels with OpenAI Triton-Profiling with Triton Proton Profilerdouble-buffering pipelining, Software Pipelining and Double Buffering with Triton-Software Pipelining and Double Buffering with Tritonlanguage and compiler, C++ and Python CUDA LibrariesNsight Compute and Systems support, Profiling with Triton Proton Profileropen source, C++ and Python CUDA Librariespersistent matmul kernel, Tiled and Persistent GEMM Kernel (Triton)-Tiled and Persistent GEMM Kernel (Triton)profiling with Triton Proton Profiler, Profiling with Triton Proton Profilerregistering custom kernels with PyTorch, TorchInductor Backend Code Generation, Registering Custom Kernels with PyTorch-Registering Custom Kernels with PyTorchsmart compilers, Smart Compilers and Automated Code Optimizations-Smart Compilers and Automated Code OptimizationsTorchInductor basis, PyTorch and Higher-Level AI Frameworks, Using the PyTorch Compilertuning kernel launch parameters, Tuning Kernel-Launch Parameterswarp specialization, Warp Specialization with Triton
- Triton Proton Profiler, Profiling with Triton Proton Profiler

- accessing shared memory, Accessing Shared Memory in Triton
- advanced kernel implementations, Advanced Triton Kernel Implementations-Software Pipelining and Double Buffering with Triton
- CUDA kernels in Python, NVIDIA Software Stack
- custom kernels, Writing Custom Kernels with OpenAI Triton-Profiling with Triton Proton Profiler
- double-buffering pipelining, Software Pipelining and Double Buffering with Triton-Software Pipelining and Double Buffering with Triton
- language and compiler, C++ and Python CUDA Libraries
- Nsight Compute and Systems support, Profiling with Triton Proton Profiler
- open source, C++ and Python CUDA Libraries
- persistent matmul kernel, Tiled and Persistent GEMM Kernel (Triton)-Tiled and Persistent GEMM Kernel (Triton)
- profiling with Triton Proton Profiler, Profiling with Triton Proton Profiler
- registering custom kernels with PyTorch, TorchInductor Backend Code Generation, Registering Custom Kernels with PyTorch-Registering Custom Kernels with PyTorch
- smart compilers, Smart Compilers and Automated Code Optimizations-Smart Compilers and Automated Code Optimizations
- TorchInductor basis, PyTorch and Higher-Level AI Frameworks, Using the PyTorch Compiler
- tuning kernel launch parameters, Tuning Kernel-Launch Parameters
- warp specialization, Warp Specialization with Triton

- about, Configuring the CPUs and OS for GPU Environments
- CPU frequency and C-states, CPU Frequency and C-states
- filesystem caching and write-back, Filesystem Caching and Write-Back
- NUMA awareness and CPU pinning, NUMA Awareness and CPU Pinning-NUMA Awareness and CPU PinningCPU + GPU superchip CPU-to-GPU data transfers, NUMA Awareness and CPU PinningCPU pinning, NUMA Awareness and CPU Pinning-NUMA Awareness and CPU PinningNUMA nodes, NUMA Awareness and CPU Pinning
- NUMA-friendly memory allocation and pinning, NUMA-Friendly Memory Allocation and Memory Pinning-NUMA-Friendly Memory Allocation and Memory Pinningefficiency of pinned memory, NUMA-Friendly Memory Allocation and Memory Pinningmax locked memory setting, Transparent HugepagesOS limit on pinned memory, NUMA-Friendly Memory Allocation and Memory Pinningpinned memory for data loaders, NUMA-Friendly Memory Allocation and Memory Pinning
- scheduler and interrupt affinity, Scheduler and Interrupt Affinityirqbalance daemon, Scheduler and Interrupt Affinity
- transparent hugepages, Transparent Hugepages
- tuning host CPU memory allocator, Tune Host CPU Memory Allocator
- virtual memory and swapping, Virtual Memory and Swapping

- CPU + GPU superchip CPU-to-GPU data transfers, NUMA Awareness and CPU Pinning
- CPU pinning, NUMA Awareness and CPU Pinning-NUMA Awareness and CPU Pinning
- NUMA nodes, NUMA Awareness and CPU Pinning

- efficiency of pinned memory, NUMA-Friendly Memory Allocation and Memory Pinning
- max locked memory setting, Transparent Hugepages
- OS limit on pinned memory, NUMA-Friendly Memory Allocation and Memory Pinning
- pinned memory for data loaders, NUMA-Friendly Memory Allocation and Memory Pinning

- irqbalance daemon, Scheduler and Interrupt Affinity

- CPU + GPU architecture, The CPU and GPU Superchip
- max locked memory setting, Transparent Hugepages
- memory fragmentation or excessive caching, GPU Memory Oversubscription, Fragmentation, and Out-of-Memory Handling
- monitoring GPU memory usage, GPU Memory Oversubscription, Fragmentation, and Out-of-Memory Handling

- streaming responses, Streaming Responses-Streaming Responses
- structured outputs, Constrained Decoding Performance Implications
- token output limits and timeouts, Token Output Limits and Timeouts

- automating, Smart Compilers and Automated Code Optimizations
- CUDA streams, Using Streams to Overlap Compute with Data Transfers-Using Streams to Overlap Compute with Data Transfersconcurrency, Overlapping Communication and Computation-Overlapping Communication and Computationmultiple GPUs, Multi-GPU Compute and Data Transfer Overlap with CUDA Streams-Multi-GPU Compute and Data Transfer Overlap with CUDA Streams
- distributed networking communication, Overlapping Communication and Computation (Pipelining)-Achieving Maximal Overlap in PracticeDistributed Data Parallel strategies, Distributed Data Parallel Strategies-Distributed Data Parallel Strategies
- inference tuning, Overlapping Communication and Computation-Overlapping Communication and Computation
- NCCL communication and GEMM computations, Optimizing Collective Communication with NCCL
- offloaded KV cache, Speculative KV Prefetching for Faster TTFT
- PyTorch pattern, Overlapping Communication and Computation

- concurrency, Overlapping Communication and Computation-Overlapping Communication and Computation
- multiple GPUs, Multi-GPU Compute and Data Transfer Overlap with CUDA Streams-Multi-GPU Compute and Data Transfer Overlap with CUDA Streams

- Distributed Data Parallel strategies, Distributed Data Parallel Strategies-Distributed Data Parallel Strategies

- bypassing page cache, Filesystem Caching and Write-Back, Using NVIDIA GDS
- GDS, Using NVIDIA GDS

### P

- P2P (peer-to-peer) DMA, Enabling Peer-to-Peer DMA and UCXP2P (peer-to-peer) GPU copies, Pitfall #3: Avoid overtuning or disabling NCCL features with environment variables
- padding shared arrays, Avoid Shared-Memory Bank Conflictsswizzling as alternative, Avoid Shared-Memory Bank Conflicts
- page cacheDeepSeek Fire-Flyer File System bypassing, DeepSeek’s Fire-Flyer File Systemlatency-sensitive training workflows, Filesystem Caching and Write-BackO_DIRECT bypassing, Filesystem Caching and Write-Back, Using NVIDIA GDS
- Page Migration Engine (PME; NVIDIA), GPU Memory Oversubscription, Fragmentation, and Out-of-Memory Handling
- page-cache size, Filesystem Caching and Write-Back
- page-locked memory, NUMA-Friendly Memory Allocation and Memory Pinning
- PagedAttention (vLLM), Continuous Scheduling, KV Cache Offloading and Memory Pool Allocation, Memory management for the KV cacheFlexDecoding, FlexDecoding (PyTorch)
- parallel aggregated tree (PAT), NCCL Communication Algorithms
- parallel filesystems, Distributed, Parallel Filesystems and Object Stores
- parallel thread execution (see PTX (parallel thread execution))
- parallelismadaptive strategies for ultrascale inference, Adaptive Parallelism Strategies (TP Versus PP Versus Hybrid)-Adaptive Parallelism Strategies (TP Versus PP Versus Hybrid)adding two vectors sequentially and in parallel, Maintaining High Occupancy and GPU Utilization-Maintaining High Occupancy and GPU Utilizationarchitecture of GPU, Understanding GPU ArchitecturecuTile, C++ and Python CUDA LibrariesData Parallel versus Distributed Data Parallel, Distributed Data Parallel Strategies-Distributed Data Parallel Strategiesdynamic parallelism, Dynamic Parallelism-Dynamic Parallelismbenchmarks, Dynamic Parallelismchild-kernel launch limit, Dynamic Parallelismdevice memory usage monitored, Dynamic Parallelismprofiling before and after a change, Dynamic Parallelism“stack overflow” errors avoided, Dynamic Parallelismwhen to capture CUDA Graph instead, Dynamic Parallelismfully sharded data parallel, Distributed Data Parallel StrategiesPTX, Inline PTX and SASS Tuning for Microoptimizations(see also PTX (parallel thread execution))speculative decoding, Disaggregated Prefill and Decode Architecture, Speculative Decoding and Parallel Token Generation Techniques-Combining Decoding Techniques and Evaluating Complexitystrategies, Parallelism Strategies for Serving Massive MoE Models-Hybrid Parallelismcontext parallelism, Parallelism Strategies for Serving Massive MoE Models, Context (Sequence) Parallelismdata parallelism, Parallelism Strategies for Serving Massive MoE Models, Data Parallelismexpert parallelism, Parallelism Strategies for Serving Massive MoE Models, Expert Parallelism-Expert Parallelismhybrid parallelism, Hybrid Parallelism-Hybrid Parallelismmodel weights and data split over GPUs, Parallelism Strategies for Serving Massive MoE Modelspipeline parallelism, Parallelism Strategies for Serving Massive MoE Models, Pipeline Parallelismprofiling, Hybrid Parallelismsummary table, Parallelism Strategies for Serving Massive MoE Modelstensor parallelism, Parallelism Strategies for Serving Massive MoE Models, Tensor Parallelismthread block cluster algorithms, Designing Efficient Algorithms with Thread Block Clusters-Designing Efficient Algorithms with Thread Block Clusters
- parameters offloaded to CPU and NVMe, Offloading Parameters to CPU and NVMe
- partitioned global address space (PGAS), Multi-GPU Programming
- PAT (parallel aggregated tree), NCCL Communication Algorithms
- PCIe, Multi-GPU Compute and Data Transfer Overlap with CUDA Streams-Multi-GPU Compute and Data Transfer Overlap with CUDA Streams
- PDUs (power distribution units), Compute Density and Power Requirements
- peer-to-peer (see P2P (peer-to-peer) DMA)
- per-thread default streams (PTDS), Modern Per-Thread Default Stream
- perf (Linux), NVTX Markers and Profiling Tools, NVTX Markers and Profiling Toolsexample MoE model, CPU and GPU Profiling with Linux perf-CPU and GPU Profiling with Linux perfNVIDIA Performance Monitoring Unit, CPU and GPU Profiling with Linux perf
- Perfetto Trace Viewer, NVTX Markers and Profiling Tools, NVTX Markers and Profiling Tools, Multi-GPU Profiling with HTA
- performance boostscompressing data, Tuning, Replicating, and Compressing DataCUDA Graphs, Capturing a CUDA Graph with a CUDA Streamconditional graph nodes, Conditional Graph Nodesdisaggregating prefill and decode, Prefill-Decode Interferencedynamic batching, Dynamic Batching-Dynamic Batchinghigh occupancy with high ILP, Loop Unrolling, Interleaving, and Compiler Hintingkernel optimization, Optimizing the Kernel-Optimizing the Kernelmax-autotune compile mode, PyTorch Compiler Deep Divememory pinning, NUMA-Friendly Memory Allocation and Memory Pinningnetwork reads eliminated, Tuning, Replicating, and Compressing Datapitfall of repeatedly reading same global memory data, Tiling and Data Reuse Using Shared Memoryrunning many warps concurrently, Maintaining High Occupancy and GPU Utilizationstream 0 legacy default stream avoided, Legacy Default StreamTORCH_LOGS performance hints, Performance Hints and Debugging Generated CodeUCX with NCCL over multinode topologies, Enabling Peer-to-Peer DMA and UCX
- performance engineers (see AI systems performance engineers)
- performance metricsabout ultimate goal of performance tuning, Maximizing GPU Utilization and Throughput Versus Latency Trade-Offsautomated performance tests, Benchmarking and Profilingbenchmarking and profiling, Benchmarking and Profiling(see also benchmarks; profiling)cache hits and misses, Profiling, Debugging, and Tuning Inference Performancecommunication versus compute bottlenecks, Diagnosing Communication- Versus Compute-Bound Workloadsdisaggregation of prefill and decode, Impact on Latency (TTFT) and Throughput (TPOT)goodput as throughput metric, Measuring “Goodput” Useful ThroughputGPU idle time measurement, Monitoring Storage I/OGPU utilization percentage, Maximizing GPU Utilization and Throughput Versus Latency Trade-OffsGPUs with NVIDIA Container Toolkit, Container Runtime Optimizations for GPUsinference tuningerror handling, Error Handlingfull-stack optimizations, Full-Stack Inference Optimizations-Full-Stack Inference Optimizationsmonitoring system metrics and counters, Monitoring System Metrics and Counters-Monitoring System Metrics and Countersobserve-hypothesize-tune loop, Profiling, Debugging, and Tuning Inference Performance-Profiling, Debugging, and Tuning Inference Performanceprofiling with Nsight, Profiling with Nsight Systems and Nsight Compute-Profiling with Nsight Systems and Nsight Computetroubleshooting, Inference Troubleshooting RecipesI/O continuous profiling and tuning, Continuous Profiling and Tuning Workflow-Continuous Profiling and Tuning Workflowlatency hiding as important tool, Streaming Multiprocessor, Threads, and WarpsMLPerf Logging, Performance Benchmarks and MLPerf Logging-Performance Benchmarks and MLPerf Loggingperformance benchmarking, Continuous Integration and Performance Benchmarking-Performance Benchmarks and MLPerf Loggingforcing an error when full graph not captured, Graph Breaks and TorchDynamo explain()performance monitoring, Performance Monitoring and Utilization in PracticeGPUs near 100% utilized, Performance Monitoring and Utilization in PracticeNVLink usage, Performance Monitoring and Utilization in Practiceprefix-merge metrics, Profiling, Debugging, and Tuning Inference PerformancePyTorch performance heads-up display, Continuous Integration and Performance Benchmarking-PyTorch HUD Performance Dashboardroofline model, Roofline Model: Compute-Bound or Memory-Bound Workloads-Roofline Model: Compute-Bound or Memory-Bound WorkloadsNsight Compute Roofline analysis, Nsight Compute and Roofline AnalysisRPS (requests per second), Dynamic Batchingthread block size, Choosing Threads-per-Block and Blocks-per-Grid SizesTPOT (time per output token)disaggregation of prefill and decode, Impact on Latency (TTFT) and Throughput (TPOT)prefill and decode phases, Scaling Disaggregated Prefill and Decode for InferenceTTFT (time to first token)constrained outputs, Constrained Decoding Performance Implicationscontext parallelism, Context (Sequence) Parallelismdisaggregation of prefill and decode, Impact on Latency (TTFT) and Throughput (TPOT)prefill and decode phases, Scaling Disaggregated Prefill and Decode for Inference
- performance regression continuous integration, Continuous Integration and Performance Benchmarking-Performance Benchmarks and MLPerf Logging
- persistence mode for GPU, GPU Persistence Modenvidia-persistenced daemon, GPU Persistence Moderecommended when using MIG, Slicing a GPU with MIG
- persistent kernels, Persistent Kernels and Megakernels-Persistent Kernels and Warp Specializationcommon workloads for, Common Workloads for Persistent Kernelscooperative groups via grid sync, Cooperative Grid Synchronization and Persistent Kernels-Cooperative Grid Synchronization and Persistent Kernelswhen to combine, When to Combine Persistent Kernels and Cooperative Groupsmegakernels, Megakernels for Inferenceoccupancy, Persistent Kernels and Megakernelswarp specialization, Persistent Kernels and Warp Specialization
- persistent threads (see persistent kernels)
- PGAS (partitioned global address space), Multi-GPU Programming
- pipeline parallel (PP), Parallelism Strategies for Serving Massive MoE Models, Pipeline ParallelismFSDP with, Combining FSDP with Tensor Parallel and Pipeline Parallelhybrid parallelism, Hybrid ParallelismPyTorch compiler with, Tensor and Pipeline Parallelism with torch.compiletensor parallel versus, Pipeline Parallelismversus TP versus hybrid, Adaptive Parallelism Strategies (TP Versus PP Versus Hybrid)-Adaptive Parallelism Strategies (TP Versus PP Versus Hybrid)
- pipelinescomparison of no pipeline with pipeline, Achieving Maximal Overlap in Practice-Achieving Maximal Overlap in PracticeCUDA C++ Pipeline API, Asynchronous Memory Prefetching and Tensor Memory Accelerator-Asynchronous Memory Prefetching and Tensor Memory Acceleratorhardware-software codesign, Asynchronous Memory Prefetching and Tensor Memory AcceleratorCUDA Graphs, Capturing a CUDA Graph with a CUDA Stream(see also CUDA Graphs)data compression, Tuning, Replicating, and Compressing Datadata pipeline tuning, Tuning the Data Pipelinedata loading and preprocessing, Efficient Data Loading and Preprocessing-Efficient Data Loading and PreprocessingNeMo Curator for training datasets, Creating High-Quality LLM Datasets with NVIDIA NeMo CuratorNVIDIA Data Loading Library, Multimodal Data Processing with NVIDIA DALIscaling out workers as GPUs scaled, Scaling Out Workers as You Scale Out Number of GPUsdistributed data parallel strategies, Distributed Data Parallel Strategies-Distributed Data Parallel Strategiesinter-kernel pipeliningabout inter-kernel pipelining, Inter-Kernel Pipelining, Synchronization, and CUDA Stream-Ordered Memory Allocationscompute overlapping with data transfers via streams, Using Streams to Overlap Compute with Data Transfers-Using Streams to Overlap Compute with Data Transfers, Multi-GPU Compute and Data Transfer Overlap with CUDA Streams-Multi-GPU Compute and Data Transfer Overlap with CUDA Streams, Overlapping Communication and Computation-Overlapping Communication and Computationcross-stream synchronization, Using CUDA Events for Cross-Stream SynchronizationCUDA stream ordered memory allocator, Stream-Ordered Memory Allocator-Stream-Ordered Memory AllocatorCUDA stream ordered memory allocator used, Using CUDA Streams and Stream-Ordered Memory Allocator with LLMs-Using CUDA Streams and Stream-Ordered Memory Allocator with LLMsCUDA stream ordered memory allocator with multiple GPUs, Multi-GPU Compute and Data Transfer Overlap with CUDA Streams-Multi-GPU Compute and Data Transfer Overlap with CUDA Streamsdefault stream usage, Best Practices for Default Stream Usage-Best Practices for Default Stream Usagedefault versus explicit streams, Default Versus Explicit (Nondefault) Streamskernel execution overlapped with CUDA streams, Overlapping Kernel Execution with CUDA Streams-Overlapping Kernel Execution with CUDA Streamslaunching five kernels on two CUDA streams, Overlapping Kernel Execution with CUDA Streams-Overlapping Kernel Execution with CUDA Streamslegacy default streams, Legacy Default Streammulti-GPU overlap of compute and data, Multi-GPU Compute and Data Transfer Overlap with CUDA Streams-Multi-GPU Compute and Data Transfer Overlap with CUDA StreamsPDL with thread block clusters and warp specialization, Combining PDL and Thread Block Clusters with Warp Specialization-Combining PDL and Thread Block Clusters with Warp Specializationper-thread default streams, Modern Per-Thread Default StreamProgrammatic Dependent Launch, Programmatic Dependent Launch-Programmatic Dependent Launchsynchronization with events and callbacks, Fine-Grained Synchronization with Events and Callbackswarp specialization and CUDA streams, Pipelining with Warp Specialization (Intra-Kernel) and CUDA Streams (Inter-Kernel)-Pipelining with Warp Specialization (Intra-Kernel) and CUDA Streams (Inter-Kernel)warp specialization replaced with CUDA streams, Overlapping Kernel Execution with CUDA Streams-Overlapping Kernel Execution with CUDA Streamswarp specialization, thread block clusters, and CUDA streams, Warp Specialization with Thread Block Clusters and CUDA Streamsintra-kernel pipeliningabout intra-kernel pipelining, Intra-Kernel Pipelining Techniquescomparing techniques, Intra-Kernel Pipelining Techniques, Using CUDA Pipeline API for Warp Specialization, Warp Specialization with Thread Block Clusters-Warp Specialization with Thread Block Clusterscooperative groups, Thread Block Clusters and Distributed Shared Memory(see also cooperative groups (CGs))double-buffering with Pipeline API, Cooperative Tiling and Double-Buffering with the CUDA Pipeline API-Cooperative Tiling and Double-Buffering with the CUDA Pipeline APIpersistent kernels, Persistent Kernels and Megakernels-Persistent Kernels and Warp SpecializationPyTorch, PyTorch, CUDA Pipeline API, and Warp Specializationthread block clusters, Intra-Kernel Pipelining, Warp Specialization, and Cooperative Thread Block Clusters(see also thread block clusters)warp specialization, Warp Specialization and the Producer-Consumer Model-Warp Specialization and the Producer-Consumer Modelwarp specialization and CUDA streams, Pipelining with Warp Specialization (Intra-Kernel) and CUDA Streams (Inter-Kernel)-Pipelining with Warp Specialization (Intra-Kernel) and CUDA Streams (Inter-Kernel)warp specialization using CUDA Pipeline API, Using CUDA Pipeline API for Warp Specialization-Using CUDA Pipeline API for Warp Specializationwarp specialization with persistent kernels, Persistent Kernels and Warp Specializationwhen to combine persistent kernels and cooperative groups, When to Combine Persistent Kernels and Cooperative Groupswhen to use warp specialization versus double-buffering, Warp Specialization and the Producer-Consumer Modelmonitoring I/O, Monitoring Storage I/O-Monitoring Storage I/Ooverlapping communication and computation, Overlapping Communication and Computation (Pipelining)-Achieving Maximal Overlap in Practicereducing frequency and volume, Reducing Communication Frequency and Volumestreams for asynchronous execution, Asynchronous Execution with StreamsTriton double-buffering, Software Pipelining and Double Buffering with Triton-Software Pipelining and Double Buffering with Triton
- PMU (NVIDIA Performance Monitoring Unit), CPU and GPU Profiling with Linux perf
- PoC (point of coherency), Understanding GPU Memory Hierarchy
- POD-Attention, Optimized KV Cache Memory Layout
- Podman with NVIDIA Container Toolkit, NVIDIA Container Runtime
- point of coherency (PoC), Understanding GPU Memory Hierarchy
- portable cluster size maximum, Distributed Shared Memory
- power and thermal constraints on inference, Power and Thermal Constraints
- power distribution units (PDUs), Compute Density and Power Requirements
- power monitoring, Performance Monitoring and Utilization in PracticeNVIDIA Management Library, Kernel Compute Throughput Versus Peak GPU FLOPSpower limits, Performance Monitoring and Utilization in Practice, Power and Thermal Constraintssetting slightly below TDP, GPU Clock Speeds and ECC
- power requirementsGPU persistence mode, GPU Persistence ModeNVL72, Compute Density and Power Requirementsliquid cooling versus air cooling, Liquid Cooling Versus Air Cooling-Liquid Cooling Versus Air Coolingpower monitoring, Performance Monitoring and Utilization in Practice
- power-saving mode for CPUs, CPU Frequency and C-states
- PP (see pipeline parallel (PP))
- #pragma unroll, Exposing Instruction-Level Parallelism, Loop Unrolling, Interleaving, and Compiler Hintingunrolling too many iterations, Profiling and Mitigating Register Pressure
- precisiondynamic precision changes, Dynamic Precision Changes-Dynamic Precision ChangesFP4, FP8, FP16, NVIDIA GPU Tensor Cores and Transformer Engine, BF16/FP16, FP8, and FP4 Reduced Precisioninference quantization, Reducing Precision from FP16 to FP8 and FP4weight-only quantization, Weight-Only Quantization (GPTQ, AWQ)mixed precisionautomatic mixed precision, TF32 and Automatic Mixed Precision (PyTorch)-TF32 and Automatic Mixed Precision (PyTorch)reduced precision, BF16/FP16, FP8, and FP4 Reduced PrecisionTensor Cores, Mixed Precision and Utilizing Tensor Cores-Transformer Engine and TMEM in Depthquantization (see quantization)
- Predibase, Reinforcement Learning Approach to Generating Optimized GPU Kernels (Predibase)-Reinforcement Learning Approach to Generating Optimized GPU Kernels (Predibase)
- predication, Techniques to Avoid Warp Divergencewarp divergence avoided, Techniques to Avoid Warp Divergencewarp divergence minimized, Using Predication to Minimize Divergence-Using Predication to Minimize Divergence
- prefillabout prefill phase, Scaling Disaggregated Prefill and Decode for Inferencechunked prefill, Stall-Free Scheduling (Chunked Prefill), Adaptive Batching and Chunked Prefill Scheduling-Adaptive Batching and Chunked Prefill Schedulingcontext parallelism, Parallelism Strategies for Serving Massive MoE Models, Context (Sequence) ParallelismContext Parallel (PyTorch), PyTorch Optimized Attention Mechanismsdisaggregated from decodeabout disaggregation, Multinode Inference, Parallelism, Decoding, and Routing Optimizations, Prefill-Decode Interference, Scaling Disaggregated Prefill and Decode for Inferencearchitecture, Disaggregated Prefill and Decode Architecturebatch size optimizations, Adaptive Batching and Chunked Prefill Schedulingcluster pools, Disaggregated Prefill and Decode Cluster Pools-Memory management for the KV cachedeploying with Kubernetes, Deploying Disaggregated Prefill and Decode with Kubernetes-Deploying Disaggregated Prefill and Decode with Kubernetes, Full-Stack Inference Optimizationslatency and throughput, Impact on Latency (TTFT) and Throughput (TPOT)load balancing, Dynamic Scheduling and Load Balancing-Dynamic resource scalingprefill to decode data transfers, KV Cache Data Transfer and NIXL, Fast KV Cache Transfer Between Prefill and Decode-Zero-Copy GPU-to-GPU Transferscaling, Scaling Prefill and Worker Nodes Independentlyseparate prefill and decode stages, Separate Prefill and Decode Inference Stages-Separate Prefill and Decode Inference Stages, Disaggregated Prefill and Decode Architecturedisaggregated routing policies, Disaggregated Routing and Scheduling Policies-QoS and early rejection policiescapacity-aware routing, Capacity-aware routing-Capacity-aware routingexample dynamic routing policy configuration, Example dynamic routing policy configurationexample dynamic routing policy in code, Example dynamic routing policy in codelatency-aware routing, Latency-aware routing-Latency-aware routingmultipath inference (racing), Multipath inference (racing)QoS and early rejection policies, QoS and early rejection policies-QoS and early rejection policiesrouting factors, Routing factors-Routing factorsspeculative decoding, Multibranch, parallel speculative decoding across workersFlashAttention, Mechanical Sympathy: Hardware-Software Codesignhardware and parallelism, Heterogeneous Hardware and Parallelism Strategies for Prefill and Decode-Different precision for prefill and decodehybrid prefill with GPU-CPU collaboration, Hybrid Prefill with GPU-CPU Collaboration-Hybrid Prefill with GPU-CPU CollaborationKV cache transfer to decode, Fast KV Cache Transfer Between Prefill and Decode-Zero-Copy GPU-to-GPU Transferconnector and data path design, Connector and Data Path Design-Connector and Data Path DesignKV cache size, KV Cache Sizezero-copy GPU-to-GPU, Zero-Copy GPU-to-GPU Transfer-Zero-Copy GPU-to-GPU Transferoptimized kernels via FlexDecoding, FlexDecoding (PyTorch)-FlexDecoding (PyTorch)scale-out of workers, Continuous Prewarming of CUDA Graphs and Caches Using Time-Series Predictionwhy disaggregation, Why Prefill-Decode Disaggregation?-Scalability of Disaggregated Prefill and Decodeadvantages of disaggregation, Advantages of Disaggregation-Phase-specific optimizationsprefill workers, Prefill workers design-Latency-aware scheduling and batchingrouting and scheduling policies, Disaggregated Routing and Scheduling Policies-QoS and early rejection policiesscalability, Scalability of Disaggregated Prefill and Decode
- prefill-decode interference, Prefill-Decode Interference
- prefix caching, Prefix Caching-Prefix CachingSGLang RadixAttention, Prefix Caching-Prefix Caching
- prewarming graphs and caches, Continuous Prewarming of CUDA Graphs and Caches Using Time-Series Prediction-Continuous Prewarming of CUDA Graphs and Caches Using Time-Series Prediction
- PrimTorch Intermediate Representation (IR), PrimTorch IR (Prims) Simplified Operator SetFX graph output, PrimTorch IR (Prims) Simplified Operator SetPyTorch compiler pipeline, PyTorch Compiler Deep Dive
- process-GPU mapping, Adaptive Process-GPU Mapping-Adaptive Process-GPU Mapping
- profiling, Benchmarking and Profilingachieved occupancy, Inspecting Achieved Occupancy and GPU Utilization-Optimizing the Kernelkernel compute throughput versus GPU FLOPS, Kernel Compute Throughput Versus Peak GPU FLOPSkernel memory throughput versus HBM memory bandwidth, Kernel Memory Throughput Versus Peak HBM Memory Bandwidthbefore and after a change, Dynamic Parallelismcompiler performance issues, Profiling and Debugging Compiler Performance IssuesCUDA events for, Using CUDA Events for Cross-Stream SynchronizationDataLoader, Efficient Data Loading and Preprocessingexample mixture-of-experts model, Profiling PyTorch to Identify Bottlenecks-CPU and GPU Profiling with Linux perfabout MoE models, Profiling PyTorch to Identify BottlenecksLinux perf on CPU and GPU, CPU and GPU Profiling with Linux perf-CPU and GPU Profiling with Linux perfNsight Systems and NVTX Timelines, System Profiling with Nsight Systems and NVTX Timelines-System Profiling with Nsight Systems and NVTX TimelinesPyTorch profiler, Using PyTorch Profiler-Using PyTorch Profilerroofline analysis for GEMM kernels, Kernel Roofline Analysis for General Matrix Multiply (GEMM)-Kernel Roofline Analysis for General Matrix Multiply (GEMM)GPU bottlenecks, Profiling and Diagnosing GPU Bottlenecks-Profiler-Guided Analysisachieved occupancy and GPU utilization, Inspecting Achieved Occupancy and GPU Utilization-Optimizing the KernelCUDA Profiling Tools Interface, PyTorch Profiler and Visualization Toolsiteratively profiling, Iteratively Profiling and Determining the Kernel Bottleneck-Iteratively Profiling and Determining the Kernel Bottleneckkernel optimization, Optimizing the Kernel-Optimizing the KernelNsight Compute, Profiling and Diagnosing GPU BottlenecksNsight Compute and Roofline analysis, Nsight Compute and Roofline AnalysisNsight Systems, Profiling and Diagnosing GPU BottlenecksNsight Systems timeline view, Nsight Systems Timeline ViewPyTorch profiler via Kineto, PyTorch Profiler and Visualization Tools-Profiler-Guided Analysiswarp stall reasons, Analyzing Warp Stall Reasons with Nsight Compute-Other Stall Reasonsgraph break recompilations monitored, TorchDynamo for Bytecode Capture and Graph Extraction, Minimize Graph RecompilationsHolistic Trace Analysis, Multi-GPU Profiling with HTAinference, Profiling with Nsight Systems and Nsight Compute-Profiling with Nsight Systems and Nsight Computeproduction environment, Inference Troubleshooting RecipesI/O continuous profiling and tuning, Continuous Profiling and Tuning Workflow-Continuous Profiling and Tuning Workflowcommunication versus compute bottlenecks, Diagnosing Communication- Versus Compute-Bound Workloadskernels generated by TorchInductor, Profiling and Debugging Compiler Performance Issueslatency with streaming enabled, Streaming Responsesmemory via PyTorch profiler, Profiling and Tuning Memory in PyTorch-Enabling Peer-to-Peer DMA and UCXactivation checkpointing, Activation Checkpointing for Memory SavingsCUDA memory allocator tuned, Tuning the CUDA Memory AllocatorFSDP automatic checkpointing, FSDP Automatic Checkpointing and Offloading-FSDP Automatic Checkpointing and OffloadingFSDP with tensor parallel and pipeline parallel, Combining FSDP with Tensor Parallel and Pipeline ParallelNCCL with UCX over multinode topologies, Enabling Peer-to-Peer DMA and UCXoffloading parameters, Offloading Parameters to CPU and NVMepeer-to-peer DMA, Enabling Peer-to-Peer DMA and UCXpluggable memory allocators, Pluggable Memory Allocators and Cross-GPU Data Transfers-Pluggable Memory Allocators and Cross-GPU Data Transfersmulti-GPU profilingHTA, Multi-GPU Profiling with HTANsight Systems, Real-Time Link Telemetry and MonitoringNCCL, Profiling and Debugging NCCLNCCL logs, In-Network SHARP AggregationNsight Compute occupancy limiters, Inspecting Achieved Occupancy and GPU UtilizationLimited by Registers, Profiling and Mitigating Register PressureNsight Compute warp stall reasons, Analyzing Warp Stall Reasons with Nsight Compute-Other Stall ReasonsStall: Compute Unit Busy, Execution Unit ContentionStall: Exec Dependency, Execution-Dependency Stalls, Profiling and Mitigating Register PressureStall: Idle, Other Stall ReasonsStall: Long Scoreboard, Memory-Related StallsStall: Math Pipe Throttle, Execution Unit ContentionStall: Memory Dependency, Other Stall ReasonsStall: Memory Throttle, Memory-Related StallsStall: No Eligible, Other Stall ReasonsStall: Not Selected, Memory-Related StallsStall: Short Scoreboard, Memory-Related StallsStall: Texture Throttle, Other Stall Reasonsstalling for other reasons, Other Stall Reasons-Other Stall ReasonsNVTX markers, NVTX Markers and Profiling Toolsprofiling tools that use markers, NVTX Markers and Profiling Tools-NVTX Markers and Profiling Toolsoccupancy tuning, Find the Right Occupancy for Your Workloadparallelism strategies, Hybrid Parallelismtools for profiling, NVTX Markers and Profiling Tools-NVTX Markers and Profiling ToolsTriton Proton Profiler, Profiling with Triton Proton Profilerwarp divergence, Profiling and Detecting Warp Divergenceworkload under insufficient bandwidth, Pitfall #4: Insufficient network bandwidth or misconfigured NICs
- Programmatic Dependent Launch (PDL; NVIDIA), Programmatic Dependent Launch-Programmatic Dependent Launchwith thread block clusters and warp specialization, Combining PDL and Thread Block Clusters with Warp Specialization-Combining PDL and Thread Block Clusters with Warp Specialization
- programmingautomated code optimization, Smart Compilers and Automated Code Optimizations-Smart Compilers and Automated Code OptimizationsCUDAasynchronous memory allocation, Asynchronous Memory Allocation and Memory Pools-Asynchronous Memory Allocation and Memory PoolsCUDA streams, Asynchronous Memory Allocation and Memory Poolsflow between CPU and GPU, Understanding GPU Architecture__global__, CUDA Programming RefresherGPU memory hierarchy, Understanding GPU Memory Hierarchy-Understanding GPU Memory Hierarchyhigh occupancy and GPU utilization, Maintaining High Occupancy and GPU Utilization-Maintaining High Occupancy and GPU Utilizationhighest-level and most-recent APIs available, Asynchronous Memory Prefetching and Tensor Memory Acceleratorkernel errors, CUDA Programming Refresherkernel inputs in 1D, 2D and 3D Kernel Inputskernel inputs in 2D and 3D, 2D and 3D Kernel Inputskernels for parallel work, CUDA Programming Refresher-CUDA Programming Refresherlaunch parameters, Configuring Launch Parameters: Blocks per Grid and Threads per Block-Configuring Launch Parameters: Blocks per Grid and Threads per Blockmaximum threadsPerBlock compile time parameter, Tuning Occupancy with Launch Bounds-Tuning Occupancy with Launch Boundsmemory pools, Asynchronous Memory Allocation and Memory Pools-Asynchronous Memory Allocation and Memory Poolsoccupancy tuning with __launch_bounds__, Tuning Occupancy with Launch Bounds-Tuning Occupancy with Launch Boundsroofline model, Roofline Model: Compute-Bound or Memory-Bound Workloads-Roofline Model: Compute-Bound or Memory-Bound WorkloadsUnified Memory, Unified Memory-Unified Memoryinstruction-level parallelismcompiler hints, Loop Unrolling, Interleaving, and Compiler Hintinginterleaving independent operations, Loop Unrolling, Interleaving, and Compiler Hintingunrolling loops, Loop Unrolling, Interleaving, and Compiler Hinting, Kernel Fusionkernel fusion, Kernel Fusion-Kernel Fusionloop unrolling, Loop Unrolling, Interleaving, and Compiler Hinting, Kernel FusionMMA API, Transformer Engine and TMEM in Depth-Transformer Engine and TMEM in Depthmultiple GPUs, Multi-GPU Programming-Multi-GPU ProgrammingNVIDIA Compute Sanitizer, Debugging Functional Correctness with NVIDIA Compute SanitizerNVLink-C2C CPU to GPU memory access, NUMA Awareness and CPU Pinningparallelizing workloads with threadblock clusters, Designing Efficient Algorithms with Thread Block Clusters-Designing Efficient Algorithms with Thread Block ClustersPTX, Feeding Tensor Cores with TMEM and TMA, Inline PTX and SASS Tuning for Microoptimizationscode from compiler, CUDA Forward and Backward Compatibility Across GPU Hardware Generations, CUDA GPU Backward and Forward Compatibility Modelcompatibility across GPU generations, CUDA Forward and Backward Compatibility Across GPU Hardware Generations, CUDA GPU Backward and Forward Compatibility Modelinline PTX code, Inline PTX and SASS Tuning for Microoptimizations-Inline PTX and SASS Tuning for Microoptimizationsinline PTX code in DeepSeek DeepEP, DeepSeek’s Use of Inline PTX for Memory Allocation Optimizationstreaming assembler, Inline PTX and SASS Tuning for Microoptimizations-Inline PTX and SASS Tuning for Microoptimizationsstreaming assembler changing across generations, Inline PTX and SASS Tuning for MicrooptimizationsTriton, Triton Programming ModelPython-centric CUDA libraries, Writing Custom Kernels with OpenAI Tritonsmart compilers, Smart Compilers and Automated Code Optimizations-Smart Compilers and Automated Code OptimizationsTriton programming model, Triton Programming Model
- Prometheusinference monitoring, Monitoring System Metrics and Counters, Monitoring System Metrics and Countersalert manager, Debugging Correctness IssuesData Center GPU Manager metrics, Monitoring System Metrics and CountersNVLink/NVSwitch monitoring, Real-Time Link Telemetry and Monitoring
- prompt cleansing, Prompt CleansingCTRL approach, Prompt Cleansing
- prompt compression, Prompt Compression
- prompt servers, Prefill workers design
- prompts extremely longcontext parallelism, Context (Sequence) Parallelismprompt compression, Prompt Compressionstall-free scheduling, Stall-Free Scheduling (Chunked Prefill)
- pruning, Structured Sparsitydynamic token pruning, Dynamic Token Pruning with LazyLLMTorchAO, PyTorch Architecture Optimization (torchao), Quantization, Sparsity, and Pruning
- PTDS (per-thread default streams), Modern Per-Thread Default Stream
- PTX (parallel thread execution), Inline PTX and SASS Tuning for Microoptimizations, TorchInductor Backend Code Generationcode from compiler, CUDA Forward and Backward Compatibility Across GPU Hardware Generations, CUDA GPU Backward and Forward Compatibility Model, TorchInductor Backend Code GenerationCUDA compatibility across GPU generations, CUDA Forward and Backward Compatibility Across GPU Hardware Generations, CUDA GPU Backward and Forward Compatibility ModelJIT compiling, CUDA Forward and Backward Compatibility Across GPU Hardware Generationsinline PTX code, Inline PTX and SASS Tuning for Microoptimizations-Inline PTX and SASS Tuning for MicrooptimizationsDeepSeek memory allocation optimization, DeepSeek’s Use of Inline PTX for Memory Allocation Optimizationstreaming assembler, Inline PTX and SASS Tuning for Microoptimizations-Inline PTX and SASS Tuning for Microoptimizationschanging per GPU architecture generations, Inline PTX and SASS Tuning for MicrooptimizationsTMEM and CUTLASS, Feeding Tensor Cores with TMEM and TMATorchInductor, TorchInductor Backend Code GenerationTriton, Triton Programming Model
- ptxas warning, Tuning Occupancy with Launch Bounds
- Pythonconditionals as tensor operations, TorchDynamo for Bytecode Capture and Graph ExtractionCUDA kernels created, NVIDIA Software StackCUDA libraries, Writing Custom Kernels with OpenAI TritonCUDA Toolkit libraries, C++ and Python CUDA Librariesnumpy replacement cuPyNumeric, C++ and Python CUDA Librariesframeworks built on CUDA, PyTorch and Higher-Level AI Frameworks(see also PyTorch)Nsight Systems backtrace sampling, PyTorch Profiler and Visualization ToolsTriton language and compiler, C++ and Python CUDA Libraries
- PyTorcharithmetic intensity, PyTorch and Arithmetic Intensityattention mechanisms optimized, PyTorch Optimized Attention Mechanismsautomatic mixed precision, TF32 and Automatic Mixed Precision (PyTorch)-TF32 and Automatic Mixed Precision (PyTorch)backend support, Pitfall #1: Using a CPU-bound Gloo backend instead of NCCLverifying which backend, Pitfall #1: Using a CPU-bound Gloo backend instead of NCCLcompiler (see PyTorch compiler)CUDA Graphs, PyTorch, Inference Engines, and CUDA Graphs, Capturing a CUDA Graph with a CUDA Streamstatic memory pools, Memory Pools for CUDA Graphscustom kernel registration, Registering Custom Kernels with PyTorch-Registering Custom Kernels with PyTorchDataLoaderCPU affinity for each worker process, NUMA Awareness and CPU Pinning-NUMA Awareness and CPU Pinningdata input pipeline optimization, Optimizing the Data Input Pipeline-Optimizing the Data Input Pipelinedata loading and preprocessing, Efficient Data Loading and Preprocessing-Efficient Data Loading and Preprocessingmultiple workers, Tuning NVMe and Filesystem for Throughput, Efficient Data Loading and Preprocessing, Scaling Out Workers as You Scale Out Number of GPUs, Optimizing the Data Input Pipelinepersistent workers, Efficient Data Loading and Preprocessingpinned memory flag, NUMA-Friendly Memory Allocation and Memory Pinning, Using Streams to Overlap Compute with Data Transfers, Optimizing the Data Input Pipelineprefetch factor, Optimizing the Data Input PipelineDistributed Data Parallel communicationbucketing, Reducing Communication Frequency and Volumedata parallel, Distributed Data Parallel StrategiesData Parallel versus Distributed Data Parallel, Distributed Data Parallel Strategies-Distributed Data Parallel Strategiesdefinition of Data Parallel, Distributed Data Parallel Strategiesdefinition of Distributed Data Parallel, Distributed Data Parallel Strategiesmodel parallel, Distributed Data Parallel StrategiesDistributedSampler, Fast Storage and Data Localityfilesystem caching, Filesystem Caching and Write-BackGPU idle time measurement, Monitoring Storage I/Ointra-kernel pipelining, PyTorch, CUDA Pipeline API, and Warp Specializationintroduction, PyTorch and Higher-Level AI Frameworkscompiler stack, PyTorch and Higher-Level AI Frameworks, PyTorch Compiler (torch.compile), PyTorch Compiler, OpenAI Triton, and XLA BackendsCUDA complexity abstracted away, PyTorch and Higher-Level AI Frameworkstensor operations using GPUs, PyTorch and Higher-Level AI FrameworksTorchInductor backend, PyTorch and Higher-Level AI Frameworksmemory allocator best-fit with coalescing, Dynamic Memory-Allocation Switching (Slab Versus Caching Versus Stream-Ordered)memory allocator plugin, Pluggable Memory Allocators and Cross-GPU Data Transfers-Pluggable Memory Allocators and Cross-GPU Data Transfersmemory caching allocator, Asynchronous Memory Allocation and Memory Poolsmemory profiler, NVTX Markers and Profiling Tools, NVTX Markers and Profiling ToolsNCCL collectives running on dedicates streams, Multi-GPU Compute and Data Transfer Overlap with CUDA Streamsnested tensors, PyTorch and Arithmetic IntensityNVLink domain, NVIDIA’s “AI Supercomputer in a Rack”persistent kernels and megakernels, Common Workloads for Persistent Kernelsprofiler (see PyTorch profiler)sparse format from trained dense models, Structured Sparsitystreamscreating, Concurrency with CUDA Streamsdefault versus explicit streams, Default Versus Explicit (Nondefault) Streamsmultiple GPUs with multiple streams, Multi-GPU Compute and Data Transfer Overlap with CUDA Streamsoverlapping computation and data transfer, Overlapping Communication and Computationprofiling when adding streams, Overlapping Communication and Computationsymmetric memory, PyTorch Symmetric Memory
- PyTorch Architecture Optimization (torchao), PyTorch Architecture Optimization (torchao), Quantization, Sparsity, and Pruningpruning, PyTorch Architecture Optimization (torchao), Quantization, Sparsity, and Pruningquantization, PyTorch Architecture Optimization (torchao), Quantization, Sparsity, and PruningINT8 quantization, Transformer Engine and TMEM in Depthsparsity, PyTorch Architecture Optimization (torchao), Quantization, Sparsity, and Pruning
- PyTorch compiler (torch.compile), PyTorch Compiler (torch.compile)-Profiling and Debugging Compiler Performance Issuesabout the compiler, PyTorch and Higher-Level AI Frameworks, PyTorch Compiler (torch.compile)arithmetic intensity, PyTorch and Arithmetic IntensityAsyncTP enabled, TorchTitan, AsyncTP, AutoParallel, and SimpleFSDPcompilation pipeline, PyTorch Compiler Deep Dive-Debugging Numerical Correctness and Accuracyabout the compilation pipeline, PyTorch Compiler Deep DiveAOT Autograd, AOT Autograd Fusion for Forward and Backward PassesPrimTorch IR, PrimTorch IR (Prims) Simplified Operator SetTorchDynamo, TorchDynamo for Bytecode Capture and Graph Extraction-TorchDynamo for Bytecode Capture and Graph ExtractionTorchInductor, TorchInductor Backend Code Generationcompiler modes, Compilation Modes and Trade-Offs in Speed, Memory, and Compile Time-Compilation Modes and Trade-Offs in Speed, Memory, and Compile TimeCUDA Graphs triggered, Autotuning with TorchInductoreager mode, Disabling the PyTorch Compiler and Reverting Back to Eager Modemax-autotune mode, PyTorch Compiler Deep Divestatic shapes required for CUDA Graphs, Autotuning with TorchInductor, Dynamic Shapes and Variable Sequence Lengthscompiler stack, PyTorch and Higher-Level AI Frameworks, PyTorch Compiler (torch.compile), PyTorch Compiler, OpenAI Triton, and XLA Backendscompiling versus writing custom kernels, Compiling Versus Writing Custom KernelsCUDA Graph Trees, CUDA Graph Trees (PyTorch Compiler Internal)debugging, Performance Hints and Debugging Generated Code, Debugging Numerical Correctness and Accuracy, Debugging Compiler Phases, Graph Breaks, and Performancenumerical correctness and accuracy, Debugging Numerical Correctness and Accuracy-Debugging Numerical Correctness and AccuracyTORCH_LOGS, Profiling and Debugging Compiler Performance Issues, Performance Hints and Debugging Generated Codedisabling, Disabling the PyTorch Compiler and Reverting Back to Eager Modeeager mode, Disabling the PyTorch Compiler and Reverting Back to Eager Modedynamic shapes, Dynamic Shapes and Variable Sequence Lengths-Dynamic Shapes and Variable Sequence Lengthsexport, TorchInductor Backend Code Generationkernel fusion, Tuning Occupancy with PyTorch, Kernel Fusion, TorchDynamo for Bytecode Capture and Graph Extractionlaunch overhead reduced via CUDA Graphs, PyTorch, Inference Engines, and CUDA GraphsMega-Cache, Using the PyTorch Compilerper-thread default stream enabled, Modern Per-Thread Default Streamperformance hints, Performance Hints and Debugging Generated Codepipeline parallel with, Tensor and Pipeline Parallelism with torch.compileprofiling and debugging compiler performance, Profiling and Debugging Compiler Performance Issuesquantization with torchao, PyTorch Architecture Optimization (torchao), Quantization, Sparsity, and Pruningscaling with DDP, DDP with torch.compilescaling with FSDP, FSDP with torch.compile-FSDP with torch.compileSimpleFSDP, TorchTitan, AsyncTP, AutoParallel, and SimpleFSDPsmart compilers, Smart Compilers and Automated Code Optimizations-Smart Compilers and Automated Code Optimizationstensor parallel with, Tensor and Pipeline Parallelism with torch.compileTorchInductor, Coalesced Versus Uncoalesced Global Memory Access, Using the PyTorch Compiler, Using the PyTorch Compilerprofiling and benchmarking kernels generated, Profiling and Debugging Compiler Performance IssuesTransformer Engine, Autotuning with TorchInductorusing, Using the PyTorch Compiler-Using the PyTorch Compilervectorized memory operations, Vectorized Memory Access
- PyTorch DataLoader (see DataLoader (PyTorch))
- PyTorch Distributed Data Parallel (see Distributed Data Parallel (DDP; PyTorch))
- PyTorch Fully Sharded Data Parallel (see FSDP (Fully Sharded Data Parallel; PyTorch))
- PyTorch Holistic Trace Analysis (see Holistic Trace Analysis (HTA; PyTorch))
- PyTorch Kineto, PyTorch Profiler and Visualization Tools, NVTX Markers and Profiling Tools, NVTX Markers and Profiling ToolsNCCL profiler plugin, Profiling and Debugging NCCL
- PyTorch Mega-Cache, Using the PyTorch Compiler
- PyTorch profilerCUDA Performance Tools Interface, PyTorch Profiler and Visualization Toolsexample mixture-of-experts model, Using PyTorch Profiler-Using PyTorch ProfilerGPU bottlenecks, PyTorch Profiler and Visualization Tools-Profiler-Guided AnalysisKineto library, PyTorch Profiler and Visualization Tools, NVTX Markers and Profiling Tools, NVTX Markers and Profiling ToolsNCCL profiler plugin, Profiling and Debugging NCCLmemory profiler, NVTX Markers and Profiling Tools, NVTX Markers and Profiling Toolsmemory profiling and tuning, Profiling and Tuning Memory in PyTorch-Enabling Peer-to-Peer DMA and UCXactivation checkpointing, Activation Checkpointing for Memory SavingsCUDA memory allocator tuned, Tuning the CUDA Memory AllocatorFSDP automatic checkpointing, FSDP Automatic Checkpointing and Offloading-FSDP Automatic Checkpointing and OffloadingFSDP with tensor parallel and pipeline parallel, Combining FSDP with Tensor Parallel and Pipeline ParallelNCCL with UCX over multinode topologies, Enabling Peer-to-Peer DMA and UCXoffloading parameters, Offloading Parameters to CPU and NVMepeer-to-peer DMA, Enabling Peer-to-Peer DMA and UCXpluggable memory allocators, Pluggable Memory Allocators and Cross-GPU Data Transfers-Pluggable Memory Allocators and Cross-GPU Data TransfersNCCL profiler plugin, Profiling and Debugging NCCLNsight Systems PyTorch focused mode, PyTorch Profiler and Visualization Tools
- PyTorch TorchDynamo (see TorchDynamo (PyTorch))
- PyTorch TorchEval (see TorchEval (PyTorch))
- PyTorch TorchInductor (see TorchInductor (PyTorch))
- PYTORCH_CUDA_ALLOC_CONF, GPU Memory Oversubscription, Fragmentation, and Out-of-Memory Handling, Stream-Ordered Memory Allocator

- P2P (peer-to-peer) GPU copies, Pitfall #3: Avoid overtuning or disabling NCCL features with environment variables

- swizzling as alternative, Avoid Shared-Memory Bank Conflicts

- DeepSeek Fire-Flyer File System bypassing, DeepSeek’s Fire-Flyer File System
- latency-sensitive training workflows, Filesystem Caching and Write-Back
- O_DIRECT bypassing, Filesystem Caching and Write-Back, Using NVIDIA GDS

- FlexDecoding, FlexDecoding (PyTorch)

- adaptive strategies for ultrascale inference, Adaptive Parallelism Strategies (TP Versus PP Versus Hybrid)-Adaptive Parallelism Strategies (TP Versus PP Versus Hybrid)
- adding two vectors sequentially and in parallel, Maintaining High Occupancy and GPU Utilization-Maintaining High Occupancy and GPU Utilization
- architecture of GPU, Understanding GPU Architecture
- cuTile, C++ and Python CUDA Libraries
- Data Parallel versus Distributed Data Parallel, Distributed Data Parallel Strategies-Distributed Data Parallel Strategies
- dynamic parallelism, Dynamic Parallelism-Dynamic Parallelismbenchmarks, Dynamic Parallelismchild-kernel launch limit, Dynamic Parallelismdevice memory usage monitored, Dynamic Parallelismprofiling before and after a change, Dynamic Parallelism“stack overflow” errors avoided, Dynamic Parallelismwhen to capture CUDA Graph instead, Dynamic Parallelism
- fully sharded data parallel, Distributed Data Parallel Strategies
- PTX, Inline PTX and SASS Tuning for Microoptimizations(see also PTX (parallel thread execution))
- speculative decoding, Disaggregated Prefill and Decode Architecture, Speculative Decoding and Parallel Token Generation Techniques-Combining Decoding Techniques and Evaluating Complexity
- strategies, Parallelism Strategies for Serving Massive MoE Models-Hybrid Parallelismcontext parallelism, Parallelism Strategies for Serving Massive MoE Models, Context (Sequence) Parallelismdata parallelism, Parallelism Strategies for Serving Massive MoE Models, Data Parallelismexpert parallelism, Parallelism Strategies for Serving Massive MoE Models, Expert Parallelism-Expert Parallelismhybrid parallelism, Hybrid Parallelism-Hybrid Parallelismmodel weights and data split over GPUs, Parallelism Strategies for Serving Massive MoE Modelspipeline parallelism, Parallelism Strategies for Serving Massive MoE Models, Pipeline Parallelismprofiling, Hybrid Parallelismsummary table, Parallelism Strategies for Serving Massive MoE Modelstensor parallelism, Parallelism Strategies for Serving Massive MoE Models, Tensor Parallelism
- thread block cluster algorithms, Designing Efficient Algorithms with Thread Block Clusters-Designing Efficient Algorithms with Thread Block Clusters

- benchmarks, Dynamic Parallelism
- child-kernel launch limit, Dynamic Parallelism
- device memory usage monitored, Dynamic Parallelism
- profiling before and after a change, Dynamic Parallelism
- “stack overflow” errors avoided, Dynamic Parallelism
- when to capture CUDA Graph instead, Dynamic Parallelism

- (see also PTX (parallel thread execution))

- context parallelism, Parallelism Strategies for Serving Massive MoE Models, Context (Sequence) Parallelism
- data parallelism, Parallelism Strategies for Serving Massive MoE Models, Data Parallelism
- expert parallelism, Parallelism Strategies for Serving Massive MoE Models, Expert Parallelism-Expert Parallelism
- hybrid parallelism, Hybrid Parallelism-Hybrid Parallelism
- model weights and data split over GPUs, Parallelism Strategies for Serving Massive MoE Models
- pipeline parallelism, Parallelism Strategies for Serving Massive MoE Models, Pipeline Parallelism
- profiling, Hybrid Parallelism
- summary table, Parallelism Strategies for Serving Massive MoE Models
- tensor parallelism, Parallelism Strategies for Serving Massive MoE Models, Tensor Parallelism

- example MoE model, CPU and GPU Profiling with Linux perf-CPU and GPU Profiling with Linux perf
- NVIDIA Performance Monitoring Unit, CPU and GPU Profiling with Linux perf

- compressing data, Tuning, Replicating, and Compressing Data
- CUDA Graphs, Capturing a CUDA Graph with a CUDA Streamconditional graph nodes, Conditional Graph Nodes
- disaggregating prefill and decode, Prefill-Decode Interference
- dynamic batching, Dynamic Batching-Dynamic Batching
- high occupancy with high ILP, Loop Unrolling, Interleaving, and Compiler Hinting
- kernel optimization, Optimizing the Kernel-Optimizing the Kernel
- max-autotune compile mode, PyTorch Compiler Deep Dive
- memory pinning, NUMA-Friendly Memory Allocation and Memory Pinning
- network reads eliminated, Tuning, Replicating, and Compressing Data
- pitfall of repeatedly reading same global memory data, Tiling and Data Reuse Using Shared Memory
- running many warps concurrently, Maintaining High Occupancy and GPU Utilization
- stream 0 legacy default stream avoided, Legacy Default Stream
- TORCH_LOGS performance hints, Performance Hints and Debugging Generated Code
- UCX with NCCL over multinode topologies, Enabling Peer-to-Peer DMA and UCX

- conditional graph nodes, Conditional Graph Nodes

- about ultimate goal of performance tuning, Maximizing GPU Utilization and Throughput Versus Latency Trade-Offs
- automated performance tests, Benchmarking and Profiling
- benchmarking and profiling, Benchmarking and Profiling(see also benchmarks; profiling)
- cache hits and misses, Profiling, Debugging, and Tuning Inference Performance
- communication versus compute bottlenecks, Diagnosing Communication- Versus Compute-Bound Workloads
- disaggregation of prefill and decode, Impact on Latency (TTFT) and Throughput (TPOT)
- goodput as throughput metric, Measuring “Goodput” Useful Throughput
- GPU idle time measurement, Monitoring Storage I/O
- GPU utilization percentage, Maximizing GPU Utilization and Throughput Versus Latency Trade-Offs
- GPUs with NVIDIA Container Toolkit, Container Runtime Optimizations for GPUs
- inference tuningerror handling, Error Handlingfull-stack optimizations, Full-Stack Inference Optimizations-Full-Stack Inference Optimizationsmonitoring system metrics and counters, Monitoring System Metrics and Counters-Monitoring System Metrics and Countersobserve-hypothesize-tune loop, Profiling, Debugging, and Tuning Inference Performance-Profiling, Debugging, and Tuning Inference Performanceprofiling with Nsight, Profiling with Nsight Systems and Nsight Compute-Profiling with Nsight Systems and Nsight Computetroubleshooting, Inference Troubleshooting Recipes
- I/O continuous profiling and tuning, Continuous Profiling and Tuning Workflow-Continuous Profiling and Tuning Workflow
- latency hiding as important tool, Streaming Multiprocessor, Threads, and Warps
- MLPerf Logging, Performance Benchmarks and MLPerf Logging-Performance Benchmarks and MLPerf Logging
- performance benchmarking, Continuous Integration and Performance Benchmarking-Performance Benchmarks and MLPerf Loggingforcing an error when full graph not captured, Graph Breaks and TorchDynamo explain()
- performance monitoring, Performance Monitoring and Utilization in PracticeGPUs near 100% utilized, Performance Monitoring and Utilization in PracticeNVLink usage, Performance Monitoring and Utilization in Practice
- prefix-merge metrics, Profiling, Debugging, and Tuning Inference Performance
- PyTorch performance heads-up display, Continuous Integration and Performance Benchmarking-PyTorch HUD Performance Dashboard
- roofline model, Roofline Model: Compute-Bound or Memory-Bound Workloads-Roofline Model: Compute-Bound or Memory-Bound WorkloadsNsight Compute Roofline analysis, Nsight Compute and Roofline Analysis
- RPS (requests per second), Dynamic Batching
- thread block size, Choosing Threads-per-Block and Blocks-per-Grid Sizes
- TPOT (time per output token)disaggregation of prefill and decode, Impact on Latency (TTFT) and Throughput (TPOT)prefill and decode phases, Scaling Disaggregated Prefill and Decode for Inference
- TTFT (time to first token)constrained outputs, Constrained Decoding Performance Implicationscontext parallelism, Context (Sequence) Parallelismdisaggregation of prefill and decode, Impact on Latency (TTFT) and Throughput (TPOT)prefill and decode phases, Scaling Disaggregated Prefill and Decode for Inference

- (see also benchmarks; profiling)

- error handling, Error Handling
- full-stack optimizations, Full-Stack Inference Optimizations-Full-Stack Inference Optimizations
- monitoring system metrics and counters, Monitoring System Metrics and Counters-Monitoring System Metrics and Counters
- observe-hypothesize-tune loop, Profiling, Debugging, and Tuning Inference Performance-Profiling, Debugging, and Tuning Inference Performance
- profiling with Nsight, Profiling with Nsight Systems and Nsight Compute-Profiling with Nsight Systems and Nsight Compute
- troubleshooting, Inference Troubleshooting Recipes

- forcing an error when full graph not captured, Graph Breaks and TorchDynamo explain()

- GPUs near 100% utilized, Performance Monitoring and Utilization in Practice
- NVLink usage, Performance Monitoring and Utilization in Practice

- Nsight Compute Roofline analysis, Nsight Compute and Roofline Analysis

- disaggregation of prefill and decode, Impact on Latency (TTFT) and Throughput (TPOT)
- prefill and decode phases, Scaling Disaggregated Prefill and Decode for Inference

- constrained outputs, Constrained Decoding Performance Implications
- context parallelism, Context (Sequence) Parallelism
- disaggregation of prefill and decode, Impact on Latency (TTFT) and Throughput (TPOT)
- prefill and decode phases, Scaling Disaggregated Prefill and Decode for Inference

- nvidia-persistenced daemon, GPU Persistence Mode
- recommended when using MIG, Slicing a GPU with MIG

- common workloads for, Common Workloads for Persistent Kernels
- cooperative groups via grid sync, Cooperative Grid Synchronization and Persistent Kernels-Cooperative Grid Synchronization and Persistent Kernelswhen to combine, When to Combine Persistent Kernels and Cooperative Groups
- megakernels, Megakernels for Inference
- occupancy, Persistent Kernels and Megakernels
- warp specialization, Persistent Kernels and Warp Specialization

- when to combine, When to Combine Persistent Kernels and Cooperative Groups

- FSDP with, Combining FSDP with Tensor Parallel and Pipeline Parallel
- hybrid parallelism, Hybrid Parallelism
- PyTorch compiler with, Tensor and Pipeline Parallelism with torch.compile
- tensor parallel versus, Pipeline Parallelism
- versus TP versus hybrid, Adaptive Parallelism Strategies (TP Versus PP Versus Hybrid)-Adaptive Parallelism Strategies (TP Versus PP Versus Hybrid)

- comparison of no pipeline with pipeline, Achieving Maximal Overlap in Practice-Achieving Maximal Overlap in Practice
- CUDA C++ Pipeline API, Asynchronous Memory Prefetching and Tensor Memory Accelerator-Asynchronous Memory Prefetching and Tensor Memory Acceleratorhardware-software codesign, Asynchronous Memory Prefetching and Tensor Memory Accelerator
- CUDA Graphs, Capturing a CUDA Graph with a CUDA Stream(see also CUDA Graphs)
- data compression, Tuning, Replicating, and Compressing Data
- data pipeline tuning, Tuning the Data Pipelinedata loading and preprocessing, Efficient Data Loading and Preprocessing-Efficient Data Loading and PreprocessingNeMo Curator for training datasets, Creating High-Quality LLM Datasets with NVIDIA NeMo CuratorNVIDIA Data Loading Library, Multimodal Data Processing with NVIDIA DALIscaling out workers as GPUs scaled, Scaling Out Workers as You Scale Out Number of GPUs
- distributed data parallel strategies, Distributed Data Parallel Strategies-Distributed Data Parallel Strategies
- inter-kernel pipeliningabout inter-kernel pipelining, Inter-Kernel Pipelining, Synchronization, and CUDA Stream-Ordered Memory Allocationscompute overlapping with data transfers via streams, Using Streams to Overlap Compute with Data Transfers-Using Streams to Overlap Compute with Data Transfers, Multi-GPU Compute and Data Transfer Overlap with CUDA Streams-Multi-GPU Compute and Data Transfer Overlap with CUDA Streams, Overlapping Communication and Computation-Overlapping Communication and Computationcross-stream synchronization, Using CUDA Events for Cross-Stream SynchronizationCUDA stream ordered memory allocator, Stream-Ordered Memory Allocator-Stream-Ordered Memory AllocatorCUDA stream ordered memory allocator used, Using CUDA Streams and Stream-Ordered Memory Allocator with LLMs-Using CUDA Streams and Stream-Ordered Memory Allocator with LLMsCUDA stream ordered memory allocator with multiple GPUs, Multi-GPU Compute and Data Transfer Overlap with CUDA Streams-Multi-GPU Compute and Data Transfer Overlap with CUDA Streamsdefault stream usage, Best Practices for Default Stream Usage-Best Practices for Default Stream Usagedefault versus explicit streams, Default Versus Explicit (Nondefault) Streamskernel execution overlapped with CUDA streams, Overlapping Kernel Execution with CUDA Streams-Overlapping Kernel Execution with CUDA Streamslaunching five kernels on two CUDA streams, Overlapping Kernel Execution with CUDA Streams-Overlapping Kernel Execution with CUDA Streamslegacy default streams, Legacy Default Streammulti-GPU overlap of compute and data, Multi-GPU Compute and Data Transfer Overlap with CUDA Streams-Multi-GPU Compute and Data Transfer Overlap with CUDA StreamsPDL with thread block clusters and warp specialization, Combining PDL and Thread Block Clusters with Warp Specialization-Combining PDL and Thread Block Clusters with Warp Specializationper-thread default streams, Modern Per-Thread Default StreamProgrammatic Dependent Launch, Programmatic Dependent Launch-Programmatic Dependent Launchsynchronization with events and callbacks, Fine-Grained Synchronization with Events and Callbackswarp specialization and CUDA streams, Pipelining with Warp Specialization (Intra-Kernel) and CUDA Streams (Inter-Kernel)-Pipelining with Warp Specialization (Intra-Kernel) and CUDA Streams (Inter-Kernel)warp specialization replaced with CUDA streams, Overlapping Kernel Execution with CUDA Streams-Overlapping Kernel Execution with CUDA Streamswarp specialization, thread block clusters, and CUDA streams, Warp Specialization with Thread Block Clusters and CUDA Streams
- intra-kernel pipeliningabout intra-kernel pipelining, Intra-Kernel Pipelining Techniquescomparing techniques, Intra-Kernel Pipelining Techniques, Using CUDA Pipeline API for Warp Specialization, Warp Specialization with Thread Block Clusters-Warp Specialization with Thread Block Clusterscooperative groups, Thread Block Clusters and Distributed Shared Memory(see also cooperative groups (CGs))double-buffering with Pipeline API, Cooperative Tiling and Double-Buffering with the CUDA Pipeline API-Cooperative Tiling and Double-Buffering with the CUDA Pipeline APIpersistent kernels, Persistent Kernels and Megakernels-Persistent Kernels and Warp SpecializationPyTorch, PyTorch, CUDA Pipeline API, and Warp Specializationthread block clusters, Intra-Kernel Pipelining, Warp Specialization, and Cooperative Thread Block Clusters(see also thread block clusters)warp specialization, Warp Specialization and the Producer-Consumer Model-Warp Specialization and the Producer-Consumer Modelwarp specialization and CUDA streams, Pipelining with Warp Specialization (Intra-Kernel) and CUDA Streams (Inter-Kernel)-Pipelining with Warp Specialization (Intra-Kernel) and CUDA Streams (Inter-Kernel)warp specialization using CUDA Pipeline API, Using CUDA Pipeline API for Warp Specialization-Using CUDA Pipeline API for Warp Specializationwarp specialization with persistent kernels, Persistent Kernels and Warp Specializationwhen to combine persistent kernels and cooperative groups, When to Combine Persistent Kernels and Cooperative Groupswhen to use warp specialization versus double-buffering, Warp Specialization and the Producer-Consumer Model
- monitoring I/O, Monitoring Storage I/O-Monitoring Storage I/O
- overlapping communication and computation, Overlapping Communication and Computation (Pipelining)-Achieving Maximal Overlap in Practice
- reducing frequency and volume, Reducing Communication Frequency and Volume
- streams for asynchronous execution, Asynchronous Execution with Streams
- Triton double-buffering, Software Pipelining and Double Buffering with Triton-Software Pipelining and Double Buffering with Triton

- hardware-software codesign, Asynchronous Memory Prefetching and Tensor Memory Accelerator

- (see also CUDA Graphs)

- data loading and preprocessing, Efficient Data Loading and Preprocessing-Efficient Data Loading and Preprocessing
- NeMo Curator for training datasets, Creating High-Quality LLM Datasets with NVIDIA NeMo Curator
- NVIDIA Data Loading Library, Multimodal Data Processing with NVIDIA DALI
- scaling out workers as GPUs scaled, Scaling Out Workers as You Scale Out Number of GPUs

- about inter-kernel pipelining, Inter-Kernel Pipelining, Synchronization, and CUDA Stream-Ordered Memory Allocations
- compute overlapping with data transfers via streams, Using Streams to Overlap Compute with Data Transfers-Using Streams to Overlap Compute with Data Transfers, Multi-GPU Compute and Data Transfer Overlap with CUDA Streams-Multi-GPU Compute and Data Transfer Overlap with CUDA Streams, Overlapping Communication and Computation-Overlapping Communication and Computation
- cross-stream synchronization, Using CUDA Events for Cross-Stream Synchronization
- CUDA stream ordered memory allocator, Stream-Ordered Memory Allocator-Stream-Ordered Memory Allocator
- CUDA stream ordered memory allocator used, Using CUDA Streams and Stream-Ordered Memory Allocator with LLMs-Using CUDA Streams and Stream-Ordered Memory Allocator with LLMs
- CUDA stream ordered memory allocator with multiple GPUs, Multi-GPU Compute and Data Transfer Overlap with CUDA Streams-Multi-GPU Compute and Data Transfer Overlap with CUDA Streams
- default stream usage, Best Practices for Default Stream Usage-Best Practices for Default Stream Usage
- default versus explicit streams, Default Versus Explicit (Nondefault) Streams
- kernel execution overlapped with CUDA streams, Overlapping Kernel Execution with CUDA Streams-Overlapping Kernel Execution with CUDA Streams
- launching five kernels on two CUDA streams, Overlapping Kernel Execution with CUDA Streams-Overlapping Kernel Execution with CUDA Streams
- legacy default streams, Legacy Default Stream
- multi-GPU overlap of compute and data, Multi-GPU Compute and Data Transfer Overlap with CUDA Streams-Multi-GPU Compute and Data Transfer Overlap with CUDA Streams
- PDL with thread block clusters and warp specialization, Combining PDL and Thread Block Clusters with Warp Specialization-Combining PDL and Thread Block Clusters with Warp Specialization
- per-thread default streams, Modern Per-Thread Default Stream
- Programmatic Dependent Launch, Programmatic Dependent Launch-Programmatic Dependent Launch
- synchronization with events and callbacks, Fine-Grained Synchronization with Events and Callbacks
- warp specialization and CUDA streams, Pipelining with Warp Specialization (Intra-Kernel) and CUDA Streams (Inter-Kernel)-Pipelining with Warp Specialization (Intra-Kernel) and CUDA Streams (Inter-Kernel)
- warp specialization replaced with CUDA streams, Overlapping Kernel Execution with CUDA Streams-Overlapping Kernel Execution with CUDA Streams
- warp specialization, thread block clusters, and CUDA streams, Warp Specialization with Thread Block Clusters and CUDA Streams

- about intra-kernel pipelining, Intra-Kernel Pipelining Techniques
- comparing techniques, Intra-Kernel Pipelining Techniques, Using CUDA Pipeline API for Warp Specialization, Warp Specialization with Thread Block Clusters-Warp Specialization with Thread Block Clusters
- cooperative groups, Thread Block Clusters and Distributed Shared Memory(see also cooperative groups (CGs))
- double-buffering with Pipeline API, Cooperative Tiling and Double-Buffering with the CUDA Pipeline API-Cooperative Tiling and Double-Buffering with the CUDA Pipeline API
- persistent kernels, Persistent Kernels and Megakernels-Persistent Kernels and Warp Specialization
- PyTorch, PyTorch, CUDA Pipeline API, and Warp Specialization
- thread block clusters, Intra-Kernel Pipelining, Warp Specialization, and Cooperative Thread Block Clusters(see also thread block clusters)
- warp specialization, Warp Specialization and the Producer-Consumer Model-Warp Specialization and the Producer-Consumer Model
- warp specialization and CUDA streams, Pipelining with Warp Specialization (Intra-Kernel) and CUDA Streams (Inter-Kernel)-Pipelining with Warp Specialization (Intra-Kernel) and CUDA Streams (Inter-Kernel)
- warp specialization using CUDA Pipeline API, Using CUDA Pipeline API for Warp Specialization-Using CUDA Pipeline API for Warp Specialization
- warp specialization with persistent kernels, Persistent Kernels and Warp Specialization
- when to combine persistent kernels and cooperative groups, When to Combine Persistent Kernels and Cooperative Groups
- when to use warp specialization versus double-buffering, Warp Specialization and the Producer-Consumer Model

- (see also cooperative groups (CGs))

- (see also thread block clusters)

- NVIDIA Management Library, Kernel Compute Throughput Versus Peak GPU FLOPS
- power limits, Performance Monitoring and Utilization in Practice, Power and Thermal Constraintssetting slightly below TDP, GPU Clock Speeds and ECC

- setting slightly below TDP, GPU Clock Speeds and ECC

- GPU persistence mode, GPU Persistence Mode
- NVL72, Compute Density and Power Requirementsliquid cooling versus air cooling, Liquid Cooling Versus Air Cooling-Liquid Cooling Versus Air Coolingpower monitoring, Performance Monitoring and Utilization in Practice

- liquid cooling versus air cooling, Liquid Cooling Versus Air Cooling-Liquid Cooling Versus Air Cooling
- power monitoring, Performance Monitoring and Utilization in Practice

- unrolling too many iterations, Profiling and Mitigating Register Pressure

- dynamic precision changes, Dynamic Precision Changes-Dynamic Precision Changes
- FP4, FP8, FP16, NVIDIA GPU Tensor Cores and Transformer Engine, BF16/FP16, FP8, and FP4 Reduced Precision
- inference quantization, Reducing Precision from FP16 to FP8 and FP4weight-only quantization, Weight-Only Quantization (GPTQ, AWQ)
- mixed precisionautomatic mixed precision, TF32 and Automatic Mixed Precision (PyTorch)-TF32 and Automatic Mixed Precision (PyTorch)reduced precision, BF16/FP16, FP8, and FP4 Reduced PrecisionTensor Cores, Mixed Precision and Utilizing Tensor Cores-Transformer Engine and TMEM in Depth
- quantization (see quantization)

- weight-only quantization, Weight-Only Quantization (GPTQ, AWQ)

- automatic mixed precision, TF32 and Automatic Mixed Precision (PyTorch)-TF32 and Automatic Mixed Precision (PyTorch)
- reduced precision, BF16/FP16, FP8, and FP4 Reduced Precision
- Tensor Cores, Mixed Precision and Utilizing Tensor Cores-Transformer Engine and TMEM in Depth

- warp divergence avoided, Techniques to Avoid Warp Divergence
- warp divergence minimized, Using Predication to Minimize Divergence-Using Predication to Minimize Divergence

- about prefill phase, Scaling Disaggregated Prefill and Decode for Inference
- chunked prefill, Stall-Free Scheduling (Chunked Prefill), Adaptive Batching and Chunked Prefill Scheduling-Adaptive Batching and Chunked Prefill Scheduling
- context parallelism, Parallelism Strategies for Serving Massive MoE Models, Context (Sequence) ParallelismContext Parallel (PyTorch), PyTorch Optimized Attention Mechanisms
- disaggregated from decodeabout disaggregation, Multinode Inference, Parallelism, Decoding, and Routing Optimizations, Prefill-Decode Interference, Scaling Disaggregated Prefill and Decode for Inferencearchitecture, Disaggregated Prefill and Decode Architecturebatch size optimizations, Adaptive Batching and Chunked Prefill Schedulingcluster pools, Disaggregated Prefill and Decode Cluster Pools-Memory management for the KV cachedeploying with Kubernetes, Deploying Disaggregated Prefill and Decode with Kubernetes-Deploying Disaggregated Prefill and Decode with Kubernetes, Full-Stack Inference Optimizationslatency and throughput, Impact on Latency (TTFT) and Throughput (TPOT)load balancing, Dynamic Scheduling and Load Balancing-Dynamic resource scalingprefill to decode data transfers, KV Cache Data Transfer and NIXL, Fast KV Cache Transfer Between Prefill and Decode-Zero-Copy GPU-to-GPU Transferscaling, Scaling Prefill and Worker Nodes Independentlyseparate prefill and decode stages, Separate Prefill and Decode Inference Stages-Separate Prefill and Decode Inference Stages, Disaggregated Prefill and Decode Architecture
- disaggregated routing policies, Disaggregated Routing and Scheduling Policies-QoS and early rejection policiescapacity-aware routing, Capacity-aware routing-Capacity-aware routingexample dynamic routing policy configuration, Example dynamic routing policy configurationexample dynamic routing policy in code, Example dynamic routing policy in codelatency-aware routing, Latency-aware routing-Latency-aware routingmultipath inference (racing), Multipath inference (racing)QoS and early rejection policies, QoS and early rejection policies-QoS and early rejection policiesrouting factors, Routing factors-Routing factorsspeculative decoding, Multibranch, parallel speculative decoding across workers
- FlashAttention, Mechanical Sympathy: Hardware-Software Codesign
- hardware and parallelism, Heterogeneous Hardware and Parallelism Strategies for Prefill and Decode-Different precision for prefill and decode
- hybrid prefill with GPU-CPU collaboration, Hybrid Prefill with GPU-CPU Collaboration-Hybrid Prefill with GPU-CPU Collaboration
- KV cache transfer to decode, Fast KV Cache Transfer Between Prefill and Decode-Zero-Copy GPU-to-GPU Transferconnector and data path design, Connector and Data Path Design-Connector and Data Path DesignKV cache size, KV Cache Sizezero-copy GPU-to-GPU, Zero-Copy GPU-to-GPU Transfer-Zero-Copy GPU-to-GPU Transfer
- optimized kernels via FlexDecoding, FlexDecoding (PyTorch)-FlexDecoding (PyTorch)
- scale-out of workers, Continuous Prewarming of CUDA Graphs and Caches Using Time-Series Prediction
- why disaggregation, Why Prefill-Decode Disaggregation?-Scalability of Disaggregated Prefill and Decodeadvantages of disaggregation, Advantages of Disaggregation-Phase-specific optimizationsprefill workers, Prefill workers design-Latency-aware scheduling and batchingrouting and scheduling policies, Disaggregated Routing and Scheduling Policies-QoS and early rejection policiesscalability, Scalability of Disaggregated Prefill and Decode

- Context Parallel (PyTorch), PyTorch Optimized Attention Mechanisms

- about disaggregation, Multinode Inference, Parallelism, Decoding, and Routing Optimizations, Prefill-Decode Interference, Scaling Disaggregated Prefill and Decode for Inference
- architecture, Disaggregated Prefill and Decode Architecture
- batch size optimizations, Adaptive Batching and Chunked Prefill Scheduling
- cluster pools, Disaggregated Prefill and Decode Cluster Pools-Memory management for the KV cache
- deploying with Kubernetes, Deploying Disaggregated Prefill and Decode with Kubernetes-Deploying Disaggregated Prefill and Decode with Kubernetes, Full-Stack Inference Optimizations
- latency and throughput, Impact on Latency (TTFT) and Throughput (TPOT)
- load balancing, Dynamic Scheduling and Load Balancing-Dynamic resource scaling
- prefill to decode data transfers, KV Cache Data Transfer and NIXL, Fast KV Cache Transfer Between Prefill and Decode-Zero-Copy GPU-to-GPU Transfer
- scaling, Scaling Prefill and Worker Nodes Independently
- separate prefill and decode stages, Separate Prefill and Decode Inference Stages-Separate Prefill and Decode Inference Stages, Disaggregated Prefill and Decode Architecture

- capacity-aware routing, Capacity-aware routing-Capacity-aware routing
- example dynamic routing policy configuration, Example dynamic routing policy configuration
- example dynamic routing policy in code, Example dynamic routing policy in code
- latency-aware routing, Latency-aware routing-Latency-aware routing
- multipath inference (racing), Multipath inference (racing)
- QoS and early rejection policies, QoS and early rejection policies-QoS and early rejection policies
- routing factors, Routing factors-Routing factors
- speculative decoding, Multibranch, parallel speculative decoding across workers

- connector and data path design, Connector and Data Path Design-Connector and Data Path Design
- KV cache size, KV Cache Size
- zero-copy GPU-to-GPU, Zero-Copy GPU-to-GPU Transfer-Zero-Copy GPU-to-GPU Transfer

- advantages of disaggregation, Advantages of Disaggregation-Phase-specific optimizations
- prefill workers, Prefill workers design-Latency-aware scheduling and batching
- routing and scheduling policies, Disaggregated Routing and Scheduling Policies-QoS and early rejection policies
- scalability, Scalability of Disaggregated Prefill and Decode

- SGLang RadixAttention, Prefix Caching-Prefix Caching

- FX graph output, PrimTorch IR (Prims) Simplified Operator Set
- PyTorch compiler pipeline, PyTorch Compiler Deep Dive

- achieved occupancy, Inspecting Achieved Occupancy and GPU Utilization-Optimizing the Kernelkernel compute throughput versus GPU FLOPS, Kernel Compute Throughput Versus Peak GPU FLOPSkernel memory throughput versus HBM memory bandwidth, Kernel Memory Throughput Versus Peak HBM Memory Bandwidth
- before and after a change, Dynamic Parallelism
- compiler performance issues, Profiling and Debugging Compiler Performance Issues
- CUDA events for, Using CUDA Events for Cross-Stream Synchronization
- DataLoader, Efficient Data Loading and Preprocessing
- example mixture-of-experts model, Profiling PyTorch to Identify Bottlenecks-CPU and GPU Profiling with Linux perfabout MoE models, Profiling PyTorch to Identify BottlenecksLinux perf on CPU and GPU, CPU and GPU Profiling with Linux perf-CPU and GPU Profiling with Linux perfNsight Systems and NVTX Timelines, System Profiling with Nsight Systems and NVTX Timelines-System Profiling with Nsight Systems and NVTX TimelinesPyTorch profiler, Using PyTorch Profiler-Using PyTorch Profilerroofline analysis for GEMM kernels, Kernel Roofline Analysis for General Matrix Multiply (GEMM)-Kernel Roofline Analysis for General Matrix Multiply (GEMM)
- GPU bottlenecks, Profiling and Diagnosing GPU Bottlenecks-Profiler-Guided Analysisachieved occupancy and GPU utilization, Inspecting Achieved Occupancy and GPU Utilization-Optimizing the KernelCUDA Profiling Tools Interface, PyTorch Profiler and Visualization Toolsiteratively profiling, Iteratively Profiling and Determining the Kernel Bottleneck-Iteratively Profiling and Determining the Kernel Bottleneckkernel optimization, Optimizing the Kernel-Optimizing the KernelNsight Compute, Profiling and Diagnosing GPU BottlenecksNsight Compute and Roofline analysis, Nsight Compute and Roofline AnalysisNsight Systems, Profiling and Diagnosing GPU BottlenecksNsight Systems timeline view, Nsight Systems Timeline ViewPyTorch profiler via Kineto, PyTorch Profiler and Visualization Tools-Profiler-Guided Analysiswarp stall reasons, Analyzing Warp Stall Reasons with Nsight Compute-Other Stall Reasons
- graph break recompilations monitored, TorchDynamo for Bytecode Capture and Graph Extraction, Minimize Graph Recompilations
- Holistic Trace Analysis, Multi-GPU Profiling with HTA
- inference, Profiling with Nsight Systems and Nsight Compute-Profiling with Nsight Systems and Nsight Computeproduction environment, Inference Troubleshooting Recipes
- I/O continuous profiling and tuning, Continuous Profiling and Tuning Workflow-Continuous Profiling and Tuning Workflowcommunication versus compute bottlenecks, Diagnosing Communication- Versus Compute-Bound Workloads
- kernels generated by TorchInductor, Profiling and Debugging Compiler Performance Issues
- latency with streaming enabled, Streaming Responses
- memory via PyTorch profiler, Profiling and Tuning Memory in PyTorch-Enabling Peer-to-Peer DMA and UCXactivation checkpointing, Activation Checkpointing for Memory SavingsCUDA memory allocator tuned, Tuning the CUDA Memory AllocatorFSDP automatic checkpointing, FSDP Automatic Checkpointing and Offloading-FSDP Automatic Checkpointing and OffloadingFSDP with tensor parallel and pipeline parallel, Combining FSDP with Tensor Parallel and Pipeline ParallelNCCL with UCX over multinode topologies, Enabling Peer-to-Peer DMA and UCXoffloading parameters, Offloading Parameters to CPU and NVMepeer-to-peer DMA, Enabling Peer-to-Peer DMA and UCXpluggable memory allocators, Pluggable Memory Allocators and Cross-GPU Data Transfers-Pluggable Memory Allocators and Cross-GPU Data Transfers
- multi-GPU profilingHTA, Multi-GPU Profiling with HTANsight Systems, Real-Time Link Telemetry and Monitoring
- NCCL, Profiling and Debugging NCCLNCCL logs, In-Network SHARP Aggregation
- Nsight Compute occupancy limiters, Inspecting Achieved Occupancy and GPU UtilizationLimited by Registers, Profiling and Mitigating Register Pressure
- Nsight Compute warp stall reasons, Analyzing Warp Stall Reasons with Nsight Compute-Other Stall ReasonsStall: Compute Unit Busy, Execution Unit ContentionStall: Exec Dependency, Execution-Dependency Stalls, Profiling and Mitigating Register PressureStall: Idle, Other Stall ReasonsStall: Long Scoreboard, Memory-Related StallsStall: Math Pipe Throttle, Execution Unit ContentionStall: Memory Dependency, Other Stall ReasonsStall: Memory Throttle, Memory-Related StallsStall: No Eligible, Other Stall ReasonsStall: Not Selected, Memory-Related StallsStall: Short Scoreboard, Memory-Related StallsStall: Texture Throttle, Other Stall Reasonsstalling for other reasons, Other Stall Reasons-Other Stall Reasons
- NVTX markers, NVTX Markers and Profiling Toolsprofiling tools that use markers, NVTX Markers and Profiling Tools-NVTX Markers and Profiling Tools
- occupancy tuning, Find the Right Occupancy for Your Workload
- parallelism strategies, Hybrid Parallelism
- tools for profiling, NVTX Markers and Profiling Tools-NVTX Markers and Profiling Tools
- Triton Proton Profiler, Profiling with Triton Proton Profiler
- warp divergence, Profiling and Detecting Warp Divergence
- workload under insufficient bandwidth, Pitfall #4: Insufficient network bandwidth or misconfigured NICs

- kernel compute throughput versus GPU FLOPS, Kernel Compute Throughput Versus Peak GPU FLOPS
- kernel memory throughput versus HBM memory bandwidth, Kernel Memory Throughput Versus Peak HBM Memory Bandwidth

- about MoE models, Profiling PyTorch to Identify Bottlenecks
- Linux perf on CPU and GPU, CPU and GPU Profiling with Linux perf-CPU and GPU Profiling with Linux perf
- Nsight Systems and NVTX Timelines, System Profiling with Nsight Systems and NVTX Timelines-System Profiling with Nsight Systems and NVTX Timelines
- PyTorch profiler, Using PyTorch Profiler-Using PyTorch Profiler
- roofline analysis for GEMM kernels, Kernel Roofline Analysis for General Matrix Multiply (GEMM)-Kernel Roofline Analysis for General Matrix Multiply (GEMM)

- achieved occupancy and GPU utilization, Inspecting Achieved Occupancy and GPU Utilization-Optimizing the Kernel
- CUDA Profiling Tools Interface, PyTorch Profiler and Visualization Tools
- iteratively profiling, Iteratively Profiling and Determining the Kernel Bottleneck-Iteratively Profiling and Determining the Kernel Bottleneck
- kernel optimization, Optimizing the Kernel-Optimizing the Kernel
- Nsight Compute, Profiling and Diagnosing GPU Bottlenecks
- Nsight Compute and Roofline analysis, Nsight Compute and Roofline Analysis
- Nsight Systems, Profiling and Diagnosing GPU Bottlenecks
- Nsight Systems timeline view, Nsight Systems Timeline View
- PyTorch profiler via Kineto, PyTorch Profiler and Visualization Tools-Profiler-Guided Analysis
- warp stall reasons, Analyzing Warp Stall Reasons with Nsight Compute-Other Stall Reasons

- production environment, Inference Troubleshooting Recipes

- communication versus compute bottlenecks, Diagnosing Communication- Versus Compute-Bound Workloads

- activation checkpointing, Activation Checkpointing for Memory Savings
- CUDA memory allocator tuned, Tuning the CUDA Memory Allocator
- FSDP automatic checkpointing, FSDP Automatic Checkpointing and Offloading-FSDP Automatic Checkpointing and Offloading
- FSDP with tensor parallel and pipeline parallel, Combining FSDP with Tensor Parallel and Pipeline Parallel
- NCCL with UCX over multinode topologies, Enabling Peer-to-Peer DMA and UCX
- offloading parameters, Offloading Parameters to CPU and NVMe
- peer-to-peer DMA, Enabling Peer-to-Peer DMA and UCX
- pluggable memory allocators, Pluggable Memory Allocators and Cross-GPU Data Transfers-Pluggable Memory Allocators and Cross-GPU Data Transfers

- HTA, Multi-GPU Profiling with HTA
- Nsight Systems, Real-Time Link Telemetry and Monitoring

- NCCL logs, In-Network SHARP Aggregation

- Limited by Registers, Profiling and Mitigating Register Pressure

- Stall: Compute Unit Busy, Execution Unit Contention
- Stall: Exec Dependency, Execution-Dependency Stalls, Profiling and Mitigating Register Pressure
- Stall: Idle, Other Stall Reasons
- Stall: Long Scoreboard, Memory-Related Stalls
- Stall: Math Pipe Throttle, Execution Unit Contention
- Stall: Memory Dependency, Other Stall Reasons
- Stall: Memory Throttle, Memory-Related Stalls
- Stall: No Eligible, Other Stall Reasons
- Stall: Not Selected, Memory-Related Stalls
- Stall: Short Scoreboard, Memory-Related Stalls
- Stall: Texture Throttle, Other Stall Reasons
- stalling for other reasons, Other Stall Reasons-Other Stall Reasons

- profiling tools that use markers, NVTX Markers and Profiling Tools-NVTX Markers and Profiling Tools

- with thread block clusters and warp specialization, Combining PDL and Thread Block Clusters with Warp Specialization-Combining PDL and Thread Block Clusters with Warp Specialization

- automated code optimization, Smart Compilers and Automated Code Optimizations-Smart Compilers and Automated Code Optimizations
- CUDAasynchronous memory allocation, Asynchronous Memory Allocation and Memory Pools-Asynchronous Memory Allocation and Memory PoolsCUDA streams, Asynchronous Memory Allocation and Memory Poolsflow between CPU and GPU, Understanding GPU Architecture__global__, CUDA Programming RefresherGPU memory hierarchy, Understanding GPU Memory Hierarchy-Understanding GPU Memory Hierarchyhigh occupancy and GPU utilization, Maintaining High Occupancy and GPU Utilization-Maintaining High Occupancy and GPU Utilizationhighest-level and most-recent APIs available, Asynchronous Memory Prefetching and Tensor Memory Acceleratorkernel errors, CUDA Programming Refresherkernel inputs in 1D, 2D and 3D Kernel Inputskernel inputs in 2D and 3D, 2D and 3D Kernel Inputskernels for parallel work, CUDA Programming Refresher-CUDA Programming Refresherlaunch parameters, Configuring Launch Parameters: Blocks per Grid and Threads per Block-Configuring Launch Parameters: Blocks per Grid and Threads per Blockmaximum threadsPerBlock compile time parameter, Tuning Occupancy with Launch Bounds-Tuning Occupancy with Launch Boundsmemory pools, Asynchronous Memory Allocation and Memory Pools-Asynchronous Memory Allocation and Memory Poolsoccupancy tuning with __launch_bounds__, Tuning Occupancy with Launch Bounds-Tuning Occupancy with Launch Boundsroofline model, Roofline Model: Compute-Bound or Memory-Bound Workloads-Roofline Model: Compute-Bound or Memory-Bound WorkloadsUnified Memory, Unified Memory-Unified Memory
- instruction-level parallelismcompiler hints, Loop Unrolling, Interleaving, and Compiler Hintinginterleaving independent operations, Loop Unrolling, Interleaving, and Compiler Hintingunrolling loops, Loop Unrolling, Interleaving, and Compiler Hinting, Kernel Fusion
- kernel fusion, Kernel Fusion-Kernel Fusion
- loop unrolling, Loop Unrolling, Interleaving, and Compiler Hinting, Kernel Fusion
- MMA API, Transformer Engine and TMEM in Depth-Transformer Engine and TMEM in Depth
- multiple GPUs, Multi-GPU Programming-Multi-GPU Programming
- NVIDIA Compute Sanitizer, Debugging Functional Correctness with NVIDIA Compute Sanitizer
- NVLink-C2C CPU to GPU memory access, NUMA Awareness and CPU Pinning
- parallelizing workloads with threadblock clusters, Designing Efficient Algorithms with Thread Block Clusters-Designing Efficient Algorithms with Thread Block Clusters
- PTX, Feeding Tensor Cores with TMEM and TMA, Inline PTX and SASS Tuning for Microoptimizationscode from compiler, CUDA Forward and Backward Compatibility Across GPU Hardware Generations, CUDA GPU Backward and Forward Compatibility Modelcompatibility across GPU generations, CUDA Forward and Backward Compatibility Across GPU Hardware Generations, CUDA GPU Backward and Forward Compatibility Modelinline PTX code, Inline PTX and SASS Tuning for Microoptimizations-Inline PTX and SASS Tuning for Microoptimizationsinline PTX code in DeepSeek DeepEP, DeepSeek’s Use of Inline PTX for Memory Allocation Optimizationstreaming assembler, Inline PTX and SASS Tuning for Microoptimizations-Inline PTX and SASS Tuning for Microoptimizationsstreaming assembler changing across generations, Inline PTX and SASS Tuning for MicrooptimizationsTriton, Triton Programming Model
- Python-centric CUDA libraries, Writing Custom Kernels with OpenAI Triton
- smart compilers, Smart Compilers and Automated Code Optimizations-Smart Compilers and Automated Code Optimizations
- Triton programming model, Triton Programming Model

- asynchronous memory allocation, Asynchronous Memory Allocation and Memory Pools-Asynchronous Memory Allocation and Memory Pools
- CUDA streams, Asynchronous Memory Allocation and Memory Pools
- flow between CPU and GPU, Understanding GPU Architecture
- __global__, CUDA Programming Refresher
- GPU memory hierarchy, Understanding GPU Memory Hierarchy-Understanding GPU Memory Hierarchy
- high occupancy and GPU utilization, Maintaining High Occupancy and GPU Utilization-Maintaining High Occupancy and GPU Utilization
- highest-level and most-recent APIs available, Asynchronous Memory Prefetching and Tensor Memory Accelerator
- kernel errors, CUDA Programming Refresher
- kernel inputs in 1D, 2D and 3D Kernel Inputs
- kernel inputs in 2D and 3D, 2D and 3D Kernel Inputs
- kernels for parallel work, CUDA Programming Refresher-CUDA Programming Refresher
- launch parameters, Configuring Launch Parameters: Blocks per Grid and Threads per Block-Configuring Launch Parameters: Blocks per Grid and Threads per Block
- maximum threadsPerBlock compile time parameter, Tuning Occupancy with Launch Bounds-Tuning Occupancy with Launch Bounds
- memory pools, Asynchronous Memory Allocation and Memory Pools-Asynchronous Memory Allocation and Memory Pools
- occupancy tuning with __launch_bounds__, Tuning Occupancy with Launch Bounds-Tuning Occupancy with Launch Bounds
- roofline model, Roofline Model: Compute-Bound or Memory-Bound Workloads-Roofline Model: Compute-Bound or Memory-Bound Workloads
- Unified Memory, Unified Memory-Unified Memory

- compiler hints, Loop Unrolling, Interleaving, and Compiler Hinting
- interleaving independent operations, Loop Unrolling, Interleaving, and Compiler Hinting
- unrolling loops, Loop Unrolling, Interleaving, and Compiler Hinting, Kernel Fusion

- code from compiler, CUDA Forward and Backward Compatibility Across GPU Hardware Generations, CUDA GPU Backward and Forward Compatibility Model
- compatibility across GPU generations, CUDA Forward and Backward Compatibility Across GPU Hardware Generations, CUDA GPU Backward and Forward Compatibility Model
- inline PTX code, Inline PTX and SASS Tuning for Microoptimizations-Inline PTX and SASS Tuning for Microoptimizations
- inline PTX code in DeepSeek DeepEP, DeepSeek’s Use of Inline PTX for Memory Allocation Optimization
- streaming assembler, Inline PTX and SASS Tuning for Microoptimizations-Inline PTX and SASS Tuning for Microoptimizations
- streaming assembler changing across generations, Inline PTX and SASS Tuning for Microoptimizations
- Triton, Triton Programming Model

- inference monitoring, Monitoring System Metrics and Counters, Monitoring System Metrics and Countersalert manager, Debugging Correctness IssuesData Center GPU Manager metrics, Monitoring System Metrics and Counters
- NVLink/NVSwitch monitoring, Real-Time Link Telemetry and Monitoring

- alert manager, Debugging Correctness Issues
- Data Center GPU Manager metrics, Monitoring System Metrics and Counters

- CTRL approach, Prompt Cleansing

- context parallelism, Context (Sequence) Parallelism
- prompt compression, Prompt Compression
- stall-free scheduling, Stall-Free Scheduling (Chunked Prefill)

- dynamic token pruning, Dynamic Token Pruning with LazyLLM
- TorchAO, PyTorch Architecture Optimization (torchao), Quantization, Sparsity, and Pruning

- code from compiler, CUDA Forward and Backward Compatibility Across GPU Hardware Generations, CUDA GPU Backward and Forward Compatibility Model, TorchInductor Backend Code GenerationCUDA compatibility across GPU generations, CUDA Forward and Backward Compatibility Across GPU Hardware Generations, CUDA GPU Backward and Forward Compatibility ModelJIT compiling, CUDA Forward and Backward Compatibility Across GPU Hardware Generations
- inline PTX code, Inline PTX and SASS Tuning for Microoptimizations-Inline PTX and SASS Tuning for MicrooptimizationsDeepSeek memory allocation optimization, DeepSeek’s Use of Inline PTX for Memory Allocation Optimization
- streaming assembler, Inline PTX and SASS Tuning for Microoptimizations-Inline PTX and SASS Tuning for Microoptimizationschanging per GPU architecture generations, Inline PTX and SASS Tuning for Microoptimizations
- TMEM and CUTLASS, Feeding Tensor Cores with TMEM and TMA
- TorchInductor, TorchInductor Backend Code Generation
- Triton, Triton Programming Model

- CUDA compatibility across GPU generations, CUDA Forward and Backward Compatibility Across GPU Hardware Generations, CUDA GPU Backward and Forward Compatibility Model
- JIT compiling, CUDA Forward and Backward Compatibility Across GPU Hardware Generations

- DeepSeek memory allocation optimization, DeepSeek’s Use of Inline PTX for Memory Allocation Optimization

- changing per GPU architecture generations, Inline PTX and SASS Tuning for Microoptimizations

- conditionals as tensor operations, TorchDynamo for Bytecode Capture and Graph Extraction
- CUDA kernels created, NVIDIA Software Stack
- CUDA libraries, Writing Custom Kernels with OpenAI Triton
- CUDA Toolkit libraries, C++ and Python CUDA Librariesnumpy replacement cuPyNumeric, C++ and Python CUDA Libraries
- frameworks built on CUDA, PyTorch and Higher-Level AI Frameworks(see also PyTorch)
- Nsight Systems backtrace sampling, PyTorch Profiler and Visualization Tools
- Triton language and compiler, C++ and Python CUDA Libraries

- numpy replacement cuPyNumeric, C++ and Python CUDA Libraries

- (see also PyTorch)

- arithmetic intensity, PyTorch and Arithmetic Intensity
- attention mechanisms optimized, PyTorch Optimized Attention Mechanisms
- automatic mixed precision, TF32 and Automatic Mixed Precision (PyTorch)-TF32 and Automatic Mixed Precision (PyTorch)
- backend support, Pitfall #1: Using a CPU-bound Gloo backend instead of NCCLverifying which backend, Pitfall #1: Using a CPU-bound Gloo backend instead of NCCL
- compiler (see PyTorch compiler)
- CUDA Graphs, PyTorch, Inference Engines, and CUDA Graphs, Capturing a CUDA Graph with a CUDA Streamstatic memory pools, Memory Pools for CUDA Graphs
- custom kernel registration, Registering Custom Kernels with PyTorch-Registering Custom Kernels with PyTorch
- DataLoaderCPU affinity for each worker process, NUMA Awareness and CPU Pinning-NUMA Awareness and CPU Pinningdata input pipeline optimization, Optimizing the Data Input Pipeline-Optimizing the Data Input Pipelinedata loading and preprocessing, Efficient Data Loading and Preprocessing-Efficient Data Loading and Preprocessingmultiple workers, Tuning NVMe and Filesystem for Throughput, Efficient Data Loading and Preprocessing, Scaling Out Workers as You Scale Out Number of GPUs, Optimizing the Data Input Pipelinepersistent workers, Efficient Data Loading and Preprocessingpinned memory flag, NUMA-Friendly Memory Allocation and Memory Pinning, Using Streams to Overlap Compute with Data Transfers, Optimizing the Data Input Pipelineprefetch factor, Optimizing the Data Input Pipeline
- Distributed Data Parallel communicationbucketing, Reducing Communication Frequency and Volumedata parallel, Distributed Data Parallel StrategiesData Parallel versus Distributed Data Parallel, Distributed Data Parallel Strategies-Distributed Data Parallel Strategiesdefinition of Data Parallel, Distributed Data Parallel Strategiesdefinition of Distributed Data Parallel, Distributed Data Parallel Strategiesmodel parallel, Distributed Data Parallel Strategies
- DistributedSampler, Fast Storage and Data Locality
- filesystem caching, Filesystem Caching and Write-Back
- GPU idle time measurement, Monitoring Storage I/O
- intra-kernel pipelining, PyTorch, CUDA Pipeline API, and Warp Specialization
- introduction, PyTorch and Higher-Level AI Frameworkscompiler stack, PyTorch and Higher-Level AI Frameworks, PyTorch Compiler (torch.compile), PyTorch Compiler, OpenAI Triton, and XLA BackendsCUDA complexity abstracted away, PyTorch and Higher-Level AI Frameworkstensor operations using GPUs, PyTorch and Higher-Level AI FrameworksTorchInductor backend, PyTorch and Higher-Level AI Frameworks
- memory allocator best-fit with coalescing, Dynamic Memory-Allocation Switching (Slab Versus Caching Versus Stream-Ordered)
- memory allocator plugin, Pluggable Memory Allocators and Cross-GPU Data Transfers-Pluggable Memory Allocators and Cross-GPU Data Transfers
- memory caching allocator, Asynchronous Memory Allocation and Memory Pools
- memory profiler, NVTX Markers and Profiling Tools, NVTX Markers and Profiling Tools
- NCCL collectives running on dedicates streams, Multi-GPU Compute and Data Transfer Overlap with CUDA Streams
- nested tensors, PyTorch and Arithmetic Intensity
- NVLink domain, NVIDIA’s “AI Supercomputer in a Rack”
- persistent kernels and megakernels, Common Workloads for Persistent Kernels
- profiler (see PyTorch profiler)
- sparse format from trained dense models, Structured Sparsity
- streamscreating, Concurrency with CUDA Streamsdefault versus explicit streams, Default Versus Explicit (Nondefault) Streamsmultiple GPUs with multiple streams, Multi-GPU Compute and Data Transfer Overlap with CUDA Streamsoverlapping computation and data transfer, Overlapping Communication and Computationprofiling when adding streams, Overlapping Communication and Computation
- symmetric memory, PyTorch Symmetric Memory

- verifying which backend, Pitfall #1: Using a CPU-bound Gloo backend instead of NCCL

- static memory pools, Memory Pools for CUDA Graphs

- CPU affinity for each worker process, NUMA Awareness and CPU Pinning-NUMA Awareness and CPU Pinning
- data input pipeline optimization, Optimizing the Data Input Pipeline-Optimizing the Data Input Pipeline
- data loading and preprocessing, Efficient Data Loading and Preprocessing-Efficient Data Loading and Preprocessing
- multiple workers, Tuning NVMe and Filesystem for Throughput, Efficient Data Loading and Preprocessing, Scaling Out Workers as You Scale Out Number of GPUs, Optimizing the Data Input Pipeline
- persistent workers, Efficient Data Loading and Preprocessing
- pinned memory flag, NUMA-Friendly Memory Allocation and Memory Pinning, Using Streams to Overlap Compute with Data Transfers, Optimizing the Data Input Pipeline
- prefetch factor, Optimizing the Data Input Pipeline

- bucketing, Reducing Communication Frequency and Volume
- data parallel, Distributed Data Parallel Strategies
- Data Parallel versus Distributed Data Parallel, Distributed Data Parallel Strategies-Distributed Data Parallel Strategies
- definition of Data Parallel, Distributed Data Parallel Strategies
- definition of Distributed Data Parallel, Distributed Data Parallel Strategies
- model parallel, Distributed Data Parallel Strategies

- compiler stack, PyTorch and Higher-Level AI Frameworks, PyTorch Compiler (torch.compile), PyTorch Compiler, OpenAI Triton, and XLA Backends
- CUDA complexity abstracted away, PyTorch and Higher-Level AI Frameworks
- tensor operations using GPUs, PyTorch and Higher-Level AI Frameworks
- TorchInductor backend, PyTorch and Higher-Level AI Frameworks

- creating, Concurrency with CUDA Streams
- default versus explicit streams, Default Versus Explicit (Nondefault) Streams
- multiple GPUs with multiple streams, Multi-GPU Compute and Data Transfer Overlap with CUDA Streams
- overlapping computation and data transfer, Overlapping Communication and Computation
- profiling when adding streams, Overlapping Communication and Computation

- pruning, PyTorch Architecture Optimization (torchao), Quantization, Sparsity, and Pruning
- quantization, PyTorch Architecture Optimization (torchao), Quantization, Sparsity, and PruningINT8 quantization, Transformer Engine and TMEM in Depth
- sparsity, PyTorch Architecture Optimization (torchao), Quantization, Sparsity, and Pruning

- INT8 quantization, Transformer Engine and TMEM in Depth

- about the compiler, PyTorch and Higher-Level AI Frameworks, PyTorch Compiler (torch.compile)
- arithmetic intensity, PyTorch and Arithmetic Intensity
- AsyncTP enabled, TorchTitan, AsyncTP, AutoParallel, and SimpleFSDP
- compilation pipeline, PyTorch Compiler Deep Dive-Debugging Numerical Correctness and Accuracyabout the compilation pipeline, PyTorch Compiler Deep DiveAOT Autograd, AOT Autograd Fusion for Forward and Backward PassesPrimTorch IR, PrimTorch IR (Prims) Simplified Operator SetTorchDynamo, TorchDynamo for Bytecode Capture and Graph Extraction-TorchDynamo for Bytecode Capture and Graph ExtractionTorchInductor, TorchInductor Backend Code Generation
- compiler modes, Compilation Modes and Trade-Offs in Speed, Memory, and Compile Time-Compilation Modes and Trade-Offs in Speed, Memory, and Compile TimeCUDA Graphs triggered, Autotuning with TorchInductoreager mode, Disabling the PyTorch Compiler and Reverting Back to Eager Modemax-autotune mode, PyTorch Compiler Deep Divestatic shapes required for CUDA Graphs, Autotuning with TorchInductor, Dynamic Shapes and Variable Sequence Lengths
- compiler stack, PyTorch and Higher-Level AI Frameworks, PyTorch Compiler (torch.compile), PyTorch Compiler, OpenAI Triton, and XLA Backends
- compiling versus writing custom kernels, Compiling Versus Writing Custom Kernels
- CUDA Graph Trees, CUDA Graph Trees (PyTorch Compiler Internal)
- debugging, Performance Hints and Debugging Generated Code, Debugging Numerical Correctness and Accuracy, Debugging Compiler Phases, Graph Breaks, and Performancenumerical correctness and accuracy, Debugging Numerical Correctness and Accuracy-Debugging Numerical Correctness and AccuracyTORCH_LOGS, Profiling and Debugging Compiler Performance Issues, Performance Hints and Debugging Generated Code
- disabling, Disabling the PyTorch Compiler and Reverting Back to Eager Modeeager mode, Disabling the PyTorch Compiler and Reverting Back to Eager Mode
- dynamic shapes, Dynamic Shapes and Variable Sequence Lengths-Dynamic Shapes and Variable Sequence Lengths
- export, TorchInductor Backend Code Generation
- kernel fusion, Tuning Occupancy with PyTorch, Kernel Fusion, TorchDynamo for Bytecode Capture and Graph Extraction
- launch overhead reduced via CUDA Graphs, PyTorch, Inference Engines, and CUDA Graphs
- Mega-Cache, Using the PyTorch Compiler
- per-thread default stream enabled, Modern Per-Thread Default Stream
- performance hints, Performance Hints and Debugging Generated Code
- pipeline parallel with, Tensor and Pipeline Parallelism with torch.compile
- profiling and debugging compiler performance, Profiling and Debugging Compiler Performance Issues
- quantization with torchao, PyTorch Architecture Optimization (torchao), Quantization, Sparsity, and Pruning
- scaling with DDP, DDP with torch.compile
- scaling with FSDP, FSDP with torch.compile-FSDP with torch.compile
- SimpleFSDP, TorchTitan, AsyncTP, AutoParallel, and SimpleFSDP
- smart compilers, Smart Compilers and Automated Code Optimizations-Smart Compilers and Automated Code Optimizations
- tensor parallel with, Tensor and Pipeline Parallelism with torch.compile
- TorchInductor, Coalesced Versus Uncoalesced Global Memory Access, Using the PyTorch Compiler, Using the PyTorch Compilerprofiling and benchmarking kernels generated, Profiling and Debugging Compiler Performance Issues
- Transformer Engine, Autotuning with TorchInductor
- using, Using the PyTorch Compiler-Using the PyTorch Compiler
- vectorized memory operations, Vectorized Memory Access

- about the compilation pipeline, PyTorch Compiler Deep Dive
- AOT Autograd, AOT Autograd Fusion for Forward and Backward Passes
- PrimTorch IR, PrimTorch IR (Prims) Simplified Operator Set
- TorchDynamo, TorchDynamo for Bytecode Capture and Graph Extraction-TorchDynamo for Bytecode Capture and Graph Extraction
- TorchInductor, TorchInductor Backend Code Generation

- CUDA Graphs triggered, Autotuning with TorchInductor
- eager mode, Disabling the PyTorch Compiler and Reverting Back to Eager Mode
- max-autotune mode, PyTorch Compiler Deep Dive
- static shapes required for CUDA Graphs, Autotuning with TorchInductor, Dynamic Shapes and Variable Sequence Lengths

- numerical correctness and accuracy, Debugging Numerical Correctness and Accuracy-Debugging Numerical Correctness and Accuracy
- TORCH_LOGS, Profiling and Debugging Compiler Performance Issues, Performance Hints and Debugging Generated Code

- eager mode, Disabling the PyTorch Compiler and Reverting Back to Eager Mode

- profiling and benchmarking kernels generated, Profiling and Debugging Compiler Performance Issues

- NCCL profiler plugin, Profiling and Debugging NCCL

- CUDA Performance Tools Interface, PyTorch Profiler and Visualization Tools
- example mixture-of-experts model, Using PyTorch Profiler-Using PyTorch Profiler
- GPU bottlenecks, PyTorch Profiler and Visualization Tools-Profiler-Guided Analysis
- Kineto library, PyTorch Profiler and Visualization Tools, NVTX Markers and Profiling Tools, NVTX Markers and Profiling ToolsNCCL profiler plugin, Profiling and Debugging NCCL
- memory profiler, NVTX Markers and Profiling Tools, NVTX Markers and Profiling Tools
- memory profiling and tuning, Profiling and Tuning Memory in PyTorch-Enabling Peer-to-Peer DMA and UCXactivation checkpointing, Activation Checkpointing for Memory SavingsCUDA memory allocator tuned, Tuning the CUDA Memory AllocatorFSDP automatic checkpointing, FSDP Automatic Checkpointing and Offloading-FSDP Automatic Checkpointing and OffloadingFSDP with tensor parallel and pipeline parallel, Combining FSDP with Tensor Parallel and Pipeline ParallelNCCL with UCX over multinode topologies, Enabling Peer-to-Peer DMA and UCXoffloading parameters, Offloading Parameters to CPU and NVMepeer-to-peer DMA, Enabling Peer-to-Peer DMA and UCXpluggable memory allocators, Pluggable Memory Allocators and Cross-GPU Data Transfers-Pluggable Memory Allocators and Cross-GPU Data Transfers
- NCCL profiler plugin, Profiling and Debugging NCCL
- Nsight Systems PyTorch focused mode, PyTorch Profiler and Visualization Tools

- NCCL profiler plugin, Profiling and Debugging NCCL

- activation checkpointing, Activation Checkpointing for Memory Savings
- CUDA memory allocator tuned, Tuning the CUDA Memory Allocator
- FSDP automatic checkpointing, FSDP Automatic Checkpointing and Offloading-FSDP Automatic Checkpointing and Offloading
- FSDP with tensor parallel and pipeline parallel, Combining FSDP with Tensor Parallel and Pipeline Parallel
- NCCL with UCX over multinode topologies, Enabling Peer-to-Peer DMA and UCX
- offloading parameters, Offloading Parameters to CPU and NVMe
- peer-to-peer DMA, Enabling Peer-to-Peer DMA and UCX
- pluggable memory allocators, Pluggable Memory Allocators and Cross-GPU Data Transfers-Pluggable Memory Allocators and Cross-GPU Data Transfers

### Q

- QoS (quality of service), Quality of Servicedynamic routing policies, Example dynamic routing policy configurationearly rejection policies, QoS and early rejection policies-QoS and early rejection policies
- quad-die Rubin GPU, Rubin Ultra and Vera Rubin Ultra (2027)GPC “partitions”, Thread Block Clusters and Distributed Shared Memory
- quality of service (see QoS (quality of service))
- quantizationdynamic activation quantization, Dynamic Quantization and Activation Range Adjustmentinference optimization, Quantization Approaches for Real-Time Inference-Fusing Quantization-Dequantization Steps into the Execution Graphabout quantization, Quantization Approaches for Real-Time Inferenceactivation quantization, Activation Quantizationcombining weight and activation quantization, Combining Weight and Activation Quantizationpost-training quantization workflow, Post-Training Quantization Workflowquant-dequant steps into execution graph, Fusing Quantization-Dequantization Steps into the Execution Graphreducing precision, Reducing Precision from FP16 to FP8 and FP4weight-only quantization, Weight-Only Quantization (GPTQ, AWQ)KV compression, Real-Time KV Cache Compression and Policy SwitchingTorchAO, PyTorch Architecture Optimization (torchao), Quantization, Sparsity, and Pruning
- Quantum InfiniBand (NVIDIA), Co-Packaged Optics: Future of Networking Hardware

- dynamic routing policies, Example dynamic routing policy configuration
- early rejection policies, QoS and early rejection policies-QoS and early rejection policies

- GPC “partitions”, Thread Block Clusters and Distributed Shared Memory

- dynamic activation quantization, Dynamic Quantization and Activation Range Adjustment
- inference optimization, Quantization Approaches for Real-Time Inference-Fusing Quantization-Dequantization Steps into the Execution Graphabout quantization, Quantization Approaches for Real-Time Inferenceactivation quantization, Activation Quantizationcombining weight and activation quantization, Combining Weight and Activation Quantizationpost-training quantization workflow, Post-Training Quantization Workflowquant-dequant steps into execution graph, Fusing Quantization-Dequantization Steps into the Execution Graphreducing precision, Reducing Precision from FP16 to FP8 and FP4weight-only quantization, Weight-Only Quantization (GPTQ, AWQ)
- KV compression, Real-Time KV Cache Compression and Policy Switching
- TorchAO, PyTorch Architecture Optimization (torchao), Quantization, Sparsity, and Pruning

- about quantization, Quantization Approaches for Real-Time Inference
- activation quantization, Activation Quantization
- combining weight and activation quantization, Combining Weight and Activation Quantization
- post-training quantization workflow, Post-Training Quantization Workflow
- quant-dequant steps into execution graph, Fusing Quantization-Dequantization Steps into the Execution Graph
- reducing precision, Reducing Precision from FP16 to FP8 and FP4
- weight-only quantization, Weight-Only Quantization (GPTQ, AWQ)

### R

- R300 Rubin Ultra GPU, Rubin Ultra and Vera Rubin Ultra (2027)
- RadixAttention (SGLang), Continuous Schedulingprefix caching, Prefix Caching
- random versus sequential reads, Sequential Versus Random Read Patterns-Sequential Versus Random Read Patterns
- RDMA (remote direct memory access), Multi-GPU Programming, High-Speed, Low-Overhead Data Transfers with RDMA-Pitfall #6: GPU memory fragmentation under UCX/RDMADeepSeek Fire-Flyer File System, DeepSeek’s Fire-Flyer File SystemGPUDirect RDMA, Multi-GPU Programming, High-Speed, Low-Overhead Data Transfers with RDMAmultinode, multirack communication, Multinode and Multirack Communication with GPUDirect RDMA-Multinode and Multirack Communication with GPUDirect RDMANCCL’s native allocator in PyTorch MemPool, Pluggable Memory Allocators and Cross-GPU Data Transfersverify true GPUDirect RDMA, High-Speed, Low-Overhead Data Transfers with RDMAInfiniBand support, High-Speed, Low-Overhead Data Transfers with RDMAzero-copy GPU-to-GPU transfer, Zero-Copy GPU-to-GPU Transfer-Zero-Copy GPU-to-GPU Transfer
- RDMA over Converged Ethernet (RoCE), Multi-GPU Programming
- read-only data caches, Read-Only Data Caches-Read-Only Data Cachesconst __restrict__, Read-Only Data Caches, Read-Only Data Caches, Read-Only Data Cachespitfall of not stating read-only, Read-Only Data Caches
- reading dataDeepSeek Fire-Flyer File System, DeepSeek’s Fire-Flyer File System-DeepSeek’s Fire-Flyer File Systemnetwork reads eliminated, Tuning, Replicating, and Compressing Dataread ahead, Tuning NVMe and Filesystem for Throughputsequential versus random pattern, Sequential Versus Random Read Patterns-Sequential Versus Random Read Patternsfilesystem optimized, Sequential Versus Random Read Patternsread size tuned, Sequential Versus Random Read Patterns
- recomputation versus memory trade-off, Recomputation Versus Memory Trade-Off
- reduced precision, BF16/FP16, FP8, and FP4 Reduced Precision
- registersGPUDirect RDMA, Multi-GPU Programmingmemory hierarchy, Streaming Multiprocessor, Threads, and Warpsoccupancy trade-off, Tuning Occupancy with Launch Boundsregister pressure from ILP, Profiling and Mitigating Register Pressureregister spilling, Tuning Occupancy with Launch Bounds
- reinforcement learning (RL)agents tuning AI at runtime, Reinforcement Learning Agents for Tuning AI Systems at RuntimeGPU kernel optimization via, Reinforcement Learning Approach to Generating Optimized GPU Kernels (Predibase)-Reinforcement Learning Approach to Generating Optimized GPU Kernels (Predibase)
- remote direct memory access (see RDMA (remote direct memory access))
- reproducibility, Transparency and Reproducibility, Reproducibility and Documentation Best Practices-Reproducibility and Documentation Best Practices
- request coalescing, Debouncing and Request Coalescing
- requests per second (RPS), Dynamic Batching
- resources onlinebook supplemental material, Using Code Examplesbook web page, How to Contact UsFireworksAI CUDA Graphs post, Replaying the GraphGPU failures at scale per Meta, Pitfall #6: NCCL communicator hangs, errors, or shuts down completelyNCCL environment variables, NCCL Communicator Lifecycle and Environment GotchasPyTorch overlapping computation and data transfer, Overlapping Communication and Computation
- ring communication algorithm, NCCL Communication Algorithms
- RL (see reinforcement learning (RL))
- RoCE (RDMA over Converged Ethernet), Multi-GPU Programmingenabling GPUDirect RDMA, Optimizing Network Communication for Kubernetes
- Roofline analysis (Nsight Compute), Nsight Compute and Roofline Analysisexample MoE model, Kernel Roofline Analysis for General Matrix Multiply (GEMM)-Kernel Roofline Analysis for General Matrix Multiply (GEMM)
- roofline model, Roofline Model: Compute-Bound or Memory-Bound Workloads-Roofline Model: Compute-Bound or Memory-Bound Workloadsroofline-guided scheduling, Roofline-Guided Scheduling and Orchestration Decisions
- Roofline performance model, Increasing CUDA Kernel Efficiency and Arithmetic Intensityarithmetic intensity, Increasing CUDA Kernel Efficiency and Arithmetic Intensity-Increasing CUDA Kernel Efficiency and Arithmetic Intensity, Roofline-Guided Scheduling and Orchestration Decisionslower precision with Tensor Cores, Mixed Precision and Utilizing Tensor Cores
- RPS (requests per second), Dynamic Batching
- Rubin Ultra GPU (R300), Rubin Ultra and Vera Rubin Ultra (2027)
- runtime optimization, Full-Stack Inference Optimizations

- prefix caching, Prefix Caching

- DeepSeek Fire-Flyer File System, DeepSeek’s Fire-Flyer File System
- GPUDirect RDMA, Multi-GPU Programming, High-Speed, Low-Overhead Data Transfers with RDMAmultinode, multirack communication, Multinode and Multirack Communication with GPUDirect RDMA-Multinode and Multirack Communication with GPUDirect RDMANCCL’s native allocator in PyTorch MemPool, Pluggable Memory Allocators and Cross-GPU Data Transfersverify true GPUDirect RDMA, High-Speed, Low-Overhead Data Transfers with RDMA
- InfiniBand support, High-Speed, Low-Overhead Data Transfers with RDMA
- zero-copy GPU-to-GPU transfer, Zero-Copy GPU-to-GPU Transfer-Zero-Copy GPU-to-GPU Transfer

- multinode, multirack communication, Multinode and Multirack Communication with GPUDirect RDMA-Multinode and Multirack Communication with GPUDirect RDMA
- NCCL’s native allocator in PyTorch MemPool, Pluggable Memory Allocators and Cross-GPU Data Transfers
- verify true GPUDirect RDMA, High-Speed, Low-Overhead Data Transfers with RDMA

- const __restrict__, Read-Only Data Caches, Read-Only Data Caches, Read-Only Data Caches
- pitfall of not stating read-only, Read-Only Data Caches

- DeepSeek Fire-Flyer File System, DeepSeek’s Fire-Flyer File System-DeepSeek’s Fire-Flyer File System
- network reads eliminated, Tuning, Replicating, and Compressing Data
- read ahead, Tuning NVMe and Filesystem for Throughput
- sequential versus random pattern, Sequential Versus Random Read Patterns-Sequential Versus Random Read Patternsfilesystem optimized, Sequential Versus Random Read Patternsread size tuned, Sequential Versus Random Read Patterns

- filesystem optimized, Sequential Versus Random Read Patterns
- read size tuned, Sequential Versus Random Read Patterns

- GPUDirect RDMA, Multi-GPU Programming
- memory hierarchy, Streaming Multiprocessor, Threads, and Warps
- occupancy trade-off, Tuning Occupancy with Launch Bounds
- register pressure from ILP, Profiling and Mitigating Register Pressure
- register spilling, Tuning Occupancy with Launch Bounds

- agents tuning AI at runtime, Reinforcement Learning Agents for Tuning AI Systems at Runtime
- GPU kernel optimization via, Reinforcement Learning Approach to Generating Optimized GPU Kernels (Predibase)-Reinforcement Learning Approach to Generating Optimized GPU Kernels (Predibase)

- book supplemental material, Using Code Examples
- book web page, How to Contact Us
- FireworksAI CUDA Graphs post, Replaying the Graph
- GPU failures at scale per Meta, Pitfall #6: NCCL communicator hangs, errors, or shuts down completely
- NCCL environment variables, NCCL Communicator Lifecycle and Environment Gotchas
- PyTorch overlapping computation and data transfer, Overlapping Communication and Computation

- enabling GPUDirect RDMA, Optimizing Network Communication for Kubernetes

- example MoE model, Kernel Roofline Analysis for General Matrix Multiply (GEMM)-Kernel Roofline Analysis for General Matrix Multiply (GEMM)

- roofline-guided scheduling, Roofline-Guided Scheduling and Orchestration Decisions

- arithmetic intensity, Increasing CUDA Kernel Efficiency and Arithmetic Intensity-Increasing CUDA Kernel Efficiency and Arithmetic Intensity, Roofline-Guided Scheduling and Orchestration Decisionslower precision with Tensor Cores, Mixed Precision and Utilizing Tensor Cores

- lower precision with Tensor Cores, Mixed Precision and Utilizing Tensor Cores

### S

- SASS (see streaming assembler (SASS))
- Scalable Hierarchical Aggregation and Reduction Protocol (see SHARP (Scalable Hierarchical Aggregation and Reduction Protocol; NVIDIA))
- scaled dot product attention (SDPA; PyTorch), PyTorch Optimized Attention Mechanisms
- scaling, Scaling Distributed Training and Inferencedisaggregated prefill and decode, Scalability of Disaggregated Prefill and Decodeprefill and decode workers, Continuous Prewarming of CUDA Graphs and Caches Using Time-Series Predictionn-GPU scaling pattern, Pattern for N-GPU ScalingPyTorch distributed libraries, Scaling with PyTorch Distributed-Multi-GPU Profiling with HTADDP with torch.compile, DDP with torch.compileFSDP with torch.compile, FSDP with torch.compile-FSDP with torch.compile
- scheduler and interrupt affinity, Scheduler and Interrupt Affinity
- schedulingbutterfly schedules, Expert Communication Optimizationcluster scheduler, Sharing and Schedulingcongestion- and topology-aware with multiple GPUs, Congestion-Aware and Topology-Aware Scheduling with Multiple GPUs-Coordinating NVSwitch Transfers with Fine-Tuned Schedulingadaptive process-GPU mapping, Adaptive Process-GPU Mapping-Adaptive Process-GPU Mappingdynamic congestion-aware scheduling, Dynamic Congestion-Aware Scheduling-Dynamic Congestion-Aware SchedulingGPUDirect RDMA, Multinode and Multirack Communication with GPUDirect RDMA-Multinode and Multirack Communication with GPUDirect RDMAMoE expert rebalancing and regrouping, MoE Expert Rebalancing and Regroupingmultinode and multirack communication, Multinode and Multirack Communication with GPUDirect RDMANCCL collective communication, Optimizing Collective Communication with NCCL-Wave scheduling of collectivesNVSwitch and fine-tuned scheduling, Coordinating NVSwitch Transfers with Fine-Tuned Scheduling-Coordinating NVSwitch Transfers with Fine-Tuned Schedulingtelemetry and monitoring, Real-Time Link Telemetry and Monitoringcontinuous scheduling, Continuous Schedulingdynamic batching merged with, Continuous Schedulingdynamic scheduling and load balancing, Dynamic Scheduling and Load Balancing-Dynamic resource scalingdynamic scheduling that is congestion-aware, Dynamic Congestion-Aware Scheduling-Dynamic Congestion-Aware Schedulingdynamic scheduling with atomic work queues, Dynamic Scheduling with Atomic Work Queues-Atomic Queuesatomic counters, Atomic Countersatomic queues, Atomic Queues-Atomic Queuesatomic queues allowing dynamic work allocation, Atomic CountersNsight Compute, Atomic Counters-Atomic Countersfirst in, first out (FIFO) scheduler, Latency-Aware Scheduling and Dynamic Routing-Latency-Aware Scheduling and Dynamic Routingin-kernel persistent scheduling, Atomic Queues and Device-Initiated CUDA Graphs for In-Kernel Persistent Schedulingiteration-level scheduling, Continuous BatchingKubernetes job scheduling, Job Scheduling with Kubernetes and SLURMlatency-aware scheduling and dynamic routing, Latency-Aware Scheduling and Dynamic Routing-Latency-Aware Scheduling and Dynamic RoutingLinux schedulerscompletely fair queueing obsolete, Tuning NVMe and Filesystem for ThroughputCompletely Fair Scheduler, Scheduler and Interrupt Affinitymultiqueue block I/O scheduler, Tuning NVMe and Filesystem for Throughputnone scheduler, Tuning NVMe and Filesystem for Throughputsetting for, Tuning NVMe and Filesystem for Throughputroofline-guided scheduling, Roofline-Guided Scheduling and Orchestration Decisionssharing workloads, Sharing and Schedulingstall-free scheduling, Stall-Free Scheduling (Chunked Prefill)thread block scheduling, Thread Block Clusters and Distributed Shared Memorywave scheduling of NCCL collectives, Wave scheduling of collectives
- scratch memory, Scratch Memory
- SDPA (scaled dot product attention; PyTorch), PyTorch Optimized Attention Mechanisms
- sequence parallelism (see context parallelism (CP))
- sequential versus random reads, Sequential Versus Random Read Patterns-Sequential Versus Random Read Patterns
- SFU (see Special Function Unit (SFU))
- SGLangdynamic batching merged with continuous scheduling, Continuous SchedulingRadixAttention, Continuous Schedulingmegakernel decode throughput, Megakernels for InferenceRadixAttentiondynamic batching merged with continuous scheduling, Continuous Schedulingprefix caching, Prefix Caching-Prefix Cachingspeculative decoding, Two-Model, Draft-Based Speculative Decoding and EAGLE
- shardscombining small files into large shards, Sequential Versus Random Read PatternsDeepSeek Fire-Flyer File System sharding metadata, DeepSeek’s Fire-Flyer File Systemsharding data across nodes, Fast Storage and Data Locality
- shared filesystems, Distributed, Parallel Filesystems and Object Stores
- shared memory (SMEM)avoiding shared memory, Warp Shuffle Intrinsics: Avoid Shared Memory and Explicit Synchronization-Warp Shuffle Intrinsics: Avoid Shared Memory and Explicit Synchronizationavoiding shared-memory bank conflicts, Avoid Shared-Memory Bank Conflicts-Avoid Shared-Memory Bank Conflictspadding shared arrays, Avoid Shared-Memory Bank ConflictsBlackwell GPU shared memory, Avoid Shared-Memory Bank Conflictsdata reuse, Tiling and Data Reuse Using Shared Memory-Tiling and Data Reuse Using Shared Memorydistributed shared memory, Tiling with Thread Block ClustersDRAM as, Tiling and Data Reuse Using Shared Memorydynamic shared-memory allocation, Dynamic Shared-Memory Allocation and Occupancy-Aware Kernel Selection-Dynamic Shared-Memory Allocation and Occupancy-Aware Kernel Selectionoccupancy balanced with resource usage, Tuning OccupancyPyTorch, PyTorch and Arithmetic Intensitytiling, Tiling and Data Reuse Using Shared Memory-Tiling and Data Reuse Using Shared Memory, Multilevel Microtiling and Software PrefetchingCUTLASS, Using CUTLASS for Optimal Arithmetic Intensity and Tensor Core PerformanceTriton programming, Accessing Shared Memory in Triton
- __shared__ regions, Distributed Shared Memory
- sharing and scheduling workloads, Sharing and Scheduling
- SHARP (Scalable Hierarchical Aggregation and Reduction Protocol; NVIDIA), In-Network Aggregations with NVIDIA SHARP-In-Network Aggregations with NVIDIA SHARPNCCL and, In-Network SHARP Aggregationoffloading aggregation operations to switch, Multinode and Multirack Communication with GPUDirect RDMA
- __shfl_sync, Warp Shuffle Intrinsics: Avoid Shared Memory and Explicit Synchronization
- shifted all-to-all schedule, Expert Communication Optimization
- silicon photonics, Co-Packaged Optics: Future of Networking Hardware
- Simple Linux Utility for Resource Management (see SLURM (Simple Linux Utility for Resource Management))
- SimpleFSDP, TorchTitan, AsyncTP, AutoParallel, and SimpleFSDP
- SIMT (single instruction, multiple threads) model, Streaming Multiprocessor, Threads, and Warps, Triton Programming Model
- simultaneous multithreading (SMT), NUMA Awareness and CPU Pinning
- single instruction, multiple threads (SIMT) model, Streaming Multiprocessor, Threads, and Warps, Triton Programming Model
- single-kernel tiling, Tiled and Persistent GEMM Kernel (Triton)-Tiled and Persistent GEMM Kernel (Triton)
- single-program multiple-data (SPMD), Triton Programming Model
- Singularity, Reduce Image Size for Faster Container Startup
- sleep state for CPU, CPU Frequency and C-states
- slicing a GPU (see MIG (Multi-Instance GPU; NVIDIA))
- Slinky, Job Scheduling with Kubernetes and SLURM
- SLO-aware request managementearly rejection, Early Rejection (Admission Control)-Early Rejection (Admission Control)fault tolerance, Fault Tolerancequality of service, Quality of Service
- SLURM (Simple Linux Utility for Resource Management), Preintegrated Rack Appliancecluster scheduler, Sharing and Schedulingjob scheduling, Job Scheduling with Kubernetes and SLURM
- SM-resident limits, Choosing Threads-per-Block and Blocks-per-Grid Sizes
- SmoothQuant, Activation Quantization
- SMs (streaming multiprocessors), Streaming Multiprocessor, Threads, and Warps-Streaming Multiprocessor, Threads, and Warpsconcurrent kernels across all SMs maximum, Stream-Ordered Memory AllocatorGPU architecture, Understanding GPU Architecture-Understanding GPU Architecturethreads and registers per SM, Choosing Threads-per-Block and Blocks-per-Grid Sizeswarps, Understanding GPU Architecture-Understanding GPU Architecturehigh occupancy, Threads, Warps, Blocks, and Gridsprogramming CUDA, Maintaining High Occupancy and GPU Utilization-Maintaining High Occupancy and GPU Utilizationminimum thread blocks resident, Tuning Occupancy with Launch Bounds-Tuning Occupancy with Launch BoundsSM-resident limits, Choosing Threads-per-Block and Blocks-per-Grid SizesTensor Cores, NVIDIA GPU Tensor Cores and Transformer Engine-NVIDIA GPU Tensor Cores and Transformer Enginethreads executed in warps, Streaming Multiprocessor, Threads, and Warpslatency hiding, Streaming Multiprocessor, Threads, and Warpswarp schedulers, Warp Scheduling and Dual Issue Instructions
- SMT (simultaneous multithreading), NUMA Awareness and CPU Pinning
- softwarehardware-software codesign, Mechanical Sympathy: Hardware-Software Codesign-Mechanical Sympathy: Hardware-Software CodesignCUDA Pipeline API plus TMA, Asynchronous Memory Prefetching and Tensor Memory AcceleratorDeepSeek Multi-Head Latent Attention, Mechanical Sympathy: Hardware-Software CodesignNVIDIA Kubernetes GPU Operator automating management, Kubernetes for Topology-Aware Container Orchestration and Networking
- software stack, NVIDIA Software Stackabout, PyTorch and Higher-Level AI FrameworksCUDA kernels created in Python, NVIDIA Software StackCUDA Runtime and Toolkit, CUDA Toolkit and RuntimeToolkit C++ and Python libraries, C++ and Python CUDA LibrariesGPU driver, GPU Drivernvidia-smi, GPU DriverPyTorch introduction, PyTorch and Higher-Level AI Frameworks
- sparse modelsmixture-of-experts as, DeepSeek Scales to ~680-Billion Parameter Models Despite US Export Hardware Restrictions in China, Toward 100-Trillion-Parameter Modelsreducing effective compute requirements, DeepSeek Scales to ~680-Billion Parameter Models Despite US Export Hardware Restrictions in China, Toward 100-Trillion-Parameter Models
- sparsityoptimized sparse GEMM kernels, Structured Sparsitypruning, Structured Sparsityarithmetic intensity increased, Structured Sparsitystructured sparsity, Structured Sparsity-Structured Sparsityinference workloads, Structured SparsityTorchAO, PyTorch Architecture Optimization (torchao), Quantization, Sparsity, and Pruning
- SpeCache, Speculative KV Prefetching for Faster TTFT
- Special Function Unit (SFU), Understanding GPU Architecture
- Spectrum-X800 Ethernet, Co-Packaged Optics: Future of Networking Hardware
- speculative decoding, Disaggregated Prefill and Decode Architecture, Speculative Decoding and Parallel Token Generation Techniques-Combining Decoding Techniques and Evaluating Complexitycombining techniques, Combining Decoding Techniques and Evaluating Complexityinterleaving decode steps, Interleaving Decode Steps from Multiple RequestsMedusa multiple heads, Speculative Decoding and Parallel Token Generation Techniquesmultitoken decoding, Multitoken Decoding with Medusa’s Multiple Heads-Multitoken Decoding with Medusa’s Multiple Headssingle-model self-speculative decoding, Single-Model Self-Speculative Decodingtwo-model draft-based, Two-Model, Draft-Based Speculative Decoding and EAGLE-Two-Model, Draft-Based Speculative Decoding and EAGLEEAGLE algorithm, Two-Model, Draft-Based Speculative Decoding and EAGLE
- speculative KV prefetching, Speculative KV Prefetching for Faster TTFT
- speculative MoE expert routing, Speculative MoE Expert Routing and Communication Reduction
- SPMD (single-program multiple-data), Triton Programming Model
- stall-free scheduling, Stall-Free Scheduling (Chunked Prefill)
- static batching, Dynamic Batching
- static indexing replaced with atomic-driven work queue, Atomic Queues
- storage communication, Multirack and Storage Communication
- storage I/O optimizationscloud storage caches, Distributed, Parallel Filesystems and Object Storescommunication versus compute bottlenecks, Diagnosing Communication- Versus Compute-Bound Workloadscompressing data, Tuning, Replicating, and Compressing Datacontinuous profiling and tuning workflow, Continuous Profiling and Tuning Workflow-Continuous Profiling and Tuning Workflowdata pipeline tuning, Tuning the Data Pipelinedata loading and preprocessing, Efficient Data Loading and Preprocessing-Efficient Data Loading and PreprocessingNeMo Curator for training datasets, Creating High-Quality LLM Datasets with NVIDIA NeMo CuratorNVIDIA Data Loading Library, Multimodal Data Processing with NVIDIA DALIscaling out workers as GPUs scaled, Scaling Out Workers as You Scale Out Number of GPUsDeepSeek Fire-Flyer File System, DeepSeek’s Fire-Flyer File System-DeepSeek’s Fire-Flyer File Systemfast storage and data locality, Fast Storage and Data Localitymonitoring I/O, Monitoring Storage I/O-Monitoring Storage I/ONFS servers, Distributed, Parallel Filesystems and Object Storestuning parameters, Distributed, Parallel Filesystems and Object StoresNVIDIA GDS, Using NVIDIA GDS-Measuring GDS with gdsioNVMe and filesystem tuned for throughput, Tuning NVMe and Filesystem for Throughputmultiqueue block I/O scheduler, Tuning NVMe and Filesystem for Throughputobject storage, Distributed, Parallel Filesystems and Object Storessequential versus random reads, Sequential Versus Random Read Patterns-Sequential Versus Random Read Patternsfilesystem optimized, Sequential Versus Random Read Patternsread size tuned, Sequential Versus Random Read Patternsshared filesystems, Distributed, Parallel Filesystems and Object Storesstriping files, Distributed, Parallel Filesystems and Object Storestuning filesystems, Tuning, Replicating, and Compressing Data
- straggler effect MoE models, Expert Parallelism
- stream 0, Legacy Default Stream
- streaming assembler (SASS), Inline PTX and SASS Tuning for Microoptimizations-Inline PTX and SASS Tuning for Microoptimizationschanging per GPU architecture generations, Inline PTX and SASS Tuning for Microoptimizations
- streaming multiprocessors (see SMs (streaming multiprocessors))
- streaming responses, Streaming Responses-Streaming Responses
- streams for asynchronous execution, Asynchronous Execution with Streams(see also CUDA streams)
- strided indexing, Coalesced Versus Uncoalesced Global Memory Access
- striping files, Distributed, Parallel Filesystems and Object Stores
- structured outputs, Constrained Decoding Performance Implications
- structured sparsity, Structured Sparsity-Structured Sparsity
- surface object APIs, Read-Only Data Caches
- swizzling, Avoid Shared-Memory Bank Conflictsthread block swizzling, Thread Block Swizzling
- symmetric memory, PyTorch Symmetric Memory
- SymPy, Dynamic Shapes and Variable Sequence Lengthsabout dynamic shapes, Dynamic Shapes and Variable Sequence Lengths
- __syncthreads(), Cooperative Groups
- systemctl, GPU Persistence Mode
- systems performance engineers (see AI systems performance engineers)
- systems-level optimizations for tuning inference, Systems-Level Optimizationserror handling, Error HandlingGPU utilization versus latency, Maximizing GPU Utilization and Throughput Versus Latency Trade-OffsKV cache, KV Cache Offloading and Memory Pool Allocationmemory, Memoryoverlapping communication and computation, Overlapping Communication and Computation-Overlapping Communication and Computationpower and thermal constraints, Power and Thermal Constraints

- disaggregated prefill and decode, Scalability of Disaggregated Prefill and Decodeprefill and decode workers, Continuous Prewarming of CUDA Graphs and Caches Using Time-Series Prediction
- n-GPU scaling pattern, Pattern for N-GPU Scaling
- PyTorch distributed libraries, Scaling with PyTorch Distributed-Multi-GPU Profiling with HTADDP with torch.compile, DDP with torch.compileFSDP with torch.compile, FSDP with torch.compile-FSDP with torch.compile

- prefill and decode workers, Continuous Prewarming of CUDA Graphs and Caches Using Time-Series Prediction

- DDP with torch.compile, DDP with torch.compile
- FSDP with torch.compile, FSDP with torch.compile-FSDP with torch.compile

- butterfly schedules, Expert Communication Optimization
- cluster scheduler, Sharing and Scheduling
- congestion- and topology-aware with multiple GPUs, Congestion-Aware and Topology-Aware Scheduling with Multiple GPUs-Coordinating NVSwitch Transfers with Fine-Tuned Schedulingadaptive process-GPU mapping, Adaptive Process-GPU Mapping-Adaptive Process-GPU Mappingdynamic congestion-aware scheduling, Dynamic Congestion-Aware Scheduling-Dynamic Congestion-Aware SchedulingGPUDirect RDMA, Multinode and Multirack Communication with GPUDirect RDMA-Multinode and Multirack Communication with GPUDirect RDMAMoE expert rebalancing and regrouping, MoE Expert Rebalancing and Regroupingmultinode and multirack communication, Multinode and Multirack Communication with GPUDirect RDMANCCL collective communication, Optimizing Collective Communication with NCCL-Wave scheduling of collectivesNVSwitch and fine-tuned scheduling, Coordinating NVSwitch Transfers with Fine-Tuned Scheduling-Coordinating NVSwitch Transfers with Fine-Tuned Schedulingtelemetry and monitoring, Real-Time Link Telemetry and Monitoring
- continuous scheduling, Continuous Schedulingdynamic batching merged with, Continuous Scheduling
- dynamic scheduling and load balancing, Dynamic Scheduling and Load Balancing-Dynamic resource scaling
- dynamic scheduling that is congestion-aware, Dynamic Congestion-Aware Scheduling-Dynamic Congestion-Aware Scheduling
- dynamic scheduling with atomic work queues, Dynamic Scheduling with Atomic Work Queues-Atomic Queuesatomic counters, Atomic Countersatomic queues, Atomic Queues-Atomic Queuesatomic queues allowing dynamic work allocation, Atomic CountersNsight Compute, Atomic Counters-Atomic Counters
- first in, first out (FIFO) scheduler, Latency-Aware Scheduling and Dynamic Routing-Latency-Aware Scheduling and Dynamic Routing
- in-kernel persistent scheduling, Atomic Queues and Device-Initiated CUDA Graphs for In-Kernel Persistent Scheduling
- iteration-level scheduling, Continuous Batching
- Kubernetes job scheduling, Job Scheduling with Kubernetes and SLURM
- latency-aware scheduling and dynamic routing, Latency-Aware Scheduling and Dynamic Routing-Latency-Aware Scheduling and Dynamic Routing
- Linux schedulerscompletely fair queueing obsolete, Tuning NVMe and Filesystem for ThroughputCompletely Fair Scheduler, Scheduler and Interrupt Affinitymultiqueue block I/O scheduler, Tuning NVMe and Filesystem for Throughputnone scheduler, Tuning NVMe and Filesystem for Throughputsetting for, Tuning NVMe and Filesystem for Throughput
- roofline-guided scheduling, Roofline-Guided Scheduling and Orchestration Decisions
- sharing workloads, Sharing and Scheduling
- stall-free scheduling, Stall-Free Scheduling (Chunked Prefill)
- thread block scheduling, Thread Block Clusters and Distributed Shared Memory
- wave scheduling of NCCL collectives, Wave scheduling of collectives

- adaptive process-GPU mapping, Adaptive Process-GPU Mapping-Adaptive Process-GPU Mapping
- dynamic congestion-aware scheduling, Dynamic Congestion-Aware Scheduling-Dynamic Congestion-Aware Scheduling
- GPUDirect RDMA, Multinode and Multirack Communication with GPUDirect RDMA-Multinode and Multirack Communication with GPUDirect RDMA
- MoE expert rebalancing and regrouping, MoE Expert Rebalancing and Regrouping
- multinode and multirack communication, Multinode and Multirack Communication with GPUDirect RDMA
- NCCL collective communication, Optimizing Collective Communication with NCCL-Wave scheduling of collectives
- NVSwitch and fine-tuned scheduling, Coordinating NVSwitch Transfers with Fine-Tuned Scheduling-Coordinating NVSwitch Transfers with Fine-Tuned Scheduling
- telemetry and monitoring, Real-Time Link Telemetry and Monitoring

- dynamic batching merged with, Continuous Scheduling

- atomic counters, Atomic Counters
- atomic queues, Atomic Queues-Atomic Queues
- atomic queues allowing dynamic work allocation, Atomic Counters
- Nsight Compute, Atomic Counters-Atomic Counters

- completely fair queueing obsolete, Tuning NVMe and Filesystem for Throughput
- Completely Fair Scheduler, Scheduler and Interrupt Affinity
- multiqueue block I/O scheduler, Tuning NVMe and Filesystem for Throughput
- none scheduler, Tuning NVMe and Filesystem for Throughput
- setting for, Tuning NVMe and Filesystem for Throughput

- dynamic batching merged with continuous scheduling, Continuous SchedulingRadixAttention, Continuous Scheduling
- megakernel decode throughput, Megakernels for Inference
- RadixAttentiondynamic batching merged with continuous scheduling, Continuous Schedulingprefix caching, Prefix Caching-Prefix Caching
- speculative decoding, Two-Model, Draft-Based Speculative Decoding and EAGLE

- RadixAttention, Continuous Scheduling

- dynamic batching merged with continuous scheduling, Continuous Scheduling
- prefix caching, Prefix Caching-Prefix Caching

- combining small files into large shards, Sequential Versus Random Read Patterns
- DeepSeek Fire-Flyer File System sharding metadata, DeepSeek’s Fire-Flyer File System
- sharding data across nodes, Fast Storage and Data Locality

- avoiding shared memory, Warp Shuffle Intrinsics: Avoid Shared Memory and Explicit Synchronization-Warp Shuffle Intrinsics: Avoid Shared Memory and Explicit Synchronization
- avoiding shared-memory bank conflicts, Avoid Shared-Memory Bank Conflicts-Avoid Shared-Memory Bank Conflictspadding shared arrays, Avoid Shared-Memory Bank Conflicts
- Blackwell GPU shared memory, Avoid Shared-Memory Bank Conflicts
- data reuse, Tiling and Data Reuse Using Shared Memory-Tiling and Data Reuse Using Shared Memory
- distributed shared memory, Tiling with Thread Block Clusters
- DRAM as, Tiling and Data Reuse Using Shared Memory
- dynamic shared-memory allocation, Dynamic Shared-Memory Allocation and Occupancy-Aware Kernel Selection-Dynamic Shared-Memory Allocation and Occupancy-Aware Kernel Selection
- occupancy balanced with resource usage, Tuning Occupancy
- PyTorch, PyTorch and Arithmetic Intensity
- tiling, Tiling and Data Reuse Using Shared Memory-Tiling and Data Reuse Using Shared Memory, Multilevel Microtiling and Software PrefetchingCUTLASS, Using CUTLASS for Optimal Arithmetic Intensity and Tensor Core Performance
- Triton programming, Accessing Shared Memory in Triton

- padding shared arrays, Avoid Shared-Memory Bank Conflicts

- CUTLASS, Using CUTLASS for Optimal Arithmetic Intensity and Tensor Core Performance

- NCCL and, In-Network SHARP Aggregation
- offloading aggregation operations to switch, Multinode and Multirack Communication with GPUDirect RDMA

- early rejection, Early Rejection (Admission Control)-Early Rejection (Admission Control)
- fault tolerance, Fault Tolerance
- quality of service, Quality of Service

- cluster scheduler, Sharing and Scheduling
- job scheduling, Job Scheduling with Kubernetes and SLURM

- concurrent kernels across all SMs maximum, Stream-Ordered Memory Allocator
- GPU architecture, Understanding GPU Architecture-Understanding GPU Architecturethreads and registers per SM, Choosing Threads-per-Block and Blocks-per-Grid Sizeswarps, Understanding GPU Architecture-Understanding GPU Architecture
- high occupancy, Threads, Warps, Blocks, and Gridsprogramming CUDA, Maintaining High Occupancy and GPU Utilization-Maintaining High Occupancy and GPU Utilization
- minimum thread blocks resident, Tuning Occupancy with Launch Bounds-Tuning Occupancy with Launch Bounds
- SM-resident limits, Choosing Threads-per-Block and Blocks-per-Grid Sizes
- Tensor Cores, NVIDIA GPU Tensor Cores and Transformer Engine-NVIDIA GPU Tensor Cores and Transformer Engine
- threads executed in warps, Streaming Multiprocessor, Threads, and Warpslatency hiding, Streaming Multiprocessor, Threads, and Warps
- warp schedulers, Warp Scheduling and Dual Issue Instructions

- threads and registers per SM, Choosing Threads-per-Block and Blocks-per-Grid Sizes
- warps, Understanding GPU Architecture-Understanding GPU Architecture

- programming CUDA, Maintaining High Occupancy and GPU Utilization-Maintaining High Occupancy and GPU Utilization

- latency hiding, Streaming Multiprocessor, Threads, and Warps

- hardware-software codesign, Mechanical Sympathy: Hardware-Software Codesign-Mechanical Sympathy: Hardware-Software CodesignCUDA Pipeline API plus TMA, Asynchronous Memory Prefetching and Tensor Memory AcceleratorDeepSeek Multi-Head Latent Attention, Mechanical Sympathy: Hardware-Software Codesign
- NVIDIA Kubernetes GPU Operator automating management, Kubernetes for Topology-Aware Container Orchestration and Networking

- CUDA Pipeline API plus TMA, Asynchronous Memory Prefetching and Tensor Memory Accelerator
- DeepSeek Multi-Head Latent Attention, Mechanical Sympathy: Hardware-Software Codesign

- about, PyTorch and Higher-Level AI Frameworks
- CUDA kernels created in Python, NVIDIA Software Stack
- CUDA Runtime and Toolkit, CUDA Toolkit and RuntimeToolkit C++ and Python libraries, C++ and Python CUDA Libraries
- GPU driver, GPU Drivernvidia-smi, GPU Driver
- PyTorch introduction, PyTorch and Higher-Level AI Frameworks

- Toolkit C++ and Python libraries, C++ and Python CUDA Libraries

- nvidia-smi, GPU Driver

- mixture-of-experts as, DeepSeek Scales to ~680-Billion Parameter Models Despite US Export Hardware Restrictions in China, Toward 100-Trillion-Parameter Models
- reducing effective compute requirements, DeepSeek Scales to ~680-Billion Parameter Models Despite US Export Hardware Restrictions in China, Toward 100-Trillion-Parameter Models

- optimized sparse GEMM kernels, Structured Sparsity
- pruning, Structured Sparsityarithmetic intensity increased, Structured Sparsity
- structured sparsity, Structured Sparsity-Structured Sparsityinference workloads, Structured Sparsity
- TorchAO, PyTorch Architecture Optimization (torchao), Quantization, Sparsity, and Pruning

- arithmetic intensity increased, Structured Sparsity

- inference workloads, Structured Sparsity

- combining techniques, Combining Decoding Techniques and Evaluating Complexity
- interleaving decode steps, Interleaving Decode Steps from Multiple Requests
- Medusa multiple heads, Speculative Decoding and Parallel Token Generation Techniquesmultitoken decoding, Multitoken Decoding with Medusa’s Multiple Heads-Multitoken Decoding with Medusa’s Multiple Heads
- single-model self-speculative decoding, Single-Model Self-Speculative Decoding
- two-model draft-based, Two-Model, Draft-Based Speculative Decoding and EAGLE-Two-Model, Draft-Based Speculative Decoding and EAGLEEAGLE algorithm, Two-Model, Draft-Based Speculative Decoding and EAGLE

- multitoken decoding, Multitoken Decoding with Medusa’s Multiple Heads-Multitoken Decoding with Medusa’s Multiple Heads

- EAGLE algorithm, Two-Model, Draft-Based Speculative Decoding and EAGLE

- cloud storage caches, Distributed, Parallel Filesystems and Object Stores
- communication versus compute bottlenecks, Diagnosing Communication- Versus Compute-Bound Workloads
- compressing data, Tuning, Replicating, and Compressing Data
- continuous profiling and tuning workflow, Continuous Profiling and Tuning Workflow-Continuous Profiling and Tuning Workflow
- data pipeline tuning, Tuning the Data Pipelinedata loading and preprocessing, Efficient Data Loading and Preprocessing-Efficient Data Loading and PreprocessingNeMo Curator for training datasets, Creating High-Quality LLM Datasets with NVIDIA NeMo CuratorNVIDIA Data Loading Library, Multimodal Data Processing with NVIDIA DALIscaling out workers as GPUs scaled, Scaling Out Workers as You Scale Out Number of GPUs
- DeepSeek Fire-Flyer File System, DeepSeek’s Fire-Flyer File System-DeepSeek’s Fire-Flyer File System
- fast storage and data locality, Fast Storage and Data Locality
- monitoring I/O, Monitoring Storage I/O-Monitoring Storage I/O
- NFS servers, Distributed, Parallel Filesystems and Object Storestuning parameters, Distributed, Parallel Filesystems and Object Stores
- NVIDIA GDS, Using NVIDIA GDS-Measuring GDS with gdsio
- NVMe and filesystem tuned for throughput, Tuning NVMe and Filesystem for Throughputmultiqueue block I/O scheduler, Tuning NVMe and Filesystem for Throughput
- object storage, Distributed, Parallel Filesystems and Object Stores
- sequential versus random reads, Sequential Versus Random Read Patterns-Sequential Versus Random Read Patternsfilesystem optimized, Sequential Versus Random Read Patternsread size tuned, Sequential Versus Random Read Patterns
- shared filesystems, Distributed, Parallel Filesystems and Object Stores
- striping files, Distributed, Parallel Filesystems and Object Stores
- tuning filesystems, Tuning, Replicating, and Compressing Data

- data loading and preprocessing, Efficient Data Loading and Preprocessing-Efficient Data Loading and Preprocessing
- NeMo Curator for training datasets, Creating High-Quality LLM Datasets with NVIDIA NeMo Curator
- NVIDIA Data Loading Library, Multimodal Data Processing with NVIDIA DALI
- scaling out workers as GPUs scaled, Scaling Out Workers as You Scale Out Number of GPUs

- tuning parameters, Distributed, Parallel Filesystems and Object Stores

- multiqueue block I/O scheduler, Tuning NVMe and Filesystem for Throughput

- filesystem optimized, Sequential Versus Random Read Patterns
- read size tuned, Sequential Versus Random Read Patterns

- changing per GPU architecture generations, Inline PTX and SASS Tuning for Microoptimizations

- (see also CUDA streams)

- thread block swizzling, Thread Block Swizzling

- about dynamic shapes, Dynamic Shapes and Variable Sequence Lengths

- error handling, Error Handling
- GPU utilization versus latency, Maximizing GPU Utilization and Throughput Versus Latency Trade-Offs
- KV cache, KV Cache Offloading and Memory Pool Allocation
- memory, Memory
- overlapping communication and computation, Overlapping Communication and Computation-Overlapping Communication and Computation
- power and thermal constraints, Power and Thermal Constraints

### T

- tcmalloc, Tune Host CPU Memory Allocator
- TCMALLOC_MAX_TOTAL_THREAD_CACHE_BYTES, Tune Host CPU Memory Allocator
- TCMALLOC_RELEASE_RATE, Tune Host CPU Memory Allocator
- TDP (thermal design power), GPU Clock Speeds and ECC
- temporal load balancing, Dynamic Congestion-Aware Scheduling
- Tensor Cores, NVIDIA GPU Tensor Cores and Transformer Engine-NVIDIA GPU Tensor Cores and Transformer EngineCUTLASS, Using CUTLASS for Optimal Arithmetic Intensity and Tensor Core Performance-Using CUTLASS for Optimal Arithmetic Intensity and Tensor Core Performancemixed precision, Mixed Precision and Utilizing Tensor Cores-Transformer Engine and TMEM in DepthMMA API, Transformer Engine and TMEM in Depth-Transformer Engine and TMEM in DepthTF32, TF32 and Automatic Mixed Precision (PyTorch)enabling, TF32 and Automatic Mixed Precision (PyTorch)theoretical throughput of NVL72 rack, NVIDIA GPU Tensor Cores and Transformer EngineTMEM, Feeding Tensor Cores with TMEM and TMA-Feeding Tensor Cores with TMEM and TMATransformer Engine, Transformer Engine and TMEM in Depthvery high active utilization, Execution Unit Contention
- Tensor Memory Accelerator (TMA), Feeding Tensor Cores with TMEM and TMAcapabilities, Asynchronous Memory Prefetching and Tensor Memory Acceleratorfetching tile from global into shared memory, Asynchronous Memory Prefetching and Tensor Memory Accelerator-Asynchronous Memory Prefetching and Tensor Memory Acceleratorglobal memory traffic reduced, Reducing Global Memory Traffic with Thread Block Clusters-Reducing Global Memory Traffic with Thread Block Clustershardware-software codesign, Asynchronous Memory Prefetching and Tensor Memory Acceleratorstarting a TMA transfer, Asynchronous Memory Prefetching and Tensor Memory Accelerator
- tensor parallel (TP), Parallelism Strategies for Serving Massive MoE Models, Tensor ParallelismAsyncTP, TorchTitan, AsyncTP, AutoParallel, and SimpleFSDPFSDP with, Combining FSDP with Tensor Parallel and Pipeline Parallelhybrid parallelism, Hybrid Parallelismpipeline parallel versus, Pipeline ParallelismPyTorch compiler with, Tensor and Pipeline Parallelism with torch.compileversus PP versus hybrid, Adaptive Parallelism Strategies (TP Versus PP Versus Hybrid)-Adaptive Parallelism Strategies (TP Versus PP Versus Hybrid)
- TensorBoard trace viewer, Multi-GPU Profiling with HTA
- TensorRT-LLM (NVIDIA), PyTorch, Inference Engines, and CUDA Graphs
- TetriInfer’s two-level scheduler, TetriInfer’s two-level scheduler
- texture object APIs, Read-Only Data Caches
- TF32, TF32 and Automatic Mixed Precision (PyTorch)enabling, TF32 and Automatic Mixed Precision (PyTorch)
- TFLOPS (teraFLOPS), Tiling and Data Reuse Using Shared Memory
- thermal constraints on inference, Power and Thermal Constraints
- thermal design power (TDP), GPU Clock Speeds and ECC
- Thompson, Martin, Mechanical Sympathy: Hardware-Software Codesign
- THP (transparent hugepages), Transparent Hugepages
- thread block clusters, Threads, Warps, Blocks, and Grids, Intra-Kernel Pipelining, Warp Specialization, and Cooperative Thread Block Clusterscooperative groups contrasted with, Thread Block Clusters and Distributed Shared Memorydescribed, Thread Block Clusters and Distributed Shared Memorydistributed shared memory, Thread Block Clusters and Distributed Shared Memory-Thread Block Pairalgorithms for parallelizing workloads, Designing Efficient Algorithms with Thread Block Clusters-Designing Efficient Algorithms with Thread Block Clusterscoordinating thread block clusters, Coordinating Thread Block Clusters with Cooperative Groups API-Coordinating Thread Block Clusters with Cooperative Groups APIglobal memory traffic reduced, Reducing Global Memory Traffic with Thread Block Clusters-Reducing Global Memory Traffic with Thread Block Clusterslaunching a thread block cluster, Launching a Thread Block Cluster, Designing Efficient Algorithms with Thread Block Clustersscratch memory, Scratch Memorysharing state between thread blocks, Distributed Shared Memorythread block clusters described, Intra-Kernel Pipelining, Warp Specialization, and Cooperative Thread Block Clustersthread block pair, Thread Block Pair-Thread Block Pairthread block swizzling, Thread Block Swizzlingthreads accessing each other’s shared memory, Threads, Warps, Blocks, and Grids, Intra-Kernel Pipelining, Warp Specialization, and Cooperative Thread Block Clusters, Thread Block Clusters and Distributed Shared Memorythreads using thread block clusters and DSMEM, Thread Block Clusters and Distributed Shared Memorywarp specialization with thread block clusters, Warp Specialization with Thread Block Clusters-Warp Specialization with Thread Block Clustersintra-kernel pipelining with inter-kernel overlap, Combining PDL and Thread Block Clusters with Warp Specialization-Combining PDL and Thread Block Clusters with Warp Specializationlaunching a thread block cluster, Launching a Thread Block Cluster, Designing Efficient Algorithms with Thread Block Clusterssharing state between thread blocks, Distributed Shared Memorythread block pairs, Launching a Thread Block Cluster, Thread Block Pair-Thread Block Pairtiling, Tiling with Thread Block Clusters-Tiling with Thread Block Clustersup to 16 thread blocks, Tiling with Thread Block Clusters
- thread blocks, Threads, Warps, Blocks, and Grids-Threads, Warps, Blocks, and Gridscooperative mode, Cooperative Groupscoscheduled on same GPC, Thread Block Clusters and Distributed Shared Memorydistributed shared memory, Thread Block Clusters and Distributed Shared Memory-Thread Block Pairsharing state between thread blocks, Distributed Shared Memorythread block clusters described, Intra-Kernel Pipelining, Warp Specialization, and Cooperative Thread Block Clustersthreads accessing each other’s shared memory, Threads, Warps, Blocks, and Grids, Thread Block Clusters and Distributed Shared Memorythreads using thread block clusters and DSMEM, Thread Block Clusters and Distributed Shared Memorydistributed shared memory andthreads accessing each other’s shared memory, Intra-Kernel Pipelining, Warp Specialization, and Cooperative Thread Block Clustersprogramming CUDAblocksPerGrid, CUDA Programming Refresher, CUDA Programming RefresherblocksPerGrid as launch parameter, Configuring Launch Parameters: Blocks per Grid and Threads per Block-Configuring Launch Parameters: Blocks per Grid and Threads per Blockminimum thread blocks resident on each SM, Tuning Occupancy with Launch Bounds-Tuning Occupancy with Launch BoundsthreadsPerBlock, CUDA Programming Refresher, CUDA Programming RefresherthreadsPerBlock as launch parameter, Configuring Launch Parameters: Blocks per Grid and Threads per Block-Configuring Launch Parameters: Blocks per Grid and Threads per Blocksharing state between thread blocks, Distributed Shared Memorythread block pairs, Launching a Thread Block Cluster, Thread Block Pair-Thread Block Pair(see also thread block clusters)thread block size, Choosing Threads-per-Block and Blocks-per-Grid Sizestiling, Tiling and Data Reuse Using Shared Memory
- threadscoalesced versus uncoalesced global memory access, Coalesced Versus Uncoalesced Global Memory Access-Coalesced Versus Uncoalesced Global Memory Accesscooperative groups, Cooperative Groups-When to Combine Persistent Kernels and Cooperative Groupsdistributed shared memory, Thread Block Clusters and Distributed Shared MemoryGPU architecture, Threads, Warps, Blocks, and Grids-Threads, Warps, Blocks, and Gridsintrawarp communication, Warp Shuffle Intrinsics: Avoid Shared Memory and Explicit Synchronization-Warp Shuffle Intrinsics: Avoid Shared Memory and Explicit Synchronizationprogramming CUDAchevron syntax (<<< >>>), CUDA Programming Refreshermaximum threadsPerBlock compile time parameter, Tuning Occupancy with Launch Bounds-Tuning Occupancy with Launch BoundsthreadsPerBlock, CUDA Programming Refresher, CUDA Programming RefresherthreadsPerBlock as launch parameter, Configuring Launch Parameters: Blocks per Grid and Threads per Block-Configuring Launch Parameters: Blocks per Grid and Threads per Blocktiling, Tiling and Data Reuse Using Shared Memorywarps, Streaming Multiprocessor, Threads, and Warps, Understanding GPU Architecture-Understanding GPU Architecturewarp divergence, Threads, Warps, Blocks, and Grids
- threadsPerBlock, CUDA Programming Refresher, CUDA Programming Refresherlaunch parameter, Configuring Launch Parameters: Blocks per Grid and Threads per Block-Configuring Launch Parameters: Blocks per Grid and Threads per Block
- throughput metricsdisaggregation of prefill and decode, Impact on Latency (TTFT) and Throughput (TPOT)FP4, FP8, FP16, NVIDIA GPU Tensor Cores and Transformer Engine“goodput”, Measuring “Goodput” Useful ThroughputNVL72 rack Tensor Core throughput, NVIDIA GPU Tensor Cores and Transformer EngineNVMe and filesystem tuned for throughput, Tuning NVMe and Filesystem for Throughput
- ThunderMLA (Stanford), ThunderMLA (Stanford)
- TileIR, Tiling and Data Reuse Using Shared Memory
- tilingarithmetic intensity increased, Tiling and Data Reuse Using Shared Memory, Multilevel Microtiling and Software Prefetchingmultilevel tiling, Multilevel Microtiling and Software Prefetchingdouble-buffered cooperative tiling, Cooperative Tiling and Double-Buffering with the CUDA Pipeline API-Cooperative Tiling and Double-Buffering with the CUDA Pipeline APImultilevel tiling, Multilevel Microtiling and Software PrefetchingPyTorch, PyTorch and Arithmetic Intensityshared memory for tiling, Tiling and Data Reuse Using Shared Memory-Tiling and Data Reuse Using Shared Memory, Multilevel Microtiling and Software PrefetchingCUTLASS, Using CUTLASS for Optimal Arithmetic Intensity and Tensor Core Performancesingle-kernel tiling, Tiled and Persistent GEMM Kernel (Triton)-Tiled and Persistent GEMM Kernel (Triton)software prefetching, Multilevel Microtiling and Software Prefetchingthread block clusters, Tiling with Thread Block Clusters-Tiling with Thread Block Clusterstiles, Tiling and Data Reuse Using Shared MemoryTMA fetching from global into shared memory, Asynchronous Memory Prefetching and Tensor Memory Accelerator-Asynchronous Memory Prefetching and Tensor Memory Accelerator
- time per output token (see TPOT (time per output token))
- time to first token (see TTFT (time to first token))
- TLB (see translation lookaside buffer (TLB))
- TMA (see Tensor Memory Accelerator (TMA))
- TMEM, Feeding Tensor Cores with TMEM and TMA-Feeding Tensor Cores with TMEM and TMAdouble buffering with CUTLASS, Using CUTLASS for Optimal Arithmetic Intensity and Tensor Core PerformanceTransformer Engine, Transformer Engine and TMEM in Depth
- toolkit (see CUDA Toolkit)
- torch.compile (see PyTorch compiler (torch.compile))
- torch.profiler (see PyTorch profiler)
- TorchAO, PyTorch Architecture Optimization (torchao), Quantization, Sparsity, and Pruningpruning, PyTorch Architecture Optimization (torchao), Quantization, Sparsity, and Pruningquantization, PyTorch Architecture Optimization (torchao), Quantization, Sparsity, and PruningINT8 quantization, Transformer Engine and TMEM in Depthsparsity, PyTorch Architecture Optimization (torchao), Quantization, Sparsity, and Pruning
- TorchBench, Continuous Integration and Performance Benchmarking
- TorchDynamo (PyTorch), Using the PyTorch Compiler, TorchDynamo for Bytecode Capture and Graph Extraction-TorchDynamo for Bytecode Capture and Graph Extractiondebugging numerical accuracy, Debugging Numerical Correctness and Accuracydynamic shapes, Profiling and Debugging Compiler Performance Issues, TorchDynamo for Bytecode Capture and Graph Extractionexplain() output, Profiling and Debugging Compiler Performance Issues, Graph Breaks and TorchDynamo explain()all-reduce-related warnings, Tensor and Pipeline Parallelism with torch.compilegraph breaks, Profiling and Debugging Compiler Performance Issues, FSDP with torch.compilegraph breaks debugged, Graph Breaks and TorchDynamo explain()FX Graph output, TorchDynamo for Bytecode Capture and Graph Extractiondumping graph after each stage, Debugging Compiler Phases, Graph Breaks, and Performancegraph breaks, TorchDynamo for Bytecode Capture and Graph Extractioncompiler conditionals as tensor operations, TorchDynamo for Bytecode Capture and Graph Extractionforcing error to be raised at compile, TorchDynamo for Bytecode Capture and Graph Extractionminifier tool, Debugging Numerical Correctness and Accuracyprofiling and debugging compiler issues, Profiling and Debugging Compiler Performance IssuesPyTorch compiler pipeline, PyTorch Compiler Deep DivePyTorch compiler stack, PyTorch and Higher-Level AI Frameworks, PyTorch Compiler (torch.compile), PyTorch Compiler, OpenAI Triton, and XLA Backendsstances for errors or fallbacks, TorchDynamo for Bytecode Capture and Graph ExtractionTORCH_LOGS, Profiling and Debugging Compiler Performance Issues
- TORCHDYNAMO_REPRO_*, Debugging Compiler Phases, Graph Breaks, and Performance
- TorchEval (PyTorch), NVTX Markers and Profiling Tools, NVTX Markers and Profiling Tools, Optimizing the Data Input Pipeline
- TorchInductor (PyTorch), TorchInductor Backend Code Generationarithmetic intensity increased, Increasing CUDA Kernel Efficiency and Arithmetic Intensityautomatic mixed precision, TF32 and Automatic Mixed Precision (PyTorch)-TF32 and Automatic Mixed Precision (PyTorch)autotuning, Using the PyTorch Compiler, Autotuning with TorchInductor-Autotuning with TorchInductormax-autotune compile mode, Using the PyTorch Compiler, PyTorch Compiler Deep Dive, Autotuning with TorchInductorCUDA Graphs, Autotuning with TorchInductordebug mode, Performance Hints and Debugging Generated Code, Debugging Compiler Phases, Graph Breaks, and Performanceinspecting TorchInductor code, Performance Hints and Debugging Generated CodeFX Graph, AOT Autograd Fusion for Forward and Backward Passesintroduction, PyTorch and Higher-Level AI FrameworksIR, TorchInductor Backend Code Generationkernel fusion, Tuning Occupancy with PyTorch, Kernel Fusion, TorchDynamo for Bytecode Capture and Graph Extractionfused attention kernels, PyTorch, CUDA Pipeline API, and Warp SpecializationGEMM kernel prologue and epilogue, Using the PyTorch Compilerminifier tool, Debugging Numerical Correctness and Accuracyoptimizations versus writing custom kernels, Compiling Versus Writing Custom KernelsPrimTorch IR simplifying operations, PrimTorch IR (Prims) Simplified Operator Setprofiling and benchmarking kernels generated, Profiling and Debugging Compiler Performance IssuesPyTorch compiler pipeline, PyTorch Compiler Deep DivePyTorch compiler stack, PyTorch and Higher-Level AI Frameworks, PyTorch Compiler (torch.compile), PyTorch Compiler, OpenAI Triton, and XLA BackendsSimpleFSDP, TorchTitan, AsyncTP, AutoParallel, and SimpleFSDPsymbolic shapes, TorchInductor Backend Code GenerationTriton based, PyTorch and Higher-Level AI Frameworks, Using the PyTorch Compiler, TorchInductor Backend Code Generation, Writing Custom Kernels with OpenAI Triton
- TORCHINDUCTOR_BENCHMARK_KERNEL, Profiling and Debugging Compiler Performance Issues
- TORCHINDUCTOR_UNIQUE_KERNEL_NAMES, Profiling and Debugging Compiler Performance Issues
- TorchTitan (PyTorch), TorchTitan, AsyncTP, AutoParallel, and SimpleFSDPAsyncTP, TorchTitan, AsyncTP, AutoParallel, and SimpleFSDPSimpleFSDP reducing memory usage, TorchTitan, AsyncTP, AutoParallel, and SimpleFSDP
- TORCH_COMPILE_DEBUG, Debugging Compiler Phases, Graph Breaks, and Performance
- TORCH_LOGS, Profiling and Debugging Compiler Performance Issuescompiler debugging, Debugging Compiler Phases, Graph Breaks, and Performancelogging options summary, Debugging Compiler Phases, Graph Breaks, and Performanceperformance hints, Performance Hints and Debugging Generated Code
- TP (see tensor parallel (TP))
- TPOT (time per output token)disaggregation of prefill and decode, Impact on Latency (TTFT) and Throughput (TPOT)prefill and decode phases, Scaling Disaggregated Prefill and Decode for Inference
- training100-trillion parameter models, Toward 100-Trillion-Parameter Modelsloading training data, Toward 100-Trillion-Parameter Modelscold start strategy, DeepSeek Scales to ~680-Billion Parameter Models Despite US Export Hardware Restrictions in ChinacostsDeepSeek-R1, Introduction and AI System OverviewDeepSeek-V3, DeepSeek Scales to ~680-Billion Parameter Models Despite US Export Hardware Restrictions in ChinaGoogle Gemini Ultra, Introduction and AI System OverviewOpenAI GPT-4, Introduction and AI System Overviewdatasets via NVIDIA NeMo Curator, Creating High-Quality LLM Datasets with NVIDIA NeMo Curator
- Transformer Engine (TE; NVIDIA), NVIDIA GPU Tensor Cores and Transformer Engine-NVIDIA GPU Tensor Cores and Transformer Enginebalancing accuracy and precision, NVIDIA GPU Tensor Cores and Transformer Engineinstalling only if calling TE modules directly, Autotuning with TorchInductormechanical sympathy, Mechanical Sympathy: Hardware-Software Codesignmodern GPUs using, Mechanical Sympathy: Hardware-Software CodesignPyTorch compiler, Autotuning with TorchInductorTMEM, Transformer Engine and TMEM in Depth
- translation lookaside buffer (TLB), Transparent Hugepages
- transparencyAI systems performance engineers, Transparency and ReproducibilityDeepSeek Open-Source Week, Transparency and Reproducibility
- transparent hugepages (THP), Transparent Hugepages
- tree communication algorithm, NCCL Communication Algorithms
- tree parallelism communication algorithm, NCCL Communication Algorithms
- Triton (OpenAI)advanced kernel implementations, Advanced Triton Kernel Implementations-Software Pipelining and Double Buffering with Tritondouble-buffering pipelining, Software Pipelining and Double Buffering with Triton-Software Pipelining and Double Buffering with Tritonpersistent matmul kernel, Tiled and Persistent GEMM Kernel (Triton)-Tiled and Persistent GEMM Kernel (Triton)warp specialization, Warp Specialization with TritonCUDA kernels in Python, NVIDIA Software Stackcustom kernels, Writing Custom Kernels with OpenAI Triton-Profiling with Triton Proton Profilerlanguage and compiler, C++ and Python CUDA Librariesopen source, C++ and Python CUDA Librariesprogrammingaccessing shared memory, Accessing Shared Memory in Tritonautotuning Triton kernels, Autotuning Triton Kernelscustom kernels, Writing Custom Kernels with OpenAI Triton-Profiling with Triton Proton ProfilerNsight Compute and Systems support, Profiling with Triton Proton Profilerprofiling with Triton Proton Profiler, Profiling with Triton Proton Profilerprogramming model, Triton Programming Modelregistering custom kernels with PyTorch, TorchInductor Backend Code Generation, Registering Custom Kernels with PyTorch-Registering Custom Kernels with PyTorchtuning kernel launch parameters, Tuning Kernel-Launch ParametersProton Profiler, Profiling with Triton Proton Profilersmart compilers, Smart Compilers and Automated Code Optimizations-Smart Compilers and Automated Code OptimizationsTorchInductor basis, PyTorch and Higher-Level AI Frameworks, Using the PyTorch Compiler, TorchInductor Backend Code Generation, Writing Custom Kernels with OpenAI Triton
- troubleshooting inference, Inference Troubleshooting Recipes
- TTFT (time to first token)constrained outputs, Constrained Decoding Performance Implicationscontext parallelism, Context (Sequence) Parallelismdisaggregation of prefill and decode, Impact on Latency (TTFT) and Throughput (TPOT)prefill and decode phases, Scaling Disaggregated Prefill and Decode for Inferencespeculative KV prefetching, Speculative KV Prefetching for Faster TTFT-Speculative KV Prefetching for Faster TTFT

- CUTLASS, Using CUTLASS for Optimal Arithmetic Intensity and Tensor Core Performance-Using CUTLASS for Optimal Arithmetic Intensity and Tensor Core Performance
- mixed precision, Mixed Precision and Utilizing Tensor Cores-Transformer Engine and TMEM in Depth
- MMA API, Transformer Engine and TMEM in Depth-Transformer Engine and TMEM in Depth
- TF32, TF32 and Automatic Mixed Precision (PyTorch)enabling, TF32 and Automatic Mixed Precision (PyTorch)
- theoretical throughput of NVL72 rack, NVIDIA GPU Tensor Cores and Transformer Engine
- TMEM, Feeding Tensor Cores with TMEM and TMA-Feeding Tensor Cores with TMEM and TMA
- Transformer Engine, Transformer Engine and TMEM in Depth
- very high active utilization, Execution Unit Contention

- enabling, TF32 and Automatic Mixed Precision (PyTorch)

- capabilities, Asynchronous Memory Prefetching and Tensor Memory Accelerator
- fetching tile from global into shared memory, Asynchronous Memory Prefetching and Tensor Memory Accelerator-Asynchronous Memory Prefetching and Tensor Memory Accelerator
- global memory traffic reduced, Reducing Global Memory Traffic with Thread Block Clusters-Reducing Global Memory Traffic with Thread Block Clusters
- hardware-software codesign, Asynchronous Memory Prefetching and Tensor Memory Accelerator
- starting a TMA transfer, Asynchronous Memory Prefetching and Tensor Memory Accelerator

- AsyncTP, TorchTitan, AsyncTP, AutoParallel, and SimpleFSDP
- FSDP with, Combining FSDP with Tensor Parallel and Pipeline Parallel
- hybrid parallelism, Hybrid Parallelism
- pipeline parallel versus, Pipeline Parallelism
- PyTorch compiler with, Tensor and Pipeline Parallelism with torch.compile
- versus PP versus hybrid, Adaptive Parallelism Strategies (TP Versus PP Versus Hybrid)-Adaptive Parallelism Strategies (TP Versus PP Versus Hybrid)

- enabling, TF32 and Automatic Mixed Precision (PyTorch)

- cooperative groups contrasted with, Thread Block Clusters and Distributed Shared Memory
- described, Thread Block Clusters and Distributed Shared Memory
- distributed shared memory, Thread Block Clusters and Distributed Shared Memory-Thread Block Pairalgorithms for parallelizing workloads, Designing Efficient Algorithms with Thread Block Clusters-Designing Efficient Algorithms with Thread Block Clusterscoordinating thread block clusters, Coordinating Thread Block Clusters with Cooperative Groups API-Coordinating Thread Block Clusters with Cooperative Groups APIglobal memory traffic reduced, Reducing Global Memory Traffic with Thread Block Clusters-Reducing Global Memory Traffic with Thread Block Clusterslaunching a thread block cluster, Launching a Thread Block Cluster, Designing Efficient Algorithms with Thread Block Clustersscratch memory, Scratch Memorysharing state between thread blocks, Distributed Shared Memorythread block clusters described, Intra-Kernel Pipelining, Warp Specialization, and Cooperative Thread Block Clustersthread block pair, Thread Block Pair-Thread Block Pairthread block swizzling, Thread Block Swizzlingthreads accessing each other’s shared memory, Threads, Warps, Blocks, and Grids, Intra-Kernel Pipelining, Warp Specialization, and Cooperative Thread Block Clusters, Thread Block Clusters and Distributed Shared Memorythreads using thread block clusters and DSMEM, Thread Block Clusters and Distributed Shared Memorywarp specialization with thread block clusters, Warp Specialization with Thread Block Clusters-Warp Specialization with Thread Block Clusters
- intra-kernel pipelining with inter-kernel overlap, Combining PDL and Thread Block Clusters with Warp Specialization-Combining PDL and Thread Block Clusters with Warp Specialization
- launching a thread block cluster, Launching a Thread Block Cluster, Designing Efficient Algorithms with Thread Block Clusters
- sharing state between thread blocks, Distributed Shared Memory
- thread block pairs, Launching a Thread Block Cluster, Thread Block Pair-Thread Block Pair
- tiling, Tiling with Thread Block Clusters-Tiling with Thread Block Clusters
- up to 16 thread blocks, Tiling with Thread Block Clusters

- algorithms for parallelizing workloads, Designing Efficient Algorithms with Thread Block Clusters-Designing Efficient Algorithms with Thread Block Clusters
- coordinating thread block clusters, Coordinating Thread Block Clusters with Cooperative Groups API-Coordinating Thread Block Clusters with Cooperative Groups API
- global memory traffic reduced, Reducing Global Memory Traffic with Thread Block Clusters-Reducing Global Memory Traffic with Thread Block Clusters
- launching a thread block cluster, Launching a Thread Block Cluster, Designing Efficient Algorithms with Thread Block Clusters
- scratch memory, Scratch Memory
- sharing state between thread blocks, Distributed Shared Memory
- thread block clusters described, Intra-Kernel Pipelining, Warp Specialization, and Cooperative Thread Block Clusters
- thread block pair, Thread Block Pair-Thread Block Pair
- thread block swizzling, Thread Block Swizzling
- threads accessing each other’s shared memory, Threads, Warps, Blocks, and Grids, Intra-Kernel Pipelining, Warp Specialization, and Cooperative Thread Block Clusters, Thread Block Clusters and Distributed Shared Memory
- threads using thread block clusters and DSMEM, Thread Block Clusters and Distributed Shared Memory
- warp specialization with thread block clusters, Warp Specialization with Thread Block Clusters-Warp Specialization with Thread Block Clusters

- cooperative mode, Cooperative Groups
- coscheduled on same GPC, Thread Block Clusters and Distributed Shared Memory
- distributed shared memory, Thread Block Clusters and Distributed Shared Memory-Thread Block Pairsharing state between thread blocks, Distributed Shared Memorythread block clusters described, Intra-Kernel Pipelining, Warp Specialization, and Cooperative Thread Block Clustersthreads accessing each other’s shared memory, Threads, Warps, Blocks, and Grids, Thread Block Clusters and Distributed Shared Memorythreads using thread block clusters and DSMEM, Thread Block Clusters and Distributed Shared Memory
- distributed shared memory andthreads accessing each other’s shared memory, Intra-Kernel Pipelining, Warp Specialization, and Cooperative Thread Block Clusters
- programming CUDAblocksPerGrid, CUDA Programming Refresher, CUDA Programming RefresherblocksPerGrid as launch parameter, Configuring Launch Parameters: Blocks per Grid and Threads per Block-Configuring Launch Parameters: Blocks per Grid and Threads per Blockminimum thread blocks resident on each SM, Tuning Occupancy with Launch Bounds-Tuning Occupancy with Launch BoundsthreadsPerBlock, CUDA Programming Refresher, CUDA Programming RefresherthreadsPerBlock as launch parameter, Configuring Launch Parameters: Blocks per Grid and Threads per Block-Configuring Launch Parameters: Blocks per Grid and Threads per Block
- sharing state between thread blocks, Distributed Shared Memory
- thread block pairs, Launching a Thread Block Cluster, Thread Block Pair-Thread Block Pair(see also thread block clusters)
- thread block size, Choosing Threads-per-Block and Blocks-per-Grid Sizes
- tiling, Tiling and Data Reuse Using Shared Memory

- sharing state between thread blocks, Distributed Shared Memory
- thread block clusters described, Intra-Kernel Pipelining, Warp Specialization, and Cooperative Thread Block Clusters
- threads accessing each other’s shared memory, Threads, Warps, Blocks, and Grids, Thread Block Clusters and Distributed Shared Memory
- threads using thread block clusters and DSMEM, Thread Block Clusters and Distributed Shared Memory

- threads accessing each other’s shared memory, Intra-Kernel Pipelining, Warp Specialization, and Cooperative Thread Block Clusters

- blocksPerGrid, CUDA Programming Refresher, CUDA Programming Refresher
- blocksPerGrid as launch parameter, Configuring Launch Parameters: Blocks per Grid and Threads per Block-Configuring Launch Parameters: Blocks per Grid and Threads per Block
- minimum thread blocks resident on each SM, Tuning Occupancy with Launch Bounds-Tuning Occupancy with Launch Bounds
- threadsPerBlock, CUDA Programming Refresher, CUDA Programming Refresher
- threadsPerBlock as launch parameter, Configuring Launch Parameters: Blocks per Grid and Threads per Block-Configuring Launch Parameters: Blocks per Grid and Threads per Block

- (see also thread block clusters)

- coalesced versus uncoalesced global memory access, Coalesced Versus Uncoalesced Global Memory Access-Coalesced Versus Uncoalesced Global Memory Access
- cooperative groups, Cooperative Groups-When to Combine Persistent Kernels and Cooperative Groups
- distributed shared memory, Thread Block Clusters and Distributed Shared Memory
- GPU architecture, Threads, Warps, Blocks, and Grids-Threads, Warps, Blocks, and Grids
- intrawarp communication, Warp Shuffle Intrinsics: Avoid Shared Memory and Explicit Synchronization-Warp Shuffle Intrinsics: Avoid Shared Memory and Explicit Synchronization
- programming CUDAchevron syntax (<<< >>>), CUDA Programming Refreshermaximum threadsPerBlock compile time parameter, Tuning Occupancy with Launch Bounds-Tuning Occupancy with Launch BoundsthreadsPerBlock, CUDA Programming Refresher, CUDA Programming RefresherthreadsPerBlock as launch parameter, Configuring Launch Parameters: Blocks per Grid and Threads per Block-Configuring Launch Parameters: Blocks per Grid and Threads per Block
- tiling, Tiling and Data Reuse Using Shared Memory
- warps, Streaming Multiprocessor, Threads, and Warps, Understanding GPU Architecture-Understanding GPU Architecturewarp divergence, Threads, Warps, Blocks, and Grids

- chevron syntax (<<< >>>), CUDA Programming Refresher
- maximum threadsPerBlock compile time parameter, Tuning Occupancy with Launch Bounds-Tuning Occupancy with Launch Bounds
- threadsPerBlock, CUDA Programming Refresher, CUDA Programming Refresher
- threadsPerBlock as launch parameter, Configuring Launch Parameters: Blocks per Grid and Threads per Block-Configuring Launch Parameters: Blocks per Grid and Threads per Block

- warp divergence, Threads, Warps, Blocks, and Grids

- launch parameter, Configuring Launch Parameters: Blocks per Grid and Threads per Block-Configuring Launch Parameters: Blocks per Grid and Threads per Block

- disaggregation of prefill and decode, Impact on Latency (TTFT) and Throughput (TPOT)
- FP4, FP8, FP16, NVIDIA GPU Tensor Cores and Transformer Engine
- “goodput”, Measuring “Goodput” Useful Throughput
- NVL72 rack Tensor Core throughput, NVIDIA GPU Tensor Cores and Transformer Engine
- NVMe and filesystem tuned for throughput, Tuning NVMe and Filesystem for Throughput

- arithmetic intensity increased, Tiling and Data Reuse Using Shared Memory, Multilevel Microtiling and Software Prefetchingmultilevel tiling, Multilevel Microtiling and Software Prefetching
- double-buffered cooperative tiling, Cooperative Tiling and Double-Buffering with the CUDA Pipeline API-Cooperative Tiling and Double-Buffering with the CUDA Pipeline API
- multilevel tiling, Multilevel Microtiling and Software Prefetching
- PyTorch, PyTorch and Arithmetic Intensity
- shared memory for tiling, Tiling and Data Reuse Using Shared Memory-Tiling and Data Reuse Using Shared Memory, Multilevel Microtiling and Software PrefetchingCUTLASS, Using CUTLASS for Optimal Arithmetic Intensity and Tensor Core Performance
- single-kernel tiling, Tiled and Persistent GEMM Kernel (Triton)-Tiled and Persistent GEMM Kernel (Triton)
- software prefetching, Multilevel Microtiling and Software Prefetching
- thread block clusters, Tiling with Thread Block Clusters-Tiling with Thread Block Clusters
- tiles, Tiling and Data Reuse Using Shared Memory
- TMA fetching from global into shared memory, Asynchronous Memory Prefetching and Tensor Memory Accelerator-Asynchronous Memory Prefetching and Tensor Memory Accelerator

- multilevel tiling, Multilevel Microtiling and Software Prefetching

- CUTLASS, Using CUTLASS for Optimal Arithmetic Intensity and Tensor Core Performance

- double buffering with CUTLASS, Using CUTLASS for Optimal Arithmetic Intensity and Tensor Core Performance
- Transformer Engine, Transformer Engine and TMEM in Depth

- pruning, PyTorch Architecture Optimization (torchao), Quantization, Sparsity, and Pruning
- quantization, PyTorch Architecture Optimization (torchao), Quantization, Sparsity, and PruningINT8 quantization, Transformer Engine and TMEM in Depth
- sparsity, PyTorch Architecture Optimization (torchao), Quantization, Sparsity, and Pruning

- INT8 quantization, Transformer Engine and TMEM in Depth

- debugging numerical accuracy, Debugging Numerical Correctness and Accuracy
- dynamic shapes, Profiling and Debugging Compiler Performance Issues, TorchDynamo for Bytecode Capture and Graph Extraction
- explain() output, Profiling and Debugging Compiler Performance Issues, Graph Breaks and TorchDynamo explain()all-reduce-related warnings, Tensor and Pipeline Parallelism with torch.compilegraph breaks, Profiling and Debugging Compiler Performance Issues, FSDP with torch.compilegraph breaks debugged, Graph Breaks and TorchDynamo explain()
- FX Graph output, TorchDynamo for Bytecode Capture and Graph Extractiondumping graph after each stage, Debugging Compiler Phases, Graph Breaks, and Performance
- graph breaks, TorchDynamo for Bytecode Capture and Graph Extractioncompiler conditionals as tensor operations, TorchDynamo for Bytecode Capture and Graph Extractionforcing error to be raised at compile, TorchDynamo for Bytecode Capture and Graph Extraction
- minifier tool, Debugging Numerical Correctness and Accuracy
- profiling and debugging compiler issues, Profiling and Debugging Compiler Performance Issues
- PyTorch compiler pipeline, PyTorch Compiler Deep Dive
- PyTorch compiler stack, PyTorch and Higher-Level AI Frameworks, PyTorch Compiler (torch.compile), PyTorch Compiler, OpenAI Triton, and XLA Backends
- stances for errors or fallbacks, TorchDynamo for Bytecode Capture and Graph Extraction
- TORCH_LOGS, Profiling and Debugging Compiler Performance Issues

- all-reduce-related warnings, Tensor and Pipeline Parallelism with torch.compile
- graph breaks, Profiling and Debugging Compiler Performance Issues, FSDP with torch.compile
- graph breaks debugged, Graph Breaks and TorchDynamo explain()

- dumping graph after each stage, Debugging Compiler Phases, Graph Breaks, and Performance

- compiler conditionals as tensor operations, TorchDynamo for Bytecode Capture and Graph Extraction
- forcing error to be raised at compile, TorchDynamo for Bytecode Capture and Graph Extraction

- arithmetic intensity increased, Increasing CUDA Kernel Efficiency and Arithmetic Intensity
- automatic mixed precision, TF32 and Automatic Mixed Precision (PyTorch)-TF32 and Automatic Mixed Precision (PyTorch)
- autotuning, Using the PyTorch Compiler, Autotuning with TorchInductor-Autotuning with TorchInductormax-autotune compile mode, Using the PyTorch Compiler, PyTorch Compiler Deep Dive, Autotuning with TorchInductor
- CUDA Graphs, Autotuning with TorchInductor
- debug mode, Performance Hints and Debugging Generated Code, Debugging Compiler Phases, Graph Breaks, and Performanceinspecting TorchInductor code, Performance Hints and Debugging Generated Code
- FX Graph, AOT Autograd Fusion for Forward and Backward Passes
- introduction, PyTorch and Higher-Level AI Frameworks
- IR, TorchInductor Backend Code Generation
- kernel fusion, Tuning Occupancy with PyTorch, Kernel Fusion, TorchDynamo for Bytecode Capture and Graph Extractionfused attention kernels, PyTorch, CUDA Pipeline API, and Warp SpecializationGEMM kernel prologue and epilogue, Using the PyTorch Compiler
- minifier tool, Debugging Numerical Correctness and Accuracy
- optimizations versus writing custom kernels, Compiling Versus Writing Custom Kernels
- PrimTorch IR simplifying operations, PrimTorch IR (Prims) Simplified Operator Set
- profiling and benchmarking kernels generated, Profiling and Debugging Compiler Performance Issues
- PyTorch compiler pipeline, PyTorch Compiler Deep Dive
- PyTorch compiler stack, PyTorch and Higher-Level AI Frameworks, PyTorch Compiler (torch.compile), PyTorch Compiler, OpenAI Triton, and XLA Backends
- SimpleFSDP, TorchTitan, AsyncTP, AutoParallel, and SimpleFSDP
- symbolic shapes, TorchInductor Backend Code Generation
- Triton based, PyTorch and Higher-Level AI Frameworks, Using the PyTorch Compiler, TorchInductor Backend Code Generation, Writing Custom Kernels with OpenAI Triton

- max-autotune compile mode, Using the PyTorch Compiler, PyTorch Compiler Deep Dive, Autotuning with TorchInductor

- inspecting TorchInductor code, Performance Hints and Debugging Generated Code

- fused attention kernels, PyTorch, CUDA Pipeline API, and Warp Specialization
- GEMM kernel prologue and epilogue, Using the PyTorch Compiler

- AsyncTP, TorchTitan, AsyncTP, AutoParallel, and SimpleFSDP
- SimpleFSDP reducing memory usage, TorchTitan, AsyncTP, AutoParallel, and SimpleFSDP

- compiler debugging, Debugging Compiler Phases, Graph Breaks, and Performance
- logging options summary, Debugging Compiler Phases, Graph Breaks, and Performance
- performance hints, Performance Hints and Debugging Generated Code

- disaggregation of prefill and decode, Impact on Latency (TTFT) and Throughput (TPOT)
- prefill and decode phases, Scaling Disaggregated Prefill and Decode for Inference

- 100-trillion parameter models, Toward 100-Trillion-Parameter Modelsloading training data, Toward 100-Trillion-Parameter Models
- cold start strategy, DeepSeek Scales to ~680-Billion Parameter Models Despite US Export Hardware Restrictions in China
- costsDeepSeek-R1, Introduction and AI System OverviewDeepSeek-V3, DeepSeek Scales to ~680-Billion Parameter Models Despite US Export Hardware Restrictions in ChinaGoogle Gemini Ultra, Introduction and AI System OverviewOpenAI GPT-4, Introduction and AI System Overview
- datasets via NVIDIA NeMo Curator, Creating High-Quality LLM Datasets with NVIDIA NeMo Curator

- loading training data, Toward 100-Trillion-Parameter Models

- DeepSeek-R1, Introduction and AI System Overview
- DeepSeek-V3, DeepSeek Scales to ~680-Billion Parameter Models Despite US Export Hardware Restrictions in China
- Google Gemini Ultra, Introduction and AI System Overview
- OpenAI GPT-4, Introduction and AI System Overview

- balancing accuracy and precision, NVIDIA GPU Tensor Cores and Transformer Engine
- installing only if calling TE modules directly, Autotuning with TorchInductor
- mechanical sympathy, Mechanical Sympathy: Hardware-Software Codesign
- modern GPUs using, Mechanical Sympathy: Hardware-Software Codesign
- PyTorch compiler, Autotuning with TorchInductor
- TMEM, Transformer Engine and TMEM in Depth

- AI systems performance engineers, Transparency and Reproducibility
- DeepSeek Open-Source Week, Transparency and Reproducibility

- advanced kernel implementations, Advanced Triton Kernel Implementations-Software Pipelining and Double Buffering with Tritondouble-buffering pipelining, Software Pipelining and Double Buffering with Triton-Software Pipelining and Double Buffering with Tritonpersistent matmul kernel, Tiled and Persistent GEMM Kernel (Triton)-Tiled and Persistent GEMM Kernel (Triton)warp specialization, Warp Specialization with Triton
- CUDA kernels in Python, NVIDIA Software Stack
- custom kernels, Writing Custom Kernels with OpenAI Triton-Profiling with Triton Proton Profiler
- language and compiler, C++ and Python CUDA Libraries
- open source, C++ and Python CUDA Libraries
- programmingaccessing shared memory, Accessing Shared Memory in Tritonautotuning Triton kernels, Autotuning Triton Kernelscustom kernels, Writing Custom Kernels with OpenAI Triton-Profiling with Triton Proton ProfilerNsight Compute and Systems support, Profiling with Triton Proton Profilerprofiling with Triton Proton Profiler, Profiling with Triton Proton Profilerprogramming model, Triton Programming Modelregistering custom kernels with PyTorch, TorchInductor Backend Code Generation, Registering Custom Kernels with PyTorch-Registering Custom Kernels with PyTorchtuning kernel launch parameters, Tuning Kernel-Launch Parameters
- Proton Profiler, Profiling with Triton Proton Profiler
- smart compilers, Smart Compilers and Automated Code Optimizations-Smart Compilers and Automated Code Optimizations
- TorchInductor basis, PyTorch and Higher-Level AI Frameworks, Using the PyTorch Compiler, TorchInductor Backend Code Generation, Writing Custom Kernels with OpenAI Triton

- double-buffering pipelining, Software Pipelining and Double Buffering with Triton-Software Pipelining and Double Buffering with Triton
- persistent matmul kernel, Tiled and Persistent GEMM Kernel (Triton)-Tiled and Persistent GEMM Kernel (Triton)
- warp specialization, Warp Specialization with Triton

- accessing shared memory, Accessing Shared Memory in Triton
- autotuning Triton kernels, Autotuning Triton Kernels
- custom kernels, Writing Custom Kernels with OpenAI Triton-Profiling with Triton Proton Profiler
- Nsight Compute and Systems support, Profiling with Triton Proton Profiler
- profiling with Triton Proton Profiler, Profiling with Triton Proton Profiler
- programming model, Triton Programming Model
- registering custom kernels with PyTorch, TorchInductor Backend Code Generation, Registering Custom Kernels with PyTorch-Registering Custom Kernels with PyTorch
- tuning kernel launch parameters, Tuning Kernel-Launch Parameters

- constrained outputs, Constrained Decoding Performance Implications
- context parallelism, Context (Sequence) Parallelism
- disaggregation of prefill and decode, Impact on Latency (TTFT) and Throughput (TPOT)
- prefill and decode phases, Scaling Disaggregated Prefill and Decode for Inference
- speculative KV prefetching, Speculative KV Prefetching for Faster TTFT-Speculative KV Prefetching for Faster TTFT

### U

- UCX (Unified Communication X; NVIDIA), Enabling Peer-to-Peer DMA and UCX
- underutilization GPU stalls, Iteratively Profiling and Determining the Kernel Bottleneck-Iteratively Profiling and Determining the Kernel Bottleneckkernel optimization, Optimizing the Kernel-Optimizing the KernelNsight Compute metrics, Iteratively Profiling and Determining the Kernel Bottleneck
- Unified Communication X (UCX; NVIDIA), Enabling Peer-to-Peer DMA and UCX
- Unified CPU-GPU Memory, The CPU and GPU Superchip
- upgrades and updateshighest-level and most-recent APIs available, Asynchronous Memory Prefetching and Tensor Memory AcceleratorNCCL release notes, Pitfall #6: NCCL communicator hangs, errors, or shuts down completelyNsight Systems and Nsight Compute with latest GPUs, Profiler-Guided AnalysisNVIDIA continually updating libraries, Using CUTLASS for Optimal Arithmetic Intensity and Tensor Core PerformanceROI of upgrading hardware, ROI of Upgrading Your Hardwareupdate but verify, Continuous Profiling and Tuning Workflow
- utilization-maximization policy, Adaptive Batching and Chunked Prefill Scheduling

- kernel optimization, Optimizing the Kernel-Optimizing the Kernel
- Nsight Compute metrics, Iteratively Profiling and Determining the Kernel Bottleneck

- highest-level and most-recent APIs available, Asynchronous Memory Prefetching and Tensor Memory Accelerator
- NCCL release notes, Pitfall #6: NCCL communicator hangs, errors, or shuts down completely
- Nsight Systems and Nsight Compute with latest GPUs, Profiler-Guided Analysis
- NVIDIA continually updating libraries, Using CUTLASS for Optimal Arithmetic Intensity and Tensor Core Performance
- ROI of upgrading hardware, ROI of Upgrading Your Hardware
- update but verify, Continuous Profiling and Tuning Workflow

### V

- variable-length sequences via dynamic shapes, Dynamic Shapes and Variable Sequence Lengths-Dynamic Shapes and Variable Sequence Lengths
- vectorized memory access, Vectorized Memory Access-Vectorized Memory Access
- Vera Rubin Superchip (VR200), Vera Rubin Superchip (2026)
- vLLMconstrained decoding, Constrained Decoding Performance ImplicationsCUDA Graphs, PyTorch, Inference Engines, and CUDA Graphsdisaggregated prefill, KV Cache Data Transfer and NIXL, Deploying Disaggregated Prefill and Decode with Kubernetesdynamic batching merged with continuous scheduling, Continuous SchedulingPagedAttention, Continuous SchedulingLMCache, Profiling, Debugging, and Tuning Inference Performancedisaggregated prefill and decode, Scaling Disaggregated Prefill and Decode for Inference, Zero-Copy GPU-to-GPU Transfermegakernel decode throughput, Megakernels for InferenceNIXL-vLLM integration, KV Cache Data Transfer and NIXLNVLink domain, NVIDIA’s “AI Supercomputer in a Rack”PagedAttention, Continuous Scheduling, KV Cache Offloading and Memory Pool Allocation, Memory management for the KV cacheparallel decoding, Interleaving Decode Steps from Multiple Requestsprefix caching, Prefix Caching, KV Cache Reuse and Prefix Sharingspeculative decoding, Two-Model, Draft-Based Speculative Decoding and EAGLEtroubleshooting inference, Inference Troubleshooting Recipes
- VR200 (Vera Rubin Superchip), Vera Rubin Superchip (2026)

- constrained decoding, Constrained Decoding Performance Implications
- CUDA Graphs, PyTorch, Inference Engines, and CUDA Graphs
- disaggregated prefill, KV Cache Data Transfer and NIXL, Deploying Disaggregated Prefill and Decode with Kubernetes
- dynamic batching merged with continuous scheduling, Continuous SchedulingPagedAttention, Continuous Scheduling
- LMCache, Profiling, Debugging, and Tuning Inference Performancedisaggregated prefill and decode, Scaling Disaggregated Prefill and Decode for Inference, Zero-Copy GPU-to-GPU Transfer
- megakernel decode throughput, Megakernels for Inference
- NIXL-vLLM integration, KV Cache Data Transfer and NIXL
- NVLink domain, NVIDIA’s “AI Supercomputer in a Rack”
- PagedAttention, Continuous Scheduling, KV Cache Offloading and Memory Pool Allocation, Memory management for the KV cache
- parallel decoding, Interleaving Decode Steps from Multiple Requests
- prefix caching, Prefix Caching, KV Cache Reuse and Prefix Sharing
- speculative decoding, Two-Model, Draft-Based Speculative Decoding and EAGLE
- troubleshooting inference, Inference Troubleshooting Recipes

- PagedAttention, Continuous Scheduling

- disaggregated prefill and decode, Scaling Disaggregated Prefill and Decode for Inference, Zero-Copy GPU-to-GPU Transfer

### W

- Warp (see NVIDIA Warp)
- warp stall reasons, Analyzing Warp Stall Reasons with Nsight Compute-Other Stall Reasons
- warps, Streaming Multiprocessor, Threads, and Warpsefficiency via PyTorch, PyTorch Considerations for Warp-Level Efficiency-PyTorch Considerations for Warp-Level Efficiencyefficient intrawarp communication, Efficient Intrawarp Communication with Warp Intrinsicsexecution efficiency improved, Improving Warp Execution Efficiency (Warp Divergence)-PyTorch Considerations for Warp-Level EfficiencyGPU architecture, Understanding GPU Architecture-Understanding GPU Architecturehigh occupancy, Threads, Warps, Blocks, and GridsILP increased for latency-bound GPUs, Iteratively Profiling and Determining the Kernel Bottlenecklatency hiding, Streaming Multiprocessor, Threads, and Warpsmaintaining high occupancy, Maintaining High Occupancy and GPU Utilizationshuffle intrinsics avoiding shared memory, Warp Shuffle Intrinsics: Avoid Shared Memory and Explicit Synchronization-Warp Shuffle Intrinsics: Avoid Shared Memory and Explicit Synchronizationstalls or frequent pauses, Analyzing Warp Stall Reasons with Nsight Compute-Other Stall Reasonsexecution-dependency stalls, Execution-Dependency Stallsmemory-related stalls, Memory-Related Stallswarp divergence, Threads, Warps, Blocks, and Grids, Improving Warp Execution Efficiency (Warp Divergence)avoiding, Techniques to Avoid Warp Divergencecauses of, Causes of Warp Divergenceefficient intrawarp communication, Efficient Intrawarp Communication with Warp Intrinsicsimproving warp execution efficiency, Improving Warp Execution Efficiency (Warp Divergence)-PyTorch Considerations for Warp-Level Efficiencypredication to minimize, Using Predication to Minimize Divergence-Using Predication to Minimize Divergenceprofiling and detecting, Profiling and Detecting Warp DivergencePyTorch considerations for efficiency, PyTorch Considerations for Warp-Level Efficiency-PyTorch Considerations for Warp-Level Efficiencywarp specializationCUDA streams and warp specialization, Pipelining with Warp Specialization (Intra-Kernel) and CUDA Streams (Inter-Kernel)-Pipelining with Warp Specialization (Intra-Kernel) and CUDA Streams (Inter-Kernel)CUTLASS, Using CUTLASS for Optimal Arithmetic Intensity and Tensor Core Performanceinter-kernel pipelining with CUDA streams, Overlapping Kernel Execution with CUDA Streams-Overlapping Kernel Execution with CUDA Streamsinter-kernel pipelining with CUDA streams and thread block clusters, Warp Specialization with Thread Block Clusters and CUDA Streamsintra-kernel pipelining, Warp Specialization and the Producer-Consumer Model-Warp Specialization and the Producer-Consumer Modelintra-kernel pipelining with CUDA Pipeline API, Using CUDA Pipeline API for Warp Specialization-Using CUDA Pipeline API for Warp Specializationintra-kernel pipelining with inter-kernel overlap, Combining PDL and Thread Block Clusters with Warp Specialization-Combining PDL and Thread Block Clusters with Warp Specializationintra-kernel pipelining with persistent kernels, Persistent Kernels and Warp Specializationintra-kernel pipelining with thread block clusters, Warp Specialization with Thread Block Clusters-Warp Specialization with Thread Block ClustersTriton, Warp Specialization with Triton
- web page for book, How to Contact Us
- weight-splitting parallelism strategies, Parallelism Strategies for Serving Massive MoE Models
- workloadsdynamic work queues, Dynamic Scheduling with Atomic Work Queues-Atomic Queuesatomic counters, Atomic Countersatomic queues, Atomic Counters, Atomic Queues-Atomic QueuesNsight Compute, Atomic Counters-Atomic Countersmemory-bound workload example, Maintaining High Occupancy and GPU Utilizationpersistent kernels, Common Workloads for Persistent Kernelsprofiling under unsufficient bandwidth, Pitfall #4: Insufficient network bandwidth or misconfigured NICsright occupancy for workload, Find the Right Occupancy for Your Workloadsharing and scheduling, Sharing and Schedulingthread block clusters parallelizing workloads, Designing Efficient Algorithms with Thread Block Clusters-Designing Efficient Algorithms with Thread Block Clusters
- write-back, Filesystem Caching and Write-Back

- efficiency via PyTorch, PyTorch Considerations for Warp-Level Efficiency-PyTorch Considerations for Warp-Level Efficiency
- efficient intrawarp communication, Efficient Intrawarp Communication with Warp Intrinsics
- execution efficiency improved, Improving Warp Execution Efficiency (Warp Divergence)-PyTorch Considerations for Warp-Level Efficiency
- GPU architecture, Understanding GPU Architecture-Understanding GPU Architecture
- high occupancy, Threads, Warps, Blocks, and Grids
- ILP increased for latency-bound GPUs, Iteratively Profiling and Determining the Kernel Bottleneck
- latency hiding, Streaming Multiprocessor, Threads, and Warps
- maintaining high occupancy, Maintaining High Occupancy and GPU Utilization
- shuffle intrinsics avoiding shared memory, Warp Shuffle Intrinsics: Avoid Shared Memory and Explicit Synchronization-Warp Shuffle Intrinsics: Avoid Shared Memory and Explicit Synchronization
- stalls or frequent pauses, Analyzing Warp Stall Reasons with Nsight Compute-Other Stall Reasonsexecution-dependency stalls, Execution-Dependency Stallsmemory-related stalls, Memory-Related Stalls
- warp divergence, Threads, Warps, Blocks, and Grids, Improving Warp Execution Efficiency (Warp Divergence)avoiding, Techniques to Avoid Warp Divergencecauses of, Causes of Warp Divergenceefficient intrawarp communication, Efficient Intrawarp Communication with Warp Intrinsicsimproving warp execution efficiency, Improving Warp Execution Efficiency (Warp Divergence)-PyTorch Considerations for Warp-Level Efficiencypredication to minimize, Using Predication to Minimize Divergence-Using Predication to Minimize Divergenceprofiling and detecting, Profiling and Detecting Warp DivergencePyTorch considerations for efficiency, PyTorch Considerations for Warp-Level Efficiency-PyTorch Considerations for Warp-Level Efficiency
- warp specializationCUDA streams and warp specialization, Pipelining with Warp Specialization (Intra-Kernel) and CUDA Streams (Inter-Kernel)-Pipelining with Warp Specialization (Intra-Kernel) and CUDA Streams (Inter-Kernel)CUTLASS, Using CUTLASS for Optimal Arithmetic Intensity and Tensor Core Performanceinter-kernel pipelining with CUDA streams, Overlapping Kernel Execution with CUDA Streams-Overlapping Kernel Execution with CUDA Streamsinter-kernel pipelining with CUDA streams and thread block clusters, Warp Specialization with Thread Block Clusters and CUDA Streamsintra-kernel pipelining, Warp Specialization and the Producer-Consumer Model-Warp Specialization and the Producer-Consumer Modelintra-kernel pipelining with CUDA Pipeline API, Using CUDA Pipeline API for Warp Specialization-Using CUDA Pipeline API for Warp Specializationintra-kernel pipelining with inter-kernel overlap, Combining PDL and Thread Block Clusters with Warp Specialization-Combining PDL and Thread Block Clusters with Warp Specializationintra-kernel pipelining with persistent kernels, Persistent Kernels and Warp Specializationintra-kernel pipelining with thread block clusters, Warp Specialization with Thread Block Clusters-Warp Specialization with Thread Block ClustersTriton, Warp Specialization with Triton

- execution-dependency stalls, Execution-Dependency Stalls
- memory-related stalls, Memory-Related Stalls

- avoiding, Techniques to Avoid Warp Divergence
- causes of, Causes of Warp Divergence
- efficient intrawarp communication, Efficient Intrawarp Communication with Warp Intrinsics
- improving warp execution efficiency, Improving Warp Execution Efficiency (Warp Divergence)-PyTorch Considerations for Warp-Level Efficiency
- predication to minimize, Using Predication to Minimize Divergence-Using Predication to Minimize Divergence
- profiling and detecting, Profiling and Detecting Warp Divergence
- PyTorch considerations for efficiency, PyTorch Considerations for Warp-Level Efficiency-PyTorch Considerations for Warp-Level Efficiency

- CUDA streams and warp specialization, Pipelining with Warp Specialization (Intra-Kernel) and CUDA Streams (Inter-Kernel)-Pipelining with Warp Specialization (Intra-Kernel) and CUDA Streams (Inter-Kernel)
- CUTLASS, Using CUTLASS for Optimal Arithmetic Intensity and Tensor Core Performance
- inter-kernel pipelining with CUDA streams, Overlapping Kernel Execution with CUDA Streams-Overlapping Kernel Execution with CUDA Streams
- inter-kernel pipelining with CUDA streams and thread block clusters, Warp Specialization with Thread Block Clusters and CUDA Streams
- intra-kernel pipelining, Warp Specialization and the Producer-Consumer Model-Warp Specialization and the Producer-Consumer Model
- intra-kernel pipelining with CUDA Pipeline API, Using CUDA Pipeline API for Warp Specialization-Using CUDA Pipeline API for Warp Specialization
- intra-kernel pipelining with inter-kernel overlap, Combining PDL and Thread Block Clusters with Warp Specialization-Combining PDL and Thread Block Clusters with Warp Specialization
- intra-kernel pipelining with persistent kernels, Persistent Kernels and Warp Specialization
- intra-kernel pipelining with thread block clusters, Warp Specialization with Thread Block Clusters-Warp Specialization with Thread Block Clusters
- Triton, Warp Specialization with Triton

- dynamic work queues, Dynamic Scheduling with Atomic Work Queues-Atomic Queuesatomic counters, Atomic Countersatomic queues, Atomic Counters, Atomic Queues-Atomic QueuesNsight Compute, Atomic Counters-Atomic Counters
- memory-bound workload example, Maintaining High Occupancy and GPU Utilization
- persistent kernels, Common Workloads for Persistent Kernels
- profiling under unsufficient bandwidth, Pitfall #4: Insufficient network bandwidth or misconfigured NICs
- right occupancy for workload, Find the Right Occupancy for Your Workload
- sharing and scheduling, Sharing and Scheduling
- thread block clusters parallelizing workloads, Designing Efficient Algorithms with Thread Block Clusters-Designing Efficient Algorithms with Thread Block Clusters

- atomic counters, Atomic Counters
- atomic queues, Atomic Counters, Atomic Queues-Atomic Queues
- Nsight Compute, Atomic Counters-Atomic Counters

### X

- XFSoptimizing, Sequential Versus Random Read Patternstuning for throughput, Tuning NVMe and Filesystem for Throughput
- XLA, PyTorch XLA Backendactivating, PyTorch XLA BackendGoogle Cloud TPU, TorchInductor Backend Code Generation, PyTorch XLA BackendPyTorch compiler pipeline, PyTorch Compiler Deep Dive, TorchDynamo for Bytecode Capture and Graph Extraction, AOT Autograd Fusion for Forward and Backward Passes

- optimizing, Sequential Versus Random Read Patterns
- tuning for throughput, Tuning NVMe and Filesystem for Throughput

- activating, PyTorch XLA Backend
- Google Cloud TPU, TorchInductor Backend Code Generation, PyTorch XLA Backend
- PyTorch compiler pipeline, PyTorch Compiler Deep Dive, TorchDynamo for Bytecode Capture and Graph Extraction, AOT Autograd Fusion for Forward and Backward Passes

### Z

- ZeRO Stage-3 in FSDP, FSDP Automatic Checkpointing and Offloading
- zero-copy GPU-to-GPU transfer, Zero-Copy GPU-to-GPU Transfer-Zero-Copy GPU-to-GPU Transfer
- zero-copy registration, Persistent NCCL User Buffers and Zero-Copy Registration

---

## Chapter ?

# About the Author _ AI Systems Performance Engineering
