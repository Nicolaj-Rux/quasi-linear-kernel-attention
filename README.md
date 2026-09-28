# Kernel attention experiments

Code for the four experiments of the paper: associative recall, a vision transformer on CIFAR-10, distillation of
nomic-embed-text-v1, and a speed/memory benchmark.

```
Experiments/        experiment code; all experiments share the attention module kernel_attention/
Weighted_Abs_Sum/     CUDA extension for weighted L1 distance sums (used by the add_riesz and add_bump kernels)
Weighted_Laplace_Sum/ CUDA extension for weighted additive Laplace sums, forward only (add_laplace inference)
install.sh          conda env + extension build
```

## Installation

Requires Anaconda (or Miniconda), Linux, an NVIDIA GPU and driver >= 575.

```bash
bash install.sh
```

This creates the conda env `kernel_attn` and builds `Weighted_Abs_Sum` and `Weighted_Laplace_Sum` into it. Step by step:

```bash
conda create -y -n kernel_attn --override-channels -c nvidia -c conda-forge python=3.12.12 cuda-toolkit=12.9.2 cuda-version=12.9 gcc_linux-64=13.4.0 gxx_linux-64=13.4.0
conda activate kernel_attn
export CUDA_HOME=$CONDA_PREFIX CUDA_PATH=$CONDA_PREFIX
pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu129
pip install pykeops==2.3 numpy==2.3.5 matplotlib==3.10.8 datasets==4.8.4 transformers==5.5.4 sentence-transformers==5.5.0 mteb==2.12.30
pip install --no-build-isolation ./Weighted_Abs_Sum
pip install --no-build-isolation ./Weighted_Laplace_Sum
```

| package | version |
|---|---|
| Python | 3.12.12 |
| PyTorch / torchvision | 2.13.0 / 0.28.0 (CUDA 12.9 wheels) |
| CUDA toolkit (nvcc) | 12.9.2 (nvcc 12.9.86) |
| gcc / g++ (conda-forge) | 13.4.0 |
| pykeops | 2.3 |
| numpy | 2.3.5 |
| matplotlib | 3.10.8 |
| datasets | 4.8.4 |
| transformers | 5.5.4 |
| sentence-transformers | 5.5.0 |
| mteb | 2.12.30 |

### Weighted_Abs_Sum

`pip install --no-build-isolation ./Weighted_Abs_Sum` compiles the extension against the installed PyTorch
(`--no-build-isolation` is required for that) for all supported GPU architectures with the env's gcc 13.4.0. For a faster
build for one GPU set its architecture, e.g. `TORCH_CUDA_ARCH_LIST="8.6" pip install --no-build-isolation ./Weighted_Abs_Sum`. It needs float32 CUDA tensors, compute capability >= 7.5.

### Weighted_Laplace_Sum

Built the same way (`pip install --no-build-isolation ./Weighted_Laplace_Sum`, same `TORCH_CUDA_ARCH_LIST` option).
`weighted_laplace_sum(q, k, v)` computes the exact O(N log N) sums `sum_{n,d} exp(-|q_d - k_d|) [v_n, 1]`, forward
only. `kernel_attention` uses it for `add_laplace` whenever no gradient is needed (evaluation, `pareto.py`);
training runs `add_laplace` on the chunked PyTorch path.

## Running

Every shell needs the env and `CUDA_PATH` (pykeops compiles against its headers). All commands run from `Experiments/`:

```bash
conda activate kernel_attn
export CUDA_HOME=$CONDA_PREFIX CUDA_PATH=$CONDA_PREFIX
cd Experiments
```

Every hyperparameter is a command-line argument whose default is the paper's setting (`--help`). Runs are seeded
(`--seed`, default 0) and use fp32. Kernels: `softmax gauss laplace riesz add_laplace add_riesz add_bump tri elu relu dpfp`.
Checkpoints go to `checkpoints/` and CIFAR-10 is downloaded to `data/` (both next to `Experiments/`; override with
`--ckpt_dir`, `--data_dir`). Rerunning a command skips finished work.

### 1. Associative recall

```bash
associative_recall/run_sweep.sh   # all kernels x N = 20, 40, ..., 600 x seeds 0, 1, 2, one run after another
python -m associative_recall.plot   # -> associative_recall/loss_vs_N_best.pdf
```

Single run: `python -m associative_recall.train --kernel add_bump --N 600 --seed 0`. Each run appends one row to
`associative_recall/results.csv` (runs are not skipped on a rerun, so start from an empty file). Outputs: `associative_recall/results.csv`, `associative_recall/loss_vs_N_best.pdf` (best seed per kernel and N).

### 2. Vision transformer (CIFAR-10)

```bash
STUDENTS="gauss laplace riesz add_laplace add_riesz add_bump tri elu relu dpfp"
for k in softmax $STUDENTS; do python -m vision_transformer.scratch --kernel $k; done
for k in $STUDENTS; do
  python -m vision_transformer.copy_eval --kernel $k        # softmax teacher weights, student kernel
  python -m vision_transformer.layer_by_layer --kernel $k   # needs the softmax scratch checkpoint
  python -m vision_transformer.end_to_end --kernel $k       # needs the layer_by_layer checkpoint
done
```

Outputs: `checkpoints/vision_{scratch,lbl,e2e}_{kernel}_tau{tau}_{runid}.pt` (copy saves none) and
`Experiments/vision_transformer/results.csv` with the test accuracy of the last weights per stage. Training loss and
test accuracy curves are printed to stdout.

### 3. Text transformer (nomic-embed-text-v1)

The teacher is the stock softmax model, evaluated with its original weights.

```bash
STUDENTS="gauss laplace riesz add_laplace add_riesz add_bump tri elu relu dpfp"
for k in $STUDENTS; do python -m text_transformer.train --kernel $k; done
for k in softmax $STUDENTS; do
  python -m text_transformer.mteb_eval --kernel $k   # MTEB(eng, v2), 41 tasks; --stage lbl for the lbl checkpoint
  python -m text_transformer.loco_eval --kernel $k   # LoCo at 2048, 4096, 8192
done
for L in 8192 16384; do
  for k in softmax add_laplace add_riesz add_bump tri elu dpfp; do
    python -m text_transformer.pareto --kernel $k --seq_len $L   # accuracy vs GPU time
  done
done
python -m text_transformer.visualize.plot_pareto                 # -> visualize/pareto_all.pdf (8192, 16384)
```

`train.py` needs up to ~18.5 GB GPU memory; MTEB's MindSmallReranking needs ~75 GB RAM. Datasets and the model are
downloaded from the Hugging Face Hub.

Outputs: `checkpoints/text_{lbl,e2e}_{kernel}_tau{tau}_{runid}.pt`; per-task/subset JSON in
`Experiments/text_transformer/results/{mteb,loco}/{kernel}_tau{tau}_{runid}/`; scores, runtime and peak memory in
`Experiments/text_transformer/results/{mteb,loco,pareto}_summary.csv`; plots in `Experiments/text_transformer/visualize/`.

### 4. Speed and memory

```bash
python -m speed_and_memory.bench          # alone on an otherwise idle GPU
```

add_riesz attention through `Weighted_Abs_Sum` (`cuda-global`, `cuda-fused`, `keops`) and add_laplace attention through
`Weighted_Laplace_Sum` vs PyTorch SDPA softmax (memory-efficient fp32/fp16, flash fp16, cuDNN fp16), forward and
forward+backward (add_laplace: forward only, it has no backward), B = 4, H = 12, D = C = 64, N = 128 ... 131072.

Output: rows appended to `Experiments/speed_and_memory/results.csv`.
