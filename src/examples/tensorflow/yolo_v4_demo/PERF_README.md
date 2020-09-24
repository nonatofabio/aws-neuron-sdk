# README for the Performance benchmark

These are the steps to perform a benchmark using the scripts version of the tutorial.

## Setup

For GPU based instances, use the DLAMI `ami-06a25ee8966373068` to create a new instance and SSH into it. Activate the `tensorflow_p36` environment and install the dependencies with:

```bash
conda activate tensorflow_p36 
pip install -r requirements.txt
```
For AWS inferentia type instances, replace the requirements file with _requirements_inf.txt_.

To collect the require COCO 2017 validation dataset and save a pre-trained version of the model, utiliz the [setup script](src/examples/tensorflow/yolo_v4_demo/setup.sh).

## Compiling the model - Inf1 instance only

To compile the model, run the script `yolo_compile.py`

## Run the benchmark

After all above steps complete, call the performance benchmark script:

```bash 
TF_XLA_FLAGS=--tf_xla_auto_jit=2 python perf_benchmark.py <path_to_saved_model> 2>&1 | tee log.txt
```
