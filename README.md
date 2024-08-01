# LoRAT

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/tracking-meets-lora-faster-training-larger/visual-object-tracking-on-lasot)](https://paperswithcode.com/sota/visual-object-tracking-on-lasot?p=tracking-meets-lora-faster-training-larger)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/tracking-meets-lora-faster-training-larger/visual-object-tracking-on-lasot-ext)](https://paperswithcode.com/sota/visual-object-tracking-on-lasot-ext?p=tracking-meets-lora-faster-training-larger)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/tracking-meets-lora-faster-training-larger/visual-object-tracking-on-tnl2k)](https://paperswithcode.com/sota/visual-object-tracking-on-tnl2k?p=tracking-meets-lora-faster-training-larger)

This is the official repository for **ECCV 2024** [Tracking Meets LoRA: Faster Training, Larger Model, Stronger Performance (LoRAT)](https://arxiv.org/abs/2403.05231).

[[Models](https://drive.google.com/drive/folders/1FvViP0MCSiAu2FSrNjg7XEORn74yOBdD?usp=sharing)]
[[Raw Results](https://drive.google.com/drive/folders/1zWlWfpWwBoomaRFNt7rTJlr6UsxfYU1T?usp=sharing)]

![banner](https://raw.githubusercontent.com/wiki/LitingLin/LoRAT/images/banner.svg)
## Prerequisites
### Environment
Assuming you have a working python environment with pip installed.

#### system packages (ubuntu)
```shell
apt update
apt install -y libturbojpeg
```
#### install pytorch
Can be skipped if using NGC container. PyTorch version should be >= 2.0.
```shell
pip install torch torchvision
```
#### extra python packages
```shell
pip install -r requirements.txt
```
This codebase should also work on Windows and macOS for debugging purposes.
### Dataset
#### Download
- [LaSOT](https://huggingface.co/datasets/l-lt/LaSOT)
- [LaSOT_Extension](https://huggingface.co/datasets/l-lt/LaSOT-ext)
- [GOT-10k](http://got-10k.aitestunion.com/downloads)
- [TrackingNet](https://github.com/SilvioGiancola/TrackingNet-devkit)
- [COCO 2017](https://cocodataset.org/#download)
- [TNL2K](https://github.com/wangxiao5791509/TNL2K_evaluation_toolkit)
#### Unzip
The paths should be organized as follows:
```
LaSOT
├── airplane
├── basketball
...
├── training_set.txt
└── testing_set.txt

LaSOT_Extension
├── atv
├── badminton
...
└── wingsuit

GOT-10k
├── train
│   ├── GOT-10k_Train_000001
│   ...
├── val
│   ├── GOT-10k_Val_000001
│   ...
└── test
    ├── GOT-10k_Test_000001
    ...
    
TrackingNet
├── TEST
├── TRAIN_0
...
└── TRAIN_11

COCO
├── annotations
│   ├── instances_train2017.json
│   └── instances_val2017.json
└── images
    ├── train2017
    │   ├── 000000000009.jpg
    │   ├── 000000000025.jpg
    │   ...
    └── val2017
        ├── 000000000139.jpg
        ├── 000000000285.jpg
        ...
TNL2K_TEST
├── advSamp_Baseball_game_002-Done
├── advSamp_Baseball_video_01-Done
...
```
#### Prepare ```consts.yaml```
Copy ```consts.template.yaml``` as ```consts.yaml``` and fill in the paths.
```yaml
LaSOT_PATH: '/path/to/lasot'
LaSOT_Extension_PATH: '/path/to/lasot_ext'
GOT10k_PATH: '/path/to/got10k'
TrackingNet_PATH: '/path/to/trackingnet'
COCO_2017_PATH: '/path/to/coco2017'
TNL2K_TEST_PATH: '/path/to/tnl2k_test'
```
#### Login to wandb (optional)
Register an account at [wandb](https://wandb.ai/), then login with the command:
```shell
wandb login
```
## Training & Evaluation

Note: Our code performs evaluation automatically when model training is complete.

- **Model weight** is saved in ```/path/to/output/run_id/checkpoint/epoch_{last}/model.bin```.
- **Performance metrics** can be found on terminal output and wandb dashboard.
- **Tracking results** are saved in ```/path/to/output/run_id/eval/epoch_{last}/```.

### Using run.sh helper script (Linux with NVIDIA GPU only)
```shell
# Train and evaluate LoRAT-B-224 model on all GPUs
./run.sh LoRAT dinov2 --output_dir /path/to/output
# Train and evaluate LoRAT-L-224 model on all GPUs
./run.sh LoRAT dinov2 --output_dir /path/to/output --mixin large
# Train and evaluate LoRAT-g-378 model on all GPUs
./run.sh LoRAT dinov2 --output_dir /path/to/output --mixin giant_378
# Train and evaluate LoRAT-L-224 model following GOT-10k protocol on all GPUs
./run.sh LoRAT dinov2 --output_dir /path/to/output --mixin large --mixin got10k
# Train and evaluate on specific GPUs
./run.sh LoRAT dinov2 --output_dir /path/to/output --device_ids 0,1,2,3
# Train and evaluate on multiple nodes
./run.sh LoRAT dinov2 --output_dir /path/to/output --nnodes $num_nodes --node_rank $node_rank --master_address $master_node_ip --date 2024.03.07-04.59.08-976343
```
You can set the default settings, e.g. `output_dir`, in ```run.sh```.
### Call main.py directly
```shell
# Train and evaluate LoRAT-B-224 model on single GPU
python main.py LoRAT dinov2 --output_dir /path/to/output

# Train and evaluate LoRAT-B-224 model on CPU
python main.py LoRAT dinov2 --output_dir /path/to/output --device cpu

# Train and evaluate LoRAT-B-224 model on all GPUs
python main.py LoRAT dinov2 --distributed_nproc_per_node $num_gpus --distributed_do_spawn_workers --output_dir /path/to/output

# Train and evaluate LoRAT-B-224 model on multiple nodes, run_id need to be set manually
python main.py LoRAT dinov2 --master_address $master_address --distributed_node_rank $node_rank distributed_nnodes $num_nodes --distributed_nproc_per_node $num_gpus --distributed_do_spawn_workers --output_dir /path/to/output --run_id $run_id
```
See ```python main.py --help``` for more options.

Note: If you encounter any issues with torch.compile, disable is with ```--mixin disable_torch_compile```.

Note: You can disable wandb logging with ```--disable_wandb```.

### Evaluation
Our code performs evaluation automatically when model training is complete. You can run evaluation only with the following command:
```shell
# evaluation only, on all datasets, defined in config/_dataset/test.yaml
./run.sh LoRAT dinov2 --output_dir /path/to/output --mixin evaluation --weight_path /path/to/weight.bin
```
The evaluated datasets are defined in ```config/_dataset/test.yaml```.

Note that, as defined in ```config/LoRAT/run.yaml```, we evaluate LaSOT Extension dataset three times. The final performance is the average of the three evaluations.
#### TrackingNet evaluation
Once the full evaluation is done, result files are saved in ```/path/to/output/run_id/eval/epoch_{last}/TrackingNet-test.zip```.

Submit this file to the [TrackingNet evaluation server](https://tracking-net.org/) to get the result of TrackingNet test split.

### Train and evaluate with GOT-10k dataset
```shell
# Train and evaluate LoRAT-B-224 model following GOT-10k protocol on all GPUs
./run.sh LoRAT dinov2 --output_dir /path/to/output --mixin got10k
```
Submit ```/path/to/output/run_id/eval/epoch_{last}/GOT10k-test.zip``` to the [GOT-10k evaluation server](http://got-10k.aitestunion.com/) to get the result of GOT-10k test split.

Evaluation only:
```shell
# evaluation only, on GOT-10k dataset
./run.sh LoRAT dinov2 --output_dir /path/to/output --mixin got10k --mixin evaluation --weight_path /path/to/weight.bin
```

Note that, as defined in ```config/LoRAT/_mixin/got10k.yaml```, we evaluate GOT-10k dataset three times.

## VOT toolkit integration
### Install VOT toolkit
```shell
pip install vot-toolkit
```
### Download VOT dataset
prepare the VOT dataset by running the following command:
```shell
cd /path/to/vot_workspace
vot initialize vot_stack(vots2024/main|tests/multiobject)
```
fill the path to the VOT dataset in ```consts.yaml```
```yaml
VOTS2023_PATH: '/path/to/vots2023_workspace/sequences'
VOT_TESTS_MULTIOBJECT_PATH: '/path/to/vot_tests_workspace/sequences'
```
### Run VOT experiments
```shell
# Run VOT experiment (vots2024/main stack) on LoRAT-g-378 with SAM-H segmentation model
python vot_main.py vots2024/main LoRAT dinov2 /path/to/output --mixin giant_378 --mixin segmentify_sam_h --tracker_name LoRAT  --weight_path /path/to/lorat_model_weight.bin
```

## Custom Dataset
[This page](DATASET.md) describes how to create a custom dataset for training and evaluation.

## Citation
```bibtex
@inproceedings{lorat,
  title={Tracking Meets LoRA: Faster Training, Larger Model, Stronger Performance},
  author={Lin, Liting and Fan, Heng and Zhang, Zhipeng and Wang, Yaowei and Xu, Yong and Ling, Haibin},
  booktitle={ECCV},
  year={2024}
}
