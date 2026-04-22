## SkyDrive: Learning to Drive in an New City with Drone-Based Vehicle Trajectories

TL; DR: We adapt autonomous driving models to a new city using massive vehicle trajectories from drone-captured videos.

## Implementation Idea

This project provide the preprocessing and analysis tool for the Songdo Traffic dataset, containing geo-referenced vehicle trajectories.

The codes in this repo are non-intrusive, the preprocessed dataset will be used for training and evaluation in Unitraj and RAP projects.

Unitraj is for vehicle trajectory prediction, and therefore making songdo dataset compatible with it is a strightforward idea (but with some engineering efforts).

RAP is trained with rasterized images from 3D bounding boxes. To construct such training data, we assume a flat ground and some common-sense height of vehicles, and then combine with the 2D planar trajectories and vehicle sizes to obtain a pseudo-3D world. Then we can render the rasterized obstacle view for a chosen ego vehicle.

## Recommended Environment Setup

The data preprocessing requires common python data processing libraries, and it's not needed after the initial processing. When the training job is actually running, we primarily need environments for unitraj and rap separately (not combined together).

So we can have a unitraj environment and a rap environment, which will have less dependency conflicts, and then install skydrive in both environments for the dataset loading. Skydrive itself has no heavy dependencies, and the dataset loading code is separate from the preprocessing code, so it should be fine to install skydrive in both environments.

UniTraj Setup

```

# create env for unitraj
conda create -n unitraj python=3.9

# install metadrive and scenarionet
# follow https://github.com/metadriverse/scenarionet

# clone unitraj repo
git clone https://github.com/vita-epfl/UniTraj.git
cd UniTraj

# make an empty init file because we need to import from unitraj
touch unitraj/__init__.py

# you may checkout the commit 9ee9a27 to have less dependencies
python -m pip install -r requirements.txt

# check that your pytorch has cuda later than your installed cuda toolkit
# otherwise there may be compatibility issues building MTR's dependencies
# as of March 2026, default installation pulls pytorch 2.8 with CUDA 12.6, which is fine.

# install unitraj
python -m pip install -e . --no-build-isolation

# if you are on a slurm based cluster, you may need something like the following
# this is because a login node may not have GCC, CUDA to build the CUDA-based KNN in MTR
module load gcc/13.2.0 cuda/12.4.1 && srun --qos debug --partition l40s --gres=gpu:1 python -m pip install -e . --no-build-isolation

# install skydrive 
cd SkyDrive
python -m pip install -e .
```

RAP setup takes longer time because of mmcv needs a source build, but it's usually fine ... 

On a Slurm-based cluster, you will need the slurm commands similar to the unitraj setup.

```
# create env for rap
conda create -n rap python=3.9
conda activate rap

# install nuplan devkit 
# https://nuplan-devkit.readthedocs.io/en/latest/installation.html#install-the-devkit
cd ~/git 
git clone https://github.com/motional/nuplan-devkit.git
cd nuplan-devkit
python -m pip install -e .

# install pytorch, adjust cuda version https://pytorch.org/get-started/previous-versions/
# don't use an old pytorch like 2.1.0, because the dinov3 backbone needs a new version of huggingface transformers,
python -m pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121 

# install huggingface transformers and wandb since they are not written in rap's requirements.txt
# you may need to pin the version of transformer like 
python -m conda pip install transformers wandb

# clone rap repo
https://github.com/vita-epfl/RAP.git
cd RAP
python -m pip install -e . --no-build-isolation

# install skydrive
cd SkyDrive
python -m pip install -e .
```

## Dataset

The dataset Songdo Traffic is placed at `datasets/Songdo_Traffic`, with a README file `datasets/Songdo_Traffic/README.md` describing the dataset in detail.

The dataset folder has the following contents:

* Monitoring sessions organized by date and intersection number, e.g.,`2022-10-04_A` is the monitoring session at intersection A on October 4, 2022.
* Each monitoring session folder contains a few CSV files, each corresponding to a monitoring session, e.g., `2022-10-04_A_AM1.csv` is the 1st monitoring session in the morning.
* Folder `master_frames` contains the BEV images of the intersection and annotations.

Terminology:

* Monitoring session: the vehicle trajectory data at a certain time period and intersection, contained in a CSV file like `2022-10-04_A_AM1.csv`.
* Frame: the vehicle locations at a certain timestamp within each monitoring session.
* Trajectory: the trajectory of a vehicle within a monitoring session, which is a sequence of locations over time.

## Data Pipeline

Effectively, this repo builds custom dataset for UniTraj and RAP.

The raw data is vehicle trajectories at multiple intersections, which is organized monitoring sessions of roughly 30 minutes. The overall idea is: create cut scenes from the long sessions, choose some vehicles as the ego vehicles, and train a model using Unitraj/RAP to predict their future trajectories.

### Preprocess

The preprocess step transforms the raw Songdo Traffic dataset into a compact format for autonomous driving, saved in `datasets/songdo_drive`

The core of the first part is in [process_songdo_traffic.py](./skydrive/preprocess/process_songdo_traffic.py), where `session_frames` are the vehicle trajectory data, and `veh_time_pairs` are the vehicles chosen to be ego-vehicles and the corresponding time of the scenes.

```bash
# generate train-test splits in a deterministic way using hash values
python skydrive/preprocess/generate_dataset_splits.py

# derive various map information using trajectory data and the original lane bounding boxes
python skydrive/preprocess/derive_map_info.py

# process the raw data into per-session scenes and candidate ego vehicles
python skydrive/preprocess/process_songdo_traffic.py
```

### Unitraj Cache-Building and Training

The pipeline for caching the dataset is divided to three parts:

[songdo_cache_builder.py](./skydrive/unitraj/songdo_cache_builder.py) is the entry point, and adapts the cache-buiding part of UniTraj's [base_dataset.py](https://github.com/vita-epfl/UniTraj/blob/main/unitraj/datasets/base_dataset.py)

[songdo_scene_loader.py](./skydrive/common/songdo_scene_loader.py) is responsible for reading the `session_frames `and corresponding `veh_time_pairs` of a monitoring session, and slice the long session into short scenes.

[unitraj_converter.py](./skydrive/unitraj/unitraj_converter.py) is responsible for converting a scene to the internal format required by unitraj (what's after the `preprocess` function of [base_dataset.py](https://github.com/vita-epfl/UniTraj/blob/main/unitraj/datasets/base_dataset.py)

[viz.py](./skydrive/unitraj/viz.py) has an example that converts scenes from raw data and animate them

**IMPORTANT**: Training a model on a cache built for another model is error prone. Because Some configurations are model-dependent, e.g., the numbers of considered nearby vehicles.

```plaintext
# Open a Terminal in SkyDrive repo root, build cache 
python skydrive/unitraj/songdo_cache_builder.py --method autobot \
        --train-data-path ./datasets/songdo_drive/train \
        --val-data-path ./datasets/songdo_drive/test \
        --unitraj-config-dir ../UniTraj/unitraj/configs \
		--cache-path ./cache/autobot \
        --overwrite-cache True

# Open a terminal in UniTraj repo directory
# create a symbolic link to the generated cache folder under `unitraj`
ln -s SKY_DRIVE_FOLDER/cache unitraj/cache

# run training
cd unitraj
python train.py 'train_data_path=[ ../storage/songdo_drive/train ]' \
	'val_data_path=[ ../storage/songdo_drive/test ]' \
	cache_path=./cache/autobot \
	use_cache=True \
	method.max_epochs=3 \
	load_num_workers=4 \
	exp_name=debug
```

### RAP Cache-Building and Training

The current Songdo-to-RAP adaptation is divided to five parts:

[songdo_rap_cache_builder.py](./skydrive/rap/songdo_rap_cache_builder.py) is the entry point, and writes the Songdo scenes directly into the cache layout consumed by RAP's `CacheOnlyDataset`.

[songdo_scene_loader.py](./skydrive/common/songdo_scene_loader.py) is responsible for reading the `session_frames` and corresponding `veh_time_pairs` of a monitoring session, and slice the long session into short scenes.

[songdo_rap_converter.py](./skydrive/rap/songdo_rap_converter.py) is responsible for converting a Songdo ego scene into the RAP feature and target dictionaries. It downsamples the 30 Hz scene into the 2 Hz history and future required by RAP, converts everything into the ego-local frame, renders synthetic camera views, and packs the tensors saved into `rap_feature.gz` and `rap_target.gz`.

[img_renderer.py](./skydrive/rap/img_renderer.py) contains the raster image renderer used to synthesize the camera views from trajectory and map geometry.

[visualize_songdo_scene_render.py](./skydrive/rap/visualize_songdo_scene_render.py) has an example that reads one Songdo scene and plots the BEV view together with the rendered camera images for multiple frames.

**IMPORTANT**: The current RAP adaptation only builds the main training cache. It does **not** build the perturbed cache, the auxiliary image cache, the PDM metric cache, or the evaluation cache. Therefore the first intended training setup is the reduced-feature version with `agent.config.pdm_scorer=False` and `agent.config.distill_feature=False`.

**IMPORTANT**: RAP's current `run_training.py` still assumes the auxiliary caches exist when `use_cache_without_dataset=True`. So Songdo training is **not** plug-and-play yet on the RAP side. The cache-building pipeline below is implemented. The training command below is the intended minimal RAP setup **after** adapting RAP's training script to train only from the main cache.

```plaintext
# Open a Terminal in SkyDrive repo root, build the main RAP cache
python skydrive/rap/songdo_rap_cache_builder.py \
        --train-data-path ./datasets/songdo_drive/train \
        --val-data-path ./datasets/songdo_drive/test \
        --cache-path ./cache/songdo_rap \
        --overwrite-cache \
        --num-workers 8

# Optional: visualize one ego scene and the rendered camera views
python skydrive/rap/visualize_songdo_scene_render.py \
        --split-name train \
        --session-filename 2022-10-05_K_AM1.csv \
        --scene-index 0 \
        --num-frames 3

# Open a terminal in RAP repo directory
# create a symbolic link to the generated cache folder under RAP
ln -s SKY_DRIVE_FOLDER/cache cache

# The cache builder also writes .songdo_rap_manifest.json under the cache root.
# Use its val_logs list when training with use_cache_without_dataset=True.
#
# After adapting RAP's run_training.py so it only loads the main cache,
# run the reduced-feature Songdo training:
cd ../RAP
python navsim/planning/script/run_training.py \
        agent=rap_agent \
        dataset=navsim_dataset \
        use_cache_without_dataset=True \
        force_cache_computation=False \
        cache_path=./cache/songdo_rap \
        agent.config.pdm_scorer=False \
        agent.config.distill_feature=False \
        agent.config.trajectory_sampling.time_horizon=5 \
        split=trainval \
        train_test_split=navtrain \
        "val_logs=[REPLACE_WITH_VAL_LOGS_FROM_.songdo_rap_manifest.json]" \
        dataloader.params.batch_size=16 \
        experiment_name=songdo_rap_debug
```
