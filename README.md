## Overview

![plot](https://github.com/Chahine-Nicolas/SpectralGVpp/rerank2-1.pdf?raw=true)

This code derivates from the repository https://github.com/Deneuvi/SpectralGV_D3Feat/tree/main

# Step 1 : Dataset preparation

module load pytorch-gpu/py3/1.10.1

cd /SpectralGVpp/SpectralGV-main

```highlight
source run_save_pickle_east
```
--> Saving evaluation pickle: /lustre/fswork/projects/rech/dki/ujo91el/code/SpectralGV_D3Feat/SpectralGV-main/datasets/lidar/lidarhd_v2.pickle

```highlight
source run_save_pickle_west
```
--> Saving evaluation pickle: /lustre/fswork/projects/rech/dki/ujo91el/code/SpectralGV_D3Feat/SpectralGV-main/datasets/lidar/lidarhd_v3.pickle

# Step 2 : D3Feat features

cd /SpectralGVpp/SpectralGV-main/D3feat_features

module load tensorflow-gpu/py3/1.12

```highlight
source run_d3feat.sh
```
--> Save D3Feat features 

# Step 3 : Re-ranking

cd /SpectralGVpp/SpectralGV-main/evaluation/LoGG3D-Net

module load pytorch-gpu/py3/1.10.1

```highlight
source run_launch_pipeline.sh
```
