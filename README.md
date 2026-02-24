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

# Step 2 : Saves D3Feat features

cd /SpectralGVpp/SpectralGV-main/D3Feat_modif_1_laurent_1_moi

module load tensorflow-gpu/py3/1.12

```highlight
source d3feat_debug.sh
```
--> Save D3Feat features 

# Step 3 : Re-ranking

cd /SpectralGVpp/SpectralGV-main/evaluation/LoGG3D-Net

module load pytorch-gpu/py3/1.10.1

```highlight
source run_launch_pipeline.sh
```
