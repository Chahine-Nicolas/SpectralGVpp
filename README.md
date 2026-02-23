#######################################################
# Step 1
#######################################################
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
#######################################################


#######################################################
# Step 2
#######################################################

/lustre/fswork/projects/rech/dki/ujo91el/code/SpectralGV_D3Feat/SpectralGV-main/D3Feat_modif_1_laurent_1_moi

module load tensorflow-gpu/py3/1.12

export PYTHONUSERBASE=/lustre/fsn1/worksf/projects/rech/dki/ujo91el/envs/d3feat

source d3feat_debug.sh


--> Saving local desc to: '/lustre/fsn1/projects/rech/dki/ujo91el/descripteur_D3Feat/rotation/LHD_FXX_0656_6860_PTS_O_LAMB93_IGN69.copc_10_10_48.npz'

#######################################################
# Step 3
#######################################################

cd /lustre/fswork/projects/rech/dki/ujo91el/code/SpectralGV_D3Feat/SpectralGV-main/evaluation/LoGG3D-Net

salloc --nodes=1 --ntasks-per-node=1 --partition gpu_p13 --time=04:00:00 --gres gpu:1 --cpus-per-task 4 --hint nomultithread --mail-user chahine-nicolas.zede@ign.fr

srun --ntasks=1 --cpus-per-task=4 --gres=gpu:1 --hint=nomultithread --time=04:00:00 --pty bash

module load pytorch-gpu/py3/1.10.1

export PYTHONUSERBASE=/lustre/fsn1/worksf/projects/rech/dki/ujo91el/envs/d3feat

source run_launch_pipeline.sh
