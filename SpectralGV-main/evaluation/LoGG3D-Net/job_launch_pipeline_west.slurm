#!/bin/bash                                                      
#SBATCH --partition=gpu_p13                                                 
#SBATCH --nodes=1                                                               
#SBATCH --ntasks-per-node=1    
#SBATCH --qos=qos_gpu-t4                 
#SBATCH --time=15:00:00                                 
#SBATCH --gres=gpu:1                                                           
#SBATCH --cpus-per-task=16                                                    
#SBATCH --hint nomultithread                                                    
#SBATCH --mail-user chahine-nicolas.zede@ign.fr                          
#SBATCH --job-name=test_launch_d3featv3        # new                        
#SBATCH --output=file_job_test_launch_d3featv3.out                      # new                        
#SBATCH --error=file_job_test_launch_d3featv3.err                        

# chargement des modules                                                        
module purge                        # new                                       
module load pytorch-gpu/py3/1.10.1

export PYTHONUSERBASE=/lustre/fsn1/worksf/projects/rech/dki/ujo91el/envs/d3feat
export MASTER_NODE=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1)
 
export 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512'

python lancement_pipilene_rotation_zede.py \
    --file_rotation rotation \
    --name_descripteur_global descripteur_global_SGV_lidar_0_1_rotation \
    --d_dist 10 \
    --root_json_index json_index.json \
    --D3Feat_util True \
    --MEAN_SHIFT_p True \
    --file_reset True \
    --name_pickle lidarhd_v3.pickle \
    --top_k 10 \
    --model logg3d \
    --voxel_size 0.1 \
    --rayon 3 \
    --name_file_exel exel_donner.xlsx \
    --bandwidth 2 \
    --weights /lustre/fswork/projects/rech/dki/ujo91el/code/tool_lidar_hd/LoGG3D-Net/training/checkpoints/2025-06-23_11-22-13_run_0_4 \
    --visualisation False
