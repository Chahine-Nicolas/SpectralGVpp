#!/bin/bash                                                      
#SBATCH --partition=gpu_p13                                                 
#SBATCH --nodes=1                                                               
#SBATCH --ntasks-per-node=1    
#SBATCH --qos=qos_gpu-t3                 
#SBATCH --time=20:00:00                                 
#SBATCH --gres=gpu:1                                                           
#SBATCH --cpus-per-task=4                                                    
#SBATCH --hint nomultithread                                                    
#SBATCH --mail-user chahine-nicolas.zede@ign.fr                          
#SBATCH --job-name=test_demo_registration_v4_splits_4_2m       # new                        
#SBATCH --output=file_job_test_demo_registration_v4_splits_4_2m.out                      # new                        
#SBATCH --error=file_job_test_demo_registration_v4_splits_4_2m.err                        

# chargement des modules                                                        
module purge                        # new                                       
module load tensorflow-gpu/py3/1.12

export PYTHONUSERBASE=/lustre/fsn1/worksf/projects/rech/dki/ujo91el/envs/d3feat
export MASTER_NODE=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1)
 
export 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512'

python demo_registration_v4_without_ply_west.py \
    --dataset_root /lustre/fsn1/worksf/projects/rech/dki/ujo91el/datas/lidarhd_v3/ \
    --path_model_D3Feat SpectralGV_D3Feat/models/D3Feat \
    --dataset_type lidar_west \
    --voxel_size 2 \
    --reset_fichier True \
    --file_rotation rotation
