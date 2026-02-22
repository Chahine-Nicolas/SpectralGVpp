#!/bin/bash
#SBATCH --partition=gpu_p13
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=20:00:00
#SBATCH --hint=nomultithread
#SBATCH --mail-user chahine-nicolas.zede@ign.fr                          
#SBATCH --job-name=test_launch_d3feat_v2        # new                        
#SBATCH --output=file_job_test_launch_d3featv2.out                      # new                        
#SBATCH --error=file_job_test_launch_d3featv2.err     

module purge
module load pytorch-gpu/py3/1.10.1

export PYTHONUSERBASE=/lustre/fsn1/worksf/projects/rech/dki/ujo91el/envs/d3feat
export MASTER_NODE=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1)

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512


srun python lancement_pipilene_rotation_zede.py \
    --file_rotation rotation \
    --name_descripteur_global descripteur_global_SGV_lidar_0_1_rotation \
    --d_dist 10 \
    --root_json_index json_index.json \
    --D3Feat_util True \
    --MEAN_SHIFT_p True \
    --file_reset True \
    --name_pickle lidar_eval_test.pickle \
    --top_k 10 \
    --model logg3d \
    --voxel_size 0.1 \
    --rayon 3 \
    --name_file_exel exel_donner.xlsx \
    --bandwidth 2 \
    --weights checkpoints/2025-05-25_14-12-46_run_0_3 \
    --visualisation False
