python -m pdb demo_registration_v4_without_ply.py \
    --dataset_root /lustre/fsn1/worksf/projects/rech/dki/ujo91el/datas/lidarhd_v2/ \
    --path_model_D3Feat SpectralGV_D3Feat/models/D3Feat \
    --dataset_type lidar_east \
    --eval_set_filepath /lustre/fswork/projects/rech/dki/ujo91el/code/SpectralGV_D3Feat/SpectralGV-main/datasets/lidar/lidarhd_v2.pickle \
    --voxel_size 2 \
    --reset_fichier False \
    --file_rotation rotation