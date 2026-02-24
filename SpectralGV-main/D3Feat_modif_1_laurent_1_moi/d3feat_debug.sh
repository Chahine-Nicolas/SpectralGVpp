python -m pdb demo_registration_v4_without_ply.py \
    --dataset_root /lustre/fsn1/worksf/projects/rech/dki/ujo91el/datas/lidarhd_v2/ \
    --path_model_D3Feat SpectralGV_D3Feat/models/D3Feat \
    --dataset_type lidar_east \
    --eval_set_filepath /lustre/fswork/projects/rech/dki/ujo91el/code/SpectralGV_D3Feat/SpectralGV-main/datasets/lidar/lidarhd_v2.pickle \
    --voxel_size 2 \
    --reset_fichier False \
    --folder_rotation /lustre/fsn1/projects/rech/dki/ujo91el/descripteur_D3Feat/lidarhd_v2_2m \
    --dictio_path /lustre/fsn1/worksf/projects/rech/dki/ujo91el/datas/lidarhd_v2/splits/dsi_eval_list_part4.json