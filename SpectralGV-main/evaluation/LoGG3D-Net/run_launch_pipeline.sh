python -m pdb lancement_pipilene_rotation_zede.py \

    --file_rotation rotation \
    --name_descripteur_global descripteur_global_SGV_lidar_0_1_rotation \
    --d_dist 10 \
    --root_json_index json_index.json \
    --D3Feat_util True \
    --MEAN_SHIFT_p True \
    --file_reset True \
    --name_pickle lidarhd_v2.pickle \
    --top_k 10 \
    --model logg3d \
    --voxel_size 2 \
    --rayon 3 \
    --name_file_exel exel_donner.xlsx \
    --bandwidth 2 \
    --weights /lustre/fswork/projects/rech/dki/ujo91el/code/tool_lidar_hd/LoGG3D-Net/training/checkpoints/2025-06-23_11-22-13_run_0_4 \
    --dataset_root /lustre/fsn1/worksf/projects/rech/dki/ujo91el/datas/lidarhd_v2/bin \
    --visualisation False