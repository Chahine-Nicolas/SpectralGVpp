import subprocess
import argparse
from openpyxl import Workbook
import os
import csv
import tqdm
import json
import numpy as np


def execute_command(file_rotation,name_descripteur_global,name_file_visu,d_dist,root_json_index,D3Feat_util ,MEAN_SHIFT_p ,file_reset ,name_pickle, top_k ,min_num_feat,nb_clusteur, model ,voxel_size , name_file_exel ,bandwidth , weights ,visualisation ,utilisation_exel,rayon):
    """
    Prepares an Excel file for logging evaluation results and executes a Python script for evaluating the LoGG3D model.

    This function:
    1. Creates an Excel workbook with predefined sheets for logging evaluation results.
    2. Constructs and executes a command to run the LoGG3D evaluation script with the provided parameters.

    Args:
        file_rotation (str): Name of the folder containing rotation data for D3Feat.
        name_descripteur_global (str): Name of the global descriptor directory.
        d_dist (float): Descriptor distance threshold for re-ranking.
        D3Feat_util (bool): If True, uses D3Feat for local descriptor computation.
        MEAN_SHIFT_p (bool): If True, applies mean shift to refine keypoint positions.
        file_reset (bool): If True, recomputes and resets all embedding files.
        name_pickle (str): Name of the evaluation pickle file.
        top_k (int): Number of top-k nearest neighbors to consider during evaluation.
        model (str): Model architecture to use for evaluation (e.g., 'logg3d' or 'logg3d1k').
        voxel_size (float): Voxel size for point cloud downsampling.
        name_file_exel (str): Name of the Excel file to log evaluation results.
        bandwidth (float): Bandwidth parameter for mean shift clustering.

    Returns:
        None: The function saves an Excel file and executes the evaluation script.

    Output:
        - An Excel file with sheets for logging evaluation results.
        - Execution of the LoGG3D evaluation script with the provided parameters.

    Raises:
        subprocess.CalledProcessError: If the executed command returns a non-zero exit status.
    """

    data_structure_path = os.getenv("WORK")
    if utilisation_exel == True :
        # Créer le classeur Excel et les feuilles
        wb = Workbook()
        file_path =  os.path.join(os.path.dirname(os.path.abspath(__file__)),f"{name_file_exel}")

        # Noms des feuilles
        names_of_sheets = ["nom_fichier","nom_map","euclid_dist", "topk_rerank", "topk_rerank_inds", "euclid_dist_rr","rotation"]
        entetes = [f"top{i}" for i in range(1, top_k + 1)]

        # Créer chaque feuille avec en-têtes
        for nom in names_of_sheets:
            ws = wb.create_sheet(title=nom)
            ws.append([nom])
            if nom != "nom_fichier":
                ws.append(entetes)

        # Supprimer la feuille par défaut
        del wb["Sheet"]
        

        # Sauvegarder le fichier vide initialisé
    
        wb.save(file_path)
    else :
        names_of_sheets = ["nom_fichier", "nom_map", "euclid_dist", "topk_rerank",
                        "topk_rerank_inds", "euclid_dist_rr", "rotation"]
        entetes = [f"top{i}" for i in range(1, top_k + 1)]

        base_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(base_dir, os.path.splitext(name_file_exel)[0])
        os.makedirs(file_path, exist_ok=True)

        # Créer un CSV vide par "feuille"
        for sheet in names_of_sheets:

            file_csv = os.path.join(file_path, f"{sheet}.csv")
            with open(file_csv, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if sheet == "nom_fichier":
                    writer.writerow([sheet])
                else:
                    writer.writerow(entetes)

    


    #command = f"python SpectralGV_D3Feat/SpectralGV-main/evaluation/LoGG3D-Net/eval_logg3d_sgv_gencer_rotation_json.py --dataset_type lidar --dataset_root {data_structure_path}/SpectralGV_D3Feat/nuage_lidar --weights {weights} --model {model} --root_json_index {root_json_index}"

    print("RUNNING FILE:", os.path.abspath(__file__))

    command = f"python eval_logg3d_sgv_gencer_rotation_json_zede_edited.py --dataset_type lidar --dataset_root /lustre/fsn1/worksf/projects/rech/dki/ujo91el/datas/lidarhd_v2/bin --weights {weights} --model {model} --root_json_index {root_json_index}"
        
    command = command +f" --d_thresh {d_dist} --n_topk {top_k} --min_num_feat {min_num_feat} --nb_clusteur {nb_clusteur} --voxel_size {voxel_size} --D3Feat_util {D3Feat_util} --MEAN_SHIFT_p {MEAN_SHIFT_p} --nom_fichier_exel {file_path} --name_descripteur_global {name_descripteur_global} --name_file_visu {name_file_visu} --reset_fichier {file_reset} --eval_set {name_pickle} --file_rotation {file_rotation} --bandwidth {bandwidth} --visualisation {visualisation} --rayon {rayon} --utilisation_exel {utilisation_exel} "
    print(f"Exécution de la commande: {command}")
        
    try:
        # Exécute la commande et attend sa fin
        print("#############################")
        subprocess.run(command, shell=True, check=True, text=True)
        
    except subprocess.CalledProcessError as e:
        print("Erreur:", e.stderr)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate point cloud descriptors with rotation support.')

    parser.add_argument('--file_rotation', type=str, default='rotation', help='Folder name for rotation data (default: rotation)')
    parser.add_argument('--name_descripteur_global', type=str, default='descripteur_global_SGV_lidar_0_1_rotation', help='Directory for global descriptors (default: descripteur_global_SGV_lidar_0_1_rotation)')
    parser.add_argument('--name_file_visu', type=str, default='image_nuage_D3feat', help='name file for visualization images. Default: "image_nuage_D3feat".')    
    parser.add_argument('--d_dist', type=float, default=10, help='Descriptor distance threshold for matching (default: 10.0)')
    parser.add_argument('--root_json_index', type=str, required=False, default="?", help='Path to the JSON index file containing metadata for the dataset. Default: "json_index.json".')
    parser.add_argument('--D3Feat_util', type=str2bool, default=False, help='Use D3Feat for local descriptors (default: False)')
    parser.add_argument('--MEAN_SHIFT_p', type=str2bool, default=False, help='Apply mean shift to keypoints (default: False)')
    parser.add_argument('--file_reset', type=str2bool, default=False, help='Force recompute all embeddings (default: False)')
    parser.add_argument('--visualisation', type=str2bool, default=False, help='If True, enables visualization of keypoints and/or intermediate results. Default: False.')
    parser.add_argument('--utilisation_exel', type=str2bool, default=True, help='If True, logs evaluation results to an Excel file. Default: False.')
    parser.add_argument('--name_pickle', type=str, default="?", help='Evaluation set pickle filename (default: lidar_5_zone_rotation.pickle)')
    parser.add_argument('--top_k', type=int, default=10, help='Number of top matches to consider (default: 10)')
    parser.add_argument('--min_num_feat', type=int, default=15000, help='Minimum number of features required to perform re-ranking. Default: 15000.')
    parser.add_argument('--nb_clusteur', type=int, default=50, help='Number of clusters for the weighted K-Means algorithm used in keypoint selection.')
    parser.add_argument('--model', type=str, default='logg3d', choices=['logg3d', 'logg3d1k'], help='Model architecture to use for evaluation. Default: "logg3d1k".')
    parser.add_argument('--voxel_size', type=float, default=0.1,help='Voxel size for downsampling (default: 0.1)')
    parser.add_argument('--rayon', type=float, default=3, help='Search radius around D3Feat points for local neighborhood matching (default: 3)')
    parser.add_argument('--name_file_exel', type=str, default='donner.xlsx', help='Excel filename for results (default: exel_donner.xlsx)')
    parser.add_argument('--bandwidth', type=int, default=2, help='MeanShift bandwidth parameter (default: 2)')
    parser.add_argument('--weights', type=str, default='/2025-05-25_14-12-46_run_0_3', help='Path to model weights directory (default: /2025-05-25_14-12-46_run_0_3)')

    args = parser.parse_args()
    
    execute_command(args.file_rotation , args.name_descripteur_global ,args.name_file_visu, args.d_dist,args.root_json_index, args.D3Feat_util , args.MEAN_SHIFT_p ,args.file_reset ,args.name_pickle ,args.top_k ,args.min_num_feat, args.nb_clusteur, args.model,args.voxel_size ,args.name_file_exel , args.bandwidth , args.weights ,args.visualisation,args.utilisation_exel, args.rayon)

