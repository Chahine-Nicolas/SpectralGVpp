import subprocess
from openpyxl import Workbook
import argparse
import os

def execute_command(name_descripteur_global,d_dist,root_json_index,D3Feat_util ,MEAN_SHIFT_p ,file_reset ,name_pickle, top_k , model ,voxel_size , name_file_exel ,bandwidth ,weights):

    # Créer le classeur Excel et les feuilles
    wb = Workbook()
    file_path =  os.path.join(os.path.dirname(os.path.abspath(__file__)),f"{name_file_exel}")

    # Noms des feuilles
    names_of_sheets = ["nom_fichier","nom_map","euclid_dist", "topk_rerank", "topk_rerank_inds", "euclid_dist_rr"]
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
    
    data_structure_path = os.getenv("WORK")
    print(root_json_index)
    command = f"python SpectralGV_D3Feat/SpectralGV-main/evaluation/LoGG3D-Net/eval_logg3d_sgv_gencer_rotation_json.py --dataset_type lidar --dataset_root {data_structure_path}/SpectralGV_D3Feat/nuage_lidar --weights {weights} --model {model} --root_json_index {root_json_index} "
        
    command = command +f" --d_thresh {d_dist} --n_topk {top_k} --voxel_size {voxel_size} --D3Feat_util {D3Feat_util} --MEAN_SHIFT_p {MEAN_SHIFT_p} --nom_fichier_exel {file_path} --name_descripteur_global {name_descripteur_global} --reset_fichier {file_reset} --eval_set {name_pickle} "  
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

    parser.add_argument('--name_descripteur_global', type=str, default='descripteur_global_SGV_lidar_0_1_rotation', help='Directory for global descriptors (default: descripteur_global_SGV_lidar_0_1_rotation)')
    parser.add_argument('--d_dist', type=float, default=10, help='Descriptor distance threshold for matching (default: 10.0)')
    parser.add_argument('--root_json_index', type=str, required=False, default="?", help='Path to the JSON index file containing metadata for the dataset. Default: "json_index.json".')
    parser.add_argument('--D3Feat_util', type=str2bool, default=False, help='Use D3Feat for local descriptors (default: False)')
    parser.add_argument('--MEAN_SHIFT_p', type=str2bool, default=False, help='Apply mean shift to keypoints (default: False)')
    parser.add_argument('--file_reset', type=str2bool, default=False, help='Force recompute all embeddings (default: False)')
    parser.add_argument('--name_pickle', type=str, default="?", help='Evaluation set pickle filename (default: lidar_5_zone_rotation.pickle)')
    parser.add_argument('--top_k', type=int, default=10, help='Number of top matches to consider (default: 10)')
    parser.add_argument('--model', type=str, default='logg3d', choices=['logg3d', 'logg3d1k'], help='Model architecture to use for evaluation. Default: "logg3d1k".')
    parser.add_argument('--voxel_size', type=float, default=0.1,help='Voxel size for downsampling (default: 0.1)')
    parser.add_argument('--name_file_exel', type=str, default='exel_donner.xlsx', help='Excel filename for results (default: exel_donner.xlsx)')
    parser.add_argument('--bandwidth', type=int, default=2, help='MeanShift bandwidth parameter (default: 2)')
    parser.add_argument('--weights', type=str, default='/2025-05-25_14-12-46_run_0_3', help='Path to model weights directory (default: /2025-05-25_14-12-46_run_0_3)')


    args = parser.parse_args()
    
    execute_command(args.name_descripteur_global , args.d_dist,args.root_json_index, args.D3Feat_util , args.MEAN_SHIFT_p ,args.file_reset ,args.name_pickle ,args.top_k , args.model,args.voxel_size ,args.name_file_exel , args.bandwidth ,args.weights)


