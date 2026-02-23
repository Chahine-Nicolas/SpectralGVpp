import subprocess
from openpyxl import Workbook
import argparse
import os
import tqdm
import json
import numpy as np

def execute_command(name_pickle, root_path, file_path_query_json ,file_path_database_json, bin_path):  

    chemain_structure_donne = root_path

    json_file_path = os.path.join(chemain_structure_donne , file_path_query_json)
    json_file_path_database = os.path.join(chemain_structure_donne, file_path_database_json)
    
    # Lire les données du fichier JSON
    with open(json_file_path, 'r') as json_file:
        data = json.load(json_file)

    # Boucle for pour parcourir les éléments du JSON
    #for rotation in data['folders']:
    # Initialisation de la seed pour le random
    np.random.seed(42)

    command_creation_dataset = f"python {chemain_structure_donne}/SpectralGVpp/SpectralGV-main/datasets/lidar/generate_evaluation_sets.py --dataset_root {chemain_structure_donne}/SpectralGV_D3Feat/nuage_lidar/ --json_path_query {json_file_path} --json_path_database {json_file_path_database} --rel_lidar_path_query {bin_path} --rel_lidar_path_map {bin_path} --name_pickle {name_pickle}"


    subprocess.run(command_creation_dataset, shell=True, check=True, text=True )

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
    parser = argparse.ArgumentParser(description='Program to perform rotation on the requested data.')

    parser.add_argument('--name_pickle', type=str, default='file_traitment.pickle', help='Name of the pickle file to be processed.')
    parser.add_argument('--root_path', type=str, default='', help='root pah of both ')
    parser.add_argument('--file_path_query_json', type=str, default='SpectralGV_D3Feat/nuage_lidar/lidar_hd_v2/query_small_full_eval_list.json', help='')
    parser.add_argument('--file_path_database_json', type=str, default='SpectralGV_D3Feat/nuage_lidar/lidar_hd_v2/dataset_small_full_eval_list.json', help='Path to the database JSON file.')
    parser.add_argument('--bin_path', type=str, default='SpectralGV_D3Feat/nuage_lidar/lidar_hd_v2/bin', help='')


    args = parser.parse_args()

    execute_command(args.name_pickle, args.root_path, args.file_path_query_json,args.file_path_database_json,args.bin_path)


