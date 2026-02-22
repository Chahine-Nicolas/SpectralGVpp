import subprocess
from openpyxl import Workbook
import argparse
import os
import tqdm
import json
import numpy as np

def execute_command(file_rotation,phi_bool,theta_bool,gamma_bool ,value_gamma,name_pickle ,file_path_query_json ,json_path_database, file_path_data , reset_donne):  
    chemain_structure = os.getenv("SCRATCH")
    chemain_structure_donne = os.getenv("WORK")

    chemain_rotation_bin = os.path.join(chemain_structure_donne ,os.path.dirname(file_path_data) ,f"{file_rotation}/bin")
    print(chemain_rotation_bin)


    os.makedirs(chemain_rotation_bin, exist_ok=True)
    fichiers = os.listdir(chemain_rotation_bin)

    if reset_donne == True :
        for fichier in fichiers:
            chemin_complet = os.path.join(chemain_rotation_bin, fichier)
            
            try:
                os.remove(chemin_complet)
                
                print(f"Le fichier {chemin_complet} a été supprimé avec succès.")
            except FileNotFoundError:
                print(f"Le fichier {chemin_complet} n'a pas été trouvé.")
            except Exception as e:
                print(f"Une erreur est survenue lors de la suppression du fichier {fichier}: {e}")
        
    json_file_path = os.path.join(chemain_structure_donne , file_path_query_json)
    json_file_path_database = os.path.join(chemain_structure_donne, json_path_database)
    
    # Lire les données du fichier JSON
    with open(json_file_path, 'r') as json_file:
        data = json.load(json_file)

    # Boucle for pour parcourir les éléments du JSON
    #for rotation in data['folders']:
    # Initialisation de la seed pour le random
    np.random.seed(42)

   
    phi = 0
    theta = 0
    gamma = value_gamma
    calibration =[]
    nom_fichier =[]
    if reset_donne == True :
        """
        for rotation in tqdm.tqdm(data):
            if phi_bool == True :
                phi = np.random.uniform(0, 360)
                
            elif theta_bool == True :
                theta = np.random.uniform(0, 360)
      
            elif gamma_bool == True :
                gamma = np.random.uniform(0, 360)

                
            chemain_fichier_rotation = os.path.join(chemain_structure_donne,f'{file_path_data}' , os.path.basename(rotation))

            #command_rotation = f"python {chemain_structure_donne}/SpectralGV_D3Feat/nuage_lidar/rotation_fichier.py --name_files {chemain_fichier_rotation} --phi {phi} --theta {theta} --gamma {gamma} --file_rotation {file_rotation}" 
            command_rotation = f"python /lustre/fswork/projects/rech/dki/ujo91el/code/SpectralGV_D3Feat/nuage_lidar/rotation_fichier.py --name_files {chemain_fichier_rotation} --phi {phi} --theta {theta} --gamma {gamma} --file_rotation {file_rotation}" 

            
            result =subprocess.run(command_rotation, shell=True, check=True, text=True , capture_output=True)

            output = result.stdout.strip().split()

            # Séparation des valeurs
            rotation_file = list(map(float, output[0:3]))
            translation = list(map(float, output[3:6]))
            base_name = output[6]
            
            calibration.append([rotation_file,translation])
            nom_fichier.append(base_name)
        """
        # Construction du dictionnaire
        calibration_file = {}
        
        for i, nom_fichier in enumerate(nom_fichier):
            
            calibration_file[str(nom_fichier)] = {
                "rotation": calibration[i][0],
                "translation": calibration[i][1]
            }

        # Sauvegarde dans un fichier JSON
        with open(os.path.join(os.path.dirname(chemain_rotation_bin),f"{file_rotation}.json"), "w") as f:
            json.dump(calibration_file, f, indent=4)

        print("Fichier calibration_file.json créé.")
    
    
    #command_creation_dataset =f"python {chemain_structure_donne}/SpectralGV_D3Feat/SpectralGV-main/datasets/lidar/generate_evaluation_sets.py --dataset_root {chemain_structure_donne}/SpectralGV_D3Feat/nuage_lidar --json_path_query {json_file_path} --json_path_database {json_file_path_database} --rel_lidar_path_query lidar_hd_v2/{file_rotation}/bin --rel_lidar_path_map lidar_hd_v2/bin --name_pickle {name_pickle}"

    #import pdb; pdb.set_trace()
    command_creation_dataset =f"python /lustre/fswork/projects/rech/dki/ujo91el/code/SpectralGV_D3Feat/SpectralGV-main/datasets/lidar/generate_evaluation_sets.py --dataset_root /lustre/fswork/projects/rech/dki/ujo91el/code/SpectralGV_D3Feat/nuage_lidar/ --json_path_query {json_file_path} --json_path_database {json_file_path_database} --rel_lidar_path_query {file_path_data} --rel_lidar_path_map {file_path_data} --name_pickle {name_pickle}"


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
    parser.add_argument('--file_rotation', type=str, required=False, default='rotation',help='Path to the folder containing rotation data.')
    parser.add_argument('--phi', type=str2bool, required=False, default=False,   help='Enable rotation along the X-axis (phi angle).' )
    parser.add_argument('--theta', type=str2bool, required=False, default=False, help='Enable rotation along the Y-axis (theta angle).')
    parser.add_argument('--gamma', type=str2bool, required=False, default=False , help='Enable rotation along the Z-axis (gamma angle).')
    parser.add_argument('--value_gamma', type=int, required=False, default=0 , help='Enable rotation along the Z-axis (gamma angle).')

    parser.add_argument('--name_pickle', type=str, default='file_traitment.pickle', help='Name of the pickle file to be processed.')
    parser.add_argument('--file_path_query_json', type=str, default='SpectralGV_D3Feat/nuage_lidar/lidar_hd_v2/query_small_full_eval_list.json', help='')
    parser.add_argument('--json_path_database', type=str, default='SpectralGV_D3Feat/nuage_lidar/lidar_hd_v2/dataset_small_full_eval_list.json', help='Path to the database JSON file.')
    parser.add_argument('--file_path_data', type=str, default='SpectralGV_D3Feat/nuage_lidar/lidar_hd_v2/bin', help='')
    parser.add_argument('--reset_donne', type=str2bool, default=False, help='Reset the data to its initial state before processing.')


    args = parser.parse_args()


    execute_command( args.file_rotation , args.phi , args.theta ,args.gamma ,args.value_gamma, args.name_pickle ,args.file_path_query_json,args.json_path_database,args.file_path_data,args.reset_donne )


