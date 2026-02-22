# This script is adapted from: https://github.com/jac99/Egonn/blob/main/eval/evaluate.py

import argparse
import numpy as np
import tqdm
import os
import sys
import csv
from scipy.spatial.distance import cdist
import random
from typing import List
import open3d as o3d
from time import time
import copy
import torch
from sklearn.cluster import KMeans
from torchsparse import SparseTensor
from itertools import repeat
from typing import List, Tuple, Union
from torchsparse.utils.collate import sparse_collate
import pickle
import shutil
from sklearn.cluster import MeanShift
import json
from openpyxl import load_workbook



from models.model_factory import model_factory

import sgv_utils 
import sgv_utils_visualisation as sgv_visu

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from datasets.poses_utils import relative_pose, apply_transform
# from datasets.mulran.utils import relative_pose as mulran_relative_pose
# from datasets.kitti.utils import get_relative_pose as kitti_relative_pose
# from datasets.kitti360.utils import kitti360_relative_pose
from datasets.lidar.utils import get_relative_pose as lidar_relative_pose
from datasets.point_clouds_utils import icp, make_open3d_feature, make_open3d_point_cloud, preprocess_pointcloud
from datasets.base_datasets import EvaluationTuple, EvaluationSet, get_pointcloud_loader



print('\n' + ' '.join([sys.executable] + sys.argv))

class Evaluator:
    """
    Class for evaluating models on a point cloud dataset.
    Handles embedding computation, nearest neighbor search, and recall calculation.
    """
    def __init__(self, dataset_root: str, dataset_type: str, eval_set_pickle: str, device: str, json_index : str , 
                 radius: List[float] = (5, 20), k: int = 50, n_samples: int =None, debug: bool = False):
        """
        Initializes the Evaluator with dataset and evaluation parameters.

        Args:
            dataset_root (str): Root path to the dataset.
            dataset_type (str): Type of the dataset (e.g., 'mulran', 'southbay').
            eval_set_pickle (str): Path to the evaluation set pickle file.
            device (str): Device to run the model on (e.g., 'cpu', 'cuda').
            json_index (str): Path to the JSON index file.
            radius (List[float]): List of thresholds (in meters) to consider an element from the map sequence a true positive.
            k (int): Maximum number of nearest neighbors to consider.
            n_samples (int, optional): Number of samples taken from a query sequence (None=all query elements).
            debug (bool): If True, uses a small subset for debugging.
        """
        assert os.path.exists(dataset_root), f"Cannot access dataset root: {dataset_root}"
        self.dataset_root = dataset_root
        self.dataset_type = dataset_type
        
        self.json_index = json_index
        #self.eval_set_filepath = os.path.join(os.path.dirname(__file__), '../../datasets/',self.dataset_type, eval_set_pickle)
        self.eval_set_filepath = "/lustre/fswork/projects/rech/dki/ujo91el/code/SpectralGV_D3Feat/SpectralGV-main/datasets/lidar/lidarhd_v3.pickle"
        self.json_set_filepath =  os.path.join(os.path.dirname(__file__),json_index)
        self.device = device
        self.radius = radius
        self.k = k
        self.n_samples = n_samples
        self.debug = debug

        assert os.path.exists(self.eval_set_filepath), f'Cannot access evaluation set pickle: {self.eval_set_filepath}'
        self.eval_set = EvaluationSet()
        self.eval_set.load(self.eval_set_filepath)
        
        ############ modifier ################
        if self.json_index != "?" : 
            assert os.path.exists(self.json_set_filepath)
            with open(self.json_set_filepath, 'r', encoding='utf-8') as f:
                self.data_json_index = json.load(f)

                
            
        
        
        if debug:
            # Make the same map set and query set in debug mdoe
            self.eval_set.map_set = self.eval_set.map_set[:4]
            self.eval_set.query_set = self.eval_set.map_set[:4]
        


        if n_samples is None or len(self.eval_set.query_set) <= n_samples:
            self.n_samples = len(self.eval_set.query_set)
        else:
            self.n_samples = n_samples

        self.pc_loader = get_pointcloud_loader(self.dataset_type)


    def evaluate(self, model, *args, **kwargs):
        """
        Evaluates the model by computing embeddings, performing nearest neighbor search, and calculating recall.

        Args:
            model: The model to evaluate.
            *args: Additional positional arguments for model inference.
            **kwargs: Additional keyword arguments for model inference.

        Returns:
            dict: Dictionary containing recall values for each radius and k.
        """    

        map_embeddings = self.compute_embeddings(self.eval_set.map_set, model)

        query_embeddings = self.compute_embeddings(self.eval_set.query_set, model)

        map_positions = self.eval_set.get_map_positions()
        query_positions = self.eval_set.get_query_positions()
        
        
        # Dictionary to store the number of true positives for different radius and NN number
        tp = {r: [0] * self.k for r in self.radius}
        query_indexes = random.sample(range(len(query_embeddings)), self.n_samples)

        # Randomly sample n_samples clouds from the query sequence and NN search in the target sequence
        for query_ndx in tqdm.tqdm(query_indexes):
            # Check if the query element has a true match within each radius
            query_pos = query_positions[query_ndx]

            # Nearest neighbour search in the embedding space
            query_embedding = query_embeddings[query_ndx]
            embed_dist = np.linalg.norm(map_embeddings - query_embedding, axis=1)
            nn_ndx = np.argsort(embed_dist)[:self.k]

            # Euclidean distance between the query and nn
            delta = query_pos - map_positions[nn_ndx]  # (k, 2) array
            euclid_dist = np.linalg.norm(delta, axis=1)  # (k,) array
            # Count true positives for different radius and NN number
            tp = {r: [tp[r][nn] + (1 if (euclid_dist[:nn + 1] <= r).any() else 0) for nn in range(self.k)] for r in
                  self.radius}

        recall = {r: [tp[r][nn] / self.n_samples for nn in range(self.k)] for r in self.radius}
        # percentage of 'positive' queries (with at least one match in the map sequence within given radius)
        return {'recall': recall}

    def compute_embedding(self, pc, model, *args, **kwargs):
        # This method must be implemented in inheriting classes
        # Must return embedding as a numpy vector
        raise NotImplementedError('Not implemented')

    def model2eval(self, model):
        # This method may be overloaded when model is a tuple consisting of a few models (as in Disco)
        model.eval()

    def compute_embeddings(self, eval_subset: List[EvaluationTuple], model, *args, **kwargs):
        """
        Computes embeddings for a subset of the evaluation set.

        Args:
            eval_subset (List[EvaluationTuple]): Subset of the evaluation set.
            model: The model to use for embedding.
            *args: Additional positional arguments for model inference.
            **kwargs: Additional keyword arguments for model inference.

        Returns:
            np.ndarray: Array of embeddings for the subset.
        """
        self.model2eval(model)

        embeddings = None
        for ndx, e in tqdm.tqdm(enumerate(eval_subset)):
            scan_filepath = os.path.join(self.dataset_root, e.rel_scan_filepath)
            print("scan_filepath", scan_filepath)
            assert os.path.exists(scan_filepath)
            pc = self.pc_loader(scan_filepath)
           
            pc = torch.tensor(pc)

            embedding = self.compute_embedding(pc, model)
            if embeddings is None:
                embeddings = np.zeros((len(eval_subset), embedding.shape[1]), dtype=embedding.dtype)
            embeddings[ndx] = embedding

        return embeddings

def euclidean_distance(query, database):
    return torch.cdist(torch.tensor(query).unsqueeze(0).unsqueeze(0), torch.tensor(database).unsqueeze(0)).squeeze().numpy()

class MetLocEvaluator(Evaluator):
    """
    Evaluator class for MetLoc, extending the base Evaluator with specific parameters and logic for global/local descriptor evaluation.
    """
    def __init__(self, dataset_root: str, dataset_type: str, eval_set_pickle: str, device: str, root_json_index : str ,
                 radius: List[float], k: int = 20, n_samples=None, repeat_dist_th: float = 0.5, voxel_size: float = 0.1,
                 icp_refine: bool = True, debug: bool = False):
        """
        Initializes the MetLocEvaluator with dataset, evaluation parameters, and model settings.

        Args:
            dataset_root (str): Root directory of the dataset.
            dataset_type (str): Type of the dataset (e.g., 'mulran', 'southbay').
            eval_set_pickle (str): Path to the evaluation set pickle file.
            device (str): Device to run the model on (e.g., 'cpu', 'cuda').
            root_json_index (str): Path to the JSON index file.
            radius (List[float]): List of radius thresholds (in meters) for true positive consideration.
            k (int): Number of nearest neighbors to consider. Defaults to 20.
            n_samples (int, optional): Number of samples to use from the query set. Defaults to None (all samples).
            repeat_dist_th (float): Threshold for repeatability distance. Defaults to 0.5.
            voxel_size (float): Size of voxels for point cloud processing. Defaults to 0.1.
            icp_refine (bool): If True, enables ICP-based pose refinement. Defaults to True.
            debug (bool): If True, enables debug mode with a small subset. Defaults to False.
        """
        super().__init__(dataset_root, dataset_type, eval_set_pickle, device,root_json_index, radius, k, n_samples, debug=debug)
        self.repeat_dist_th = repeat_dist_th
        self.icp_refine = icp_refine
        self.voxel_size = voxel_size
        

    def model2eval(self, models):
        """
        Sets all models in the input tuple to evaluation mode.

        Args:
            models (tuple): Tuple of models to be set to evaluation mode.
        """
        # This method may be overloaded when model is a tuple consisting of a few models (as in Disco)
        [model.eval() for model in models]
        
    def evaluate(self, model, d_thresh,min_num_feat,MEAN_SHIFT_p ,path_model_D3Feat,D3Feat,name_file_exel,name_descripteur_global,name_file_visu,reset=False,file_rotation='',visualisation = False ,rayon =3,nb_clusteur = 50, bandwidth = 2 ,utilisation_exel=True,*args, **kwargs):
        if 'only_global' in kwargs:
            self.only_global = kwargs['only_global']
        else:
            self.only_global = False
        
        #suprime la visualisation precedente
        folder_path = os.getenv("SCRATCH")
        base_dir = "image_nuage_D3feat"    
        if os.path.exists(os.path.join(folder_path,base_dir)):
            shutil.rmtree(os.path.join(folder_path,base_dir))
        

        save_path = os.path.dirname(__file__) + '/pickles/logg3d_' + self.dataset_type + '/'

        self.compute_embeddings(self.eval_set.query_set, model,name_descripteur_global, reset)
        self.compute_embeddings(self.eval_set.map_set, model,name_descripteur_global,reset)

        
        query_embeddings =  None
        chemain_machine = os.getenv("SCRATCH")
        for ndx, e in tqdm.tqdm(enumerate(self.eval_set.query_set)):
            chemain_query_global = chemain_machine +f"/{name_descripteur_global}"
            dernier_element = os.path.basename(e.rel_scan_filepath)
            nom_sans_extension = os.path.splitext(dernier_element)[0]

            
            global_embedding = torch.load(os.path.join(chemain_query_global , nom_sans_extension+".pt"))
            if query_embeddings is None:
                query_embeddings = np.zeros((len(self.eval_set.query_set), global_embedding.shape[0]), dtype=global_embedding.dtype)
            
            query_embeddings[ndx] = global_embedding        		
        

        
        map_embeddings = None
        #seen_descriptors = []
        local_map_embeddings = []
        for ndx, e in tqdm.tqdm(enumerate(self.eval_set.map_set)):
            chemain_map_global = chemain_machine + f"/{name_descripteur_global}"
            dernier_element = os.path.basename(e.rel_scan_filepath)
            nom_sans_extension = os.path.splitext(dernier_element)[0]

            
            global_embedding = torch.load(os.path.join(chemain_map_global , nom_sans_extension+".pt"))
            if map_embeddings is None:
                map_embeddings = np.zeros((len(self.eval_set.map_set), global_embedding.shape[0]), dtype=global_embedding.dtype)

            map_embeddings[ndx] = global_embedding 

            #global_embedding = np.reshape(global_embedding, (1, -1))       		
            #seen_descriptors.append(global_embedding)
        
        # db_seen_descriptors = np.copy(seen_descriptors)
        # db_seen_descriptors = db_seen_descriptors.reshape(
        #     -1, np.shape(global_embedding)[1])

        
        map_positions = self.eval_set.get_map_positions() # Nmap x 2
        query_positions = self.eval_set.get_query_positions() # Nquery x 2
        

        if self.n_samples is None or len(query_embeddings) <= self.n_samples:
            query_indexes = list(range(len(query_embeddings)))
            self.n_samples = len(query_embeddings)
        else:
            query_indexes = random.sample(range(len(query_embeddings)), self.n_samples)

        if self.only_global:
            metrics = {}
        else:
            metrics = {eval_mode: {'rre': [], 'rte': [], 'repeatability': [],
                                'success': [], 'success_inliers': [], 'failure_inliers': [],
                                'rre_refined': [], 'rte_refined': [], 'success_refined': [],
                                'success_inliers_refined': [], 'repeatability_refined': [],
                                'failure_inliers_refined': [], 't_ransac': []}
                       for eval_mode in ['Initial', 'Re-Ranked']}

        # Dictionary to store the number of true positives (for global desc. metrics) for different radius and NN number
        global_metrics = {'tp': {r: [0] * self.k for r in self.radius}}
        global_metrics['tp_rr'] = {r: [0] * self.k for r in self.radius}
        global_metrics['RR'] = {r: [] for r in self.radius}
        global_metrics['RR_rr'] = {r: [] for r in self.radius}
        global_metrics['t_RR'] = []

        directori_json = self.eval_set.query_set[0].rel_scan_filepath.split(file_rotation, 1)[0] 
        directori_json = os.path.join(self.dataset_root,directori_json,file_rotation,file_rotation+".json")
        print("directori_json", directori_json)
        directori_json = '/lustre/fsn1/worksf/projects/rech/dki/ujo91el/datas/lidarhd_v3/rotation/rotation.json'
        if os.path.exists(directori_json):
            with open(directori_json, 'r', encoding='utf-8') as f:
                data_rotation = json.load(f)


        for query_ndx in tqdm.tqdm(query_indexes):
            
            if self.json_index != '?' :
                # Check if the query element has a true match within each radius
                id_query = self.data_json_index[query_ndx]["row_index"]
                query_pos = query_positions[id_query]
                print(os.path.basename (self.eval_set.query_set[id_query].rel_scan_filepath))
                query_pose = self.eval_set.query_set[id_query].pose
                
                nn_ndx = np.array(self.data_json_index[query_ndx]["TopN_id_list"])
        
                euclid_dist = np.array(self.data_json_index[query_ndx]["dist"])    # (k,) array
            else :
                # Check if the query element has a true match within each radius
                id_query = query_ndx
                query_pos = query_positions[id_query]
                query_pose = self.eval_set.query_set[id_query].pose
                
                # Nearest neighbour search in the embedding space
                query_embedding = query_embeddings[id_query]
                embed_dist = np.linalg.norm(map_embeddings - query_embedding, axis=1)
                nn_ndx = np.argsort(embed_dist)[:self.k]
                #print("nn_ndx", nn_ndx)

                # PLACE RECOGNITION EVALUATION
                # Euclidean distance between the query and nn
                # Here we use non-icp refined poses, but for the global descriptor it's fine
                delta = query_pos - map_positions[nn_ndx]       # (k, 2) array
                euclid_dist = np.linalg.norm(delta, axis=1)     # (k,) array
                
            # else :
            #     # Check if the query element has a true match within each radius
            #     print("je_suis_passer")
            #     id_query = query_ndx
            #     query_pos = query_positions[id_query]
            #     query_pose = self.eval_set.query_set[id_query].pose
                
            #     # Nearest neighbour search in the embedding space
            #     query_embedding = query_embeddings[id_query]
            #     embed_dist = np.linalg.norm(map_embeddings - query_embedding, axis=1)
            #     nn_ndx = np.argsort(embed_dist)[:self.k]
                
            #     # feat_dists = cdist(query_embedding, db_seen_descriptors,
            #     #            metric="cosine").reshape(-1)
            #     # min_dist, nearest_idx = np.min(feat_dists), np.argmin(feat_dists)
                
            #     #print("nn_ndx", nn_ndx)

            #     # PLACE RECOGNITION EVALUATION
            #     # Euclidean distance between the query and nn
            #     # Here we use non-icp refined poses, but for the global descriptor it's fine
            #     delta = query_pos - map_positions[nn_ndx]       # (k, 2) array
            #     euclid_dist = np.linalg.norm(delta, axis=1)     # (k,) array


            
                

            # re_rank = True
            if d_thresh > 0:
                topk = len(nn_ndx)
                fitness_list = np.zeros(topk)
                tick = time()
                
                liste_id = np.append(nn_ndx,id_query)

                for k in range(topk):
                    k_id = nn_ndx[k]

                    if D3Feat == True:
                        if k == 0 :
                            local_query_embeddings_LoGG3D = self.compute_embeddings(self.eval_set.query_set, model, id_nom =id_query)
                            coord_D3Feat_query = self.D3feat_fonction(self.eval_set.query_set , id_query , file_rotation ,nb_clusteur,MEAN_SHIFT_p ,bandwidth )

                            local_query_embeddings_modif = self.program_selection_point(coord_D3Feat_query ,local_query_embeddings_LoGG3D,rayon,min_num_feat)
                            



                        
                        local_map_embeddings_LoGG3D = self.compute_embeddings(self.eval_set.map_set, model, id_nom = k_id) 
                        coord_D3Feat_map = self.D3feat_fonction(self.eval_set.map_set , k_id , file_rotation ,nb_clusteur,MEAN_SHIFT_p , bandwidth)                 
                        local_map_embeddings_modif = self.program_selection_point(coord_D3Feat_map ,local_map_embeddings_LoGG3D,rayon,min_num_feat)

                        print(len(local_query_embeddings_modif['keypoints']))
                        print(len(local_map_embeddings_modif['keypoints']))
                        if visualisation == True  :
                            nom_dossier = os.path.join(self.dataset_root, self.eval_set.query_set[id_query].rel_scan_filepath)
                            conf_val = sgv_visu.sgv_fn(local_query_embeddings_modif, local_map_embeddings_modif,local_query_embeddings_LoGG3D,local_map_embeddings_LoGG3D,D3Feat,k,nom_dossier,name_file_visu, d_thresh=d_thresh ,min_num_feat = min_num_feat)
                        else :
                            conf_val = sgv_utils.sgv_fn(local_query_embeddings_modif, local_map_embeddings_modif,D3Feat, d_thresh=d_thresh , min_num_feat = min_num_feat)
                    else : 
                        if k == 0 :
                            local_query_embeddings_LoGG3D = self.compute_embeddings(self.eval_set.query_set, model, id_nom =id_query)
                        
                        local_map_embeddings_LoGG3D = self.compute_embeddings(self.eval_set.map_set, model, id_nom = k_id)
                        
                        if visualisation == True  :
                            nom_dossier = os.path.join(self.dataset_root, self.eval_set.query_set[id_query].rel_scan_filepath)
                            conf_val = sgv_visu.sgv_fn(local_query_embeddings_LoGG3D, local_map_embeddings_LoGG3D,local_query_embeddings_LoGG3D,local_map_embeddings_LoGG3D,D3Feat,k,nom_dossier,name_file_visu, d_thresh=d_thresh ,min_num_feat = min_num_feat)
                        else :
                            conf_val = sgv_utils.sgv_fn(local_query_embeddings_LoGG3D, local_map_embeddings_LoGG3D,D3Feat, d_thresh=d_thresh ,min_num_feat = min_num_feat)
                    fitness_list[k] = conf_val
                print(fitness_list)
                
                topk_rerank = np.flip(np.asarray(fitness_list).argsort())
                topk_rerank_inds = copy.deepcopy(nn_ndx)
                topk_rerank_inds[:topk] = nn_ndx[topk_rerank]
                t_rerank = time() - tick
                global_metrics['t_RR'].append(t_rerank)



                delta_rerank = query_pos - map_positions[topk_rerank_inds]
                euclid_dist_rr = np.linalg.norm(delta_rerank, axis=1)
                


                if utilisation_exel == True :
                    wb = load_workbook(name_file_exel)
                    
                    wb["nom_fichier"].append([str(self.eval_set.query_set[id_query].rel_scan_filepath)])
                    nom_map = [self.eval_set.map_set[i].rel_scan_filepath for i in topk_rerank_inds]
                    wb["nom_map"].append(nom_map)
                    wb["euclid_dist"].append(list(euclid_dist))
                    wb["topk_rerank"].append(list(nn_ndx))
                    wb["topk_rerank_inds"].append(list(topk_rerank_inds))
                    wb["euclid_dist_rr"].append(list(euclid_dist_rr))
                    #wb["rotation"].append(data_rotation[os.path.basename(str(self.eval_set.query_set[id_query].rel_scan_filepath))]["rotation"])

                    # Sauvegarder les ajouts
                    wb.save(name_file_exel)
                else :
                    self.append_to_sheet(f"{name_file_exel}/nom_fichier",[str(self.eval_set.query_set[id_query].rel_scan_filepath)])  
                    nom_map = [self.eval_set.map_set[i].rel_scan_filepath for i in topk_rerank_inds]
                    self.append_to_sheet(f"{name_file_exel}/nom_map",nom_map)
                    self.append_to_sheet(f"{name_file_exel}/euclid_dist",list(euclid_dist))
                    self.append_to_sheet(f"{name_file_exel}/topk_rerank",list(nn_ndx))
                    self.append_to_sheet(f"{name_file_exel}/topk_rerank_inds",list(topk_rerank_inds))
                    self.append_to_sheet(f"{name_file_exel}/euclid_dist_rr",list(euclid_dist_rr))
                    #self.append_to_sheet(f"{name_file_exel}/rotation",data_rotation[os.path.basename(str(self.eval_set.query_set[id_query].rel_scan_filepath))]["rotation"])
                                
            else:
                euclid_dist_rr = euclid_dist
                global_metrics['t_RR'].append(0)

            # Count true positives for different radius and NN number
            global_metrics['tp'] = {r: [global_metrics['tp'][r][nn] + (1 if (euclid_dist[:nn + 1] <= r).any() else 0) for nn in range(self.k)] for r in self.radius}
            
            global_metrics['tp_rr'] = {r: [global_metrics['tp_rr'][r][nn] + (1 if (euclid_dist_rr[:nn + 1] <= r).any() else 0) for nn in range(self.k)] for r in self.radius}
            global_metrics['RR'] = {r: global_metrics['RR'][r]+[next((1.0/(i+1) for i, x in enumerate(euclid_dist <= r) if x), 0)] for r in self.radius}
            global_metrics['RR_rr'] = {r: global_metrics['RR_rr'][r]+[next((1.0/(i+1) for i, x in enumerate(euclid_dist_rr <= r) if x), 0)] for r in self.radius}
            if self.only_global:
                continue

        # Calculate mean metrics
        global_metrics["recall"] = {r: [global_metrics['tp'][r][nn] / self.n_samples for nn in range(self.k)] for r in self.radius}

        global_metrics["recall_rr"] = {r: [global_metrics['tp_rr'][r][nn] / self.n_samples for nn in range(self.k)] for r in self.radius}
        global_metrics['MRR'] = {r: np.mean(np.asarray(global_metrics['RR'][r])) for r in self.radius}
        global_metrics['MRR_rr'] = {r: np.mean(np.asarray(global_metrics['RR_rr'][r])) for r in self.radius}
        global_metrics['mean_t_RR'] = np.mean(np.asarray(global_metrics['t_RR']))

        mean_metrics = {}
        if not self.only_global:
            # Calculate mean values of local descriptor metrics
            for eval_mode in ['Initial', 'Re-Ranked']:
                mean_metrics[eval_mode] = {}
                for metric in metrics[eval_mode]:
                    m_l = metrics[eval_mode][metric]
                    if len(m_l) == 0:
                        mean_metrics[eval_mode][metric] = 0.
                    else:
                        if metric == 't_ransac':
                            mean_metrics[eval_mode]["t_ransac_sd"] = np.std(m_l)
                        mean_metrics[eval_mode][metric] = np.mean(m_l)

        return global_metrics, mean_metrics        
       
    def evaluate2(self, model, d_thresh,min_num_feat,MEAN_SHIFT_p ,path_model_D3Feat,D3Feat,name_file_exel,name_descripteur_global,name_file_visu,reset=False,file_rotation='',visualisation = False ,rayon =3,nb_clusteur = 50, bandwidth = 2 ,utilisation_exel=True,*args, **kwargs):
        if 'only_global' in kwargs:
            self.only_global = kwargs['only_global']
        else:
            self.only_global = False
        
        #suprime la visualisation precedente
        folder_path = os.getenv("SCRATCH")
        base_dir = "image_nuage_D3feat"    
        if os.path.exists(os.path.join(folder_path,base_dir)):
            shutil.rmtree(os.path.join(folder_path,base_dir))
        

        save_path = os.path.dirname(__file__) + '/pickles/logg3d_' + self.dataset_type + '/'

        # re_rank = True
        D3Feat = True

        path = "/lustre/fsn1/worksf/projects/rech/dki/ujo91el/datas/lidarhd_v3/dsi_train_list.json"
        with open(path, "r") as f:
            train = json.load(f)
        print(len(train))
        traind = {path: i for i, path in enumerate(train)}
        
        path = "/lustre/fsn1/worksf/projects/rech/dki/ujo91el/datas/lidarhd_v3/poses_grid2.json"
        with open(path, "r") as f:
            poses = json.load(f)
        print(len(poses))

        poses_inv = {tuple(v): k for k, v in poses.items()}

        #path = "/lustre/fswork/projects/rech/dki/ujo91el/code/DSI_LIDAR_HD/dictio_results/results_dsi3d_moe_west_global2.json"
        path = "/lustre/fswork/projects/rech/dki/ujo91el/code/DSI_LIDAR_HD/dictio_results/results_dsi3d_west_global_approx.json"
        path = "/lustre/fswork/projects/rech/dki/ujo91el/code/DSI_LIDAR_HD/dictio_results/results_dsi3d_moe_west_global_approx.json"
        
        with open(path, "r") as f:
            results_dict = json.load(f)
        print(len(results_dict))

        
        map_positions = self.eval_set.get_map_positions() # Nmap x 2
        query_positions = self.eval_set.get_query_positions() # Nquery x 2

        hit_at_1 = 0
        hit_at_1_after_rerank = 0
        broken = 0
        
        if d_thresh > 0:
            topk = 10
            fitness_list = np.zeros(topk)
            tick = time()
            
            #liste_id = np.append(nn_ndx,id_query)

            file_rotation = "lidarhd_v3"
            #for k in range(topk):
            id_counter = 0
            for i in tqdm.tqdm(results_dict):
          
                if id_counter < 25759:
                    id_counter +=1
                    continue
    
                id_query = i['query_idx']
                top_k_og = i['Top10_id']
                
                #top_k = [ traind[x] for x in top_k_og]
                top_k = [ traind.get(x, -1) for x in top_k_og]
               
                dist = i['dist']
                """
                query_pose = i['query_pose']
                true_id = poses_inv[tuple(query_pose)]
                """
                #traind[true_id]
                #evalid[true_id]

                """
                prefix = true_id[:-6]
                matches = [k for k in traind if k.startswith(prefix)]
                
                if len(matches) == 1:
                    value = traind[matches[0]]
                    id_query = value
                else:
                    broken +=1 
                    continue
                """
                
                    #raise ValueError(f"Expected 1 match, found {len(matches), prefix}")
                    
                rank_list_dist_filter = [1 if x <= 3 else 0 for x in dist]
                if 1 not in rank_list_dist_filter:
                    print("no possible re-ranking")
                    continue
                else:
                    hits_clos = np.where(np.array(rank_list_dist_filter)[:10] == 1)[0]
                    if hits_clos.size > 0:
                        if hits_clos[0] == 0:
                            hit_at_1 += 1
                
                print("id_query ", id_query) 
                print("Top10_id ", top_k) 
                print("distances ", dist) 
                print(path)

                sequence_path = "/lustre/fsn1/worksf/projects/rech/dki/ujo91el/datas/lidarhd_v3"
                f = open(sequence_path +"/id_dsi_eval_list.json") 
                train_indices2 = json.load(f)
                f.close()
                
                try:
                    
                    local_query_embeddings_LoGG3D = self.compute_embeddings(self.eval_set.query_set, model, id_nom =id_query)
                    coord_D3Feat_query = self.D3feat_fonction(self.eval_set.map_set , id_query , file_rotation ,nb_clusteur,MEAN_SHIFT_p ,bandwidth )
                    local_query_embeddings_modif = self.program_selection_point(coord_D3Feat_query ,local_query_embeddings_LoGG3D,rayon,min_num_feat)
                except:
                    broken+=1
                    continue
                    
                counter = 0
                for k_id in top_k:
                    #k_id = nn_ndx[k]

                    if k_id == -1: # hallucinated answers
                        fitness_list[counter] = 0
                        counter+=1
                        continue

                    #print(i['Top10_id'][counter])
                    try:
                        #local_map_embeddings_LoGG3D = self.compute_embeddings(self.eval_set.map_set, model, id_nom = k_id) 
                        local_map_embeddings_LoGG3D = self.compute_embeddings(self.eval_set.query_set, model, id_nom = k_id)
                        #coord_D3Feat_map = self.D3feat_fonction(self.eval_set.map_set , k_id , file_rotation ,nb_clusteur,MEAN_SHIFT_p , bandwidth)      
                        coord_D3Feat_map = self.D3feat_fonction(self.eval_set.map_set , k_id , file_rotation ,nb_clusteur,MEAN_SHIFT_p , bandwidth) 
                        local_map_embeddings_modif = self.program_selection_point(coord_D3Feat_map ,local_map_embeddings_LoGG3D,rayon,min_num_feat)
                    except:
                        broken+=1
                        continue
                    
                    
                    #print(len(local_query_embeddings_modif['keypoints']))
                    #print(len(local_map_embeddings_modif['keypoints']))

                    conf_val = sgv_utils.sgv_fn(local_query_embeddings_modif, local_map_embeddings_modif,D3Feat, d_thresh=d_thresh , min_num_feat = min_num_feat)
                    
                    fitness_list[counter] = conf_val
                    counter+=1

                id_counter+=1
                
                print("fitness_list ",fitness_list)
                
                topk_rerank = np.flip(np.asarray(fitness_list).argsort())

                id_top10_reranked = [top_k_og[idx] for idx in topk_rerank]
                dist_top10_reranked = [dist[idx] for idx in topk_rerank]
                rerank_list_dist_filter = [1 if x <= 3 else 0 for x in dist_top10_reranked]
                
                hits_clos2 = np.where(np.array(rerank_list_dist_filter)[:10] == 1)[0]
                print("Before ",rank_list_dist_filter)
                print("After" , rerank_list_dist_filter)
                
                if hits_clos2.size > 0:
                    if hits_clos2[0] == 0:
                        hit_at_1_after_rerank += 1
                        
                #topk_rerank_inds = copy.deepcopy(nn_ndx)
                #topk_rerank_inds[:topk] = nn_ndx[topk_rerank]
                t_rerank = time() - tick
                #global_metrics['t_RR'].append(t_rerank)


                print("hit_at_1 ", hit_at_1)
                print("hit_at_1_after_rerank ", hit_at_1_after_rerank)
                print("broken ", broken)
            
            print("results_dict ", len(results_dict))
            import pdb; pdb.set_trace()

        global_metrics = None  
        return global_metrics
    
    def append_to_sheet(self ,sheet_name, row_values):
                        out_dir = os.path.dirname(__file__)
                        file_path = os.path.join(out_dir, f"{sheet_name}.csv")
                        
                        with open(file_path, mode="a", newline="", encoding="utf-8") as f:
                            writer = csv.writer(f)
                            writer.writerow(row_values)    
        

    def ransac_fn(self, query_keypoints, candidate_keypoints):
        """

        Returns fitness score and estimated transforms
        Estimation using Open3d 6dof ransac based on feature matching.
        """
        kp1 = query_keypoints['keypoints']
        kp2 = candidate_keypoints['keypoints']
        ransac_result = get_ransac_result(query_keypoints['features'], candidate_keypoints['features'],
                                          kp1, kp2)
        return ransac_result.transformation, len(ransac_result.correspondence_set), ransac_result.fitness
    
    def compute_embeddings(self, eval_subset: List[EvaluationTuple], model,name_descripteur_global='',reset=False, id_nom=None):
        """
        Computes and optionally saves global and local embeddings for a subset of the evaluation set.

        If `id_nom` is not provided, embeddings are computed for all elements in `eval_subset` and saved to disk.
        If `id_nom` is provided, embeddings are computed only for the specified element and returned as a dictionary.

        Args:
            eval_subset (List[EvaluationTuple]): List of evaluation tuples (query or map set).
            model: The model used to compute embeddings.
            name_descripteur_global (str): Name of the global descriptor directory. Defaults to ''.
            reset (bool): If True, recomputes embeddings even if they already exist. Defaults to False.
            id_nom (int, optional): Index of the element to compute embeddings for. If None, computes for all elements.

        Returns:
            Union[None, dict]:
                - If `id_nom` is None, returns None.
                - If `id_nom` is provided, returns a dictionary with:
                    - 'keypoints': (N, 3) array of keypoints.
                    - 'features': (N, D) array of feature descriptors.
        """
        
        self.model2eval((model,))
        chemain_machine = os.getenv("SCRATCH")
        
        if id_nom == None:
            for ndx, e in tqdm.tqdm(enumerate(eval_subset)):
                dernier_element = os.path.basename(e.rel_scan_filepath)
                nom_sans_extension = os.path.splitext(dernier_element)[0]
                chemin_fichier = os.path.join( chemain_machine + f"/{name_descripteur_global}/", nom_sans_extension + ".pt")
                if not os.path.exists(os.path.dirname(chemin_fichier)):
                    os.makedirs(os.path.dirname(chemin_fichier))
                
                if not os.path.exists(chemin_fichier) or reset== True : 

                    scan_filepath = os.path.join(self.dataset_root, e.rel_scan_filepath)
                    


                    assert os.path.exists(scan_filepath)
                    pc = self.pc_loader(scan_filepath)
                    
                    xyzi = pc
                        

                    
                    hash_vals, coords, indices, input , points = make_sparse_tensor(xyzi, voxel_size=self.voxel_size, num_points=80000,return_points=True, return_hash = True)

                    
                    coords = coords.astype(np.int32)
                    if torch.cuda.is_available():
                        input = input.cuda()
                    else:
                        input = input.to(torch.device("cpu"))
                    
                    points = points[:,:3]
                    
                                
                    global_descriptor , global_embedding, keypoints, key_embeddings = self.compute_embedding(input, model, coords, points, is_dense=True)
                    
                    dernier_element = os.path.basename(e.rel_scan_filepath)
                    nom_sans_extension = os.path.splitext(dernier_element)[0]
                    if not os.path.exists(os.path.dirname(chemin_fichier)):
                        os.makedirs(os.path.dirname(chemin_fichier))

                    torch.save(global_descriptor, chemin_fichier)
        
        if id_nom != None:

            scan_filepath = os.path.join(self.dataset_root, eval_subset[id_nom].rel_scan_filepath)
            print(os.path.basename(scan_filepath))
            
            
                    
            assert os.path.exists(scan_filepath)
            pc = self.pc_loader(scan_filepath)
            
            xyzi = pc
                

            
            hash_vals, coords, indices, input , points = make_sparse_tensor(xyzi, voxel_size=self.voxel_size, num_points=80000,return_points=True, return_hash = True)

            
            
            coords = coords.astype(np.int32)
            if torch.cuda.is_available():
                input = input.cuda()
            else:
                input = input.to(torch.device("cpu"))
            
            points = points[:,:3]
                                    
            global_descriptor , global_embedding, keypoints, key_embeddings = self.compute_embedding(input, model, coords, points, is_dense=True)

            return {'keypoints': keypoints, 'features': key_embeddings}  
        
            

    def compute_embedding(self, pc, model, coords, points, is_dense=False):
        """
        Computes global and local embeddings for a point cloud using the provided model.

        Extracts a global descriptor, global embedding, keypoints, and their corresponding feature descriptors
        from the model's output. Handles both dense and sparse point cloud representations.

        Args:
            pc: Input point cloud tensor.
            model: The model used to compute embeddings.
            coords: Coordinates of the point cloud.
            points: Points of the point cloud.
            is_dense (bool): If True, treats the point cloud as dense. Defaults to False.

        Returns:
            tuple: (global_descriptor, global_embedding, keypoints, key_embeddings)
                - global_descriptor (np.ndarray): Global descriptor of the point cloud.
                - global_embedding (np.ndarray): Global embedding reshaped as (1, D).
                - keypoints (torch.Tensor): (N, 3) array of keypoints.
                - key_embeddings (torch.Tensor): (N, D) array of keypoint feature descriptors.
        """


        output_desc, output_feats, xc = model(pc)
        
        if is_dense:

            output_features = output_feats.cpu().detach().numpy()
            output_points = points
        else:
            xc_coords = xc.C
            
            xc_feat = xc.F.cpu().detach().numpy()
            _, xc_counts = torch.unique(xc_coords[:, -1], return_counts=True)
            y = torch.split(xc_coords, list(xc_counts))

            assert len(y) == 1

            xc_coords = xc_coords.cpu().detach().numpy()
            xc_coords= xc_coords[:,:3]
            len_coords = len(coords)
            all_coords = np.vstack((coords,xc_coords))
            all_hash_vals = ravel_hash(all_coords)

            dense_hash = all_hash_vals[:len_coords]
            coarse_hash = all_hash_vals[len_coords:]

            sorter = np.argsort(dense_hash)
            xc_hash_inds = sorter[np.searchsorted(dense_hash, coarse_hash, sorter=sorter)]

            unique_xc_hash_inds, unique_xc_hash_inds_indices = np.unique(xc_hash_inds,
                                                return_index=True)
            

            output_points = points[unique_xc_hash_inds]
            output_features = xc_feat[unique_xc_hash_inds_indices]

        global_descriptor = output_desc.cpu().detach().numpy()


        global_embedding = np.reshape(global_descriptor, (1, -1))

        
        return global_descriptor ,global_embedding, torch.tensor(output_points, dtype=torch.float), torch.tensor(output_features, dtype=torch.float)


    def print_results(self, global_metrics, metrics):
        # Global descriptor results are saved with the last n_k entry
        print('\n','Initial Retrieval:')
        recall = global_metrics['recall']
        
        for r in recall:
            print(f"Radius: {r} [m] : ")
            print(f"Recall@N : ", end='')
            for x in recall[r]:
                print("{:0.1f}, ".format(x*100.0), end='')
            print("")
            print('MRR: {:0.1f}'.format(global_metrics['MRR'][r]*100.0))
        
        print('\n','Re-Ranking:')
        recall_rr = global_metrics['recall_rr']
        for r_rr in recall_rr:
            print(f"Radius: {r_rr} [m] : ")
            print(f"Recall@N : ", end='')
            for x in recall_rr[r_rr]:
                print("{:0.1f}, ".format(x*100.0), end='')
            print("")
            print('MRR: {:0.1f}'.format(global_metrics['MRR_rr'][r_rr]*100.0))
        print('Re-Ranking Time: {:0.3f}'.format(1000.0 *global_metrics['mean_t_RR']))

        print('\n','Metric Localization:')
        for eval_mode in ['Initial', 'Re-Ranked']:
            if eval_mode not in metrics:
                break
            print('#keypoints: {}'.format(eval_mode))
            for s in metrics[eval_mode]:
                print(f"{s}: {metrics[eval_mode][s]:0.3f}")
            print('')
    def program_selection_point(self, local_embeddings_D3feat, local_embeddings_LoGG3D, rayon , max_points = 20000):
        """
        Selects keypoints from LoGG3D embeddings based on proximity to D3Feat keypoints.

        For each keypoint in LoGG3D, checks if it lies within a specified radius of any D3Feat keypoint.
        Returns only the LoGG3D keypoints and features that satisfy this condition.

        Args:
            local_embeddings_D3feat (np.ndarray): (N, 3) array of D3Feat keypoints.
            local_embeddings_LoGG3D (dict): Dictionary containing LoGG3D keypoints and features.
                - 'keypoints': (M, 3) array of LoGG3D keypoints.
                - 'features': (M, D) array of LoGG3D feature descriptors.
            rayon (float): Radius threshold for proximity selection.

        Returns:
            dict: Dictionary containing filtered LoGG3D keypoints and features.
                - 'keypoints': (K, 3) array of selected keypoints.
                - 'features': (K, D) array of selected feature descriptors.
        """
        local_embeddings ={}
        if torch.cuda.is_available():
            coord_embeddings_D3feat = torch.from_numpy(local_embeddings_D3feat).unsqueeze(0).cuda()
            coordoner_point_LoGG3D = local_embeddings_LoGG3D['keypoints'].unsqueeze(0).cuda()
        else:
            coord_embeddings_D3feat = torch.from_numpy(local_embeddings_D3feat).unsqueeze(0).to(torch.device("cpu"))

            coordoner_point_LoGG3D = local_embeddings_LoGG3D['keypoints'].unsqueeze(0).to(torch.device("cpu"))
        print(coordoner_point_LoGG3D.shape[1])
        distances =  torch.norm((coord_embeddings_D3feat[:, :, None, :] - coordoner_point_LoGG3D[:, None, :, :]), dim=-1)
        mask = distances <= rayon
        mask_ref = mask.any(dim=1)[0]
        selected_keypoints = local_embeddings_LoGG3D['keypoints'][mask_ref]
        selected_features = local_embeddings_LoGG3D['features'][mask_ref]

        torch.manual_seed(42)
        # Limitation à max_points si trop nombreux
        if selected_keypoints.shape[0] > max_points:
            idx = torch.randperm(selected_keypoints.shape[0])[:max_points]
            selected_keypoints = selected_keypoints[idx]
            selected_features = selected_features[idx]

        local_embeddings['keypoints'] = selected_keypoints
        local_embeddings['features'] = selected_features
        return local_embeddings
    

    def D3feat_fonction(self ,eval_subset,id_nom ,file_rotation,nb_clusteur,MEAN_SHIFT_p=False ,bandwidth = 2 ): 
        """
        Extracts and optionally applies mean shift to D3Feat keypoints for a specific scan.

        Loads precomputed D3Feat keypoints and scores, selects the top-k keypoints based on scores,
        and optionally applies weighted mean shift to refine keypoint positions.

        Args:
            eval_subset (List[EvaluationTuple]): List of evaluation tuples containing scan information.
            id_nom (int): Index of the scan in `eval_subset` to process.
            MEAN_SHIFT_p (bool): If True, applies weighted mean shift to keypoints. Defaults to False.
            file_rotation (str): Identifier for the rotation file. Defaults to ','.

        Returns:
            np.ndarray: (K, 3) array of selected or refined keypoints.
        """ 
        src_file = eval_subset[id_nom].rel_scan_filepath
        
        src_name = os.path.splitext(os.path.basename(src_file))[0]
        chemain_stokage_fichier = os.getenv("SCRATCH")
        src_data = np.load(os.path.join(chemain_stokage_fichier ,f"descripteur_D3Feat/{file_rotation}/{src_name}.npz"))

        scores = src_data["scores"].reshape(-1)
        top_k = 300
        top_k = min(top_k, len(scores))


        top_indices = np.argsort(scores)[-top_k:]

        top_keypts = src_data["keypts"][top_indices]
        top_scores = src_data["scores"][top_indices]
                            
        
        if MEAN_SHIFT_p == True:
            # -------- MEAN SHIFT pondéré --------
            ms_centers = self.weighted_kmeans(
                points=src_data["keypts"],
                scores=src_data["scores"],
                n_clusters = nb_clusteur
            )
            coord_point = ms_centers
        else:
            coord_point = top_keypts     
        return   coord_point


    def weighted_kmeans(self, points, scores, n_clusters, random_seed=42):
        # Normaliser les scores entre 0 et 1
        norm_scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores) + 1e-8)
        norm_scores = np.ravel(norm_scores)

        # K-Means avec pondération directe (nécessite scikit-learn >= 1.3)
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=random_seed,
            n_init=10
        )
        kmeans.fit(points, sample_weight=norm_scores)

        return kmeans.cluster_centers_

    # def weighted_kmeans(self, points, scores, n_clusters=50, n_replicas_max=100, random_seed=42):
    #     norm_scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores) + 1e-8)
    #     replica_counts = (norm_scores * n_replicas_max).astype(int)
        
    #     weighted_points = np.vstack([
    #         np.tile(point, (int(count), 1))
    #         for point, count in zip(points, replica_counts) if count > 0
    #     ])
        
    #     kmeans = KMeans(n_clusters=n_clusters, random_state=random_seed, n_init=10)
    #     kmeans.fit(weighted_points)
    #     return kmeans.cluster_centers_


def get_ransac_result(feat1, feat2, kp1, kp2, ransac_dist_th=0.5, ransac_max_it=10000):
    feature_dim = feat1.shape[1]
    pcd_feat1 = make_open3d_feature(feat1, feature_dim, feat1.shape[0])
    pcd_feat2 = make_open3d_feature(feat2, feature_dim, feat2.shape[0])
    if not isinstance(kp1, np.ndarray):
        pcd_coord1 = make_open3d_point_cloud(kp1.numpy())
        pcd_coord2 = make_open3d_point_cloud(kp2.numpy())
    else:
        pcd_coord1 = make_open3d_point_cloud(kp1)
        pcd_coord2 = make_open3d_point_cloud(kp2)

    # ransac based eval
    ransac_result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        pcd_coord1, pcd_coord2, pcd_feat1, pcd_feat2,
        mutual_filter=True,
        max_correspondence_distance=ransac_dist_th,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=3,
        checkers=[o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.8),
                  o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(ransac_dist_th)],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(ransac_max_it, 0.999))

    return ransac_result


def calculate_repeatability(kp1, kp2, T_gt, threshold: float):
    # Transform the source point cloud to the same position as the target cloud
    kp1_pos_trans = apply_transform(kp1, torch.tensor(T_gt, dtype=torch.float))
    dist = torch.cdist(kp1_pos_trans, kp2)      # (n_keypoints1, n_keypoints2) tensor

    # *** COMPUTE REPEATABILITY ***
    # Match keypoints from the first cloud with closests keypoints in the second cloud
    min_dist, _ = torch.min(dist, dim=1)
    # Repeatability with a distance threshold th
    return torch.mean((min_dist <= threshold).float()).item()

def sparse_quantize(coords,
                    voxel_size: Union[float, Tuple[float, ...]] = 1,
                    *,
                    return_index: bool = False,
                    return_inverse: bool = False,
                    return_hash: bool = False) -> List[np.ndarray]:
    if isinstance(voxel_size, (float, int)):
        voxel_size = tuple(repeat(voxel_size, 3))
    assert isinstance(voxel_size, tuple) and len(voxel_size) == 3

    voxel_size = np.array(voxel_size)
    coords = np.floor(coords / voxel_size).astype(np.int32)

    hash_vals, indices, inverse_indices = np.unique(ravel_hash(coords),
                                            return_index=True,
                                            return_inverse=True)
    coords = coords[indices]

    if return_hash: outputs = [hash_vals, coords]
    else: outputs = [coords]

    if return_index:
        outputs += [indices]
    if return_inverse:
        outputs += [inverse_indices]
    return outputs[0] if len(outputs) == 1 else outputs

def ravel_hash(x: np.ndarray) -> np.ndarray:
    assert x.ndim == 2, x.shape

    x -= np.min(x, axis=0)
    x = x.astype(np.uint64, copy=False)
    xmax = np.max(x, axis=0).astype(np.uint64) + 1

    h = np.zeros(x.shape[0], dtype=np.uint64)
    for k in range(x.shape[1] - 1):
        h += x[:, k]
        h *= xmax[k + 1]
    h += x[:, -1]
    return h

def make_sparse_tensor(lidar_pc, voxel_size=0.10, num_points=35000,return_points=False, return_hash = False):
    
    # get rounded coordinates
    coords = np.round(lidar_pc[:, :3] / voxel_size)
    coords_min = coords.min(0, keepdims=1)
    coords -= coords_min
    feats = lidar_pc


    # sparse quantization: filter out duplicate points
    hash_vals, _, indices = sparse_quantize(coords, return_index=True, return_hash=return_hash)

    coords = coords[indices]
    feats = feats[indices]


    # construct the sparse tensor
    inputs = SparseTensor(feats, coords)
    inputs = sparse_collate([inputs])
    inputs.C = inputs.C.int()
    
    if return_points:
        return hash_vals, coords, indices, inputs , feats
    else:
        return inputs

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
    parser = argparse.ArgumentParser(description='Evaluate the MetLoc model for place recognition and localization.')
    parser.add_argument('--dataset_root', type=str, required=False, default='', help='Root directory of the dataset. If not provided, the default dataset path will be used.')
    parser.add_argument('--dataset_type', type=str, required=False, default='kitti360', choices=['mulran', 'southbay', 'kitti', 'alita', 'kitti360', 'lidar'], help='Type of the dataset to evaluate. Default: "kitti360".')
    parser.add_argument('--eval_set', type=str, required=False, default="?", help='Filename of the evaluation pickle file. This file must be located in the dataset root directory.')
    parser.add_argument('--root_json_index', type=str, required=False, default="?", help='Path to the JSON index file containing metadata for the dataset. Default: "json_index.json".')
    parser.add_argument('--radius', type=float, nargs='+', default=[5, 23, 33], help='List of distance thresholds (in meters) to consider a match as a True Positive. Default: [5, 23, 33].')
    parser.add_argument('--n_samples', type=int, default=None, help='Number of query elements to sample from the dataset. If not provided, all query elements will be used.')
    parser.add_argument('--weights', type=str, default='/logg3d.pth', help='Path to the pre-trained model weights. Default: "/logg3d.pth".')
    parser.add_argument('--model', type=str, default='logg3d', choices=['logg3d', 'logg3d1k'], help='Model architecture to use for evaluation. Default: "logg3d1k".')
    parser.add_argument('--d_thresh', type=float, default=0.4, help='Descriptor distance threshold (in meters) for re-ranking. Default: 0.4.')
    parser.add_argument('--n_topk', type=int, default=10, help='Number of top-k nearest neighbors to consider during evaluation. Default: 10.')
    parser.add_argument('--min_num_feat', type=int, default=15000, help='Minimum number of features required to perform re-ranking. Default: 15000.')
    parser.add_argument('--icp_refine', dest='icp_refine', action='store_true', help='Enable ICP-based pose refinement. Default: True.')
    parser.set_defaults(icp_refine=True)
    parser.add_argument('--voxel_size', type=float, default=0.1, help='Voxel size (in meters) for point cloud downsampling. Default: 0.1.')
    parser.add_argument('--path_model_D3Feat', type=str, default='../../D3Feat_modif_1_laurent_1_moi/results_kitti_custom/Log_11011605/', help='Path to the directory containing the D3Feat model and its outputs. Default: "../../D3Feat_modif_1_laurent_1_moi/results_kitti_custom/Log_11011605/".')
    parser.add_argument('--mulran_sequence', type=str, required=False, default='sejong', help='Name of the MulRan sequence to evaluate. Only applicable if dataset_type is "mulran". Default: "sejong".')
    parser.add_argument('--D3Feat_util', type=str2bool, default=False, help='Whether to use D3Feat for local descriptor computation. Default: False.')
    parser.add_argument('--reset_fichier', type=str2bool, default=False, help='If True, recomputes and resets all embedding files. Default: False.')
    parser.add_argument('--MEAN_SHIFT_p', type=str2bool, default=False, help='If True, applies mean shift to refine keypoint positions. Default: False.')
    parser.add_argument('--visualisation', type=str2bool, default=False, help='If True, enables visualization of keypoints and/or intermediate results. Default: False.')
    parser.add_argument('--nom_fichier_exel', type=str, default='exel', help='Name of the Excel file to log evaluation results. Default: "" (no logging).')
    parser.add_argument('--name_descripteur_global', type=str, default='descripteur_global_SGV_lidar_0_1_rotation', help='Name of the global descriptor directory where embeddings are stored. Default: "" (uses default directory).')
    parser.add_argument('--name_file_visu', type=str, default='image_nuage_D3feat', help='name file for visualization images. Default: "image_nuage_D3feat".')
    parser.add_argument('--file_rotation', type=str, required=False, default='rotation', help='Name of the folder containing rotation data for D3Feat. Default: "rotation".')
    parser.add_argument('--full_zone', type=int, default=5, help='Zone identifier for evaluation. Default: 5.')
    parser.add_argument('--rayon', type=float, default=3, help='Search radius around D3Feat points for local neighborhood matching.')
    parser.add_argument('--nb_clusteur', type=int, default=50, help='Number of clusters for the weighted K-Means algorithm used in keypoint selection.')

    parser.add_argument('--bandwidth', type=int, default=2, help='Kernel bandwidth (radius) for the MeanShift clustering. Larger values merge more points into fewer clusters; smaller values create more clusters')
    parser.add_argument('--utilisation_exel', type=str2bool, default=True, help='If True, logs evaluation results to an Excel file. Default: False.')

    args = parser.parse_args()
    if args.eval_set == "?":
        if args.dataset_type == 'kitti':
            args.eval_set = 'kitti_00_eval_tout_map.pickle'
        elif args.dataset_type == 'mulran':
            
            if args.mulran_sequence == 'sejong':
                args.eval_set = 'test_Sejong01_Sejong02.pickle'
            elif args.mulran_sequence == 'DCC' :
                args.eval_set = 'test_DCC_01_DCC_02_10.0_5.pickle'
        elif args.dataset_type == 'southbay':
            args.eval_set = 'test_SunnyvaleBigloop_1.0_5.pickle'
        elif args.dataset_type == 'alita':
            args.eval_set = 'test_val_5_0.01_5.pickle'
        elif args.dataset_type == 'kitti360':
            args.eval_set = 'kitti360_09_3.0_eval.pickle'
        elif args.dataset_type == 'lidar':
            if args.full_zone == 9 : 
                args.eval_set = 'lidar_neuf_zone.pickle'
            elif args.full_zone == 5 :
                args.eval_set = 'lidar_eval_test.pickle'
    
            
    print(f'weights: {args.weights}')
    #args.weights = os.path.dirname(__file__) + args.weights
    args.weights =  "/lustre/fswork/projects/rech/dki/ujo91el/code/tool_lidar_hd/LoGG3D-Net/training/checkpoints/2025-06-23_11-22-13_run_0_4"
    print(f'weights: {args.weights}')
    print(f'Dataset root: {args.dataset_root}')
    print(f'Dataset type: {args.dataset_type}')
    print(f'Evaluation set: {args.eval_set}')
    print(f'Radius: {args.radius} [m]')
    print(f'd_thresh: {args.d_thresh} [m]')
    print(f'n_topk: {args.n_topk} ')
    print('')

    # /lustre/fswork/projects/rech/dki/ujo91el/code/SpectralGV_D3Feat/SpectralGV-main/datasets/lidar/file_traitment.pickle
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print('Device: {}'.format(device))
    
    model = model_factory(args.model)
    if args.weights is not None:
        assert os.path.exists(args.weights), 'Cannot open network weights: {}'.format(args.weights)
        print('Loading weights: {}'.format(args.weights))
        checkpoint = torch.load(args.weights, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

    model.to(device) # logg3d-net
    

    evaluator = MetLocEvaluator(    args.dataset_root,  
                                    args.dataset_type, 
                                    args.eval_set, device ,  
                                    args.root_json_index, 
                                    radius=args.radius, 
                                    k=args.n_topk,
                                    n_samples=args.n_samples, voxel_size=args.voxel_size,
                                    icp_refine=args.icp_refine)
    

    global_metrics, metrics = evaluator.evaluate2(model, 
                                                d_thresh=args.d_thresh,
                                                min_num_feat = args.min_num_feat,
                                                MEAN_SHIFT_p = args.MEAN_SHIFT_p,
                                                path_model_D3Feat=args.path_model_D3Feat, 
                                                only_global=False ,
                                                D3Feat=args.D3Feat_util , 
                                                name_file_exel = args.nom_fichier_exel,
                                                name_descripteur_global = args.name_descripteur_global,
                                                name_file_visu = args.name_file_visu,
                                                reset = args.reset_fichier,
                                                file_rotation = args.file_rotation,
                                                visualisation = args.visualisation,
                                                rayon = args.rayon,
                                                nb_clusteur = args.nb_clusteur,
                                                bandwidth = args.bandwidth,
                                                utilisation_exel = args.utilisation_exel
                                                )

    evaluator.print_results(global_metrics, metrics)








