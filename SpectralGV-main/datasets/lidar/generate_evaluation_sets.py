# Test set for Kitti Sequence 00 dataset.
# Following procedures in EgoNN, we use 170 seconds of drive from sequence for map generation
# and the rest is left for queries
# This file is directly copied from: https://github.com/jac99/Egonn/blob/main/datasets/kitti/generate_evaluation_sets.py

import numpy as np
import argparse
from typing import List
import os
import sys
import json
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from datasets.lidar.lidar_raw import LidarSequence
from datasets.base_datasets import EvaluationTuple, EvaluationSet, filter_query_elements


MAP_TIMERANGE = (0, 170)

def get_scans(sequence_query: LidarSequence, sequence_map: LidarSequence, json_path_query: str, json_path_database: str) -> List[EvaluationTuple]:
    """
    Extracts a list of point cloud scans from query and map sequences, using provided JSON files for indexing.

    Args:
        sequence_query (LidarSequence): Lidar sequence object for the query set.
        sequence_map (LidarSequence): Lidar sequence object for the map set.
        json_path_query (str): Path to the JSON file containing query scan filenames.
        json_path_database (str): Path to the JSON file containing map scan filenames.

    Returns:
        tuple: A tuple containing two lists of EvaluationTuple objects:
            - map_set (List[EvaluationTuple]): List of EvaluationTuples for the map set.
            - query_set (List[EvaluationTuple]): List of EvaluationTuples for the query set.

    Description:
        This function reads the JSON files to get the filenames of the scans, then retrieves the corresponding
        timestamps, file paths, positions, and poses from the LidarSequence objects. It constructs EvaluationTuple
        objects for each scan and returns them as separate lists for the map and query sets.
    """
    # Get a list of all point clouds from the sequence (the full sequence or test split only)
    query_set = []
    map_set = []

    with open(json_path_query, 'r', encoding='utf-8') as f:
        data_query = json.load(f)

    with open(json_path_database, 'r', encoding='utf-8') as f:
        data_database = json.load(f)

    for ndx in range(len(data_query)):
        indice = os.path.splitext(os.path.basename(data_query[ndx]))[0]
       
        valeur = sequence_query.dictionary_id.get(indice)
        pose = sequence_query.lidar_poses[valeur]

        # Kitti poses are in camera coordinates system where y is the upward axis
        position = pose[[0, 1], 3]
        item_query = EvaluationTuple(sequence_query.rel_lidar_timestamps[valeur],
                                    sequence_query.rel_scan_filepath[valeur],
                                    position,
                                    pose)
        if os.path.splitext(os.path.basename(sequence_query.rel_scan_filepath[valeur]))[0] != indice:
            print("erreur lors de la regeration du .spickle")
            break
        query_set.append(item_query)

    for ndx in range(len(data_database)):
        indice = os.path.splitext(os.path.basename(data_database[ndx]))[0]
        valeur = sequence_map.dictionary_id.get(indice)
        pose = sequence_map.lidar_poses[valeur]

        # Kitti poses are in camera coordinates system where y is the upward axis
        position = pose[[0, 1], 3]

        item_query = EvaluationTuple(sequence_map.rel_lidar_timestamps[valeur],
                                    sequence_map.rel_scan_filepath[valeur],
                                    position,
                                    pose)
        if os.path.splitext(os.path.basename(sequence_map.rel_scan_filepath[valeur]))[0] != indice:
            print("erreur lors de la regeration du .spickle")
            break
        map_set.append(item_query)


    return map_set, query_set



def generate_evaluation_set(dataset_root: str, json_path_query: str, json_path_database: str,
                            rel_lidar_path_query: str, rel_lidar_path_map: str) -> EvaluationSet:
    """
    Generates an evaluation set containing map and query point cloud scans for evaluation purposes.

    Args:
        dataset_root (str): Root directory of the dataset.
        json_path_query (str): Path to the JSON file containing query scan filenames.
        json_path_database (str): Path to the JSON file containing map scan filenames.
        rel_lidar_path_query (str): Relative path to the LiDAR scans for the query set.
        rel_lidar_path_map (str): Relative path to the LiDAR scans for the map set.

    Returns:
        EvaluationSet: An EvaluationSet object containing:
            - query_set (List[EvaluationTuple]): List of EvaluationTuples for the query set.
            - map_set (List[EvaluationTuple]): List of EvaluationTuples for the map set.

    Description:
        This function initializes LidarSequence objects for both query and map sets, retrieves the scans
        using the provided JSON files, and constructs an EvaluationSet object. It prints the number of
        database and query elements before returning the EvaluationSet.
    """
    sequence_query = LidarSequence(dataset_root, rel_lidar_path_query)
    sequance_map = LidarSequence(dataset_root, rel_lidar_path_map)
    
    map_set, query_set = get_scans(sequence_query, sequance_map, json_path_query, json_path_database)

    print(f'{len(map_set)} database elements, {len(query_set)} query elements')
    return EvaluationSet(query_set, map_set)


    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate evaluation sets for KITTI dataset')

    parser.add_argument('--dataset_root', type=str, required=False, default='', help='Root directory path of the KITTI dataset.')
    parser.add_argument('--json_path_query', type=str, required=False, default='', help='JSON file path listing query scan filenames for the evaluation set.')
    parser.add_argument('--json_path_database', type=str, required=False, default='', help='JSON file path listing map scan filenames for the evaluation set.')
    parser.add_argument('--rel_lidar_path_query', type=str, required=False, default='', help='Relative path to the query LiDAR scans directory (relative to dataset_root).')
    parser.add_argument('--rel_lidar_path_map', type=str, required=False, default='', help='Relative path to the map LiDAR scans directory (relative to dataset_root).')
    parser.add_argument('--name_pickle', type=str, required=False, default='', help='Output pickle filename to save the generated evaluation set.')



    

    args = parser.parse_args()

    # Sequences are fixed
    
    print(f'Dataset root: {args.dataset_root}')
    
    

    lidar_eval_set = generate_evaluation_set(args.dataset_root, args.json_path_query ,args.json_path_database , args.rel_lidar_path_query , args.rel_lidar_path_map )

    file_path_name = os.path.join(os.path.dirname(__file__),args.name_pickle )
    print(f"Saving evaluation pickle: {file_path_name}")
    lidar_eval_set.save(file_path_name)
    

