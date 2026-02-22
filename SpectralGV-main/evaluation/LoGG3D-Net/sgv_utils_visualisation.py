# Functions in this file are adapted from: https://github.com/ZhiChen902/SC2-PCR/blob/main/SC2_PCR.py

import numpy as np
import torch
import json
import matplotlib.pyplot as plt
import inspect
import os
import re


def match_pair_parallel(src_keypts, tgt_keypts, src_features, tgt_features):
    # normalize:
    src_features = torch.nn.functional.normalize(src_features, p=2.0, dim=1)
    tgt_features = torch.nn.functional.normalize(tgt_features, p=2.0, dim=1)

    distance = torch.cdist(src_features, tgt_features)
    min_vals, min_ids = torch.min(distance, dim=2)
 
    min_ids = min_ids.unsqueeze(-1).expand(-1, -1, 3)
    tgt_keypts_corr = torch.gather(tgt_keypts, 1, min_ids)
    src_keypts_corr = src_keypts

    return src_keypts_corr, tgt_keypts_corr

def power_iteration(M, num_iterations=5):
    """
    Calculate the leading eigenvector using power iteration algorithm
    Input:
        - M:      [bs, num_pts, num_pts] the adjacency matrix
    Output:
        - leading_eig: [bs, num_pts] leading eigenvector
    """
    leading_eig = torch.ones_like(M[:, :, 0:1])
    leading_eig_last = leading_eig
    for i in range(num_iterations):
        leading_eig = torch.bmm(M, leading_eig)
        leading_eig = leading_eig / (torch.norm(leading_eig, dim=1, keepdim=True) + 1e-6)
        if torch.allclose(leading_eig, leading_eig_last):
            break
        leading_eig_last = leading_eig
    leading_eig = leading_eig.squeeze(-1)
    return leading_eig


def cal_spatial_consistency( M, leading_eig):
    """
    Calculate the spatial consistency based on spectral analysis.
    Input:
        - M:          [bs, num_pts, num_pts] the adjacency matrix
        - leading_eig [bs, num_pts]           the leading eigenvector of matrix M
    Output:
        - sc_score_list [bs, 1]
    """
    spatial_consistency = leading_eig[:, None, :] @ M @ leading_eig[:, :, None]
    spatial_consistency = spatial_consistency.squeeze(-1) / M.shape[1]
    return spatial_consistency


def sgv(src_keypts, tgt_keypts, src_features, tgt_features,src_keypts_tout,tgt_keypts_tout,src_features_tout,tgt_features_tout,k,nom_dossier,data_rotation,name_file_visu, d_thresh=5.0):
    """
    Input:
        - src_keypts: [1, num_pts, 3]
        - tgt_keypts: [bs, num_pts, 3]
        - src_features: [1, num_pts, D]
        - tgt_features: [bs, num_pts, D]
    Output:
        - sc_score_list:   [bs, 1], spatial consistency score for each candidate
    """
    # Correspondence Estimation: Nearest Neighbour Matching
    src_keypts_corr, tgt_keypts_corr = match_pair_parallel(src_keypts, tgt_keypts, src_features, tgt_features)
    ############################# modificat #############################

    src_keypts_np = src_keypts.cpu().numpy()
    tgt_keypts_np = tgt_keypts.cpu().numpy()
    src_keypts_np_tout = src_keypts_tout.cpu().numpy()
    tgt_keypts_np_tout = tgt_keypts_tout.cpu().numpy()
    src_keypts_corr_np = src_keypts_corr.cpu().numpy()
    tgt_keypts_corr_np = tgt_keypts_corr.cpu().numpy()
 


    # === Détection du nom du fichier courant (sans extension) ===


    # === Dossier principal et dossier spécifique au script ===
    folder_path = os.getenv("SCRATCH")
    base_dir = name_file_visu
    script_dir = os.path.join(folder_path,base_dir, os.path.basename(nom_dossier))

    # === Dossiers pour les figures ===
    fig_dirs = {
        "figure1": "nuage_point_requete",
        "figure2": "nuage_point_corelation_requete",
        "figure3": "nuage_point_corelation_map",
        "figure4": "nuage_point_map"
    }

    pattern = r'rotated_(\d+)_(\d+)_(\d+)'
    ###### il faut modifier ici pour prendre le .json #############
    rotation_x = data_rotation['rotation'][0]
    rotation_y = data_rotation['rotation'][1]
    rotation_z = data_rotation['rotation'][2]
    translation_x = data_rotation['translation'][0]
    translation_y = data_rotation['translation'][1]
    translation_z = data_rotation['translation'][2]
    """
    match = re.search(pattern, nom_dossier)

    if match:
        # Extraire les valeurs de rotation
        rotation_x = int(match.group(1))
        rotation_y = int(match.group(2))
        rotation_z = int(match.group(3))

    pattern = r'translation_(\d+)_(\d+)_(\d+)'
    match = re.search(pattern, nom_dossier)
    
    
    if match:
        # Extraire les valeurs de rotation
        translation_x = int(match.group(1))
        translation_y = int(match.group(2))
        translation_z = int(match.group(3))
    """
   
    # === Création des dossiers ===
    for fig_key, sub_dir in fig_dirs.items():
        path = os.path.join(script_dir, sub_dir)
        os.makedirs(path, exist_ok=True)
    


    def rotation_matrix_x(phi):
        return np.array([
            [1, 0, 0],
            [0, np.cos(phi), -np.sin(phi)],
            [0, np.sin(phi), np.cos(phi)]
        ])

    def rotation_matrix_y(theta):
        return np.array([
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)]
        ])

    def rotation_matrix_z(gamma):
        return np.array([
            [np.cos(gamma), -np.sin(gamma), 0],
            [np.sin(gamma), np.cos(gamma), 0],
            [0, 0, 1]
        ])

    def rotate_points(points, phi, theta, gamma, rotation_center):

        phi = np.radians(phi)
        theta = np.radians(theta)
        gamma = np.radians(gamma)

        R_x = rotation_matrix_x(phi)
        R_y = rotation_matrix_y(theta)
        R_z = rotation_matrix_z(gamma)
        R = R_z @ R_y @ R_x

        coordinates = points[:, :3]
        

        # Centrage sur le centre de rotation extrait du nom
        centered_coordinates = coordinates - rotation_center
        rotated_points = (R @ centered_coordinates.T).T + rotation_center

        

        return rotated_points
    
    # === Fonction de rotation ===
    def rotation_xyz(points_x,points_y,points_z, rot_x_deg, rot_y_deg, rot_z_deg,translation_x, translation_y, translation_z):
        rx = np.radians(rot_x_deg)
        ry = np.radians(rot_y_deg)
        rz = np.radians(rot_z_deg)
        
        Rx = np.array([[1, 0, 0],
                    [0, np.cos(rx), -np.sin(rx)],
                    [0, np.sin(rx), np.cos(rx)]])
        
        Ry = np.array([[np.cos(ry), 0, np.sin(ry)],
                    [0, 1, 0],
                    [-np.sin(ry), 0, np.cos(ry)]])
        
        Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                    [np.sin(rz), np.cos(rz), 0],
                    [0, 0, 1]])
        
        R = Rz @ Ry @ Rx  # Rotation combinée
        points = np.vstack((points_x, points_y, points_z)).T
        tranlation = np.array([translation_x, translation_y, translation_z])
        #import pdb; pdb.set_trace()
        centered_coordinates = points - tranlation

        rotated_coordinates = (R @ centered_coordinates.T).T + tranlation

        return rotated_coordinates
    
    # === 1. Nuage de points complet (requête) ===
    
    # Appliquer la rotation sur tes nuages de points avant d’afficher
    rotation_center = np.array([translation_x, translation_y, translation_z])
    points_src_keypts_np_tout = src_keypts_np_tout[0,:,:]
    points_src_keypts_np= src_keypts_np[0,:,:]
    points_src_keypts_corr_np =  src_keypts_corr_np[0,:,:]
    
    src_keypts_np_tout_rot = rotate_points(points_src_keypts_np_tout, -rotation_x, -rotation_y, -rotation_z, rotation_center)
    src_keypts_np_rot = rotate_points(points_src_keypts_np, -rotation_x, -rotation_y, -rotation_z, rotation_center)
    src_keypts_corr_np_rot = rotate_points(points_src_keypts_corr_np, -rotation_x, -rotation_y, -rotation_z, rotation_center)

    # src_keypts_np_tout_rot = points_src_keypts_np_tout
    # src_keypts_np_rot = points_src_keypts_np
    # src_keypts_corr_np_rot = points_src_keypts_corr_np
    
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111, projection='3d')
    
    # Dessiner les points bleus avec une couleur plus claire et une transparence plus élevée
    ax1.scatter(src_keypts_np_tout_rot[:,0], src_keypts_np_tout_rot[:,1], src_keypts_np_tout_rot[:,2],
                s=0.1, c='blue', alpha=0.1, label='Nuage de points complet (requête)', zorder=2, depthshade=False, antialiased=True)

    # Dessiner les points rouges avec une taille plus grande et une couleur plus visible
    ax1.scatter(src_keypts_np_rot[:,0], src_keypts_np_rot[:,1], src_keypts_np_rot[:,2],
                s=0.2, c='red', alpha=1, label='Points caractéristiques (requête)', zorder=1, depthshade=False, antialiased=True)

    ax1.set_title(f"Nuage de points complet (requête) - {src_keypts_np_rot.shape[0]} points")


    # # Dessiner les points bleus avec une couleur plus claire et une transparence plus élevée
    # ax1.scatter(points_src_keypts_np_tout[:,0], points_src_keypts_np_tout[:,1], points_src_keypts_np_tout[:,2],
    #             s=0.1, c='blue', alpha=0.1, label='Nuage de points complet (requête)', zorder=2, depthshade=False, antialiased=True)

    # # Dessiner les points rouges avec une taille plus grande et une couleur plus visible
    # ax1.scatter(points_src_keypts_np[:,0], points_src_keypts_np[:,1], points_src_keypts_np[:,2],
    #             s=0.2, c='red', alpha=1, label='Points caractéristiques (requête)', zorder=1, depthshade=False, antialiased=True)

    ax1.set_title(f"Nuage de points complet (requête) - {points_src_keypts_np.shape[0]} points")

    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_aspect('auto')
    ax1.view_init(elev=29, azim=-133)
    ax1.legend(scatterpoints=1, markerscale=5)


    fig1.savefig(os.path.join(script_dir, fig_dirs["figure1"], "figure1_nuage_point_requete.png"))

    
    # === 2. Points correspondants (requête) ===
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111, projection='3d')
    ax2.scatter(src_keypts_corr_np_rot[:,0], src_keypts_corr_np_rot[:,1], src_keypts_corr_np_rot[:,2],
                s=2, c='red', alpha=0.5)
    ax2.set_title(f"Points correspondants (Requête) - {src_keypts_corr_np_rot.shape[1]} points")
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_aspect('auto')
    ax2.view_init(elev=29, azim=-133)
    fig2.savefig(os.path.join(script_dir, fig_dirs["figure2"], "figure2_corelation_requete.png"))

    """
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111, projection='3d')
    ax1.scatter(src_keypts_np_tout[0,:,0], src_keypts_np_tout[0,:,1], src_keypts_np_tout[0,:,2],
                s=0.5, c='blue', alpha=0.5, label='Nuage de points complet (requête)')
    ax1.scatter(src_keypts_np[0,:,0], src_keypts_np[0,:,1], src_keypts_np[0,:,2],
                s=2, c='red', alpha=0.5 , label='Points caractéristiques (requête)')

    ax1.set_title(f"Nuage de points complet (requête) - {src_keypts_np.shape[1]} points")
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_aspect('auto')
    ax1.view_init(elev=29, azim=-133)
    ax1.legend(scatterpoints=1, markerscale=5)

    fig1.savefig(os.path.join(script_dir, fig_dirs["figure1"], "figure1_nuage_point_requete.png"))
    

    # === 2. Points correspondants (requête) ===
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111, projection='3d')
    ax2.scatter(src_keypts_corr_np[0,:,0], src_keypts_corr_np[0,:,1], src_keypts_corr_np[0,:,2],
                s=2, c='red', alpha=0.5)
    ax2.set_title(f"Points correspondants (Requête) - {src_keypts_corr_np_rot.shape[1]} points")
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_aspect('auto')
    ax2.view_init(elev=29, azim=-133)
    fig2.savefig(os.path.join(script_dir, fig_dirs["figure2"], "figure2_corelation_requete.png"))
    """
    
    # === 3. Points correspondants (map) ===
    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111, projection='3d')
    ax3.scatter(tgt_keypts_corr_np[0,:,0], tgt_keypts_corr_np[0,:,1], tgt_keypts_corr_np[0,:,2],
                s=2, c='green')
    ax3.set_title(f"Points correspondants (Map) - {tgt_keypts_corr_np.shape[1]} points")
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    ax3.set_aspect('auto')
    ax3.view_init(elev=29, azim=-133)
    fig3.savefig(os.path.join(script_dir, fig_dirs["figure3"], f"figure3_corelation_map_{k}.png"))
    

    # === 4. Nuage de points complet (map) ===
    fig4 = plt.figure()
    ax4 = fig4.add_subplot(111, projection='3d')
    ax4.scatter(tgt_keypts_np_tout[0,:,0], tgt_keypts_np_tout[0,:,1], tgt_keypts_np_tout[0,:,2],
                s=0.1, c='blue', alpha=0.1 ,label='Nuage de points complet (map)', zorder=2, depthshade=False, antialiased=True)
    ax4.scatter(tgt_keypts_np[0,:,0], tgt_keypts_np[0,:,1], tgt_keypts_np[0,:,2],
                s=0.2, c='red', alpha=1 , label='Points caractéristiques (map)', zorder=1, depthshade=False, antialiased=True)
    ax4.set_title(f"Nuage de points complet (map) - {tgt_keypts_np.shape[1]} points")
    ax4.set_xlabel('X')
    ax4.set_ylabel('Y')
    ax4.set_zlabel('Z')
    ax4.set_aspect('auto')
    ax4.view_init(elev=29, azim=-133)
    
    ax4.legend(scatterpoints=1, markerscale=5)
    fig4.savefig(os.path.join(script_dir, fig_dirs["figure4"], f"figure4_nuage_point_map{k}.png"))
    #print(rotation_x, rotation_y, rotation_z)
    #print(src_keypts_np_rot)
    #print(tgt_keypts_np_tout)
    #print(src_keypts_np_tout_rot)
   
    
    
    
    ##################################################################

    # Spatial Consistency Adjacency Matrix
    print(src_keypts_corr[:, :, None, :].shape, src_keypts_corr[:, None, :, :].shape)
    src_dist = torch.norm((src_keypts_corr[:, :, None, :] - src_keypts_corr[:, None, :, :]), dim=-1)
    target_dist = torch.norm((tgt_keypts_corr[:, :, None, :] - tgt_keypts_corr[:, None, :, :]), dim=-1)
    cross_dist = torch.abs(src_dist - target_dist)
    adj_mat = torch.clamp(1.0 - cross_dist ** 2 / d_thresh ** 2, min=0)

    # Spatial Consistency Score
    lead_eigvec = power_iteration(adj_mat)
    sc_score_list = cal_spatial_consistency(adj_mat, lead_eigvec)

    sc_score_list = np.squeeze(sc_score_list.cpu().detach().numpy())
    return sc_score_list

def sgv_fn(query_keypoints, candidate_keypoints,query_keypoints_tout,candidate_keypoints_tout ,D3Feat,k,nom_dossier ,name_file_visu, d_thresh=5.0, min_num_feat=15000):

    kp1 = query_keypoints['keypoints']
    kp2 = candidate_keypoints['keypoints']
    f1 = query_keypoints['features']
    f2 = candidate_keypoints['features']

    kp1_tout = query_keypoints_tout['keypoints']
    kp2_tout = candidate_keypoints_tout['keypoints']
    f1_tout = query_keypoints_tout['features']
    f2_tout = candidate_keypoints_tout['features']

    if D3Feat == False :

        kp1 = kp1[:min_num_feat]
        kp2 = kp2[:min_num_feat]
        f1 = f1[:min_num_feat]
        f2 = f2[:min_num_feat]


    ###############################
    # draw_registration_result(kp1, kp2, np.eye(4))

    if torch.cuda.is_available():
        src_keypts = kp1.unsqueeze(0).cuda()
        tgt_keypts = kp2.unsqueeze(0).cuda() 
        src_features = f1.unsqueeze(0).cuda()
        tgt_features = f2.unsqueeze(0).cuda()
        ###########modification ###########
        src_keypts_tout = kp1_tout.unsqueeze(0).cuda()
        tgt_keypts_tout = kp2_tout.unsqueeze(0).cuda()
        src_features_tout = f1_tout.unsqueeze(0).cuda()
        tgt_features_tout = f2_tout.unsqueeze(0).cuda()
        ######################################

        
    else:
        src_keypts = kp1.unsqueeze(0).to(torch.device("cpu"))
        tgt_keypts = kp2.unsqueeze(0).to(torch.device("cpu"))
        src_features = f1.unsqueeze(0).to(torch.device("cpu"))
        tgt_features = f2.unsqueeze(0).to(torch.device("cpu"))
        ###########modification ###########
        src_keypts_tout = kp1_tout.unsqueeze(0).to(torch.device("cpu"))
        tgt_keypts_tout = kp2_tout.unsqueeze(0).to(torch.device("cpu"))
        src_features_tout = f1_tout.unsqueeze(0).to(torch.device("cpu"))
        tgt_features_tout = f2_tout.unsqueeze(0).to(torch.device("cpu"))
        ######################################
    
    nom_files = os.path.basename(nom_dossier)
    fichier_json = os.path.dirname(os.path.dirname(nom_dossier))
    nom_json = os.path.join(os.path.basename(fichier_json)+".json")

    if os.path.exists(os.path.join(fichier_json, nom_json)):
        with open(os.path.join(fichier_json, nom_json), 'r', encoding='utf-8') as f:
            data_json_index = json.load(f)
        data_rotation = data_json_index[nom_files]
    else:
        print(f"Le fichier JSON {nom_json} n'existe pas dans le dossier {fichier_json}.")
        data_rotation = {'rotation': [0, 0, 0], 'translation': [0, 0, 0]}

    



    conf = sgv(src_keypts, tgt_keypts, src_features, tgt_features,src_keypts_tout,tgt_keypts_tout,src_features_tout,tgt_features_tout ,k ,nom_dossier,data_rotation,name_file_visu,d_thresh=d_thresh)



    return  conf