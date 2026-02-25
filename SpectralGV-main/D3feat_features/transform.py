import argparse
import os
import glob
import numpy as np
import open3d as o3d


TARGET_DIAGONAL = 10.0  # Taille cible pour la mise à l'échelle


def load_point_clouds(input_dir):
    ply_files = sorted(glob.glob(os.path.join(input_dir, "*.ply")))
    all_points = []
    for file in ply_files:
        pcd = o3d.io.read_point_cloud(file)
        all_points.append(np.asarray(pcd.points))
    return ply_files, all_points


def compute_global_centroid_and_scale(all_points):
    concatenated = np.concatenate(all_points, axis=0)
    centroid = np.mean(concatenated, axis=0)
    centered = concatenated - centroid
    min_corner = centered.min(axis=0)
    max_corner = centered.max(axis=0)
    diagonal = np.linalg.norm(max_corner - min_corner)
    scale = TARGET_DIAGONAL / diagonal
    return centroid, scale


def apply_random_z_rotation(points):
    angle = np.random.uniform(0, 2 * np.pi)
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    rotation_matrix = np.array([
        [cos_a, -sin_a, 0],
        [sin_a,  cos_a, 0],
        [0,      0,     1]
    ])
    return points @ rotation_matrix.T


def process_and_save(ply_files, all_points, centroid, scale_factor, output_dir, apply_scale, apply_rotation):
    os.makedirs(output_dir, exist_ok=True)

    for file_path, points in zip(ply_files, all_points):
        filename = os.path.basename(file_path)
        print(f"Processing: {filename}")

        transformed = points.copy()
        if apply_scale:
            transformed = (transformed - centroid) * scale_factor
        else:
            transformed = transformed - centroid

        if apply_rotation:
            transformed = apply_random_z_rotation(transformed)

        # Créer le nuage transformé
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(transformed)

        # Enregistrer le fichier
        output_file = os.path.join(output_dir, filename.replace(".ply", "_processed.ply"))
        o3d.io.write_point_cloud(output_file, pcd)
        print(f"Saved: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Normalize and transform 3D point clouds.")
    parser.add_argument('--input_dir', required=True, help='Répertoire contenant les fichiers .ply')
    parser.add_argument('--output_dir', required=True, help='Répertoire de sortie pour les fichiers transformés')
    parser.add_argument('--scale', action='store_true', help='Appliquer la mise à l\'échelle globale')
    parser.add_argument('--rotation', action='store_true', help='Appliquer une rotation aléatoire autour de Z')

    args = parser.parse_args()

    ply_files, all_points = load_point_clouds(args.input_dir)

    if not ply_files:
        print("Aucun fichier .ply trouvé dans le répertoire spécifié.")
        return

    centroid, scale_factor = compute_global_centroid_and_scale(all_points)
    print(f"Global centroid: {centroid}")
    print(f"Scale factor: {scale_factor}")

    process_and_save(ply_files, all_points, centroid, scale_factor,
                     args.output_dir, args.scale, args.rotation)


if __name__ == "__main__":
    main()
