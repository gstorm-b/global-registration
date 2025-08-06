import open3d as o3d
from reglib import *
import argparse

def main():
    parser = argparse.ArgumentParser(
        description="Mesh sampling pipeline: mesh -> uniform sampling -> voxel downsample -> save PCD"
    )
    parser.add_argument(
        "-i", "--input", required=True,
        help="Input mesh file path (.ply, .stl, .obj, etc.)"
    )
    parser.add_argument(
        "-n", "--num_samples", type=int, default=100000,
        help="Number of points to sample uniformly from mesh"
    )
    parser.add_argument(
        "-l", "--leaf_size", type=float, default=0.001,
        help="Voxel size for downsampling"
    )
    parser.add_argument(
        "--no_write", action="store_true",
        help="Do not write the output .pcd file"
    )
    args = parser.parse_args()

    pcd = mesh_sampling(
        mesh_path=args.input,
        num_samples=args.num_samples,
        leaf_size=args.leaf_size,
        write_pcd=not args.no_write
    )

    # Print basic info
    print(pcd)

if __name__ == "__main__":
    main()

# python create_point_cloud.py -i data/K41144.stl -n 200000 -l 0.001