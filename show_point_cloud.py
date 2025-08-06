import argparse
import open3d as o3d
from reglib import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize a .pcd point cloud.")
    parser.add_argument("--input", "-i", required=True,
                        help="Path to the input .pcd file.")
    args = parser.parse_args()

    show_point_cloud(args.input)

# python show_point_cloud.py -i data/K41144.pcd

