# -*- coding: utf-8 -*-
# author: www.pinakinathc.me

import argparse
import os
import cv2
import numpy as np
import scipy.io
import glob
import matplotlib.pyplot as plt

from utils import (get_dominant_from_gvf, get_diagonal_edges,
        get_intersections, get_selected_regions)

def main(image_path, args):
    image_name = os.path.split(image_path)[-1]
    img_color = cv2.imread(image_path)
    H, W, _ = img_color.shape
    large_edge = max(H, W)
    max_edge = min(large_edge, 500) # Max Edge shouldn't be more than 500 pixels
    aspect_ratio = max_edge / float(large_edge)
    img_color = cv2.resize(img_color, (0,0), fx=aspect_ratio, fy=aspect_ratio)
    H, W, _ = img_color.shape
    img = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    canny_edges = cv2.Canny(img, 100, 200)

    # Calculate GVF
    gvf = scipy.io.loadmat(os.path.join(args.mat_dir, image_name[:-3]+"mat"))
    dx = gvf["u"]
    dy = gvf["v"]
    # dx = cv2.Sobel(canny_edges, cv2.CV_64F, 1, 0, ksize=3)*-1 # Horizontal Gradient
    # dy = cv2.Sobel(canny_edges, cv2.CV_64F, 0, 1, ksize=3)*-1 # Vertical Gradient

    max_val = max(dy.max(), dx.max())
    dy, dx = dy/max_val, dx/max_val # normalize

    # Calculate Dominant Pixels
    dominant_pixels = get_dominant_from_gvf(canny_edges, dy, dx, args.gvf_angle_thresh)

    # Calculate Diagonal Edges
    diagonal_edges, components_list = get_diagonal_edges(dominant_pixels,
            args.CCA_size_thresh, args.pca_angles, args.pca_angle_thresh, args.CCA_aspect_ratio_thresh)

    intersections, edge_interpolation = get_intersections(diagonal_edges, components_list,
            args.intersect_area_thresh, args.intersect_num)

    # Get Tree Regions from these points
    selected_regions, radius_lines = get_selected_regions(edge_interpolation, img)

    print (selected_regions.shape, img_color.shape)
    final_img = np.zeros((H, W, 4), np.uint8)
    final_img[:,:,:3] = img_color
    final_img[:,:,3] = selected_regions
    cv2.imwrite(os.path.join(args.vis_save_path, image_name[:-4]+'.png'), final_img)
    
    # intersect_points_tmp = [elem["end"] for elem in edge_interpolation]
    # intersect_points = []
    # for point in intersect_points_tmp:
    #     if point not in intersect_points:
    #         intersect_points.append(point)

    # x = [ele[0] for ele in intersect_points]
    # y = [ele[1] for ele in intersect_points]

    # px = []
    # py = []
    # for i in range(diagonal_edges.shape[0]):
    #     for j in range(diagonal_edges.shape[1]):
    #         if diagonal_edges[i,j] > 0:
    #             px.append(j)
    #             py.append(i)

    # plt.imshow(img, cmap='gray')
    # plt.scatter(px, py, marker='.', c='blue')
    # plt.scatter(x, y, marker='+', s=77, c='red')
    # for [[cx,cy], [x,y]] in radius_lines:
    #     plt.arrow(cx, cy, x-cx, y-cy, head_width=3, length_includes_head=True, color='yellow')

    # plt.show()
    # # plt.savefig(os.path.join(args.vis_save_path, image_name[:-4]+'_image.jpg'))
    # plt.clf()

    # plt.imshow(255 - intersections, cmap='gray')
    # plt.show()
    # # plt.savefig(os.path.join(args.vis_save_path, image_name[:-4]+'_lines.jpg'))
    # plt.clf()

    # plt.imshow(255 - diagonal_edges, cmap='gray')
    # plt.scatter(x, y, marker='+', s=77, c='red')
    # plt.show()
    # # plt.savefig(os.path.join(args.vis_save_path, image_name[:-4]+'_intersection.jpg'))
    # plt.clf()

    # plt.imshow(selected_regions, cmap='gray')
    # plt.show()
    # plt.clf()

    #   if args.vis_save_path != "":
    #       image_name = os.path.split(image_path)[-1]
    #       plt.savefig(os.path.join(args.vis_save_path, image_name))
    #   else:
    #       plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Source code for ACCV project")
    parser.add_argument("image_dir", type=str,
                        help="enter root dir of input images")
    parser.add_argument("mat_dir", type=str,
                        help="enter root directory of GVF or GGVF images")
    parser.add_argument("--gvf_angle_thresh", type=float, default=5.,
                        help="Angle (in Degrees) threshold for GVF Symmetry")
    parser.add_argument("--CCA_size_thresh", type=float, default=10,
                        help="Minimum pixels inside a Connected Compoment for consideration")
    parser.add_argument("--CCA_aspect_ratio_thresh", type=float, default=1.0,
                        help="Remove Components with smaller aspect ratio")
    parser.add_argument("--pca_angles", type=float, nargs="+", default=[45, 135],
                        help="PCA Angles of Connected Components Selected")
    parser.add_argument("--pca_angle_thresh", type=float, default=10,
                        help="PCA Angles thresh for consideration")
    parser.add_argument("--intersect_area_thresh", type=float, default=20,
                        help="Area to consider 3 or 4 line intersections")
    parser.add_argument("--intersect_num", type=int, default=1,
                        help="# of intersection points that should lie inside a region")
    parser.add_argument("--vis_save_path", type=str, default="",
                        help="path to save vis results")
    args = parser.parse_args()
    
    image_path_list = glob.glob(os.path.join(args.image_dir, "*"))
    
    for image_path in image_path_list[0:]:
        main(image_path, args)
