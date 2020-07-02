# -*- coding: utf-8 -*-
# author: www.pinakinathc.me

import numpy as np
import cv2
import itertools
from numba import jit

@jit(nopython=True)
def get_dominant_from_gvf(canny_edges, dy, dx, gvf_angle_thresh):
    H, W = dx.shape
    dominant_pixels = np.zeros((H, W), dtype=np.uint8)

    for y in range(H-2):
        for x in range(W-2):
            if canny_edges[y+1, x+1] == 0: # Skip for Non-Canny Edges
                continue
            a11 = np.degrees(np.arctan2(dy[y, x], dx[y, x]))
            a12 = np.degrees(np.arctan2(dy[y, x+1], dx[y, x+1]))
            a13 = np.degrees(np.arctan2(dy[y, x+2], dx[y, x+2]))
            a21 = np.degrees(np.arctan2(dy[y+1, x], dx[y+1, x]))
            a23 = np.degrees(np.arctan2(dy[y+1, x+2], dx[y+1, x+2]))
            a31 = np.degrees(np.arctan2(dy[y+2, x], dx[y+2, x]))
            a32 = np.degrees(np.arctan2(dy[y+2, x+1], dx[y+2, x+1]))
            a33 = np.degrees(np.arctan2(dy[y+2, x+2], dx[y+2, x+2]))
            # angles = [a11, a12, a13, a21, a23, a31, a32, a33]
            diff = [abs(a11-a33), abs(a12-a32), abs(a13-a31), abs(a21-a23)]
            for val in diff:
                if val >= 180-gvf_angle_thresh and val <= 180+gvf_angle_thresh:
                    dominant_pixels[y+1, x+1] = 255
                    break
    return dominant_pixels


def filter_components_aspect_ratio(component, aspect_ratio_thresh):
    x, y, w, h = cv2.boundingRect(component)
    aspect_ratio = max(w, h) / float(min(w, h))
    if aspect_ratio <= aspect_ratio_thresh:
        return True
    return False


def get_diagonal_edges(dominant_pixels, CCA_size_thresh,
        pca_angles, pca_angle_thresh, aspect_ratio_thresh):

    diagonal_edges = np.zeros_like(dominant_pixels)
    _, contours, _ = cv2.findContours(dominant_pixels, cv2.RETR_TREE,
            cv2.CHAIN_APPROX_SIMPLE)
    components_list = [] # stores mean and angle of each component
    
    for cnt in contours:
        if filter_components_aspect_ratio(cnt, aspect_ratio_thresh):
            continue # Skip if returns True
        cnt = cnt[:,0,:]
        
        if cnt.shape[0] > CCA_size_thresh:
            mean, eigenvectors, eigenvalues = cv2.PCACompute2(cnt.astype(np.float64), np.empty((0)))
            angle = np.degrees(np.arctan2(eigenvectors[0,1], eigenvectors[0,0]))

            for pca in pca_angles:
                if angle >= pca - pca_angle_thresh and angle <= pca + pca_angle_thresh:
                    components_list.append({"mean": mean, 
                        "slope": eigenvectors[0,1]/float(eigenvectors[0,0])})
                    for [x, y] in cnt:
                        diagonal_edges[y, x] = 255
                        
    return diagonal_edges, components_list


def filter_intersection_points(edge_interpolation, H, W,
        intersect_area_thresh, intersect_num):

    new_edge_interpolation = []
    initial_intersection = np.zeros((H, W))

    for element in edge_interpolation:
        [x, y] = element["end"]
        initial_intersection[y, x] = 255

    delta = intersect_area_thresh//2
    for i in range(delta, H-delta):
        for j in range(delta, W-delta):
            if np.sum(initial_intersection[i-delta:i+delta, j-delta:j+delta])/255 >= intersect_num:
                for elem in edge_interpolation:
                    [x, y] = elem["end"]
                    if y>=i-delta and y<=i+delta and x>=j-delta and x<=j+delta:
                        elem["end"] = [j, i]
                        new_edge_interpolation.append(elem)

                        # Ignore already considered Line Interpolation
                        for point in edge_interpolation:
                            if point["start"] == elem["start"]:
                                initial_intersection[point["end"][1], point["end"][0]] = 0

    return new_edge_interpolation


def get_intersections(diagonal_edges, components_list,
        intersect_area_thresh, intersect_num):

    edge_interpolation = [] # store the interpolation of diagonal edges
    intersections = diagonal_edges.copy()
    H, W = diagonal_edges.shape
    
    for idx, source in enumerate(components_list[:-1]):
        for target in components_list[idx:]:
            """we shall represent these 2 lines in the form:
                y = mx + c
                hence, c = y_{1} - mx_{1}
                also given, y = m_{1}x + c_{1} and y = m_{2}x + c_{2}
                x_{intersect} = (c2 - c1)/(m1 - m2)
                y_{intersect} = m1 * (c2-c1)/(m1-m2) + c1
            """
            m1, m2 = source["slope"], target["slope"]
            x1, y1 = source["mean"][0]
            x2, y2 = target["mean"][0]
            c1 = y1 - m1*x1
            c2 = y2 - m2*x2
            slope_diff = m1 - m2
            if slope_diff == 0: # lines are parallel and hance will never meet
                continue
            x_inter = int(round((c2 - c1)/slope_diff))
            y_inter = int(round(m1 * x_inter + c1))
            x1, y1, x2, y2 = list(map(int, [x1, y1, x2, y2]))

            if x_inter < 0 or x_inter >= W or y_inter < 0 or y_inter >= H:
                continue # lines intersect but outside image space
            edge_interpolation.append({"start": [x1, y1], "end":[x_inter, y_inter]})
            edge_interpolation.append({"start": [x2, y2], "end":[x_inter, y_inter]})

            # intersections = cv2.line(intersections, (x1, y1), (x_inter, y_inter), 255, 1)
            # intersections = cv2.line(intersections, (x2, y2), (x_inter, y_inter), 255, 1)
            
    edge_interpolation = filter_intersection_points( 
            edge_interpolation, H, W, intersect_area_thresh, intersect_num)

    for elem in edge_interpolation:
        (x1, y1) = elem["start"]
        (x_inter, y_inter) = elem["end"]
        intersections = cv2.line(intersections, (x1, y1), (x_inter, y_inter), 255, 1)
    return intersections, edge_interpolation


def get_selected_regions(edge_interpolation, img):
    extracted_regions = np.zeros_like(img)
    radius_lines = []
    intersection_points = [elem["end"] for elem in edge_interpolation]
    diagonal_components = [elem["start"] for elem in edge_interpolation]

    for [cx, cy] in intersection_points:
        radius = np.inf
        min_x, min_y = np.inf, np.inf
        for [x, y] in diagonal_components:
            dist = ((x-cx)**2 + (y-cy)**2)**0.5
            if dist < radius:
                radius = dist
                min_x, min_y = x, y
        assert radius != np.inf, "Did not find any component for intersection point"
        radius, min_x, min_y = list(map(int, [radius, min_x, min_y]))
        extracted_regions = cv2.circle(extracted_regions, (cx, cy), radius, 1, -1)
        radius_lines.append([[cx, cy], [min_x, min_y]])

    extracted_regions = extracted_regions * img
    return extracted_regions, radius_lines
