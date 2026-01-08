import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import trimesh
import multiprocessing as mp
from tqdm import tqdm
from typing import Tuple


def normalize_disparity_map(disparity_map):
    '''Normalize disparity map for visualization 
    disparity should be larger than zero
    '''
    return np.maximum(disparity_map, 0.0) / (disparity_map.max() + 1e-10)


def visualize_disparity_map(disparity_map, gt_map, save_path=None):
    '''Visualize or save disparity map and compare with ground truth
    '''
    # Normalize disparity maps
    disparity_map = normalize_disparity_map(disparity_map)
    gt_map = normalize_disparity_map(gt_map)
    # Visualize or save to file
    if save_path is None:
        concat_map = np.concatenate([disparity_map, gt_map], axis=1)
        plt.imshow(concat_map, 'gray')
        plt.show()
    else:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        concat_map = np.concatenate([disparity_map, gt_map], axis=1)
        plt.imsave(save_path, concat_map, cmap='gray')

import time
def task1_compute_disparity_map_simple(
    ref_img: np.ndarray,        # shape (H, W)
    sec_img: np.ndarray,        # shape (H, W)
    window_size: int, 
    disparity_range: Tuple[int, int],   # (min_disparity, max_disparity)
    matching_function: str      # can be 'SSD', 'SAD', 'normalized_correlation'
):
    '''Assume image planes are parallel to each other
    Compute disparity map using simple stereo system following the steps:
    1. For each row, scan all pixels in that row
    2. Generate a window for each pixel in ref_img
    3. Search for a disparity (d) within (min_disparity, max_disparity) in sec_img 
    4. Select the best disparity that minimize window difference between ref_img[row, col] and sec_img[row, col - d]
    '''
    ref_img = ref_img.astype(np.float32)
    sec_img = sec_img.astype(np.float32)

    H,W = ref_img.shape
    disparity_map = np.zeros((H,W), dtype=np.float32)
    d_min, d_max = disparity_range
    half = window_size // 2
    for y in range(H):
        if y - half < 0 or y + half + 1 > H:
                continue
        for x in range(W):
            if x - half < 0 or x + half + 1 >W:
                continue
            window_1 = ref_img[y-half : y+half+1, x-half : x+half+1]
            best_d = 0
            diff_min = float('inf')
            for d in range(d_min, d_max+1):
                x_sec = x-d
                if x_sec-half < 0 or x_sec + half + 1 > W:
                    continue

                window_2 = sec_img[y-half : y+half+1, x_sec-half : x_sec+half+1]
                diff = 0
                if matching_function == 'SSD':
                    diff = np.sum((window_1 - window_2)**2)
                elif matching_function == 'SAD':
                    diff = np.sum(np.abs(window_1 - window_2))
                elif matching_function == 'normalized_correlation':
                    mean_1 = window_1.mean()
                    mean_2 = window_2.mean()
                    tmp1 = np.sum((window_1-mean_1) * (window_2-mean_2))
                    tmp2 = np.sqrt(np.sum((window_1 - mean_1)**2) * np.sum((window_2-mean_2)**2))
                    if tmp2 ==0:
                        diff = 0.0
                    else:
                        diff = - tmp1 / tmp2

                if diff<diff_min:
                    diff_min=diff
                    best_d=d

            disparity_map[y,x] = best_d
    return disparity_map

def task1_simple_disparity(ref_img, sec_img, gt_map, img_name='tsukuba'):
    '''Compute disparity maps for different settings
    '''
    window_sizes = [13]  # Try different window sizes
   # window_sizes =[3,5,7,9,13,19,25,31]
    disparity_range = (0,15)  # Determine appropriate disparity range
    matching_functions = ['SSD'] 
    #matching_functions = ['SSD', 'SAD', 'normalized_correlation']  # Try different matching functions

    disparity_maps = []
    
    # Generate disparity maps for different settings
    for window_size in window_sizes:
        for matching_function in matching_functions:
            start_time=time.time()
            print(f"Computing disparity map for window_size={window_size}, disparity_range={disparity_range}, matching_function={matching_function}")
            disparity_map = task1_compute_disparity_map_simple(
                ref_img, sec_img, 
                window_size, disparity_range, matching_function)
            end_time = time.time()
            disparity_maps.append((disparity_map, window_size, matching_function, disparity_range))
            dmin, dmax = disparity_range
            visualize_disparity_map(
                disparity_map, gt_map, 
                save_path=f"output/task1_{img_name}_{window_size}_{dmin}_{dmax}_{matching_function}.png")
            print("Task 1, function called, time consumption:",end_time-start_time)
    return disparity_maps


def task2_compute_depth_map(disparity_map, baseline, focal_length):
    '''
    Compute depth map by z = fB / (x - x')
    Note that a disparity less or equal to zero should be ignored (set to zero) 
    '''
    H,W = disparity_map.shape 
    depth_map = np.zeros((H,W),dtype=np.float32)
    for x in range(H):
        for y in range(W):
            disparity = disparity_map[x,y]
            if disparity > 0:
                depth_map[x,y] = baseline * focal_length / disparity
    return depth_map


def task2_visualize_pointcloud(
    ref_img: np.ndarray,        # shape (H, W, 3) 
    disparity_map: np.ndarray,  # shape (H, W)
    save_path: str = 'output/task2_tsukuba.ply'
):
    '''Visualize 3D pointcloud from disparity map following the steps:
    1. Calculate depth map from disparity
    2. Set pointcloud's XY as image's XY and and pointcloud's Z as depth
    3. Set pointcloud's color as ref_img's color
    4. Save pointcloud to ply files for visualizationh. We recommend to open ply file with MeshLab
    5. Adjust the baseline and focal_length for better performance
    6. You may need to cut some outliers for better performance
    '''
    baseline = 10
    focal_length = 120
    depth_map = task2_compute_depth_map(disparity_map, baseline, focal_length)
    H,W = disparity_map.shape
    N=H*W

    X,Y = np.meshgrid(np.arange(W),np.arange(H))
    X=X.flatten()
    Y=Y.flatten()
    Z=depth_map.flatten()
    colors = ref_img.reshape(-1, 3)[:, ::-1]
    z_valid = Z > 0
    if np.any(z_valid):
        zmin, zmax = np.percentile(Z[z_valid], [1, 99])
        valid = z_valid & (Z >= zmin) & (Z <= zmax)
    else:
        valid = z_valid
    points = np.stack([X[valid], Y[valid], Z[valid]], axis=-1).astype(np.float32)
    colors = colors[valid].astype(np.uint8)
    """
    #修改前方法：遍历每个像素点，较慢
    # Points
    points = np.zeros((N,3),dtype=np.float32)
    # Colors 
    colors = np.zeros((N,3),dtype=np.uint8)
    idx = 0
    for x in range(H):        
        for y in range(W):     
            z = depth_map[x, y]
            if z <= 0: 
                continue
            zmin, zmax = np.percentile(depth_map, [1, 99])
            if z < zmin or z > zmax: 
                continue
            points[idx] = (y,x, z)
            b, g, r = ref_img[x, y]
            colors[idx] = (r, g, b) #颜色转换
            idx += 1
    points = points[:idx]
    colors = colors[:idx]
    """
    # Save pointcloud to ply file
    pointcloud = trimesh.PointCloud(points, colors)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    pointcloud.export(save_path, file_type='ply')


def task3_compute_disparity_map_dp(ref_img, sec_img):
    ''' Conduct stereo matching with dynamic programming
    '''
    ref_img = ref_img.astype(np.float32)
    sec_img = sec_img.astype(np.float32)
    H,W= ref_img.shape
    disparity_map_dp = np.zeros((H,W), dtype=np.float32)
    half = 13//2
    max_d = 20
    occlusion = 40000
    for y in range(half,H-half):
        cost_mat = np.zeros((W,W), dtype=np.float32)
        cost_mat.fill(np.inf)
        for ref_x in range(half,W-half):
            ref_window = ref_img[y-half : y+half+1, ref_x-half : ref_x+half+1]
            for sec_x in range(max(half, ref_x - max_d), ref_x+1):
                sec_window = sec_img[y-half : y+half+1, sec_x-half : sec_x+half+1]
                #ssd
                cost_mat[ref_x][sec_x] = np.sum((ref_window-sec_window)**2)

        path = np.zeros((W,W), dtype=np.int32)
        c = np.zeros((W,W), dtype=np.float32)
        for x in range(W):
            c[x,0] = x * occlusion
            c[0,x] = x * occlusion
        for ref in range(1,W):
            for sec in range(1,W):
                min1 = c[ref-1,sec-1] + cost_mat[ref,sec]
                min2 = c[ref,sec-1] + occlusion
                min3 = c[ref-1,sec] + occlusion
                c[ref,sec] = min(min1, min2, min3)
                path[ref,sec] = [min1, min2, min3].index(c[ref,sec])

        ref_x= W-1
        sec_x= W-1
        while ref_x>0 and sec_x>0:
            choice = path[ref_x,sec_x]
            if choice == 0:
                disparity_map_dp[y,ref_x] = abs(ref_x - sec_x)
                ref_x -= 1
                sec_x -= 1
            elif choice == 1:
                sec_x -= 1
            else:
                ref_x -= 1
        for x in range(W):
            if disparity_map_dp[y,x] == 0:
                disparity_map_dp[y,x] = disparity_map_dp[y,x-1]
    return disparity_map_dp


def main(tasks): 
    
    # Read images and ground truth disparity maps
    moebius_img1 = cv2.imread("data/moebius1.png")
    moebius_img1_gray = cv2.cvtColor(moebius_img1, cv2.COLOR_BGR2GRAY)
    moebius_img2 = cv2.imread("data/moebius2.png")
    moebius_img2_gray = cv2.cvtColor(moebius_img2, cv2.COLOR_BGR2GRAY)
    moebius_gt = cv2.imread("data/moebius_gt.png", cv2.IMREAD_GRAYSCALE)

    tsukuba_img1 = cv2.imread("data/tsukuba1.jpg")
    tsukuba_img1_gray = cv2.cvtColor(tsukuba_img1, cv2.COLOR_BGR2GRAY)
    tsukuba_img2 = cv2.imread("data/tsukuba2.jpg")
    tsukuba_img2_gray = cv2.cvtColor(tsukuba_img2, cv2.COLOR_BGR2GRAY)
    tsukuba_gt = cv2.imread("data/tsukuba_gt.jpg", cv2.IMREAD_GRAYSCALE)
    
    # Task 0: Visualize cv2 Results
    if '0' in tasks:   
        # Compute disparity maps using cv2
        stereo = cv2.StereoBM.create(numDisparities=64, blockSize=15)
        moebius_disparity_cv2 = stereo.compute(moebius_img1_gray, moebius_img2_gray)
        visualize_disparity_map(moebius_disparity_cv2, moebius_gt)
        tsukuba_disparity_cv2 = stereo.compute(tsukuba_img1_gray, tsukuba_img2_gray)
        visualize_disparity_map(tsukuba_disparity_cv2, tsukuba_gt)
        
        if '2' in tasks:
            print('Running task2 with cv2 results ...')
            task2_visualize_pointcloud(tsukuba_img1, tsukuba_disparity_cv2, save_path='output/task2_tsukuba_cv2.ply')

    ######################################################################
    # Note. Running on moebius may take a long time with your own code       #
    # In this homework, you are allowed 【only to deal with tsukuba images】 #
    ######################################################################

    # Task 1: Simple Disparity Algorithm
    if '1' in tasks:
        print('Running task1 ...,picture: tsukuba')
        disparity_maps = task1_simple_disparity(tsukuba_img1_gray, tsukuba_img2_gray, tsukuba_gt, img_name='tsukuba')
        #print('Running task1 ...,picture: moebius')
        #disparity_maps = task1_simple_disparity(moebius_img1_gray, moebius_img2_gray, moebius_gt, img_name='moebius')
        
        #####################################################
        # If you want to run on moebius images,             #
        # parallelizing with multiprocessing is recommended #
        #####################################################
        # task1_simple_disparity(moebius_img1_gray, moebius_img2_gray, moebius_gt, img_name='moebius')
        
        if '2' in tasks:
            print('Running task2 with disparity maps from task1 ...')
            for (disparity_map, window_size, matching_function, disparity_range) in disparity_maps:
                dmin, dmax = disparity_range
                task2_visualize_pointcloud(
                    tsukuba_img1, disparity_map, 
                    save_path=f'output/task2_tsukuba_{window_size}_{dmin}_{dmax}_{matching_function}.ply')      
        
    # Task 3: Non-local constraints
    if '3' in tasks:
        print('----------------- Task 3 -----------------')
        start=time.time()
        tsukuba_disparity_dp = task3_compute_disparity_map_dp(tsukuba_img1_gray, tsukuba_img2_gray)
        end=time.time()
        print("Task 3, function called, time consumption:",end-start)
        visualize_disparity_map(tsukuba_disparity_dp, tsukuba_gt, save_path='output/task3_tsukuba.png')
        
        if '2' in tasks:
            print('Running task2 with disparity maps from task3 ...')
            task2_visualize_pointcloud(tsukuba_img1, tsukuba_disparity_dp, save_path='output/task2_tsukuba_dp.ply')

if __name__ == '__main__':
    # Set tasks to run
    parser = argparse.ArgumentParser(description='Homework 4')
    parser.add_argument('--tasks', type=str, default='0123')
    args = parser.parse_args()

    main(args.tasks)
