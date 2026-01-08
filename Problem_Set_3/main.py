from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.linalg
def visualize_matches(I1, I2, matches):
    # display two images side-by-side with matches
    # this code is to help you visualize the matches, you don't need
    # to use it to produce the results for the assignment

    I3 = np.zeros((I1.size[1],I1.size[0]*2,3) )
    I3[:,:I1.size[0],:] = I1
    I3[:,I1.size[0]:,:] = I2
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.imshow(np.array(I3).astype(np.uint8))
    ax.plot(matches[:,0],matches[:,1],  '+r')
    ax.plot( matches[:,2] + I1.size[0], matches[:,3], '+r')
    ax.plot([matches[:,0], matches[:,2]+I1.size[0]],[matches[:,1], matches[:,3]], 'r')
    plt.show()

def normalize_points(pts):
    # Normalize points
    # 1. calculate mean and std
    # 2. build a transformation matrix
    # :return normalized_pts: normalized points
    # :return T: transformation matrix from original to normalized points
    print("using normalized points......")
    mean=np.mean(pts,axis=0)
    std=np.std(pts)
    scale=np.sqrt(2)/std
    tx,ty=-scale*mean[0], -scale*mean[1]
    T = np.array([
        [scale, 0, tx],
        [0, scale, ty],
        [0, 0, 1]
    ])
    N = pts.shape[0]
    pts_homo = np.hstack((pts, np.ones((N, 1))))
    normalized_homo = (T @ pts_homo.T).T
    normalized_pts = normalized_homo[:, :2]
    return normalized_pts, T

def fit_fundamental(matches):
    # Calculate fundamental matrix from ground truth matches
    # 1. (normalize points if necessary)
    # 2. (x2, y2, 1) * F * (x1, y1, 1)^T = 0 -> AX = 0
    # X = (f_11, f_12, ..., f_33) 
    # build A(N x 9) from matches(N x 4) according to Eight-Point Algorithm
    # 3. use SVD (np.linalg.svd) to decomposite the matrix
    # 4. take the smallest eigen vector(9, ) as F(3 x 3)
    # 5. use SVD to decomposite F, set the smallest eigenvalue as 0, and recalculate F
    # 6. Report your fundamental matrix results
    
    # 归一化
    N=len(matches)
    A=np.zeros((N,9))
    points1=matches[:,:2]
    points2=matches[:,2:]
    norm_points1,T1=normalize_points(points1) 
    norm_points2,T2=normalize_points(points2) 
    for i in range(N):
        x1,y1=norm_points1[i] 
        x2,y2=norm_points2[i] 
        A[i]=[x2 * x1, x2 * y1, x2, y2 * x1, y2 * y1, y2, x1, y1, 1]
    _,_,V=np.linalg.svd(A)
    F = V[-1].reshape(3, 3)
    U,S,Vt = np.linalg.svd(F)
    S[2] = 0
    F_rank2 = U @ np.diag(S) @ Vt
    F = T2.T @ F_rank2 @ T1
    """
    # 不加normalization
    N=len(matches)
    A=np.zeros((N,9))
    points1=matches[:,:2]
    points2=matches[:,2:]
    for i in range(N):
        x1,y1=points1[i]
        x2,y2=points2[i]
        A[i]=[x2 * x1, x2 * y1, x2, y2 * x1, y2 * y1, y2, x1, y1, 1]
    _,_,V=scipy.linalg.svd(A)
    F = V[-1].reshape(3, 3)
    U,S,Vt = np.linalg.svd(F)
    S[2] = 0
    F_rank2 = U @ np.diag(S) @ Vt
    F=F_rank2
    """
    return F

def visualize_fundamental(matches, F, I1, I2):
    # Visualize the fundamental matrix in image 2
    N = len(matches)
    M = np.c_[matches[:,0:2], np.ones((N,1))].transpose()
    L1 = np.matmul(F, M).transpose() # transform points from 
    # the first image to get epipolar lines in the second image

    # find points on epipolar lines L closest to matches(:,3:4)
    l = np.sqrt(L1[:,0]**2 + L1[:,1]**2)
    L = np.divide(L1, np.kron(np.ones((3,1)), l).transpose())   # rescale the line
    pt_line_dist = np.multiply(L, np.c_[matches[:, 2:4], np.ones((N, 1))]).sum(axis = 1)
    closest_pt = matches[:, 2:4] - np.multiply(L[:, 0:2],np.kron(np.ones((2, 1)), pt_line_dist).transpose())

    # find endpoints of segment on epipolar line (for display purposes)
    pt1 = closest_pt - np.c_[L[:,1], -L[:,0]] * 10    # offset from the closest point is 10 pixels
    pt2 = closest_pt + np.c_[L[:,1], -L[:,0]] * 10

    # display points and segments of corresponding epipolar lines
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.imshow(np.array(I2).astype(np.uint8))
    ax.plot(matches[:, 2],matches[:, 3],  '+r')
    ax.plot([matches[:, 2], closest_pt[:, 0]],[matches[:, 3], closest_pt[:, 1]], 'r')
    ax.plot([pt1[:, 0], pt2[:, 0]],[pt1[:, 1], pt2[:, 1]], 'g')
    plt.show()

def evaluate_fundamental(matches, F):
    N = len(matches)
    points1, points2 = matches[:, :2], matches[:, 2:]
    points1_homogeneous = np.concatenate([points1, np.ones((N, 1))], axis=1)
    points2_homogeneous = np.concatenate([points2, np.ones((N, 1))], axis=1)
    product = np.dot(np.dot(points2_homogeneous, F), points1_homogeneous.T)
    diag = np.diag(product)
    residual = np.mean(diag ** 2)
    return residual

## Task 0: Load data and visualize
## load images and match files for the first example
## matches[:, :2] is a point in the first image
## matches[:, 2:] is a corresponding point in the second image

library_image1 = Image.open('data/library1.jpg')
library_image2 = Image.open('data/library2.jpg')
library_matches = np.loadtxt('data/library_matches.txt')

lab_image1 = Image.open('data/lab1.jpg')
lab_image2 = Image.open('data/lab2.jpg')
lab_matches = np.loadtxt('data/lab_matches.txt')

## Visualize matches
#visualize_matches(library_image1, library_image2, library_matches)
#visualize_matches(lab_image1, lab_image2, lab_matches)

## Task 1: Fundamental matrix
## display second image with epipolar lines reprojected from the first image

# first, fit fundamental matrix to the matches
# Report your 【fundamental matrices, visualization and evaluation results】

library_F = fit_fundamental(library_matches) # this is a function that you should write
print("Task 1, the fundamental matrix of library is:\n ",library_F)
visualize_fundamental(library_matches, library_F, library_image1, library_image2)
print("Task 1, the evaluation results of liabrary is: ",evaluate_fundamental(library_matches, library_F))
assert evaluate_fundamental(library_matches, library_F) < 0.5

lab_F = fit_fundamental(lab_matches) # this is a function that you should write
print("Task 1, the fundamental matrix of lab is:\n ",lab_F)
visualize_fundamental(lab_matches, lab_F, lab_image1, lab_image2) 
print("Task 1, the evaluation results of lab is: ",evaluate_fundamental(lab_matches, lab_F))
assert evaluate_fundamental(lab_matches, lab_F) < 0.5

print("-------end of Task 1-------")

## Task 2: Camera Calibration

def calc_projection(points_2d, points_3d):
    # Calculate camera projection matrices
    # 1. Points_2d = P * Points_3d -> AX = 0
    # X = (p_11, p_12, ..., p_34) is flatten of P
    # build matrix A(2*N, 12) from points_2d
    # 2. SVD decomposite A
    # 3. take the eigen vector(12, ) of smallest eigen value
    # 4. return projection matrix(3, 4)
    # :param points_2d: 2D points N x 2
    # :param points_3d: 3D points N x 3
    # :return P: projection matrix
    N = len(points_2d)
    A=np.zeros((2*N,12))
    homo_2d=np.ones((N,3))
    homo_2d[:,0:2]=points_2d
    homo_3d=np.ones((N,4))
    homo_3d[:,0:3]=points_3d
    for i in range(N):
        x,y,_=homo_2d[i]
        X,Y,Z,_=homo_3d[i]
        A[2*i] = [X,Y,Z,1,0,0,0,0,-x*X,-x*Y,-x*Z,-x]
        A[2*i+1] = [0,0,0,0,X,Y,Z,1,-y*X,-y*Y,-y*Z,-y]
    U, S, Vt = np.linalg.svd(A)
    P_vector = Vt[-1, :]
    P=P_vector.reshape(3,4)
    return P

def rq_decomposition(P):
    # Use RQ decomposition to calcsulte K, R, T
    # 1. perform RQ decomposition on left-most 3x3 matrix of P(3 x 4) to get K, R
    # 2. calculate T by P = K[R|T]
    # 3. normalize to set K[2, 2] = 1
    # :param P: projection matrix
    # :return K, R, T: camera matrices
    _P=P[:,:3]
    K,R=scipy.linalg.rq(_P)
    T=np.linalg.inv(K) @P[:,3]
    K/=K[2,2]
    return K, R, T

def evaluate_points(P, points_2d, points_3d):
    # Visualize the actual 2D points and the projected 2D points calculated from
    # the projection matrix
    # You do not need to modify anything in this function, although you can if you
    # want to
    # :param P: projection matrix 3 x 4
    # :param points_2d: 2D points N x 2
    # :param points_3d: 3D points N x 3
    # :return points_3d_proj: project 3D points to 2D by P
    # :return residual: residual of points_3d_proj and points_2d

    N = len(points_3d)
    points_3d = np.hstack((points_3d, np.ones((N, 1))))
    points_3d_proj = np.dot(P, points_3d.T).T
    u = points_3d_proj[:, 0] / points_3d_proj[:, 2]
    v = points_3d_proj[:, 1] / points_3d_proj[:, 2]
    residual = np.sum(np.hypot(u-points_2d[:, 0], v-points_2d[:, 1]))
    points_3d_proj = np.hstack((u[:, np.newaxis], v[:, np.newaxis]))
    return points_3d_proj, residual

def triangulate_points(P1, P2, point1, point2):
    # Use linear least squares to triangulation 3d points
    # 1. Solve: point1 = P1 * point_3d
    #           point2 = P2 * point_3d
    # 2. use SVD decomposition to solve linear equations
    # :param P1, P2 (3 x 4): projection matrix of two cameras
    # :param point1, point2: points in two images
    # :return point_3d: 3D points calculated by triangulation
    x1,y1=point1
    x2,y2=point2
    A=np.zeros((4,4))
    A[0] = x1 * P1[2] - P1[0]
    A[1] = y1 * P1[2] - P1[1]
    A[2] = x2 * P2[2] - P2[0]
    A[3] = y2 * P2[2] - P2[1]
    _,_,Vt= np.linalg.svd(A)
    X_homo = Vt[-1]
    X_homo /= X_homo[3] 
    point_3d = X_homo[:3]
    return point_3d


lab_points_3d = np.loadtxt('data/lab_3d.txt')

projection_matrix = dict()

for key, points_2d in zip(["lab_a", "lab_b"], [lab_matches[:, :2], lab_matches[:, 2:]]):
    print("image: ",key)
    P = calc_projection(points_2d, lab_points_3d)
    print("P:\n ",P)
    points_3d_proj, residual = evaluate_points(P, points_2d, lab_points_3d)
    print("corrdinates of projection points:\n ",points_3d_proj)
    distance = np.mean(np.linalg.norm(points_2d - points_3d_proj))
    # Check: residual should be < 20 and distance should be < 4 
    print("Task 2, the residual is: ",residual)
    print("Task 2, the distance is: ",distance)
    assert residual < 20.0 and distance < 4.0
    projection_matrix[key] = P

print("-------end of task 2-------")

## Task 3
## Camera Centers
projection_library_a = np.loadtxt('data/library1_camera.txt')
projection_library_b = np.loadtxt('data/library2_camera.txt')
projection_matrix["library_a"] = projection_library_a
projection_matrix["library_b"] = projection_library_b

#print(projection_matrix) 
for P in projection_matrix.values():
    # 【Paste your K, R, T results in your report】
    print("P",P)
    K, R, T = rq_decomposition(P)
    print("Task 3, result of K is :",K)
    print("Task 3, result of R is :",R)
    print("Task 3, result of T is :",T)
    print("---------------")

print("-------end of task 3-------")

## Task 4: Triangulation
lab_points_3d_estimated = []
for point_2d_a, point_2d_b, point_3d_gt in zip(lab_matches[:, :2], lab_matches[:, 2:], lab_points_3d):
    point_3d_estimated = triangulate_points(projection_matrix['lab_a'], projection_matrix['lab_b'], point_2d_a, point_2d_b)

    # Residual between ground truth and estimated 3D points
    residual_3d = np.sum(np.linalg.norm(point_3d_gt - point_3d_estimated))
    assert residual_3d < 0.1
    lab_points_3d_estimated.append(point_3d_estimated)

# Residual between re-projected and observed 2D points
lab_points_3d_estimated = np.stack(lab_points_3d_estimated)

print("Task 4, triangulated 3D points for the lab pair: ",lab_points_3d_estimated) # Sanity check
_, residual_a = evaluate_points(projection_matrix['lab_a'], lab_matches[:, :2], lab_points_3d_estimated)
_, residual_b = evaluate_points(projection_matrix['lab_b'], lab_matches[:, 2:], lab_points_3d_estimated)
print("Task 4, lab, residual_a: ",residual_a)
print("Task 4, lab, residual_b: ",residual_b)
assert residual_a < 20 and residual_b < 20

library_points_3d_estimated = []
for point_2d_a, point_2d_b in zip(library_matches[:, :2], library_matches[:, 2:]):
    point_3d_estimated = triangulate_points(projection_matrix['library_a'], projection_matrix['library_b'], point_2d_a, point_2d_b)
    library_points_3d_estimated.append(point_3d_estimated)

# Residual between re-projected and observed 2D points
library_points_3d_estimated = np.stack(library_points_3d_estimated)
_, residual_a = evaluate_points(projection_matrix['library_a'], library_matches[:, :2], library_points_3d_estimated)
_, residual_b = evaluate_points(projection_matrix['library_b'], library_matches[:, 2:], library_points_3d_estimated)
print("Task 4, library, residual_a: ",residual_a)
print("Task 4, library, residual_b: ",residual_b)
assert residual_a < 30 and residual_b < 30
print("-------end of task 4-------")

## Task 5: Fundamental matrix estimation without ground-truth matches
import cv2

def fit_fundamental_without_gt(image1, image2):
    # Calculate fundamental matrix without groundtruth matches
    # 1. convert the images to gray
    # 2. compute SIFT keypoints and descriptors
    # 3. match descriptors with Brute Force Matcher
    # 4. select good matches
    # 5. extract matched keypoints
    # 6. compute fundamental matrix with RANSAC
    # :param image1, image2: two-view images
    # :return fundamental_matrix
    # :return matches: selected matched keypoints 
    if image1.shape[2]==3:
        gray1 = cv2.cvtColor(image1,cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(image2,cv2.COLOR_RGB2GRAY)
    else:
        gray1 = image1
        gray2 = image2
    
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    raw_matches = bf.match(des1, des2)

    raw_matches = sorted(raw_matches, key=lambda x: x.distance)
    
    pts1 = np.float32([kp1[m.queryIdx].pt for m in raw_matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in raw_matches])

    _, inlier_mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)
    inliers1 = pts1[inlier_mask.ravel() == 1]
    inliers2 = pts2[inlier_mask.ravel() == 1]
    matches = np.hstack((inliers1, inliers2))  #shape:(N,4)
    fundamental_matrix=fit_fundamental(matches)

    #输出内点的数量和平均残差
    residual = np.mean(inlier_mask.ravel() == 1) 
    print(f"Number of inliers: {len(matches)}")
    print(f"Average residual: {residual}")
    #可视化：在两个图像中分别显示内点  {下面的绘制图像的代码（~394行）部分借助了大语言模型的帮助}
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    plt.subplots_adjust(wspace=0, hspace=0)  
    axes[0].imshow(image1)
    axes[0].scatter(inliers1[:, 0], inliers1[:, 1], color='red', marker='o', s=5)
    axes[0].set_title("Inliers in Image 1")
    axes[0].axis('off')
    axes[1].imshow(image2)
    axes[1].scatter(inliers2[:, 0], inliers2[:, 1], color='red', marker='o', s=5)
    axes[1].set_title("Inliers in Image 2")
    axes[1].axis('off')
    plt.show()
    #对应内点连线可视化，{运行时到这一步可能需要等待5-6s的时间}
    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]
    height = max(h1, h2)
    combined_image = np.zeros((height, w1 + w2, 3), dtype=np.uint8)
    combined_image[:h1, :w1] = image1
    combined_image[:h2, w1:] = image2
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(combined_image)
    for pt1, pt2 in zip(inliers1, inliers2):
        color = np.random.rand(3,)
        x1, y1 = pt1
        x2, y2 = pt2
        ax.plot([x1, x2 + w1], [y1, y2], color=color, linewidth=1)
        ax.scatter(x1, y1, color=color, s=5)
        ax.scatter(x2 + w1, y2, color=color, s=5)
    ax.set_title("Inlier Matches")
    ax.axis('off')
    plt.tight_layout()
    plt.show()

    return fundamental_matrix, matches

print("House pair result:")
house_image1 = Image.open('data/house1.jpg')
house_image2 = Image.open('data/house2.jpg')
house_F, house_matches = fit_fundamental_without_gt(np.array(house_image1), np.array(house_image2))
visualize_fundamental(house_matches, house_F, house_image1, house_image2)

print("Gaudi pair result:")
gaudi_image1 = Image.open('data/gaudi1.jpg')
gaudi_image2 = Image.open('data/gaudi2.jpg')
gaudi_F, gaudi_matches = fit_fundamental_without_gt(np.array(gaudi_image1), np.array(gaudi_image2))
visualize_fundamental(gaudi_matches, gaudi_F, gaudi_image1, gaudi_image2)