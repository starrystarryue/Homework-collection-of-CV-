# This is a raw framework for image stitching using Harris corner detection.
# For libraries you can use modules in numpy, scipy, cv2, os, etc.
import numpy as np
from scipy import ndimage, spatial
import cv2
from os import listdir
import matplotlib.pyplot as plt

#utils.py
from utils import gaussian_blur_kernel_2d,listpoint_homo_coordinate
# h,w = np.shape(img)
# pixel(x,y): x-axis: w direction 列索引; y-axis: h direction 行索引

np.random.seed(42)

IMGDIR = 'Problem2Images'

def gradient_x(img):
    # convert img to grayscale
    # should we use int type to calclate gradient?
    # should we conduct some pre-processing to remove noise? which kernel should we pply?
    # which kernel should we choose to calculate gradient_x?
    # TODO
    img = np.float64(img)
    gauss = ndimage.gaussian_filter(img, 3, mode='reflect')
    grad_x = ndimage.sobel(gauss, axis=1,mode='reflect')
    return grad_x  

def gradient_y(img):
    # TODO
    img = np.float64(img)
    gauss=ndimage.gaussian_filter(img, 3, mode='reflect')
    grad_y = ndimage.sobel(gauss, axis=0, mode='reflect')
    return grad_y   

def harris_response(img, alpha, win_size):
    # In this function you are going to claculate harris response R.
    # Please refer to 04_Feature_Detection.pdf page 32 for details. 
    # You have to discover how to calculate det(M) and trace(M), and
    # remember to smooth the gradients. 
    # Avoid using too much "for" loops to speed up.
    # TODO
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    w = gaussian_blur_kernel_2d(win_size, 1.0) 
    I_x = gradient_x(img)
    I_y = gradient_y(img)
    A = ndimage.convolve(I_x * I_x, w, mode='reflect')
    B = ndimage.convolve(I_x * I_y, w, mode='reflect')
    C = ndimage.convolve(I_y * I_y, w, mode='reflect')
    det = A*C - B**2
    trace = A + C
    R = det - alpha*(trace**2)
    return R   

def corner_selection(R, thresh, min_dist):
    # non-maximal suppression for R to get R_selection and transform selected corners to list of tuples
    # hint: 
    #   use ndimage.maximum_filter()  to achieve non-maximum suppression
    #   set those which aren’t **local maximum** to zero.
    # TODO
    local_max = ndimage.maximum_filter(R, size=min_dist)
    R_selection = (R== local_max) & (R > thresh)
    y_coords, x_coords = np.where(R_selection)
    pix = [(x, y) for x, y in zip(x_coords, y_coords)]
    return pix


def histogram_of_gradients(img, pix):
    # no template for coding, please implement by yourself.
    # You can refer to implementations on Github or other websites
    # Hint: 
    #   1. grad_x & grad_y
    #   2. grad_dir by arctan function
    #   3. for each interest point, choose m*m blocks with each consists of m*m pixels
    #   4. I divide the region into n directions (maybe 8).
    #   5. For each blocks, calculate the number of derivatives in those directions and normalize the Histogram. 
    #   6. After that, select the prominent gradient and take it as principle orientation.
    #   7. Then rotate it’s neighbor to fit principle orientation and calculate the histogram again. 
    # TODO
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    H,W =img.shape
    cell= 4
    block =16
 
    grad_x=gradient_x(img)
    grad_y=gradient_y(img)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2) 
    grad_dir = np.degrees(np.arctan2(grad_y, grad_x))
    grad_dir[grad_dir < 0] += 360

    tmp=[]
    L = block* block * 8
    delta = int(cell * block //2)
    for pixel in pix:
        cx,cy=pixel[0],pixel[1] 
        feature = np.empty((L))
        feature_idx=0
        y_start = cy - delta
        y_end = cy + delta
        x_start = cx - delta
        x_end = cx + delta

        #越界，全部跳过这个block
        if y_start < 0 or x_start < 0 or y_end >= H or x_end >= W:
            feature = np.zeros(block * block * 8)
            tmp.append(feature)
            features =np.array(tmp)
            continue      
        #遍历block的每个cell
        for col_start in range (x_start, x_end - cell ,cell):
            for row_start in range(y_start, y_end - cell, cell):
                col_end = col_start + cell
                row_end = row_start + cell
                #遍历cell的每个点，36个分区找主方向
                hist = np.zeros(36)
                for col in range(col_start,col_end+1):
                    for row in range(row_start,row_end+1):
                        hist[int(grad_dir[row,col] // 10) ] += grad_mag[row,col]
                main_degree = np.argmax(hist) * 10
                #再遍历cell的每个点 主方向对齐，8个分区得到特征向量
                _hist=np.zeros(8)
                for col in range(col_start,col_end+1):
                    for row in range(row_start,row_end+1):
                        rotate_dir=(grad_dir[row,col]-main_degree) % 360 
                        _hist[int(rotate_dir //45)]+=grad_mag[row,col]
                #_hist =_hist/(np.linalg.norm(_hist) + 1e-10) #【*】错误：不是对一个cell单独归一化
                feature[feature_idx : feature_idx+8]=_hist
                feature_idx += 8

        feature = np.array(feature)
        feature /= (np.linalg.norm(feature) +1e-10)
        tmp.append(feature)
        features =np.array(tmp)
    return features 

def feature_matching(img_1, img_2):
    R1 = harris_response(img_1, 0.04, 9)
    R2 = harris_response(img_2, 0.04, 9)
    cor1 = corner_selection(R1, 0.01*np.max(R1), 5)
    cor2 = corner_selection(R2, 0.01*np.max(R1), 5)
    fea1 = histogram_of_gradients(img_1, cor1)
    fea2 = histogram_of_gradients(img_2, cor2)
    dis = spatial.distance.cdist(fea1, fea2, metric='euclidean')
    dis+=1e-7
    threshold = 0.75
    pixels_1 = []
    pixels_2 = []
    p1, p2 = np.shape(dis)
    if p1 < p2:
        for p in range(p1):
            dis_min = np.min(dis[p])
            pos = np.argmin(dis[p])
            dis[p][pos] = np.max(dis)
            if dis_min/np.min(dis[p])  <= threshold:
                pixels_1.append(cor1[p])
                pixels_2.append(cor2[pos])
                dis[:, pos] = np.max(dis)

    else:
        for p in range(p2):
            dis_min = np.min(dis[:, p])
            pos = np.argmin(dis[:, p])
            dis[pos][p] = np.max(dis)
            if dis_min/np.min(dis[:, p]) <= threshold:
                pixels_2.append(cor2[p])
                pixels_1.append(cor1[pos])
                dis[pos] = np.max(dis)
    min_len = min(np.shape(cor1)[0], np.shape(cor2)[0])

    print("num of valid points:",np.shape(pixels_1)[0])
    rate = np.shape(pixels_1)[0]/min_len

    print("final rate: ",rate)
    assert rate >= 0.03, "Fail to Match!"

    return pixels_1, pixels_2

def test_matching():   
    img_1 = cv2.imread(f'{IMGDIR}/1_1.jpg')
    img_2 = cv2.imread(f'{IMGDIR}/1_2.jpg')

    img_gray_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
    img_gray_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)

    pixels_1, pixels_2 = feature_matching(img_1, img_2)

    H_1, W_1 = img_gray_1.shape
    H_2, W_2 = img_gray_2.shape

    img = np.zeros((max(H_1, H_2), W_1 + W_2, 3))
    img[:H_1, :W_1, (2, 1, 0)] = img_1 / 255
    img[:H_2, W_1:, (2, 1, 0)] = img_2 / 255
    
    plt.figure(figsize=(20, 10), dpi=300)
    plt.imshow(img)

    N = len(pixels_1)
    for i in range(N):
        x1, y1 = pixels_1[i] #【*】这里x表示列，y表示行 corner_detection返回的pix点需要列主序！
        x2, y2 = pixels_2[i]
        plt.plot([x1, x2+W_1], [y1, y2])

    # plt.show()
    plt.savefig('test.jpg')

def compute_homography(pixels_1, pixels_2):
    # compute the best-fit homography using the Singular Value Decomposition (SVD)
    # homography matrix is a (3,3) matrix consisting rotation, translation and projection information.
    # consider how to form matrix A for U, S, V = np.linalg.svd((np.transpose(A)).dot(A))
    # homo_matrix = np.reshape(V[np.argmin(S)], (3, 3))
    # TODO
    N = len(pixels_1)
    pixels_1_homo = listpoint_homo_coordinate(pixels_1)
    pixels_2_homo = listpoint_homo_coordinate(pixels_2)
    A = np.zeros((2*N, 9))
    for i in range(N):
        x1,y1,_ =pixels_1_homo[i]
        x2,y2,_ =pixels_2_homo[i]
        A[2*i] = np.array([-x1,-y1,-1,0,0,0,x2*x1,x2*y1,x2])
        A[2*i+1] = np.array([0,0,0,-x1,-y1,-1,y2*x1,y2*y1,y2])
    _, S, V = np.linalg.svd((np.transpose(A)).dot(A))
    homo_matrix = np.reshape(V[np.argmin(S)], (3, 3))
    return homo_matrix

def align_pair(pixels_1, pixels_2):
    # utilize \verb|homo_coordinates| for homogeneous pixels
    # and \verb|compute_homography| to calulate homo_matrix
    # implement RANSAC to compute the optimal alignment.
    # you can refer to implementations online.
    N = len(pixels_1)
    pixels_1_homo = listpoint_homo_coordinate(pixels_1)
    pixels_2_homo = listpoint_homo_coordinate(pixels_2)
    est_homo = np.zeros((3, 3))
    max_iterate = 1000
    threshold = 4.0
    max_inliers = 0
    for _ in range(max_iterate):
        rand = np.random.choice(N,4,replace=False)
        rand_pixels_1 = [pixels_1[j] for j in rand]
        rand_pixels_2 = [pixels_2[j] for j in rand]
        homo_matrix = compute_homography(rand_pixels_1, rand_pixels_2)

        trans_pixels_1 = homo_matrix.dot(np.transpose(pixels_1_homo))
        trans_pixels_1 = trans_pixels_1 / trans_pixels_1[2, :]
        trans_pixels_1 = np.transpose(trans_pixels_1)

        dis = spatial.distance.cdist(trans_pixels_1, pixels_2_homo, metric='euclidean')
        dis = np.diagonal(dis)
        num_inliers = np.count_nonzero(dis < threshold)
        if num_inliers > max_inliers:
            max_inliers = num_inliers
            est_homo = homo_matrix
    return est_homo

def stitch_blend(img_1, img_2, est_homo):
    # hint: 
    # First, project four corner pixels with estimated homo-matrix
    # and converting them back to Cartesian coordinates after normalization.
    # Together with four corner pixels of the other image, we can get the size of new image plane.
    # Then, remap both image to new image plane and blend two images using Alpha Blending.
    h1, w1, d1 = np.shape(img_1)  # d=3 RGB
    h2, w2, d2 = np.shape(img_2)
    p1 = est_homo.dot(np.array([0, 0, 1]))
    p2 = est_homo.dot(np.array([0, h1, 1]))
    p3 = est_homo.dot(np.array([w1, 0, 1]))
    p4 = est_homo.dot(np.array([w1, h1, 1]))
    p1 = np.int16(p1/p1[2])
    p2 = np.int16(p2/p2[2])
    p3 = np.int16(p3/p3[2])
    p4 = np.int16(p4/p4[2])
    x_min = min(0, p1[0], p2[0], p3[0], p4[0])
    x_max = max(w2, p1[0], p2[0], p3[0], p4[0])
    y_min = min(0, p1[1], p2[1], p3[1], p4[1])
    y_max = max(h2, p1[1], p2[1], p3[1], p4[1])
    x_range = np.arange(x_min, x_max+1, 1)
    y_range = np.arange(y_min, y_max+1, 1)
    x, y = np.meshgrid(x_range, y_range)
    x = np.float32(x)
    y = np.float32(y)
    homo_inv = np.linalg.pinv(est_homo)
    trans_x = homo_inv[0, 0]*x+homo_inv[0, 1]*y+homo_inv[0, 2]
    trans_y = homo_inv[1, 0]*x+homo_inv[1, 1]*y+homo_inv[1, 2]
    trans_z = homo_inv[2, 0]*x+homo_inv[2, 1]*y+homo_inv[2, 2]
    trans_x = trans_x/trans_z
    trans_y = trans_y/trans_z
    trans_x=np.float32(trans_x)
    trans_y = np.float32(trans_y)
    est_img_1 = cv2.remap(img_1, trans_x, trans_y, cv2.INTER_LINEAR)
    est_img_2 = cv2.remap(img_2, x, y, cv2.INTER_LINEAR)
    """
    #原来的alpha-blending 
    alpha1 = cv2.remap(np.ones(np.shape(img_1)), trans_x,
                      trans_y, cv2.INTER_LINEAR)
    alpha2 = cv2.remap(np.ones(np.shape(img_2)), x, y, cv2.INTER_LINEAR)
    alpha = alpha1+alpha2
    alpha[alpha == 0] = 2
    alpha1 = alpha1/alpha
    alpha2 = alpha2/alpha
    est_img = est_img_1*alpha1 + est_img_2*alpha2
    """
    #修改后
    mask1 = cv2.remap(np.ones(img_1.shape[:2], dtype=np.uint8), trans_x, trans_y, cv2.INTER_NEAREST)
    mask2 = cv2.remap(np.ones(img_2.shape[:2], dtype=np.uint8), x, y, cv2.INTER_NEAREST)
    dist1 = cv2.distanceTransform(mask1, cv2.DIST_L2, 5)
    dist2 = cv2.distanceTransform(mask2, cv2.DIST_L2, 5)
    alpha1 = dist1 / (dist1 + dist2 + 1e-8)
    alpha2 = dist2 / (dist1 + dist2 + 1e-8)
    alpha1 = np.repeat(alpha1[:, :, np.newaxis], 3, axis=2)
    alpha2 = np.repeat(alpha2[:, :, np.newaxis], 3, axis=2)
    est_img = est_img_1 * alpha1 + est_img_2 * alpha2

    return est_img


def generate_panorama(ordered_img_seq):

    len = np.shape(ordered_img_seq)[0]
    mid = int(len/2) # middle anchor
    i = mid-1
    j = mid+1
    principle_img = ordered_img_seq[mid]
    print("图片数量为",len,"mid=",mid,"i=",i,"j=",j)
    while(j < len):
        print("正在匹配图片:j=",j,"and",mid)
        pixels1, pixels2 = feature_matching(ordered_img_seq[j], principle_img)
        homo_matrix = align_pair(pixels1, pixels2)
        principle_img = stitch_blend(
            ordered_img_seq[j], principle_img, homo_matrix)
        principle_img=np.uint8(principle_img)
        j = j+1  
    while(i >= 0):
        print("正在匹配图片:i=",i,"and",mid)
        pixels1, pixels2 = feature_matching(ordered_img_seq[i], principle_img)
        homo_matrix = align_pair(pixels1, pixels2)
        principle_img = stitch_blend(
            ordered_img_seq[i], principle_img, homo_matrix)
        principle_img=np.uint8(principle_img)
        i = i-1  
    est_pano = principle_img
    return est_pano

def MakePanorama_grail():
    #超参：threshold = 0.75(in def feature_matching), max_iter=1000+threshold=4(in def align_pair)
    print("Make panorama for grail begin......")
    img_2 = cv2.imread(f'{IMGDIR}/panoramas/grail/grail02.jpg')
    img_3 = cv2.imread(f'{IMGDIR}/panoramas/grail/grail03.jpg')
    img_4 = cv2.imread(f'{IMGDIR}/panoramas/grail/grail04.jpg')
    img_5 = cv2.imread(f'{IMGDIR}/panoramas/grail/grail05.jpg')
    img_6 = cv2.imread(f'{IMGDIR}/panoramas/grail/grail06.jpg')
    img_7 = cv2.imread(f'{IMGDIR}/panoramas/grail/grail07.jpg')
    img_list=[]
    img_list.append(img_2)
    img_list.append(img_3)
    img_list.append(img_4)
    img_list.append(img_5)
    img_list.append(img_6)
    pano = generate_panorama(img_list)
    cv2.imwrite("outputs/panorama_1.png", pano)
    print("Output image: panorama_1.png saved!")

def MakePanorama_library():
    #超参1：threshold=0.8(in def feature_matching), max_iter=1100+threshold=4.0(in def align_pair)  -->拼接img_1~img_4
    #超参2：threshold = 0.8(in def feature_matching), max_iter=1200+threshold=4.0(in def align_pair)  --->拼接img_1~img_5
    #运行时间略长，约2min-2.5min不等
    print("Make panorama for library begin......")
    img_1 = cv2.imread(f'{IMGDIR}/panoramas/library/10.jpg')
    img_2 = cv2.imread(f'{IMGDIR}/panoramas/library/11.jpg')
    img_3 = cv2.imread(f'{IMGDIR}/panoramas/library/12.jpg')
    img_4 = cv2.imread(f'{IMGDIR}/panoramas/library/13.jpg')
    img_5 = cv2.imread(f'{IMGDIR}/panoramas/library/14.jpg')
    img_list=[]
    img_list.append(img_1)
    img_list.append(img_2)
    img_list.append(img_3)
    img_list.append(img_4)
    img_list.append(img_5)
    pano = generate_panorama(img_list)
    cv2.imwrite("outputs/panorama_2.png", pano)
    print("Output image: panorama_2.png saved!")

def MakePanorama_parrington():
    #超参：threshold=0.85(in def feature_matching), max_iter=1000+thresh=4.0(in def align_pair)
    print("Make panorama for parrington begin......")
    img_1 = cv2.imread(f'{IMGDIR}/panoramas/parrington/prtn01.jpg')
    img_2 = cv2.imread(f'{IMGDIR}/panoramas/parrington/prtn02.jpg')
    img_3 = cv2.imread(f'{IMGDIR}/panoramas/parrington/prtn03.jpg')
    img_4 = cv2.imread(f'{IMGDIR}/panoramas/parrington/prtn04.jpg')
    img_5 = cv2.imread(f'{IMGDIR}/panoramas/parrington/prtn05.jpg')
    img_6 = cv2.imread(f'{IMGDIR}/panoramas/parrington/prtn06.jpg')
    img_list=[]
    img_list.append(img_2)
    img_list.append(img_3)
    img_list.append(img_4)
    img_list.append(img_5)
    img_list.append(img_6)
    pano = generate_panorama(img_list)
    cv2.imwrite("outputs/panorama_3.png", pano)
    print("Output image: panorama_3.png saved!")

def MakePanorama_XueMountainEntrance():
    #超参：threshold=0.8(in def feature_matching), max_iter=1000+threshold=4.0(in def align_pair)
    print("Make panorama for Xue-Mountain-Entrance begin......")
    img_1 = cv2.imread(f'{IMGDIR}/panoramas/Xue-Mountain-Entrance/DSC_0171.jpg')
    img_2 = cv2.imread(f'{IMGDIR}/panoramas/Xue-Mountain-Entrance/DSC_0172.jpg')
    img_3 = cv2.imread(f'{IMGDIR}/panoramas/Xue-Mountain-Entrance/DSC_0173.jpg')
    img_4 = cv2.imread(f'{IMGDIR}/panoramas/Xue-Mountain-Entrance/DSC_0174.jpg')
    img_5 = cv2.imread(f'{IMGDIR}/panoramas/Xue-Mountain-Entrance/DSC_0175.jpg')
    img_list=[]
    img_list.append(img_1)
    img_list.append(img_2)
    img_list.append(img_3)
    img_list.append(img_4)
    img_list.append(img_5)
    pano = generate_panorama(img_list)
    cv2.imwrite("outputs/panorama_4.png", pano) 
    print("Output image: panorama_4.png saved!")

def pair_image_stitching(img1,img2):
    pixels1, pixels2 = feature_matching(img1,img2)
    homo_matrix = align_pair(pixels1, pixels2)
    stitched=stitch_blend(img1,img2,homo_matrix)
    return stitched

def PairStitching():
    img1 = cv2.imread(f'{IMGDIR}/1_1.jpg')
    img2 = cv2.imread(f'{IMGDIR}/1_2.jpg')
    print("Now create visualizations on pair image stitching....")
    cv2.imwrite('outputs/blend_1.png', pair_image_stitching(img1,img2))
    print("Output image: blend_1.png saved!")
    img1 = cv2.imread(f'{IMGDIR}/2_1.jpg')
    img2 = cv2.imread(f'{IMGDIR}/2_2.jpg')
    cv2.imwrite('outputs/blend_2.png', pair_image_stitching(img1,img2))
    print("Output image: blend_2.png saved!")
    img1 = cv2.imread(f'{IMGDIR}/3_1.jpg')
    img2 = cv2.imread(f'{IMGDIR}/3_2.jpg')
    cv2.imwrite('outputs/blend_3.png', pair_image_stitching(img1,img2))
    print("Output image: blend_3.png saved!")

if __name__ == '__main__':
    # make image list
    # call generate panorama and it should work well
    # save the generated image following the requirements

    #1.test
    # 目前outputs文件夹中test.jpg的效果为在超参: thrshold=0.7(in def feature_matching)下的运行结果
    #test_matching()

    #2.pair-image stitching
    # 保存1_1/1_2 2_1/2_2 3_1/3_2的pair-image stitching融合结果
    # 报告中的三张图片是按照目前上述代码的参数运行后，得到的结果
    PairStitching()

    # make panorama
    #【每个的参数与超参选择不同！】各自的选择组合分别在以下【函数的注释中】
    # 您可以在MakePanorama_xxx函数下的头两行注释中，看到拼接四个全景图各自需要的参数取值以及参数所属的函数~
    # 报告中的四张图片是按照上述参数运行后，分别得到的结果
    # 目前，上述代码中的参数是恰好【适合grail全景拼接】的组合
    MakePanorama_grail()