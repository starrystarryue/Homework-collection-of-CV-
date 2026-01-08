import numpy as np

def listpoint_homo_coordinate(list_of_points):
    list_of_points = np.asarray(list_of_points)
    N=np.shape(list_of_points)[0]
    if N == 0:
        return np.zeros((0, 3)) 
    homo_listpoint=np.ones((N,3))
    homo_listpoint[:,0:2]=list_of_points
    return homo_listpoint

def gaussian_blur_kernel_2d(size,sigma):
    kernel=np.zeros((size,size),dtype=np.float64)
    center=size//2

    normalization=0
    for i in range(size):
        for j in range(size):
            up=(i-center)**2+(j-center)**2
            down=2*(sigma**2)
            kernel[i,j]=np.exp(-up/down)
            normalization+=kernel[i,j]
    kernel/=normalization
    return kernel
