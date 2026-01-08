import numpy as np
import cv2

def cross_correlation_2d(image, kernel):
    kh,kw=kernel.shape
    h,w,c=image.shape

    pad_h = kernel.shape[0] // 2
    pad_w = kernel.shape[1] // 2
    
    new_image=np.zeros((h,w,c),dtype=np.float32)
    for color in range(3):
        padded = np.pad(image[:, :, color], ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
        for i in range(h):
            for j in range(w):
                region = padded[i:i+kh,j:j+kw]
                new_image[i, j, color] = np.sum(region * kernel)

    return new_image

def convolve_2d(image, kernel):
    flipped_kernel = np.flip(kernel)
    new_image=cross_correlation_2d(image, flipped_kernel)
    return new_image

def gaussian_blur_kernel_2d(size,sigma):
    kernel=np.zeros((size,size),dtype=np.float32)
    center=size//2

    normalization=0
    for i in range(size):
        for j in range(size):
            fenzi=(i-center)**2+(j-center)**2
            fenmu=2*(sigma**2)
            kernel[i,j]=np.exp(-fenzi/fenmu)
            normalization+=kernel[i,j]
    kernel/=normalization
    return kernel

def low_pass(image,size,sigma):
    new_image=None
    gauss_kernel=gaussian_blur_kernel_2d(size,sigma)
    new_image=convolve_2d(image,gauss_kernel)
    return new_image

def image_subsampling(image):
    return image[1::2,1::2]

def gaussian_pyramid(image,level_num,sigma,size):
    pyramid=[image]
    for level in range(1,level_num):
        blurred=low_pass(pyramid[-1],size,sigma)
        subsampling=image_subsampling(blurred)
        pyramid.append(subsampling)

    return pyramid

def main():
    level_num=4
    sigma=1.0
    size=5
    #Lena test
    image =cv2.imread('Lena.png', cv2.IMREAD_COLOR)
    pyramid = gaussian_pyramid(image, level_num, sigma, size)
    for i, level_img in enumerate(pyramid):
            if i==0:
                continue
            out_name=f'Lena_level_{i}.png'
            cv2.imwrite(out_name, level_img)
            print(f"Saved: {out_name}")
    #frog test
    image2 =cv2.imread('frog.jpg', cv2.IMREAD_COLOR)
    pyramid2 = gaussian_pyramid(image2, level_num, sigma, size)
    for i, level_img in enumerate(pyramid2):
            if i==0:
                continue
            out_name=f'frog_level_{i}.jpg'
            cv2.imwrite(out_name, level_img)
            print(f"Saved: {out_name}")

    
    print("All images saved, end.")

if __name__ == '__main__':
    main()