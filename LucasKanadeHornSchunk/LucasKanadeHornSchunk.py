from ex1_utils import *

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time


# def lucaskanade(im1, im2, N, sigma=1, epsilon=1e-7, harris=False, harris_threshold=0.01, harris_alpha=0.05,
#                  return_derivatives=False):
#
#     I1 = im1.copy().astype(np.float64)
#     I2 = im2.copy().astype(np.float64)
#     sum_kernel = np.ones((N, N))
#
#     I1 /= 255.0
#     I2 /= 255.0
#
#     Ix1, Iy1 = gaussderiv(I1, sigma)
#     Ix2, Iy2 = gaussderiv(I2, sigma)
#     Ix = (Ix1 + Ix2) / 2.0
#     Iy = (Iy1 + Iy2) / 2.0
#     It = I2 - I1
#     It = gausssmooth(It, sigma)
#
#     IxIy = Ix * Iy
#
#     I_y2 = cv2.filter2D(Iy ** 2, -1, sum_kernel)
#     I_x2 = cv2.filter2D(Ix ** 2, -1, sum_kernel)
#     Ixy = cv2.filter2D(IxIy, -1, sum_kernel)
#
#     D = I_x2 * I_y2 - Ixy ** 2
#
#     IxIt = cv2.filter2D(Ix * It, -1, sum_kernel)
#     IyIt = cv2.filter2D(Iy * It, -1, sum_kernel)
#
#     u = - (I_y2 * IxIt - Ixy * IyIt)
#     v = - (I_x2 * IyIt - Ixy * IxIt)
#
#     D[D == 0] = epsilon
#
#     if harris:
#         harris_response = D - harris_alpha * (Iy2 + Ix2) ** 2
#
#         unstable_values = harris_response < harris_threshold
#         u[unstable_values] = 0
#         v[unstable_values] = 0
#         u[~unstable_values] = u[~unstable_values] / D[~unstable_values]
#         v[~unstable_values] = v[~unstable_values] / D[~unstable_values]
#     else:
#
#         u = u / D
#         v = v / D
#
#     if return_derivatives:
#         return u, v, Ix, Iy, It
#     else:
#         return u, v

import numpy as np
import cv2


def lucaskanade(img1, img2, window_size, sigma=1.0, eps=1e-7, use_harris=False,
                      harris_thresh=0.01, harris_factor=0.05, return_derivs=False):
    """
    Estimate optical flow using the Lucas–Kanade method with optional Harris reliability check.

    Parameters:
      img1, img2: Grayscale input images.
      window_size: Size of the neighborhood (e.g., 9 for a 9x9 window).
      sigma: Standard deviation for Gaussian derivative and smoothing.
      eps: Small constant to avoid division by zero.
      use_harris: If True, perform a Harris-based check to mark unreliable flow regions.
      harris_thresh: Threshold for the Harris response.
      harris_factor: Weight factor for the squared trace in the Harris response.
      return_derivs: If True, also return the computed spatial and temporal derivatives.

    Returns:
      u, v: Optical flow components.
      Optionally returns: Ix, Iy, It.
    """
    # Convert images to float64 and normalize to [0, 1]
    I1 = img1.copy().astype(np.float64) / 255.0
    I2 = img2.copy().astype(np.float64) / 255.0

    # Define a uniform window for summing over the neighborhood.
    window = np.ones((window_size, window_size), dtype=np.float32)

    # Compute spatial derivatives using Gaussian derivative filters.
    dx1, dy1 = gaussderiv(I1, sigma)
    dx2, dy2 = gaussderiv(I2, sigma)
    # Average derivatives from both images.
    Ix = (dx1 + dx2) / 2.0
    Iy = (dy1 + dy2) / 2.0

    # Compute the temporal derivative and smooth it.
    It = gausssmooth(I2 - I1, sigma)

    # Compute neighborhood sums for squared and mixed derivatives.
    sum_Ix2 = cv2.filter2D(Ix ** 2, -1, window)
    sum_Iy2 = cv2.filter2D(Iy ** 2, -1, window)
    sum_IxIy = cv2.filter2D(Ix * Iy, -1, window)

    # Compute the determinant of the structure tensor.
    det = sum_Ix2 * sum_Iy2 - sum_IxIy ** 2

    # Compute neighborhood sums for products with the temporal derivative.
    sum_IxIt = cv2.filter2D(Ix * It, -1, window)
    sum_IyIt = cv2.filter2D(Iy * It, -1, window)

    # Compute numerators for the optical flow equations.
    num_u = - (sum_Iy2 * sum_IxIt - sum_IxIy * sum_IyIt)
    num_v = - (sum_Ix2 * sum_IyIt - sum_IxIy * sum_IxIt)

    # Prevent division by zero.
    det[det == 0] = eps

    # Optionally, use a Harris response to filter out unreliable flow estimates.
    if use_harris:
        # Compute a Harris-like response. Here, trace is the sum of squared derivatives.
        trace_val = sum_Ix2 + sum_Iy2
        harris_resp = det - harris_factor * (trace_val ** 2)
        reliable_mask = harris_resp >= harris_thresh
        # Initialize flow arrays and compute only where reliable.
        u = np.zeros_like(num_u)
        v = np.zeros_like(num_v)
        u[reliable_mask] = num_u[reliable_mask] / det[reliable_mask]
        v[reliable_mask] = num_v[reliable_mask] / det[reliable_mask]
    else:
        u = num_u / det
        v = num_v / det

    if return_derivs:
        return u, v, Ix, Iy, It
    else:
        return u, v


def warp_image(im, u, v):
    """Warp image im using flow fields u and v."""
    h, w = im.shape
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (grid_x + u).astype(np.float32)
    map_y = (grid_y + v).astype(np.float32)
    return cv2.remap(im, map_x, map_y, interpolation=cv2.INTER_LINEAR)


def pyramidal_lucaskanade(im1, im2, N, num_levels=3):
    """
    Pyramidal Lucas-Kanade optical flow.

    Parameters:
      im1, im2: input grayscale images.
      N: size of the neighborhood (e.g., 9 for a 9x9 window).
      num_levels: number of pyramid levels.

    Returns:
      u, v: optical flow components.
    """
    # Build image pyramids
    pyramid1 = [im1]
    pyramid2 = [im2]
    for _ in range(1, num_levels):
        im1 = cv2.pyrDown(im1)
        im2 = cv2.pyrDown(im2)
        pyramid1.append(im1)
        pyramid2.append(im2)

    # Initialize flow at coarsest level
    u = np.zeros_like(pyramid1[-1])
    v = np.zeros_like(pyramid1[-1])

    # Process from coarsest to finest level
    for level in range(num_levels - 1, -1, -1):
        I1 = pyramid1[level]
        I2 = pyramid2[level]
        # If not the coarsest, upsample flow to current level's resolution
        if level < num_levels - 1:
            u = cv2.resize(u, (I1.shape[1], I1.shape[0]), interpolation=cv2.INTER_LINEAR) * 2
            v = cv2.resize(v, (I1.shape[1], I1.shape[0]), interpolation=cv2.INTER_LINEAR) * 2
        # Warp I2 using the current flow estimate
        I2_warp = warp_image(I2, u, v)
        # Compute incremental flow at this level using the standard lucaskanade
        du, dv = lucaskanade(I1, I2_warp, N)
        # Update the flow estimates
        u = u + du
        v = v + dv

    return u, v



# def hornschunck(im1, im2, n_iters, lmbd, sigma=1, init_lucas_kanade=False):
#     # Preprocess images: copy, convert to float32, and normalize.
#     I1 = im1.copy().astype(np.float32) / 255.0
#     I2 = im2.copy().astype(np.float32) / 255.0
#
#     # Compute spatial derivatives using Gaussian derivative (sigma).
#     Ix1, Iy1 = gaussderiv(I1, sigma)
#     Ix2, Iy2 = gaussderiv(I2, sigma)
#     Ix = (Ix1 + Ix2) / 2.0
#     Iy = (Iy1 + Iy2) / 2.0
#
#     # Compute the temporal derivative and smooth it.
#     It = gausssmooth(I2 - I1, sigma)
#
#     # Initialize flow fields.
#     if init_lucas_kanade:
#         # Use Lucas–Kanade to provide an initial estimate and derivatives.
#         u, v, _, _, _ = lucaskanade(im1, im2, 11, sigma, return_derivatives=True)
#     else:
#         u = np.zeros_like(I1)
#         v = np.zeros_like(I1)
#
#     # Precompute the denominator for the update equation.
#     D = lmbd + Ix**2 + Iy**2
#     # Define the residual Laplacian kernel.
#     laplacian = np.array([[0, 1, 0],
#                           [1, 0, 1],
#                           [0, 1, 0]], dtype=np.float32) / 4
#
#     # Iteratively update the flow estimates.
#     for _ in range(n_iters):
#         # Compute local averages via convolution with the Laplacian kernel.
#         u_avg = cv2.filter2D(u, -1, laplacian)
#         v_avg = cv2.filter2D(v, -1, laplacian)
#         # Compute the update term.
#         P = Ix * u_avg + Iy * v_avg + It
#         # Update u and v.
#         u = u_avg - Ix * (P / D)
#         v = v_avg - Iy * (P / D)
#
#     return u, v

def hornschunck(img1, img2, iterations, lmbd, sigma=1.0, init_with_lk=False):
    """
    Compute optical flow using the Horn–Schunck algorithm.

    Parameters:
      img1, img2 : Input grayscale images.
      iterations : Number of iterative updates.
      lmbd       : Regularization parameter.
      sigma      : Standard deviation for Gaussian derivatives and smoothing.
      init_with_lk : If True, initialize flow using Lucas–Kanade.

    Returns:
      u, v : Optical flow fields in x and y directions.
    """
    # Convert images to float32 and normalize.
    I1 = img1.astype(np.float32).copy() / 255.0
    I2 = img2.astype(np.float32).copy() / 255.0

    # Compute spatial gradients for both images using Gaussian derivative.
    grad_x1, grad_y1 = gaussderiv(I1, sigma)
    grad_x2, grad_y2 = gaussderiv(I2, sigma)
    grad_x = (grad_x1 + grad_x2) / 2.0
    grad_y = (grad_y1 + grad_y2) / 2.0

    # Calculate the temporal gradient and smooth it.
    grad_t = gausssmooth(I2 - I1, sigma)

    # Initialize flow fields: either using Lucas–Kanade or zeros.
    if init_with_lk:
        u, v, _, _, _ = lucaskanade(img1, img2, 11, sigma, return_derivatives=True)
    else:
        u = np.zeros_like(I1)
        v = np.zeros_like(I1)

    # Precompute the constant part of the denominator.
    denom = lmbd + grad_x ** 2 + grad_y ** 2

    # Define a 4-connected Laplacian kernel for neighborhood averaging.
    laplacian_kernel = np.array([[0, 1, 0],
                                 [1, 0, 1],
                                 [0, 1, 0]], dtype=np.float32) / 4.0

    # Iteratively refine the flow estimates.
    for _ in range(iterations):
        u_avg = cv2.filter2D(u, -1, laplacian_kernel)
        v_avg = cv2.filter2D(v, -1, laplacian_kernel)
        # Compute the update term based on current local averages.
        update = grad_x * u_avg + grad_y * v_avg + grad_t
        u = u_avg - grad_x * (update / denom)
        v = v_avg - grad_y * (update / denom)

    return u, v


# im1 = np.random.rand(200, 200).astype(np.float32)
# im2 = im1.copy()
# im2 = rotate_image(im2, 1)
# U_lk, V_lk = lucaskanade(im1, im2, 9)
# U_hs, V_hs = hornschunck(im1, im2, 1000, 0.5)
# fig1, ((ax1_11, ax1_12), (ax1_21, ax1_22)) = plt.subplots(2, 2)
# ax1_11.imshow(im1)
# ax1_12.imshow(im2)
# show_flow(U_lk, V_lk, ax1_21, type='angle')
# show_flow(U_lk, V_lk, ax1_22, type='field', set_aspect=True)
# fig1.suptitle('Lucas Kanade Optical Flow')
# fig2, ((ax2_11, ax2_12), (ax2_21, ax2_22)) = plt.subplots(2, 2)
# ax2_11.imshow(im1)
# ax2_12.imshow(im2)
# show_flow(U_hs, V_hs, ax2_21, type='angle')
# show_flow(U_hs, V_hs, ax2_22, type='field', set_aspect=True)
# fig2.suptitle('Horn Schunck Optical Flow')
# fig1.savefig("lucas_kanade_flow9.svg", format="svg")
# fig2.savefig("horn_schunck_flow1000,05.svg", format="svg")
# plt.show()

def load_folder_images(folder_path):
    images = []
    image_names = sorted(os.listdir(folder_path))
    for image_name in image_names:
        image_path = os.path.join(folder_path, image_name)
        im = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        im = im.astype(np.float32)
        images.append(im)
    return images




def show_flow_on_image(image, u, v, figsize=(10, 10)):
    _, ax = plt.subplots(1, 1, figsize=figsize)
    show_flow(u, v, ax, set_aspect=True)
    plt.imshow(image, cmap='gray', extent=[0, image.shape[1], -image.shape[0], 0])
    plt.axis('off')

truck_images = load_folder_images('./collision')
lab_images = load_folder_images('./lab2')

# Harris with and without

# u_lk, v_lk = lucaskanade(truck_images[140], truck_images[141], 11, sigma=1, eps=1e-5,
#                           use_harris=True,
#                           harris_factor=0.1,
#                           harris_thresh=0.4)
#
# u_hs, v_hs = hornschunck(truck_images[140], truck_images[141], 1000, 0.5)
#
#
# show_flow_on_image(truck_images[140], u_lk, v_lk)
# plt.tight_layout()
# plt.savefig('./results/collision/lk_haris3.svg')
#
# u_lk, v_lk = lucaskanade(truck_images[140], truck_images[141], 11, sigma=1, eps=1e-8,
#                           use_harris=False,
#                           harris_factor=0,
#                           harris_thresh=0)
#
# show_flow_on_image(truck_images[140], u_lk, v_lk)
# plt.tight_layout()
# plt.savefig('./results/collision/lk_no_haris3.svg')

# img1 = cv2.imread('ezgif-frame-040.jpg', cv2.IMREAD_GRAYSCALE)
# img2 = cv2.imread('ezgif-frame-041.jpg', cv2.IMREAD_GRAYSCALE)
#
# u_p, v_p = pyramidal_lucaskanade(img1,
#                                   img2,
#                                   N=11,
#                                   num_levels=4)
#
# show_flow_on_image(img1, u_p, v_p)
# plt.savefig('./results/lk_pyramidal.svg', bbox_inches='tight')
#
#
# u_p, v_p = lucaskanade(img1,
#                         img2,
#                         window_size=11,
#                         sigma=1,
#                         use_harris=False,
#                         harris_factor=0.1,
#                         harris_thresh=0.04)
#
# show_flow_on_image(img1, u_p, v_p)
# plt.savefig('./results/lk_no_pyramidal.svg', bbox_inches='tight')


# Algorithm comparison:

# img_cporta1 = cv2.imread('./disparity/cporta_left.png', cv2.IMREAD_GRAYSCALE)
# img_cporta2 = cv2.imread('./disparity/cporta_right.png', cv2.IMREAD_GRAYSCALE)
#
# u_p, v_p = hornschunck(lab_images[23],
#                         lab_images[24],
#                         1000,
#                         0.5)
#
# show_flow_on_image(lab_images[23], u_p, v_p)
# plt.savefig('./results/lab_hc.pdf', bbox_inches='tight')
#
# u_p, v_p = hornschunck(truck_images[150],
#                         truck_images[151],
#                         1000,
#                         0.5)
#
# show_flow_on_image(truck_images[150], u_p, v_p)
# plt.savefig('./results/collision/hs_truck_150.pdf', bbox_inches='tight')
#
# u_p, v_p = hornschunck(img_cporta1,
#                         img_cporta2,
#                         1000,
#                         1)
#
# show_flow_on_image(img_cporta1, u_p, v_p)
# plt.savefig('./results/hs_porta.pdf', bbox_inches='tight')
#
#
#
#
#
# u_p, v_p = lucaskanade(truck_images[150],
#                         truck_images[151],
#                         window_size=11,
#                         sigma=1,
#                         use_harris=False)
#
#
# show_flow_on_image(truck_images[150], u_p, v_p)
# plt.savefig('./results/lk_truck_150.svg', bbox_inches='tight')
#
# u_p, v_p = lucaskanade(lab_images[23],
#                         lab_images[24],
#                         window_size=11,
#                         sigma=1,
#                         use_harris=False)
#
#
# show_flow_on_image(lab_images[23], u_p, v_p)
# plt.savefig('./results/lab_lk.svg', bbox_inches='tight')
#
# u_p, v_p = lucaskanade(img_cporta1,
#                         img_cporta2,
#                         window_size=11,
#                         sigma=2,
#                         use_harris=False)
#
#
#
# show_flow_on_image(img_cporta1, u_p, v_p)
# plt.savefig('./results/cporta_lk.svg', bbox_inches='tight')

# Time comparisons


# start = time.time()
# hornschunck(truck_images[130], truck_images[131], 1000, 1)
# end = time.time()
# print(end - start)
#
#
# start = time.time()
# hornschunck(truck_images[130], truck_images[131], 1000, 1, init_with_lk=True)
# end = time.time()
# print(end - start)
#
#
# start = time.time()
# hornschunck(truck_images[130], truck_images[131], 500, 1)
# end = time.time()
# print(end - start)
#
#
# start = time.time()
# lucaskanade(truck_images[130], truck_images[131], 9, sigma=1)
# end = time.time()
# print(end - start)
#
#
# start = time.time()
# lucaskanade(truck_images[130], truck_images[131], 3, sigma=1)
# end = time.time()
# print(end - start)
#
#
# start = time.time()
# pyramidal_lucaskanade(truck_images[130], truck_images[131], 9, num_levels=4)
# end = time.time()
# print(end - start)



#Hs with LucasKanade

# u_hs, v_hs = hornschunck(lab_images[23], lab_images[24], 500, 1.5, init_with_lk=True)
# show_flow_on_image(lab_images[23], u_hs, v_hs)
# plt.savefig('./results/lab_hs_LK.svg', bbox_inches='tight')
#
# u_hs, v_hs = hornschunck(lab_images[23], lab_images[24], 500, 1.5, init_with_lk=False)
# show_flow_on_image(lab_images[23], u_hs, v_hs)
# plt.savefig('./results/lab_hs_NO_LK.svg', bbox_inches='tight')



