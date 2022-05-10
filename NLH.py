import os
import time
import cv2
import utils

from NLH_precompute_estimation import NLH_precompute_estimation
from NLH_1st_stage import NLH_1st_stage
from NLH_2nd_stage import NLH_2nd_stage


def runNLH(im):
    sigma_est, dis_map = NLH_precompute_estimation(im, stage='Stage1')
    imr = NLH_1st_stage(im, sigma_est, dis_map)
    sigma_imr = NLH_precompute_estimation(imr, stage='Stage2')
    denoiseImg = NLH_2nd_stage(im, imr, dis_map, sigma_est, sigma_imr)

    return denoiseImg


if __name__ == '__main__':
    im_dir = 'image'
    im_name = 'house.png'
    # input a picture
    im_path = os.path.join(im_dir, im_name)
    img = cv2.imread(im_path, cv2.IMREAD_GRAYSCALE)
    img = utils.im2double(img)
    # create a noise image

    # set sigma value
    sigma = 30
    im_noise = utils.add_gaussian_noise(img, sigma)

    # denoise
    start = time.time()

    im_denoise = runNLH(im_noise)

    print("time is ", time.time() - start)
    # result
    PSNR = utils.compute_psnr(img, im_denoise)
    SSIM = utils.compute_ssim(img, im_denoise)
    print("PSNR is ", PSNR)
    print("SSIM is ", SSIM)

    save_name = im_name[:-4] + '_sigma_' + str(sigma) + '.png'
    cv2.imwrite(os.path.join('result', save_name), im_denoise * 255)
    cv2.imshow("The final image", im_denoise)
    cv2.waitKey(0)
