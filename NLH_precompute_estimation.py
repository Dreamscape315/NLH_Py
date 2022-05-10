import Complex


def NLH_precompute_estimation(im, stage):
    dis_map, sigma_tb = Complex.Distance(8, 16, 2, 39, 32, im)
    if sigma_tb > 70:
        dis_map1, sigma_tb1 = Complex.Distance_refine(8, 16, 2, 39, 32, im, dis_map, sigma_tb)
        sigma_est = sigma_tb1
    else:
        dis_map1, sigma_tb1 = Complex.Distance_refine(8, 16, 2, 39, 32, im, dis_map, sigma_tb)
        dis_map2, sigma_tb2 = Complex.Distance_refine(8, 16, 2, 39, 32, im, dis_map1, sigma_tb1)
        sigma_est = sigma_tb2

    if stage == 'Stage1':
        dis_map, sigma_temp = Complex.Distance(8, 16, 2, 19, 8, im)
        return sigma_est, dis_map
    elif stage == 'Stage2':
        return sigma_est
