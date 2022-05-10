import Complex
import numpy as np


def NLH_1st_stage(im, sigma, dis_map):
    # <hyper parameter> -------------------------------------------------------------------------------
    Ns = 43
    N3 = 8
    N2 = 32
    alpha = 0.618
    lamda = 0.8
    Thr = 1.35
    TTHR = 10000.15
    # <\ hyper parameter> -----------------------------------------------------------------------------
    if sigma < 7.5:
        k = 2
        N_step = 4
        N1 = 9
    elif 7.5 <= sigma < 60:
        k = 3
        N_step = 5
        N1 = 9
    elif 60 <= sigma < 75:
        k = 5
        N_step = 7
        N1 = 9
    elif 75 <= sigma < 85:
        k = 6
        N_step = 8
        N1 = 9
    else:
        k = 7
        N_step = 9
        N1 = 9

    imr = im
    for i in range(k):
        imr = Complex.Adaptive(N1, N2, N3, Ns, N_step, np.array(alpha * imr + (1 - alpha) * im), Thr, sigma / 255,
                               lamda * imr + (1 - lamda) * im, TTHR, np.array(dis_map))
        N_step -= 1
        if N_step <= 3:
            N_step = 3
    return imr
