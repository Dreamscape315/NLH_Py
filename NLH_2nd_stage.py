import Complex


def NLH_2nd_stage(im, imr, dis_map, sigmaNoise, sigmaImr):
    # <hyper parameter> -------------------------------------------------------------------------------
    N2 = 64
    N3 = 8
    Ns = 129
    beta = 0.8
    # <\ hyper parameter> -----------------------------------------------------------------------------
    if sigmaNoise < 25:
        N1 = 8
        N_step1 = 8
        N_step2 = 5
    elif 25 <= sigmaNoise < 75:
        N1 = 16
        N_step1 = 16
        N_step2 = 10
    else:
        N1 = 20
        N_step1 = 20
        N_step2 = 13

    if sigmaNoise < 7.5:
        gamma = 6.0 + abs(sigmaImr - sigmaNoise * 0.05)
    elif 7.5 <= sigmaNoise < 25:
        gamma = 6.0 + abs(sigmaImr - sigmaNoise * 0.05)
    elif 25 <= sigmaNoise < 40.0:
        gamma = 6.0 + abs(sigmaImr - sigmaNoise * 0.05)
    else:
        gamma = 6.0 + abs(sigmaImr - sigmaNoise * 0.05)
    y_est = Complex.Wiener(N1, N2, N3, Ns, N_step1, im, gamma, sigmaNoise / 255, dis_map, beta, imr * gamma,
                           imr * beta + im * (1 - beta))
    for i in range(2):
        y_est = Complex.Wiener(N1, N2, N3, Ns, N_step1, y_est, gamma, sigmaNoise / 255, dis_map, beta, imr * gamma,
                               y_est * beta + im * (1 - beta))
    y_est = Complex.Wiener(N1, N2, N3, Ns, N_step2, y_est, gamma, sigmaNoise / 255, dis_map, beta, (imr * gamma),
                           (y_est * beta + im * (1 - beta)))
    y_est = Complex.Wiener(N1, N2, N3, Ns, N_step2, y_est, gamma, sigmaNoise / 255, dis_map, beta, (imr * gamma),
                           y_est)

    return y_est
