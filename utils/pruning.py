import torch
import numpy as np
from scipy.integrate import quad
from TracyWidom import TracyWidom
import matplotlib.pyplot as plt

def bema_scheduler(epoch):
    return max(0, -1/300*epoch + 1)

def mp_density(ndf, pdim, var=1):
    gamma = ndf/pdim
    inv_gamma_sqrt = np.sqrt(1/gamma)
    a = var*(1-inv_gamma_sqrt)**2
    b = var*(1+inv_gamma_sqrt)**2
    return a, b

def dmp(x, ndf, pdim, var=1, log=False):
    gamma = ndf/pdim
    a, b = mp_density(ndf, pdim, var)

    if not log:
        if gamma == 1 and x == 0 and 1/x > 0:
            d = np.inf
        elif x <= a and x >= b:
            d = 0
        else:
            d = gamma/(2*np.pi*var*x) * np.sqrt((x-a)*(b-x))
    else:
        if gamma == 1 and x == 0 and 1/x > 0:
            d = np.inf
        elif x <= a and x >= b:
            d = -np.inf
        else:
            d = (np.log(gamma) - (np.log(2) + np.log(np.pi) + np.log(var) + np.log(x)) +
                 0.5*np.log(x-a) + 0.5*np.log(b-x))
    return d

def error(singular_values, alpha, pTilde, gamma, sigma_sq, show=False):
    """Compute L-infinity error between theoretical and empirical CDFs"""
    ind = np.arange(int(alpha*pTilde), int((1-alpha)*pTilde))
    pruned_values = singular_values[ind]

    theoretical = mp_cdf(gamma, sigma_sq, pruned_values)
    empirical = alpha + (1-2*alpha)*np.arange(len(pruned_values))/len(pruned_values)

    difference = theoretical - empirical

    if show:
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.hist(difference, bins=50, label='Difference histogram')
        plt.legend()
        plt.title('Distribution of CDF Differences')

        plt.subplot(1, 2, 2)
        x = np.arange(len(empirical))
        plt.plot(x, empirical, label='empirical')
        plt.plot(x, theoretical, label='theoretical')
        plt.legend()
        plt.title('CDF Comparison')
        plt.show()

    return np.linalg.norm(difference, np.inf)

def mp_density_wrapper(gamma, sigma_sq, sample_points):
    """Compute MP density at sample points"""
    lp = sigma_sq*np.power(1+np.sqrt(gamma), 2)
    lm = sigma_sq*np.power(1-np.sqrt(gamma), 2)

    return np.array([
        mp_density_inner(gamma, sigma_sq, x) if lm <= x <= lp else 0
        for x in sample_points
    ])

def mp_density_inner(gamma, sigma_sq, x):
    """Helper function to compute MP density at a point"""
    lp = sigma_sq*np.power(1+np.sqrt(gamma), 2)
    lm = sigma_sq*np.power(1-np.sqrt(gamma), 2)
    return np.sqrt((lp-x)*(x-lm))/(gamma*x*2*np.pi*sigma_sq)

def mp_cdf(gamma, sigma_sq, sample_points):
    """Compute MP CDF at sample points"""
    lp = sigma_sq*np.power(1+np.sqrt(gamma), 2)
    lm = sigma_sq*np.power(1-np.sqrt(gamma), 2)

    output = []
    for x in sample_points:
        if gamma <= 1:
            if x < lm or x >= lp:
                output.append(0)
            else:
                output.append(mp_cdf_inner(gamma, sigma_sq, x))
        else:
            if x < lm:
                output.append((gamma-1)/gamma)
            elif x >= lp:
                output.append(1)
            else:
                output.append((gamma-1)/(2*gamma) + mp_cdf_inner(gamma, sigma_sq, x))

    return np.array(output)

def mp_cdf_inner(gamma, sigma_sq, x):
    """Helper function to compute MP CDF at a point"""
    lp = sigma_sq*np.power(1+np.sqrt(gamma), 2)
    lm = sigma_sq*np.power(1-np.sqrt(gamma), 2)
    r = np.sqrt((lp - x)/(x - lm))

    F = np.pi * gamma + (1/sigma_sq)*np.sqrt((lp - x)* (x - lm))
    F += -(1+gamma)*np.arctan((r*r-1)/(2*r))

    if gamma != 1:
        F += (1-gamma) * np.arctan((lm *r*r - lp)/(2 *sigma_sq *(1-gamma)*r))

    F /= 2 * np.pi * gamma
    return F

def bema_inside(pdim, ndf, eigs, alpha, beta):
    """Core BEMA algorithm implementation"""
    pTilde = min(pdim, ndf)
    gamma = pdim/ndf
    ev = np.sort(eigs)
    ind = list(range(int(alpha*pTilde), int((1-alpha)*pTilde)))

    # Compute q values and corresponding eigenvalues
    q = [dmp(i/pTilde, ndf, pdim, 1) for i in ind]
    lamda = [ev[i] for i in ind]

    # Compute sigma squared
    num = np.dot(q, lamda)
    denum = np.dot(q, q)
    sigma_sq = num/denum

    # Compute lambda plus using Tracy-Widom distribution
    tw1 = TracyWidom(beta=1)
    t_b = tw1.cdfinv(1-beta)
    lamda_plus = sigma_sq*(((1+np.sqrt(gamma))**2 +
                           t_b*ndf**(-2/3)*(gamma)**(-1/6)*(1+np.sqrt(gamma))**4/3))
    l2 = sigma_sq* (1+np.sqrt(gamma))**2

    return sigma_sq, lamda_plus, l2

def bema_mat_wrapper(matrix, pReal, nReal, alpha, beta, goodnessOfFitCutoff, show=False):
    """Wrapper function for BEMA algorithm"""
    # Handle matrix transposition if necessary for eigenvalue computation
    if pReal <= nReal:
        p = pReal
        n = nReal
        matrix_norm = np.matmul(matrix, matrix.transpose())/nReal
    else:
        p = nReal
        n = pReal
        matrix_norm = np.matmul(matrix.transpose(), matrix)/nReal

    # Compute eigenvalues
    v = np.linalg.eigvalsh(matrix_norm)
    sigma_sq, lamda_plus, l2 = bema_inside(p, n, v, alpha, beta)

    # Compute error and goodness of fit
    pTilde = min(p, n)
    LinfError = error(v, alpha, pTilde, p/n, sigma_sq)
    gamma = p/n
    goodFit = LinfError < goodnessOfFitCutoff

    if show:
        visualize_spectrum(v, lamda_plus, gamma, sigma_sq, pTilde)

    return v, p/n, sigma_sq, lamda_plus, goodFit

def visualize_spectrum(v, lamda_plus, gamma, sigma_sq, pTilde):
    """Visualize eigenvalue spectrum and MP distribution"""
    plt.figure(figsize=(15, 5))

    # Full spectrum plot
    plt.subplot(1, 2, 1)
    plt.hist(v[-pTilde:], bins=100, color="black",
            label="Empirical Density", density=True)
    plt.axvline(x=lamda_plus, label="Lambda Plus", color="red")
    plt.legend()
    plt.title("Empirical Distribution Density")

    # Truncated spectrum plot
    plt.subplot(1, 2, 2)
    eigsTruncated = [i for i in v[-pTilde:] if i < lamda_plus]
    plt.hist(eigsTruncated, bins=100, color="black",
            label="Truncated Empirical Density", density=True)

    # Add theoretical density
    Z = np.linspace(min(eigsTruncated), max(eigsTruncated), 100)
    Y = mp_density_wrapper(gamma, sigma_sq, Z)
    plt.plot(Z, Y, color="orange", label="Predicted Density")
    plt.legend()
    plt.title("Density Comparison Zoomed")

    plt.tight_layout()
    plt.show()

def compute_eigs_to_keep(model, layer_matrix, dims, epoch, goodness_of_fit_cutoff, show=False):
    """Compute number of eigenvalues to keep based on BEMA algorithm"""
    p, n = layer_matrix.shape

    eigs, gamma, sigma_sq, lambda_plus, good_fit = bema_mat_wrapper(
        layer_matrix, p, n, 0.2, 0.1, goodness_of_fit_cutoff, show=show
    )

    # Find eigenvalues below lambda_plus
    lt = len(eigs) - 1
    for i in range(len(eigs)):
        if eigs[i] > lambda_plus:
            lt = i - 1
            break

    # Calculate number of eigenvalues to keep
    gt = len(eigs) - lt
    p = gt + int(bema_scheduler(epoch) * lt)

    # Ensure we keep at least as many eigenvalues as output dimension
    eigs_to_keep = max(p, dims[-1])
    lp_transformed = np.sqrt(lambda_plus) * np.sqrt(n)

    return lp_transformed, eigs_to_keep, good_fit