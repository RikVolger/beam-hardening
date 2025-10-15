import numpy as np

def birch_marshall():
    """Generate an X-ray energy spectrum
    
    Python implementation of the Birch & Marshall algorithm for generating
    theoretical X-ray spectra.
    Original paper (B&M): https://doi.org/10.1088/0031-9155/24/3/002
    Algorithm presented in: https://doi.org/10.1118/1.596223 
    by Boone.
    """
    k = 1e-14
    a0 = 0.503
    a1 = -0.946
    a2 = 0.155
    a3 = 1.163
    a4 = -0.682

    b = lambda kVeq: (.0029 * kVeq + 0.41) * 10**6
    c = lambda kVeq, i, b, theta: (kVeq**2 - i**2) / (b * np.tan(theta))

    S_l = np.sum(
        k * (a0 + a1 * )
    )