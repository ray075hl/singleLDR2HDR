# Virtual Illumination Generation (VIG)
import cv2
import numpy as np


def scale_fun(v_, mean_i_, max_i_):

    r = 1.0 - mean_i_/max_i_    
    fv = lambda v : r*( 1/(1+np.exp(-1.0*(v - mean_i_))) - 0.5 )
    
    fv_k_ = [fv(vk) for vk in v_]
    return fv_k_

def VIG(illuminace, inv_illuminace):
    """
    Generation of virtual exposure images(5 levels)
    """
    inv_illuminace /=np.max(inv_illuminace)
    mi = np.mean(illuminace)

    maxi = np.max(illuminace)
    v1 = 0.2;    v3 = mi;    v2 = 0.5 * (v1 + v3)
    v5 = 0.8;    v4 = 0.5 * (v3 + v5)
    v = [v1, v2, v3, v4, v5]
    fvk_list = scale_fun(v, mi, maxi)
    # equation (7)
    # I_k = [(1+fvk)*illuminace for fvk in fvk_list]  
    # equation (8)
    I_k = [(1 + fvk) * (illuminace + fvk * inv_illuminace) for fvk in fvk_list]  

    return I_k


# Unit test
if __name__ == '__main__':
    pass 










