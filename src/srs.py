# Selective Reflectance Scaling
import numpy as np
import cv2 


def SRS(reflectance, illuminace):
    """
    Stretch the pixel whose illuminace brighter than mean value 
    """
    r_R = 0.5
    def compare_func(r, i, m):    
        return r * (i/m)**r_R if i > m else r 

    srs_fun = np.vectorize(compare_func)
    mean_I = np.mean(illuminace)
    result = srs_fun(reflectance, illuminace, mean_I)
    return result

# Unit test
if __name__ == '__main__':
    pass 
