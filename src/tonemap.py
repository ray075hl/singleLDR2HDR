# Tone Reproduction
import numpy as np
import cv2


def tonereproduct(bgr_image, L, R_, Ik_list, FLAG):
    """
    Fusion of multiple exposure images.
    """
    Lk_list = [ np.exp(R_) * Ik for Ik in Ik_list ] 
    L = L + 1e-22 

    rt = 1.0
    b, g, r = cv2.split(bgr_image)
    # Restore color image
    if FLAG == False:
        Sk_list = [cv2.merge((Lk*(b/L)**rt, Lk*(g/L)**rt, Lk*(r/L)**rt)) for Lk in Lk_list]
        return Sk_list[2]
    else:  # Weight maps

        Wk_list = []
        for index, Ik in enumerate(Ik_list):
            if index < 3:
                wk = Ik / np.max(Ik)
            else:
                temp = 0.5*(1 - Ik)
                wk = temp / np.max(temp)
            Wk_list.append(wk)

        A = np.zeros_like(Wk_list[0])
        B = np.zeros_like(Wk_list[0])
        for lk, wk in zip(Lk_list, Wk_list):
            A = A + lk * wk 
            B = B + wk

        L_ = (A/B)
        ratio = np.clip(L_/L, 0, 3) # Clip unreasonable values
        b_ = ratio * b
        g_ = ratio * g
        r_ = ratio * r
        out = cv2.merge( ( b_, g_, r_ ) )
        return np.clip(out, 0.0, 1.0)

# Unit test
if __name__ == '__main__':
    pass 


