import sys
import cv2 
import numpy as np
from matplotlib import pyplot as plt


from src.wls_filter import wlsFilter
from src.srs import SRS
from src.virtual_ev import VIG 
from src.tonemap import *


def Show_origin_and_output(origin, I):
   """
   Show final result.
   """
   plt.figure(figsize=(12, 6))
   plt.subplots_adjust(left=0,right=1,bottom=0,top=1, wspace=0.005, hspace=0)

   plt.subplot(121),plt.imshow(np.flip(origin, 2)),plt.title('Origin')
   plt.axis('off')
   plt.subplot(122),plt.imshow(np.flip(I, 2)),plt.title('Fake HDR')
   plt.axis('off')
   plt.savefig('compare.png', bbox_inches='tight', pad_inches=0)
   plt.show()
   


class FakeHDR():
    
    def __init__(self, flag):
        self.weighted_fusion = flag
        self.wls = wlsFilter
        self.srs = SRS
        self.vig = VIG
        self.tonemap = tonereproduct

    def process(self, image):
        
        if image.shape[2] == 4:
            image = image[:,:,0:3]
        S = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)/255.0
        image = 1.0*image/255
        L = 1.0*S
    
        I = self.wls(S)
        R = np.log(L+1e-22) - np.log(I+1e-22)
        R_ = self.srs(R, L)
        I_K = self.vig(L, 1.0 - L)

        result_ = self.tonemap(image, L, R_, I_K, self.weighted_fusion)
        return result_

