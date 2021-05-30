import numpy as np
import matplotlib.pyplot as plt
import astropy.constants as con
import astropy.units as u

def aaa(per):
    ab = per*(0.984/49849)**(1/3)
    return ab

p11 = aaa(34)
print('The new period is:', p11)
