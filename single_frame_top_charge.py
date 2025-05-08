from pathlib import Path

import skimage.filters
import scipy.stats
import scipy.ndimage
import pyovf as OVF
import numpy as np
import matplotlib.pyplot as plt

import pims
import trackpy as tp


def topological_density(vector_field, sigma=0):
    dx = np.gradient(vector_field, axis=0)
    dy = np.gradient(vector_field, axis=1)

    # Calculate components of the cross product (n_x * dn/dx x dn/dy)
    cross_product = np.cross(dx, dy)
    return np.sum(vector_field * cross_product, axis=2) / (4 * np.pi)

    
X, Y, data_out = OVF.read('/cluster/home/mauricec/simulations/5x5_20x3_res32_PBC001_demag_j010_pol_0p2_10e10_run50ns/m000000.ovf')
vector_field = skimage.filters.gaussian(data_out[0], sigma=7, channel_axis=2)

f = topological_density(vector_field)

print(X.shape)
print(Y.shape)
print(data_out.shape)
print("f", f.shape)

plt.imshow(vector_field[:, :, 0])
plt.savefig('image.png')

plt.imshow(f)
plt.savefig('top_charge.png')