# import sys
# import os
# from os.path import join as pjoin
# import numpy as np
# import matplotlib.pyplot as plt
# import scipy

# import refnx
# from refnx.analysis import CurveFitter
# from refnx.reflect import Slab

# import refellips
# from refellips.dataSE import DataSE, open_EP4file
# from refellips.reflect_modelSE import ReflectModelSE
# from refellips.objectiveSE import ObjectiveSE
# from refellips.dispersion import RI, Cauchy, load_material

# print(
#     f"refellips: {refellips.version.version}\n"
#     f"refnx: {refnx.version.version}\n"
#     f"scipy: {scipy.version.version}\n"
#     f"numpy: {np.version.version}"
# )

csv_file = "../../data/KH522_Ellipso.csv"

with open(csv_file) as f:
    for i in range(5):
        print(repr(f.readline()))
