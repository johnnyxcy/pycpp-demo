# lapacke_solve()
# lapacke_solve1(
#     [
#         [1, 2, 3],
#         [4, 5, 6],
#         [7, 8, 9],
#     ],
#     [1, 2, 3],
# )
# import numpy as np
# from scipy.linalg import lu
# mat = np.diag(
#     [
# 1089.942113770775,
# 1010.096696063498,
# 58.66178651250262,
# 5661.020957489456,
# -1589555826.24471,
# -3799865302.243693,
# -162987547.4735685,
# -1771081.02947091,
# 255.3627654232152,
# 151.2319507163488,
#     ]
# )
# print(lu(mat))
import sys

from build.pycppmod import lapacke_solve, lapacke_solve1, lu2

vec = [
    1089.942113770775,
    1010.096696063498,
    58.66178651250262,
    5661.020957489456,
    -1589555826.24471,
    -3799865302.243693,
    -162987547.4735685,
    -1771081.02947091,
    255.3627654232152,
    151.2319507163488,
]

import numpy as np

lu2(np.diag(vec))
