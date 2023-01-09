### CS229 Code Starts Here ###
"""
2022-12-07 Linus Hein.
"""
import argparse

import numpy as np
from matplotlib import pyplot as plt
import sklearn.preprocessing

parser = argparse.ArgumentParser()

parser.add_argument('file', type=str)
parser.add_argument('column', type=str)
args = parser.parse_args()

file = args.file
column = int(args.column)

fig, axs = plt.subplots(1, 2)
fig.set_size_inches(8, 4)
fig.set_dpi(400)

arr = np.load(file, allow_pickle=True)

# normalize w/o noise
normalizer = sklearn.preprocessing.QuantileTransformer(
    output_distribution='normal',
    n_quantiles=max(min(arr.shape[0] // 30, 1000), 10),
    subsample=1e9,
    random_state=0,
)
normalizer.fit(arr)
arr_norm = normalizer.transform(arr)
value_norm = arr_norm[:, column]
axs[0].hist(value_norm, 100)
axs[0].set_title('Normalize w/o noise')
axs[0].grid()

# normalize w/ noise
normalizer = sklearn.preprocessing.QuantileTransformer(
    output_distribution='normal',
    n_quantiles=max(min(arr.shape[0] // 30, 1000), 10),
    subsample=1e9,
    random_state=0,
)
arr = arr + np.random.randn(*arr.shape) * 1e-4
normalizer.fit(arr)
arr_norm = normalizer.transform(arr)
value_norm = arr_norm[:, column]
axs[1].hist(value_norm, 100)
axs[1].set_title('Normalize w/ noise')
axs[1].grid()
# plt.savefig('normalization.png', dpi=400)
plt.show()
### CS229 Code Ends Here ###
