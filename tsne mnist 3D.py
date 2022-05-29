####################
# load mnist
####################

from tensorflow.keras.datasets import fashion_mnist

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

print('x_train.shape =', x_train.shape)
print('y_train.shape =', y_train.shape)
print('x_test.shape =', x_test.shape)
print('y_test.shape =', y_test.shape)
print()

# use a smaller dataset
samples = 5000

####################
# flatten
####################

x_train_flat = x_train.reshape(x_train.shape[0], -1)
x_train_flat = x_train_flat[:samples, :] # cut

print('x_train_flat.shape =', x_train_flat.shape)
print()



####################
# Dimensionality Reduction
# Timing
####################

import time
start = time.time()
"""
####################
# pca
####################

from sklearn.decomposition import PCA

pca = PCA(n_components = 3)
x_train_3d = pca.fit_transform(x_train_flat)
"""
####################
# t-SNE
####################

from sklearn.manifold import TSNE

tsne = TSNE( n_components = 3,
             # method = 'barnes_hut', # default & fast o(n*ln(n))
             
             init = 'random', # slow & better
             # init = 'pca', # fast & bad

             learning_rate = 'auto',
             )
x_train_3d = tsne.fit_transform(x_train_flat)

# end timing
end = time.time()
print('time =', end-start)



####################
# visualization
####################

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# define axes
ax = plt.axes(projection='3d')

scatter = ax.scatter(
    x_train_3d[:, 0],
    x_train_3d[:, 1],
    x_train_3d[:, 2],
    c=y_train[:samples], cmap='jet', s=5 ) # s=size, alpha=0.3

# auto legend
ax.legend( *scatter.legend_elements(prop='colors'),
           title='fashion_mnist', loc='best' )

ax.grid()
plt.tight_layout()
plt.show()
