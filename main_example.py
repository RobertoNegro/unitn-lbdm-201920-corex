import linearcorex as lc
import numpy as np

print('\nInput Matrix\n==================')
#              A     A     A     A     A     iA    C     C     A     C
X = np.array([[0.01, 0.01, 0.01, 0.01, 0.01, 1.00, 1.00, 1.00, 0.01, 1.00],
              [0.01, 0.01, 0.01, 0.01, 0.01, 1.00, 0.00, 0.00, 0.01, 0.00],
              [1.00, 1.00, 1.00, 1.00, 1.00, 0.01, 0.00, 0.00, 1.00, 0.00],
              [1.00, 1.00, 1.00, 1.00, 1.00, 0.01, 1.00, 1.00, 1.00, 1.00],
              [1.00, 1.00, 1.00, 1.00, 1.00, 0.01, 1.00, 1.00, 1.00, 1.00]])
print('%s' % str(X))

print('\nFitting...\n==================')
out = lc.Corex(n_hidden=2, max_iter=1000, verbose=True)
out.fit(X)

print('\nClusters\n==================')
print(out.clusters())

# print('\nCovariance\n==================')
# print(out.get_covariance())

print('\nTCS\n==================')
print(out.tcs)

print('\nTC\n==================')
print(out.tc)

print('\nPrediction\n==================')
sample = np.array([[1., 1., 1., 1., 1., 0., 1., 1., 1., 1.]])
p, log_z = out.transform(sample, details=True)
print(p)

print('\nEND\n==================')
