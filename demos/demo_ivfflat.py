# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import faiss


def acc_cal(nq, I, I_ref, k):
    print("calculate accuracy ...")
    rec = np.zeros(nq)
    for i in range(nq):
        rec[i] = 1.0 * len(set(I[i][:]).intersection(set(I_ref[i][:]))) / k
    acc = sum(rec) * 1. / nq * 100
    return acc


np.set_printoptions(suppress=True)

k = 4                            # number of nearest neighbors
d = 64                           # dimension
nb = 100000                      # database size
nq = 10000                       # nb of queries
np.random.seed(1234)             # make reproducible
xb = np.random.random((nb, d)).astype('float32')
xb[:, 0] += np.arange(nb) / 1000.
xq = np.random.random((nq, d)).astype('float32')
xq[:, 0] += np.arange(nq) / 1000.

# flat
index_flat = faiss.IndexFlatL2(d)
index_flat.add(xb)
D_flat, I_flat = index_flat.search(xq, k)


nlist = 100
quantizer = faiss.IndexFlatL2(d)  # the other index
index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
# here we specify METRIC_L2, by default it performs inner-product search

assert not index.is_trained
index.train(xb)
assert index.is_trained

index.add(xb)
index.nprobe = 5
D, I = index.search(xq, k)
print(acc_cal(nq, I, I_flat, k))


