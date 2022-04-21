from sklearn import preprocessing

import numpy as np
import time
import faiss

# params
n_nb = 4
n_dim = 512
n_db = 1000 * 1000
idx_probe = 5647

X = np.random.random((n_db, n_dim)).astype('float32')
X = preprocessing.normalize(X)

# x = X[20][:]
# print('|x| --> %f' % np.linalg.norm(x))

probe = np.loadtxt('/home/firefly/mnt_manu/tmp/my_Recogn.txt')
# print('|probe| --> %f' % np.linalg.norm(probe))

X[idx_probe][:] = probe

top_1 = [-1, -1]  # [idx, sim]
ts = time.time()
for i in range(n_db):
    gallery = X[i]
    sim = np.dot(gallery, probe.T)
    # print(f'idx {i} sim --> {sim}\n')
    top_1 = [i, sim] if sim > top_1[1] else top_1
te = time.time()
print(f'naive top_1 --> {top_1[0]} in {(te - ts) * 1000}ms\n')

# flat
probe = probe[np.newaxis, :]
index_flat = faiss.IndexFlatL2(n_dim)
index_flat.add(X)
ts = time.time()
D_flat, I_flat = index_flat.search(probe, n_nb)
te = time.time()
print(f'flat top_1 --> {I_flat[0][0]} in {(te - ts) * 1000}ms\n')

nlist = 100
quantizer = faiss.IndexFlatL2(n_dim)  # the other index
index = faiss.IndexIVFFlat(quantizer, n_dim, nlist)
assert not index.is_trained
index.train(X)
assert index.is_trained
index.add(X)
index.nprobe = 5
ts = time.time()
D_ivfflat, I_ivfflat = index.search(probe, n_nb)
te = time.time()
print(f'ivfflat top_1 --> {I_ivfflat[0][0]} in {(te - ts) * 1000}ms\n')






