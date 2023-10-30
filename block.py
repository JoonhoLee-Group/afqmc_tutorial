import h5py

f = h5py.File("energy.h5", "r")
E = f["energy"][()]
f.close()
E = E[5000:]

import pyblock
reblock_data = pyblock.blocking.reblock(E)
for reblock_iter in reblock_data:
    print(reblock_iter)

opt = pyblock.blocking.find_optimal_block(len(E), reblock_data)
print(opt)
print(reblock_data[opt[0]])
