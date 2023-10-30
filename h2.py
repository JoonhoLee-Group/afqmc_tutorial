from pyscf import gto, scf, fci
from afqmc import afqmc_main

r0 = 1.6
natoms = 2
mol = gto.M(
    atom=[("H", i*r0, 0, 0) for i in range(natoms)],
    basis='sto-6g',
    unit='Bohr',
    verbose=5
)

total_t = 5
nwalker = 1000
afqmc_job = afqmc_main(mol, dt=0.01, total_t=total_t, nwalkers=nwalker, scheme='hybrid energy')
time1, energy1 = afqmc_job.simulate_afqmc()

import h5py
f = h5py.File("energy.h5", "w")
f["t"] = time1
f["energy"] = energy1
f.close()


# afqmc_job = afqmc_main(mol, dt=0.01, total_t=total_t, nwalkers=1000, scheme='local energy')
# time2, energy2 = afqmc_job.simulate_afqmc()

# HF energy
# FCI energy
mf = scf.RHF(mol)
hf_energy = mf.kernel()
cisolver = fci.FCI(mf)
fci_energy = cisolver.kernel()[0]
print(fci_energy)
import matplotlib.pyplot as plt
fig, axes = plt.subplots()

# time = np.arange(0, 5, 0.)
axes.plot(time1, energy1, '--', label='afqmc (my code)')
# axes.plot(time1, energy2, '-', label='afmqc, local energy')
axes.plot(time1, [hf_energy] * len(time1), '--')
axes.plot(time1, [fci_energy] * len(time1), '--')
axes.set_ylabel("ground state energy")
axes.set_xlabel("imaginary time")
# plt.show()


import numpy
import pandas as pd
df = pd.read_csv("output.csv")
print(df["Block"])
t = 0.01 * numpy.array(df["Block"].values)
energy = df["ETotal"].values

import matplotlib.pyplot as plt

axes.plot(t, energy, '--', label='ipie')

axes.set_xlim(0, total_t)
axes.legend()
plt.show()
