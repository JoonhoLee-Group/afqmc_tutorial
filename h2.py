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

afqmc_job = afqmc_main(mol, dt=0.01, total_t=5, nwalkers=1000)
time, energy = afqmc_job.simulate_afqmc()

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
axes.plot(time, energy, 'o-', label='afqmc')
axes.plot(time, [hf_energy] * len(time), '--',label='Hartree-Fock')
axes.plot(time, [fci_energy] * len(time), '--',label='FCI')
axes.legend()
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

axes.plot(t, energy)
# axes.set_xlim(0, 1)
plt.show()
