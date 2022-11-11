from pyscf import gto
from afqmc import afqmc_main

r0 = 1.6
natoms = 2
mol = gto.M(
    atom=[("H", i*r0, 0, 0) for i in range(natoms)],
    basis='sto-6g',
    unit='Bohr',
    verbose=5
)

afqmc_job = afqmc_main(mol, dt=0.01, total_t=100, nwalkers=1000)
time, energy = afqmc_job.simulate_afqmc()
