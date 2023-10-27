from pyscf import gto, scf
from afqmc import afqmc

mol = gto.M(
    atom=[("H", 0, 0, 0), ("H", 1.6, 0, 0)],
    basis='sto-6g',
    unit='Bohr',
    verbose=4
)

nwalkers = 1
total_time = 0.1
dt = 0.01
afqmc = afqmc(mol, nwalkers, total_time, dt, verbose=5)
t_lis, computed_e = afqmc.kernel()