from pyscf import gto, scf, fci

atom = gto.M(
    atom=[("H", 1.6 * i, 0, 0) for i in range(0, 2)],
    basis="sto-6g",
    verbose=4,
    unit="Bohr",
)
mf = scf.UHF(atom)
mf.chkfile = "scf.chk"
mf.kernel()
import os
os.syste("tools/pyscf/pyscf_to_ipie.py -i scf.chk -j input.json")
