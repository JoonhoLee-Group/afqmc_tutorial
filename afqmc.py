from pyscf import gto, tools, lo, scf
import numpy as np
import scipy
import logging

logger = logging.getLogger(__name__)


def read_fcidump(fname, norb):
    """

    :param fname: electron integrals dumped by pyscf
    :param norb: number of orbitals
    :return: electron integrals for 2nd quantization
    """
    v2e = np.zeros((norb, norb, norb, norb))
    h1e = np.zeros((norb, norb))

    with open(fname, "r") as f:
        lines = f.readlines()
        for line, info in enumerate(lines):
            if line < 4:
                continue
            line_content = info.split()
            integral = float(line_content[0])
            p, q, r, s = [int(i_index) for i_index in line_content[1:5]]
            if r != 0:
                v2e[p-1, q-1, r-1, s-1] = integral
                v2e[q-1, p-1, r-1, s-1] = integral
                v2e[p-1, q-1, s-1, r-1] = integral
                v2e[q-1, p-1, s-1, r-1] = integral
            elif p != 0:
                h1e[p-1, q-1] = integral
                h1e[q-1, p-1] = integral
            else:
                nuc = integral

    return h1e, v2e, nuc


class afqmc_main(object):
    def __init__(self, mol, dt, total_t, nwalkers=100, taylor_order=6):
        self.mol = mol
        self.nwalkers = nwalkers
        self.total_t = total_t
        self.dt = dt
        self.nfields = None
        self.trial = None
        self.walker_tensor = None
        self.walker_weight = None
        self.precomputed_l_tensor = None
        self.taylor_order = taylor_order

    def hamiltonian_integral(self):
        # 1e & 2e integrals
        s_mat = self.mol.intor('int1e_ovlp')
        ao_coeff = lo.orth.lowdin(s_mat)
        norb = ao_coeff.shape[0]
        tools.fcidump.from_mo(self.mol, 'fcidump_lowdin.out', ao_coeff)
        h1e, v2e, nuc = read_fcidump('fcidump_lowdin.out', norb)

        # Cholesky decomposition
        v2e = np.moveaxis(v2e, 1, 2)
        v2e = v2e.reshape((norb**2, -1))
        l_tensor = np.linalg.cholesky(v2e)
        l_tensor = l_tensor.T
        l_tensor = l_tensor.reshape(l_tensor.shape[0], norb, norb)
        self.nfields = l_tensor.shape[0]
        return h1e, nuc, l_tensor

    def get_trial(self):
        # RHF
        mf = scf.RHF(self.mol)
        mf.kernel()
        s_mat = self.mol.intor('int1e_ovlp')
        Xinv = np.linalg.inv(lo.orth.lowdin(s_mat))
        self.trial = Xinv.dot(mf.mo_coeff[:,:self.mol.nelec[0]])

    def init_walker(self):
        self.get_trial()
        temp = self.trial.copy()
        self.walker_tensor = np.array([temp] * self.nwalkers, dtype=np.complex128)
        self.walker_weight = [1.] * self.nwalkers

    def simulate_afqmc(self):
        self.init_walker()
        h1e, nuc, l_tensor = self.hamiltonian_integral()
        self.precomputed_l_tensor = np.einsum("pr, npq->nrq", self.trial.conj(), l_tensor)
        time = 0
        energy_list = []
        time_list = []
        while time < self.total_t:
            print(f"time: {time}")
            time_list.append(time)
            ovlp = self.get_overlap()
            ovlp_inv = np.linalg.inv(ovlp)
            theta = np.einsum("zqp, zpr->zqr", self.walker_tensor, ovlp_inv)
            trace_l_theta = self.force_bias(theta)
            xbar = -1j * np.sqrt(self.dt) * trace_l_theta
            local_e = self.local_energy(theta, h1e, nuc, trace_l_theta)
            energy = sum([self.walker_weight[i]*local_e[i] for i in range(len(local_e))])
            energy = energy / sum(self.walker_weight)
            energy_list.append(energy)
            self.propagate(h1e, nuc, xbar, ovlp, l_tensor)
            print(self.walker_weight)
            if int(time / self.dt) == 50:
                self.reorthogonal()
            time = time + self.dt
            print(f"energy: {energy_list[-1]}")
        return time_list, energy_list

    def get_overlap(self):
        ovlp = np.einsum('pr, zpq->zrq', self.trial.conj(), self.walker_tensor)
        return ovlp

    def force_bias(self, theta):
        bias = np.einsum('zqr, nrq->zn', theta, self.precomputed_l_tensor)
        return bias

    def propagate(self, h1e, nuc, xbar, ovlp, l_tensor):
        # 1-body propagator propagation
        one_body_op_power = scipy.linalg.expm(-self.dt * h1e)
        self.walker_tensor = np.einsum('pq, zqr->zpr', one_body_op_power, self.walker_tensor)

        # 2-body propagator propagation
        xi = np.random.normal(0.0, 1.0, self.nfields * self.nwalkers)
        xi = xi.reshape(self.nwalkers, self.nfields)
        two_body_op_power = 1j * np.sqrt(self.dt) * np.einsum('zn, npq->zpq', xi-xbar, l_tensor)
        Temp = self.walker_tensor.copy()
        for order_i in range(1, 1+self.taylor_order):
            Temp = np.einsum('zpq, zqr->zpr', two_body_op_power, Temp) / order_i
            self.walker_tensor += Temp

        ovlp_new = self.get_overlap()
        ovlp_ratio = np.linalg.det(ovlp_new) / np.linalg.det(ovlp)
        phase_factor = np.array([max(0, np.cos(np.angle(i_ratio))) for i_ratio in ovlp_ratio])
        i_factor = np.exp(np.einsum("zn, zn->z", xi, xbar)-0.5*np.einsum("zn, zn->z", xbar, xbar))
        importance_func = np.abs(ovlp_ratio * i_factor) * phase_factor
        self.walker_weight = self.walker_weight * importance_func

    def local_energy(self, theta, h1e, nuc, trace_l_theta):
        trace_l_theta2 = trace_l_theta ** 2
        trace_l_theta2 = 2 * np.einsum("zn->z", trace_l_theta2)
        trace_l_theta_l_theta = np.einsum('nrq, zqp, nps, zsr->z',
                                          self.precomputed_l_tensor, theta,
                                          self.precomputed_l_tensor, theta)
        local_e2 = trace_l_theta2 - trace_l_theta_l_theta
        local_e1 = np.sum(h1e) # *2
        local_e = [ie + local_e1 + nuc for ie in local_e2]
        return local_e

    def reorthogonal(self):
        ortho_walkers = np.zeros_like(self.walker_tensor)
        for idx in range(self.walker_tensor.shape[0]):
            ortho_walkers[idx] = np.linalg.qr(self.walker_tensor[idx])[0]
        self.walker_tensor = ortho_walkers
