'''
    A simple program calculating the ground state energy of molecules using Auxiliary Field Quantum Monte Carlo (AFQMC). The program uses a simple eigenvalue decomposition to calculate the L tensor, instead of the modified Cholesky Decomposition (mCD) algorithm. 
    Hybrid formalism is used to update the walker weights. 
    Energy bound and population control are not implemented.  
'''

import time, sys, os, h5py
from pyscf import scf, lo, ao2mo
from pyscf.lib import logger
import numpy as np
import scipy
import itertools

class TemplateWalkers(object):
    def __init__(self, weights, states):
        self.phi = states.copy()
        self.weight = weights.copy()
        self.nwalkers = self.weight.size
        self.unscaled_weight = self.weight.copy()
        #print("self.phi.shape = {}".format(self.phi.shape))
        self.buff_size = self.phi.size + self.weight.size
        self.buff_names = ["phi", "weight"]
        self.walker_buffer = np.zeros(self.buff_size, dtype=np.complex128)
    def copy_to(self,afqmc):
        afqmc.walker_states = self.phi.copy()
        afqmc.walker_weights = self.weight.copy()

class afqmc(object):
    def __init__(self, mol, nwalkers, total_time, dt, trial = 'HF', taylor_order = 6, verbose = 4):
        self.mol = mol
        self.nwalkers = nwalkers
        self.walker_states = None
        self.walker_weights = None
        self.total_time = total_time
        self.dt = dt
        self.ltensor = None
        self.pre_Ltensor = None
        self.naux = None
        self.trial = trial
        self.trial_wf = None
        self.orth = None
        self.hcore = None
        self.eri = None
        self.stdout = mol.stdout
        self.verbose = verbose
        self.hybrid_energy = None
        self.mf_shift = None
        # For debugging use
        self.taylor_order = taylor_order


    def set_trial_wavefunction(self):
        if self.trial == 'HF':
            mf = scf.RHF(self.mol)
            mf.kernel()
            #lowdin orthogonalization
            ovlp_mat = self.mol.intor('int1e_ovlp')
            X = lo.orth.lowdin(ovlp_mat)
            self.orth = X
            orth_coeff = np.linalg.inv(X) @ mf.mo_coeff
            #What if we use non-orthogonal basis? 
            self.trial_wf = orth_coeff[:, :self.mol.nelec[0]]
        elif self.trial == 'CCSD':
            raise NotImplementedError
        elif self.trial == 'DFT':
            raise NotImplementedError
        
    def cholesky(self, eri, nao):
        '''
        eri: 4d array, (nao, nao, nao, nao), chemist notation, i.e. (ij|kl) = <ik|jl>
        uses direct diagonalization if the eri is not positive definite, using mCD is a better choice and this is to be implemented 
        '''
        log = logger.Logger(self.stdout, self.verbose)
        Vij = eri.reshape((nao**2, -1))
        log.debug("Vij = %s", Vij)
        try:
            L = scipy.linalg.cholesky(Vij, lower = True)
            ltensor = L.reshape((-1, nao, nao))
        except scipy.linalg.LinAlgError:
            # it happens, since the eri is not numerically positive definite (but Why?)
            e, v = scipy.linalg.eigh(Vij)
            log.debug("eigenvalues: %s", e)
            log.debug("eigenvectors: %s", v)
            idx = e > 1e-12
            L = (v[:,idx] * np.sqrt(e[idx]))
            L = np.flip(L, 1)
            ltensor = L.T.reshape((-1, nao, nao))
            #log.debug("ltensor = %s", ltensor)
        eri_reconstruct = np.einsum('apq, ars->pqrs', ltensor, ltensor)
        log.debug("eri difference = %s", np.linalg.norm(eri_reconstruct - eri))
        self.naux = ltensor.shape[0]
        self.ltensor = ltensor
        return ltensor

    def modify_hamiltonian(self, hcore, eri):
        '''
        calculate the modified hamiltonian and the shifted ltensor after mean field subtraction & 1 body term separation from the 2-body interaction term
        '''
        # 1 body term separation
        hcore_mod = np.zeros_like(hcore)
        for p, q in itertools.product(range(hcore.shape[0]),repeat=2):
            hcore_mod[p,q] = hcore[p,q] - 0.5 * np.trace(eri[p,:,:,q])
        return hcore_mod
    
    def mean_field_shift(self, hcore_mod, ltensor, nuclear_repulsion):
        log = logger.Logger(self.stdout, self.verbose)
        vbar = 1j * np.einsum("npp->n", ltensor)
        self.mf_shift = vbar
        # mean field subtraction
        hcore_mf = hcore_mod - 1j * np.einsum("p,pij->ij", vbar, ltensor)
        #ltensor_mf = ltensor - (-1j) * np.array([vbar[i]* np.eye(ltensor.shape[1]) for i in range(self.naux)])
        nuclear_repulsion_mf = nuclear_repulsion + 0.5 * np.einsum('i,i', vbar, vbar)
        return hcore_mf, nuclear_repulsion_mf
    
    def set_pre_Ltensor(self):
        '''Get the precomputed L tensor from the trial wave function'''
        assert self.trial_wf is not None and self.ltensor is not None
        self.pre_Ltensor = np.einsum('ij,aik->ajk', self.trial_wf.conj(), self.ltensor)
        return
    
    def overlap(self):
        '''
        Get the overlap matrices of the trial wavefunction with the walker wave functions
        returns: overlap, 3d array with size (nwalkers, nao, nao)
        '''
        return np.einsum('ij,aik->ajk', self.trial_wf.conj(), self.walker_states)

    def get_hamiltonian(self):
        '''Get the hamiltonian matrix elements with respect to the orthogonalized basis'''
        assert self.hcore is None and self.eri is None 
        hcore = self.mol.intor_symmetric('int1e_kin') + self.mol.intor_symmetric('int1e_nuc')
        hcore_orth = self.orth.conj().T @ hcore @ self.orth
        self.hcore = hcore_orth
        eri = self.mol.ao2mo(self.orth)
        #Not imposing symmetry here
        eri = ao2mo.restore(1, np.asarray(eri), self.mol.nao)
        self.eri = eri
        nuclear_repulsion = self.mol.energy_nuc()
        return hcore_orth, eri, nuclear_repulsion

    def get_gf(self):
        ovlp = self.overlap()
        overlap_inv = np.zeros_like(self.overlap())
        for a in range(self.nwalkers):
            overlap_inv[a,:,:] = np.linalg.inv(self.overlap()[a,:,:])
        gf = np.einsum('ajk, akl, il->aij', self.walker_states, overlap_inv, self.trial_wf.conj())
        return gf

    def get_theta(self):
        '''Get the theta matrix'''
        overlap_inv = np.linalg.inv(self.overlap())
        return np.einsum('aij, ajk-> aik', self.walker_states, overlap_inv)

    def propagate_taylor_ecap(self, x, xbar, hcore_shift, ltensor):
        # calculate the overlap for updating weights
        S = self.overlap()
        det_S = np.linalg.det(S)
        log = logger.Logger(self.stdout, self.verbose)
        stts = time.time()
        #Propagate the walker states
        # 1-body propagator propagation
        log.debug('h1e = %s', hcore_shift)
        hcore_exp = scipy.linalg.expm(-self.dt/2 * hcore_shift)
        self.walker_states = np.einsum('pq, zqr->zpr', hcore_exp, self.walker_states)
        # 2-body propagator propagation
        twobody_exp = 1j * np.sqrt(self.dt) * np.einsum('zn, npq->zpq', x-xbar, ltensor)
        log.debug('ltensor = %s', ltensor)
        log.debug('two body exponent = %s', twobody_exp)
        Temp = self.walker_states.copy()
        for order_i in range(1, 1+self.taylor_order):
            Temp = np.einsum('zpq, zqr->zpr', twobody_exp, Temp) / order_i
            self.walker_states += Temp
        # 1-body propagator propagation again
        hcore_exp = scipy.linalg.expm(-self.dt/2 * hcore_shift)
        self.walker_states = np.einsum('pq, zqr->zpr', hcore_exp, self.walker_states)
        log.debug("walker_states: %s", self.walker_states)

        #(self.walker_states, Rup) = np.linalg.qr(self.walker_states)
        #detR = np.linalg.det(Rup)

        S_upd = self.overlap()
        log.debug("updated overlap: %s",  S_upd)
        ends = time.time()
        sttw = time.time()
        #Propagate the walker weights
        det_Supd = np.linalg.det(S_upd) #* detR
        overlap_ratio = (det_Supd / det_S)**2
        ebound = (2.0 / self.dt) ** 0.5
        cfb = np.einsum("zn, zn->z", x, xbar)-0.5*np.einsum("zn, zn->z", xbar, xbar)
        cmf = -np.sqrt(self.dt)*np.einsum('zn, n->z', x-xbar, self.mf_shift)
        log.debug("xbar = %s", xbar)
        log.debug("mf_shift = %s", self.mf_shift)
        hybrid_energy = -(np.log(overlap_ratio) + cfb + cmf) / self.dt
        hybrid_energy = np.clip(hybrid_energy.real, a_min=-ebound, a_max=ebound, out=hybrid_energy.real)
        self.hybrid_energy = hybrid_energy if self.hybrid_energy is None else self.hybrid_energy
        importance_func = np.exp(-self.dt * 0.5 * (hybrid_energy + self.hybrid_energy))
        log.debug("hybrid energy = %s", self.hybrid_energy)
        self.hybrid_energy = hybrid_energy
        phase = (-self.dt * self.hybrid_energy-cfb).imag
        phase_factor = np.array([max(0, np.cos(iphase)) for iphase in phase])
        importance_func = np.abs(importance_func) * phase_factor
        self.walker_weights *= importance_func
        #self.walker_weights = np.clip(self.walker_weights * importance_func, a_min = None, a_max = min(100, 0.1 * self.nwalkers), out=self.walker_weights * importance_func)
        endw = time.time()
        return ends - stts, endw - sttw, det_Supd

    def propagate_taylor(self, x, xbar, hcore_shift, ltensor_shift, det_S):
        log = logger.Logger(self.stdout, self.verbose)
        stts = time.time()
        #Propagate the walker states
        # 1-body propagator propagation
        log.debug('h1e = %s', hcore_shift)
        hcore_exp = scipy.linalg.expm(-self.dt/2 * hcore_shift)
        self.walker_states = np.einsum('pq, zqr->zpr', hcore_exp, self.walker_states)
        # 2-body propagator propagation
        twobody_exp = 1j * np.sqrt(self.dt) * np.einsum('zn, npq->zpq', x-xbar, ltensor_shift)
        log.debug('ltensor = %s', ltensor_shift)
        log.debug('two body exponent = %s', twobody_exp)
        Temp = self.walker_states.copy()
        for order_i in range(1, 1+self.taylor_order):
            Temp = np.einsum('zpq, zqr->zpr', twobody_exp, Temp) / order_i
            self.walker_states += Temp
        # 1-body propagator propagation again
        hcore_exp = scipy.linalg.expm(-self.dt/2 * hcore_shift)
        self.walker_states = np.einsum('pq, zqr->zpr', hcore_exp, self.walker_states)
        log.debug("walker_states: %s", self.walker_states)

        # (self.walker_states, Rup) = np.linalg.qr(self.walker_states)
        # detR = np.linalg.det(Rup)

        S_upd = self.overlap()
        ends = time.time()
        sttw = time.time()
        #Propagate the walker weights
        det_Supd = np.linalg.det(S_upd)# * detR
        overlap_ratio = (det_Supd / det_S)**2
        for a in range(self.nwalkers):
            #Calculate the importance function
            if self.walker_weights[a] > 0. :
                log.debug("overlap ratio = %s", overlap_ratio[a])
                expn = np.exp(np.einsum('i,i', x[a,:], xbar[a,:])- 0.5 * np.einsum('i,i', xbar[a,:], xbar[a,:]))
                dtheta = np.angle(overlap_ratio[a])
                self.walker_weights[a] *= (np.abs(overlap_ratio[a]*expn) *  max(0, np.cos(dtheta)))
        endw = time.time()
        return ends - stts, endw - sttw, det_Supd

    def get_ltheta(self, theta):
        '''
        returns:
            ltheta, 4d array with size (nwalkers, naux, nao, nao), ltheta = [L^\gamma \Theta_i]_{pq}
        '''
        return np.einsum('pij, ajk->apik', self.pre_Ltensor, theta)

    def get_local_e(self, hcore, theta, ltheta, nuc):
        '''
        input:
            hcore, 2d array with size (nao, nao), the unmodified hcore 
            
        returns: 
            local_e , 1d array with size (nwalkers)
        '''
        log = logger.Logger(self.stdout, self.verbose)
        tr_ltheta = np.einsum('apii->ap', ltheta)
        tr_ltheta_ltheta = np.einsum('apij, apji->ap', ltheta, ltheta)
        local_e1 = 2 * np.einsum('ij, ajk, ik->a', hcore, theta, self.trial_wf.conj())
        local_e2 = .5 * np.einsum('ap->a', (2 *tr_ltheta)**2 - 2 * tr_ltheta_ltheta)
        local_nuc = np.array([nuc]*self.nwalkers)
        log.debug('local_e1 = %s', local_e1[0])
        log.debug('local_e2 = %s', local_e2[0])
        log.debug('local_nuc = %s', local_nuc[0])
        local_e = local_nuc + local_e1 + local_e2
        return local_e
    
    def get_local_e_gf(self, hcore, ltensor, eri, gf, nuc):
        '''
        input: 
            hcore, 2d array with size (nao, nao), the unmodified hcore 
            ltensor, 3d array with size (naux, nao, nao), the cholesky tensor 
            gf, 3d array with size (nwalkers, nao, nao), the green's function of the walkers
        '''
        log = logger.Logger(self.stdout, self.verbose)
        local_e1 = 2. * np.einsum('ij, aij -> a', hcore, gf)
        local_e2 = 2. * np.einsum('ijkl, aij, akl -> a', eri, gf, gf) - np.einsum('ijkl, ail, akj -> a', eri, gf, gf)
        log.debug('eri - sum_ltensor = %s', np.linalg.norm(eri - np.einsum('pij,pkl->ijkl', ltensor, ltensor)))
        #local_e2 = 2 * np.einsum('pij, pkl, aij, akl -> a', ltensor, ltensor, gf, gf) - 1 * np.einsum('pij, pkl, ail, akj -> a', ltensor, ltensor, gf, gf)
        local_nuc = np.array([nuc]*self.nwalkers)
        log.debug('local_e1 = %s', local_e1[0])
        log.debug('local_e2 = %s', local_e2[0])
        log.debug('local_nuc = %s', local_nuc[0])
        local_e = local_nuc + local_e1 + local_e2
        return local_e

    def force_bias(self, theta):
        '''
        Force bias for each auxiliary field
        input: 
            pre_Ltensor, 3d array with size (nwalkers, nao, nao)
            theta, 3d array with size (nwalkers, nao, nao)
        returns: 
            xbar, 2d array with size (nwalkers,naux)
        '''
        xbar= -np.sqrt(self.dt) * (1j * 2 * np.einsum('pij,aji->ap', self.pre_Ltensor, theta) - self.mf_shift)
        return xbar

    def init_walkers(self):
        assert self.trial_wf is not None
        tmp = self.trial_wf.copy()
        self.walker_states = np.array([tmp] * self.nwalkers, dtype=np.complex128)
        self.walker_weights = np.array([1.] * self.nwalkers)

    def reorthogonal(self):
        ortho_walkers = np.zeros_like(self.walker_states)
        for idx in range(self.walker_states.shape[0]):
            ortho_walkers[idx] = np.linalg.qr(self.walker_states[idx])[0]
        self.walker_states = ortho_walkers

    def kernel(self):
        log = logger.Logger(self.stdout, self.verbose)
        np.random.seed(114514)
        self.set_trial_wavefunction()
        self.init_walkers()
        log.debug("trial wavefunction = %s", self.trial_wf)
        log.debug('walker wavefunction = %s', self.walker_states)
        # get the integrals & matrix elements of the Hamiltonian
        hcore, eri, nuc = self.get_hamiltonian()
        print("eri : ", eri)
        print("nuclear repulsion energy: ", nuc)
        ltensor = self.cholesky(eri, self.mol.nao)
        #TODO: figure out why the ltensor is not stable in sign
        log.debug('ltensor = %s', ltensor)
        #log.debug('eri - sum_ltensor = %s', np.linalg.norm(eri - np.einsum('pij,pkl->ijkl', ltensor, ltensor)))
        # perform the modification of h1
        hcore_mod = self.modify_hamiltonian(hcore, eri)
        # perform the mean field subtraction
        hcore_mf, nuc_mf = self.mean_field_shift(hcore_mod, ltensor, nuc)
        #pre_Ltensor_mf = self.get_pre_Ltensor_mf(ltensor_mf)
        # simulation
        S = self.overlap()
        #(self.walker_states, Rup) = np.linalg.qr(self.walker_states)
        #detR = np.linalg.det(Rup)
        detS = np.linalg.det(S) #* detR
        t = 0
        computed_e = []
        t_lis = []
        time_cost_e = 0
        time_cost_propagate = 0
        time_cost_propagate_state = 0
        time_cost_propagate_weight = 0
        self.set_pre_Ltensor()
        from ipie.walkers.pop_controller import PopController
        pcontrol = PopController(self.nwalkers, self.total_time/self.dt)

        from ipie.qmc.comm import FakeComm
        comm = FakeComm()

        while (t < self.total_time):
            self.walker_weights = self.walker_weights / np.sum(self.walker_weights) * self.nwalkers
            fakewalkers = TemplateWalkers(self.walker_weights, self.walker_states)
            pcontrol.pop_control(fakewalkers, comm)
            fakewalkers.copy_to(self)
            log.info('time = %f， weight max = %f, weight min = %f', t, np.max(self.walker_weights), np.min(self.walker_weights))
            t_lis.append(t)
            # random variable x generation
            x = np.random.normal(0, 1, (self.nwalkers, self.naux))
            #log.debug("x = %s", x)
            # precompute the required tensors
            theta = self.get_theta()
            ltheta = self.get_ltheta(theta)
            log.debug('overlap matrix = %s', self.overlap()[0])
            #gf = self.get_gf()
            #log.debug('green\'s function = %s', gf[0])
            # compute the local energy of each walker
            stt1 = time.time()
            local_e = self.get_local_e(hcore, theta, ltheta, nuc)
            end1 = time.time()
            time_cost_e += end1 - stt1
            log.debug('walker_weights = %s', self.walker_weights)
            log.debug("local energy = %s", local_e)
            #local_e = self.get_local_e_gf(hcore, ltensor, eri, gf, nuc)
            etot = np.sum([self.walker_weights[i]*local_e[i] for i in range(self.nwalkers)]) / np.sum(self.walker_weights)
            computed_e.append(etot)
            # propagate the walkers
            xbar = self.force_bias(theta)
            log.debug('x - xbar = %s', x - xbar)
            #self.propagate(x, np.zeros_like(x), hcore, ltensor)
            stt2 = time.time()
            time_s, time_w, detS = self.propagate_taylor_ecap(x, xbar, hcore_mf, ltensor)
            #reorthogonalization
            if int(t/self.dt) == 5:
                self.reorthogonal()
            end2 = time.time()
            time_cost_propagate += end2 - stt2
            time_cost_propagate_state += time_s
            time_cost_propagate_weight += time_w
            #self.propagate(x, xbar, hcore_mod, ltensor)
            #self.propagate_taylor(x, xbar, hcore_mod, ltensor)
            t += self.dt
        log.info("local energy time = %s", time_cost_e)
        log.info("propagation time = %s", time_cost_propagate)
        log.info("propagation time for state updating= %s", time_cost_propagate_state)
        log.info("propagation time for weight updating= %s", time_cost_propagate_weight)
        return t_lis, computed_e







        



