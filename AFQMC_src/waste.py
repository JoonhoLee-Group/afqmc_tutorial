def propagate(self, x, xbar, hcore_shift, ltensor_shift):
        '''
        Propagate the walker states and weights in imaginary time.
        input:
            x: the sampled auxiliary field, 2d array with size (nwalkers, naux)
            xbar: the force bias, 2d array with size (nwalkers, naux)
            hcore_shift: the modified H_1 after mean field subtraction, 2d array with size (nao, nao)
            ltensor_shift: the shifted CD tensor of the 2-electron integrals, 3d array with size (naux, nao, nao)
        '''
        log = logger.Logger(self.stdout, self.verbose)
        stts = time.time()
        #Calculate the overlap function to update walker weights
        S = self.overlap()
        #Propagate the walker states
        for a in range(self.nwalkers):
            #Build the propagator B(x-xbar)
            B_exponent = -self.dt * hcore_shift + 1j * np.sqrt(self.dt) * np.einsum('p,pij->ij', x[a,:] - xbar[a,:], ltensor_shift) # (nao, nao) matrix
            B = scipy.linalg.expm(B_exponent)
            self.walker_states[a,:,:] = B @ self.walker_states[a,:,:]
        log.debug('walker states = %s',self.walker_states)
        S_upd = self.overlap()
        ends = time.time()
        #Propagate the walker weights
        sttw = time.time()
        for a in range(self.nwalkers):
            #Calculate the importance function
            if self.walker_weights[a] > 0. :
                overlap_ratio = (np.linalg.det(S_upd[a,:,:]) / np.linalg.det(S[a,:,:]))**2
                log.debug("overlap ratio = %s", overlap_ratio)
                expn = np.exp(np.einsum('i,i', x[a,:], xbar[a,:])- 0.5 * np.einsum('i,i', xbar[a,:], xbar[a,:]))
                dtheta = np.angle(overlap_ratio)
                self.walker_weights[a] *= (np.abs(overlap_ratio*expn) *  max(0, np.cos(dtheta)))
        endw = time.time()
        return ends - stts, endw - sttw

def get_pre_Ltensor_mf(self, ltensor_mf):
    '''Get the mean field shifted fprecomputed L tensor from the trial wave function'''
    assert self.trial_wf is not None and self.ltensor is not None
    pre_Ltensor_mf = np.einsum('ij,aik->ajk', self.trial_wf.conj(), ltensor_mf)
    return pre_Ltensor_mf