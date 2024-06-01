import numpy as np
import my_modules.direct_srv as ds
from my_modules.helpers import calculate_marginals, generate_uhoh

def give_rick_srvs(p,q):
        ps = [q,p]
        kxs = tuple(len(p) for p in ps)  # number of values for each input variable
        ks = max(kxs)  # number of states for the SRV to be constructed
        srv_ps_shape = kxs + (ks,)  # for convenience, used in multiple places below
        on_violation = 'continue'
        ls,__ = ds.compute_direct_srv_assuming_independence(ps, ks, method='joint', return_also_violation=True, 
                                                                                on_violation=on_violation, verbose=1)
        ws = [1*ls[:,:,0].reshape(9,), 1*ls[:,:,1].reshape(9,), 1*ls[:,:,2].reshape(9,)]
        return ws, ls

def give_rick_bs(srv, ps,qs):
        p_states = len(ps)
        q_states = len(qs)
        marginals = calculate_marginals(srv, ps, qs).reshape(6,3)
        print("\nFinal marginals:", marginals,'\n')
        any_mismatch = np.any(abs(marginals - marginals[0])>1e-2, axis=1)
        generate_uhoh(not np.any(any_mismatch))
        target_marginal = marginals[0]
        bs = np.array([target*np.ones(p_states+q_states) for target in target_marginal])
        return bs, target_marginal

def give_rick_ws(srvs, bs, pinv, pinv_nm):
        ws = []
        srv_states = len(srvs)
        for i in range(0, srv_states):
                t = srvs[i] - np.dot(pinv, bs[i])
                w = np.dot(pinv_nm, t)
                ws.append(w)
        return ws