import numpy as np
import warnings
from my_modules.helpers import *
import my_modules.variables as vb
np.set_printoptions(suppress=True)

#best case
# ps = np.array([0.2791386,  0.28731193, 0.43354946])
# qs = np.array([0.39314196, 0.3746973,  0.23216074])

#ahab's great whale
# qs = np.array([0.81134934, 0.15412982, 0.03452084])
# ps = np.array([0.14296979, 0.08690643, 0.77012377]) 

#control
# ps,qs = np.array([0.4, 0.32, 0.28]), np.array([0.5, 0.35, 0.15])

# ps,qs = np.zeros(3), np.zeros(3)
# for i in range(0,2):
#     ps[i] = np.random.uniform(0.2,0.6-ps[(i-1)%3])
#     qs[i] = np.random.uniform(0.2,0.6-qs[(i-1)%3])
# ps[2],qs[2] = 1-ps[0]-ps[1], 1-qs[0]-qs[1]
ps = np.random.dirichlet([1,1,1])
qs = np.random.dirichlet([1,1,1])

#### derived variables #####
srv_states = 3
p_states = ps.shape[0]
q_states = qs.shape[0]
target_marginal = ps[::-1]
bs = np.array([target*np.ones(p_states+q_states) for target in target_marginal])
#### ----------------- #####

#### analytically calculated ####
master_mat = np.concatenate((np.kron(np.eye(p_states), ps), np.kron(qs, np.eye(q_states))), axis = 0)
# pinv = np.linalg.pinv(master_mat)
# null_mat = np.eye(9) - np.dot(pinv, master_mat)
# pinv_nm = null_mat
#### ---------------------- ####

#### loop variables ####
prev_srvs = []
prev_sum = 0
srvs = []
# ws= vb.ws
# ws = [1*np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]), 1*np.array([0, 0, 1, 1, 0, 0, 0, 1, 0])]

# fixing the free parameters:
free_params = [[5,6,7,8], [5,6,7,8]]
put_one_in = [[8],[7]]
norm = [1,   1]
########################

for i in range(0, srv_states-1):
    #we find the solution assuming the solution['free_params'] = 0 and then solution['put_one_in'] = 1:
    

    if i==0:
        if len(put_one_in[i])>1:
            b = bs[i] - np.sum(master_mat[:, put_one_in[i]], axis=1).reshape(6)
        else:
            b = bs[i] - master_mat[:, put_one_in[i]].reshape(6)
        A = np.copy(master_mat)
        A[:, free_params[i]] = 0
        pinvA = np.linalg.pinv(A)
        solution = np.dot(pinvA, b)
        solution[put_one_in[i]] = 1
        solution = normalize(solution, norm[i], 0, p_states, q_states, search = 1)

    if i==1:
        b = bs[i]
        A = np.eye(9) - np.diag(prev_srvs[0])
        A = np.dot(master_mat, A)
        pinv = np.linalg.pinv(A)
        s1 = np.dot(pinv, b)
        s1 = normalize(s1, norm[i], 0, p_states, q_states, search = 1)
        solution = s1*(1-prev_srvs[0])
    

    #The above step only works when the target_marginal is chosen wisely:
    # assert np.all(abs(np.dot(master_mat, solution)-bs[i])<1e-7)
    

    #keep it between 0<x<1-prev_sum
    # solution = normalize(solution, norm[i], prev_sum, p_states, q_states, search = i)
    # solution = normalize(solution, norm[i], 0, p_states, q_states, search = 1)

    prev_sum += solution
    prev_srvs.append(solution)

    srvs.append(solution.reshape(q_states,p_states))

#calculate the last element of the srvs##
A, B, C = impurify(srvs[0], srvs[1], (1-prev_sum).reshape(3,3), do_it = 0)
srvs.append(C)
## ----------------------------------##

#### finallyyyy ####
srv = np.concatenate([matrix[..., np.newaxis] for matrix in srvs], axis=2)
srv[abs(srv)< 1e-13] = 0
##----------------#

marginals = calculate_marginals(srv, ps, qs).reshape(6,3)
print("\nFinal marginals:", marginals,'\n')
any_mismatch = np.any(abs(marginals - marginals[0])>1e-2, axis=1)
generate_uhoh(not np.any(any_mismatch))

mutual_info, entropy = calculate_mutual_info(srv, ps,qs, srv_states)
print("\nMutual info and entropy:", mutual_info, entropy, "\n")

print("\nSrv:\n",srv, "\n ps, and qs, end_marginal:", repr(ps), ", ", repr(qs), ", ", repr(marginals[0]))

# np.save('srv.npy', srv)