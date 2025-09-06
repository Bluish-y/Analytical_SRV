import numpy as np
import warnings
from my_modules.helpers import *
import my_modules.variables as vb
import my_modules.direct_srv as ds
import my_modules.analyticalsrv_func as asrv
np.set_printoptions(suppress=True)

#best case
ps = np.array([0.2791386,  0.28731193, 0.43354946])
qs = np.array([0.39314196, 0.3746973,  0.23216074])

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
# ps = np.random.dirichlet([1,1,1])
# qs = np.random.dirichlet([1,1,1])

#### derived variables #####
srv_states = 3
p_states = ps.shape[0]
q_states = qs.shape[0]
#### ----------------- #####

#### analytically calculated ####
master_mat = np.concatenate((np.kron(np.eye(p_states), ps), np.kron(qs, np.eye(q_states))), axis = 0)
pinv = np.linalg.pinv(master_mat)
null_mat = np.eye(9) - np.dot(pinv, master_mat)
pinv_nm = null_mat
#### ---------------------- ####

#### loop variables ####
prev_srvs = []
prev_sum = 0
srvs = []
<<<<<<< HEAD
# r_srvs, r_srv= vb.give_rick_srvs(ps,qs)
# bs, target_marginal = vb.give_rick_bs(r_srv,ps,qs)
# ws = vb.give_rick_ws(r_srvs, bs, pinv, pinv_nm)
# print_info(r_srv, qs,ps)

target_marginal = ps[::-1]
bs = np.array([target*np.ones(p_states+q_states) for target in target_marginal])
ws = [np.zeros(p_states*q_states)]*3
i_srv = asrv.generate_simple_srv(ps,qs,bs,ws,pinv,null_mat,master_mat)
print_info(i_srv, qs, ps)
=======
r_srvs, r_srv= vb.give_rick_srvs(ps,qs)
bs, target_marginal = vb.give_rick_bs(r_srv,ps,qs)
ws = vb.give_rick_ws(r_srvs, bs, pinv, pinv_nm)
print_info(r_srv, qs,ps)

# target_marginal = ps[::-1]
# bs = np.array([target*np.ones(p_states+q_states) for target in target_marginal])
# ws = [np.zeros(p_states*q_states)]*3
# i_srv = asrv.generate_simple_srv(ps,qs,bs,ws,pinv,null_mat,master_mat)
# print_info(i_srv, qs, ps)
>>>>>>> 9b40086 (Convex optimization to improve synergy values)

# ws = [1*np.array([0.5, 0, 0.2, 0, 0.5, 0, 0, 0, 1]), 1*np.array([0, 0, 0.7, 0.7, 0, 0, 0, 1, 0])]

########################


for i in range(0, srv_states-1):
    #we find the solution assuming the solution['free_params'] = 0 and then solution['put_one_in'] = 1:
    b,w = bs[i], ws[i]
    solution = np.dot(pinv, b) + np.dot(null_mat, w)
    solution[abs(solution)<1e-8] = 0
    solution[abs(1-solution)<1e-8] = 1
    

    #The above step only works when the target_marginal is chosen wisely:
    assert np.all(abs(np.dot(master_mat, solution)-bs[i])<1e-7)
    
<<<<<<< HEAD
    #grad calculation
    grad_b, grad_w = grad_info(solution, ps,qs, target_marginal, w, pinv, null_mat)

    step = 0.05
    target_marginal += step*grad_b
    b = target_marginal[i]*np.ones(p_states+q_states)
    w += step*grad_w

    solution = np.dot(pinv, b) + np.dot(null_mat, w)
=======
>>>>>>> 9b40086 (Convex optimization to improve synergy values)

    prev_sum += solution
    prev_srvs.append(solution)
    srvs.append(solution.reshape(q_states,p_states))
prev_srvs.append(1-prev_sum)

<<<<<<< HEAD
=======

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
print_info(srv, ps, qs)

s3 = prev_srvs[2]
prev_sum = 0
srvs = []
for i in range(0,srv_states-1):
    solution = prev_srvs[i]
    #grad calculation
    grad_b, grad_w = grad_info(solution, ps,qs, target_marginal, w, pinv, null_mat, s3)

    step = 0.13
    target_marginal += step*grad_b
    b = target_marginal[i]*np.ones(p_states+q_states)
    w += step*grad_w

    solution = np.dot(pinv, b) + np.dot(null_mat, w)
    prev_sum += solution
    srvs.append(solution.reshape(q_states,p_states))


>>>>>>> 9b40086 (Convex optimization to improve synergy values)
for k in range(0, srv_states):
    target_marginal[k] = np.dot(master_mat, prev_srvs[k])[0]

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
print_info(srv, ps, qs)

<<<<<<< HEAD
=======

>>>>>>> 9b40086 (Convex optimization to improve synergy values)
# np.save('srv.npy', srv)