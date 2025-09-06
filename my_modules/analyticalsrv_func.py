import numpy as np
import cvxpy as cp
import warnings
from my_modules.helpers import *
import my_modules.variables as vb
np.set_printoptions(suppress=True)



def new_analyticalsrv_test(ps,qs, do_it = 0, search_it = 1, do_and_search=0, print_it = 0):
    flag=True
    count=0
    x0 = np.concatenate([np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]), np.array([0, 1, 0, 0, 0, 1, 1, 0, 0]), np.array([0, 0, 1, 1, 0, 0, 0, 1, 0])])
    while(flag):
        if count>100:
            flag=False
        #### derived variables #####
        srv_states = 3
        p_states = ps.shape[0]
        q_states = qs.shape[0]
        target_marginal = qs[::-1]
        bs = np.array([target*np.ones(p_states+q_states) for target in target_marginal])

        #### ----------------- #####

        #### analytically calculated ####
        master_mat = np.concatenate((np.kron(np.eye(p_states), ps), np.kron(qs, np.eye(q_states))), axis = 0)
        cols = p_states*q_states
        master_mat = np.concatenate(((np.pad(master_mat, ((0,0), (0,2*cols)), mode="constant", constant_values = 0)), 
                                    (np.pad(master_mat, ((0,0), (cols,cols)), mode="constant", constant_values = 0)), 
                                    (np.pad(master_mat, ((0,0), (2*cols, 0)), mode="constant", constant_values = 0))), axis=0)
        # master_mat = np.concatenate((master_mat, master_mat, master_mat), axis=1)
        new_mat = np.kron(np.ones(p_states), np.eye(cols))
        master_mat = np.concatenate((master_mat, new_mat), axis = 0)
        
        # targets = np.concatenate(bs, axis=0)
        # targets = np.concatenate((targets, np.ones(new_mat.shape[0])))
        targets = np.ones(master_mat.shape[0])

        # set_vectors = np.array([[0,0,1], [0,0,1], [1,0,0], [1,0,0], [0,1,0]])
        # indices = np.array([[2,11,20], [4,13,22], [5, 14, 23], [7, 16, 25], [8, 17, 26]])
        # for i in range(set_vectors.shape[0]):
        #     for j in range(set_vectors.shape[1]):
        #         targets = np.append(targets, set_vectors[i,j])
        #         add = np.zeros((1,master_mat.shape[1]))
        #         add[0,indices[i,j]] = 1
        #         master_mat = np.concatenate((master_mat, add), axis=0)

        pinv = np.linalg.pinv(master_mat)
        null_mat = np.eye(pinv.shape[0]) - np.dot(pinv, master_mat)
        pinv_nm = null_mat
        #### ---------------------- ####
        # targets = np.ones(master_mat.shape[0])
        # x0 = np.ones(master_mat.shape[1])
        
        # x = x0 - np.dot(pinv, (np.dot(master_mat, x0)-targets))
        # if np.any(x<0) or np.any(x>1):
        #     x0 = x.reshape(3,9).copy()
        #     for j in range(x0.shape[1]):
        #         x0[:,j] -= x0[:,j].min()
        #         x0[:,j] /= x0[:,j].sum() 
        #     x0 = x0.reshape(27,)
        # else:
        #     flag = False
        # count+=1
        # Given numpy arrays:
        # A (m_a x n), b (m_a,), B (m_b x n), C (m_c x n), x0 (n,)
        # If a constraint set is absent, use empty matrices with matching width n.
        flag= False
        x, b = convex_optim(x0, master_mat, np.eye(x0.size), np.eye(x0.size), ps,qs)

    x = x.reshape(3,9)
    srv = x.reshape(3,9).T.reshape(3,3,3)
    return srv

def convex_optim(x0, A, B, C, ps,qs):
    n = x0.shape[0]
    x = cp.Variable(n)

    # Scalars b1, b2, b3
    # b1 = cp.Variable()
    # b2 = cp.Variable()
    # b3 = cp.Variable()
    b = cp.Variable(3)

    # Each block length
    m = 6
    ones = np.ones(m)

    constraints = [
        A @ x == cp.hstack([b[0]*ones, b[1]*ones, b[2]*ones, cp.Constant(np.ones(9))]),
        b[0] + b[1] + b[2] == 1
    ]
    if B.size > 0:
        constraints.append(B @ x >= 0)
    if C.size > 0:
        constraints.append(C @ x <= 1)


    objective = cp.Minimize(objective_func(x,ps,qs, b))
    # objective = cp.Minimize(0.5 * cp.sum_squares(x - x0))
    prob = cp.Problem(objective, constraints)
    prob.solve()

    x_star = x.value
    b_star = b.value
    return x_star, b_star

def objective_func(x, ps, qs, b):
    # x comes in flat (length 27)
    X = cp.reshape(x, (3, 9))

    # Step 2: transpose -> shape (9, 3)
    X = cp.transpose(X)

    # Step 3: reshape into (3, 3, 3)
    X = cp.reshape(X, (3, 3, 3))

    total = 0.0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                total -= X[i, j, k] * ps[i] * qs[j] * cp.log(X[i, j, k] / b[k])
    return total



def analyticalsrv_test(ps,qs, do_it = 0, search_it = 1, do_and_search=0, print_it = 0):
    #### derived variables #####
    srv_states = 3
    p_states = ps.shape[0]
    q_states = qs.shape[0]
    target_marginal = ps[::-1]
    bs = np.array([target*np.ones(p_states+q_states) for target in target_marginal])
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
    # ws= vb.ws
    ws = [1*np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]), 1*np.array([0, 0, 1, 1, 0, 0, 0, 1, 0])]

    #fixing the free parameters:
    free_params = [[1,5,6,7,8], [2,4,6,7,8]]
    put_one_in = [[8],[7]]
    norm = [0.9,   1]
    ########################
    flag=True
    while(flag):
        prev_sum = 0
        srvs = []
        for i in range(0, srv_states-1):
            #we find the solution assuming the solution['free_params'] = 0 and then solution['put_one_in'] = 1:
            if len(put_one_in[i])>1:
                b = bs[i] - np.sum(master_mat[:, put_one_in[i]], axis=1).reshape(6)
            else:
                b = bs[i] - master_mat[:, put_one_in[i]].reshape(6)
            A = np.copy(master_mat)
            A[:, free_params[i]] = 0
            pinvA = np.linalg.pinv(A)
            solution = np.dot(pinvA, b)
            solution[put_one_in[i]] = 1

            #The above step only works when the target_marginal is chosen wisely:
            assert np.all(abs(np.dot(master_mat, solution)-bs[i])<1e-7)
            
            #we try to make it look like the 010100001 case as much as we can:
            w = ws[i]
            w1 = np.dot(pinv_nm, (w - solution))
            solution += 1*np.dot(null_mat,w1)

            #keep it between 0<x<1-prev_sum
            if search_it:
                solution = normalize(solution, norm[i], prev_sum, p_states, q_states, search = i)
            else:
                solution = normalize(solution, norm[i], 0, p_states, q_states, search = 0)

            prev_sum += solution

            srvs.append(solution.reshape(q_states,p_states))

        #calculate the last element of the srvs##
        A, B, C = impurify(srvs[0], srvs[1], (1-prev_sum).reshape(3,3), do_it = do_it)
        srvs.append(C)
        ## ----------------------------------##

        #### finallyyyy ####
        srv = np.concatenate([matrix[..., np.newaxis] for matrix in srvs], axis=2)
        srv[abs(srv)< 1e-13] = 0
        ##----------------#

        marginals = calculate_marginals(srv, ps, qs).reshape(6,3)
        if do_and_search:
            flag = np.any(np.any(abs(marginals - marginals[0])>1e-2, axis=1))
            ls = srv
            ws = [1*ls[:,:,0].reshape(9,), 1*ls[:,:,1].reshape(9,), 1*ls[:,:,2].reshape(9,)]
        else:
            flag = False
    if print_it:
        marginals = calculate_marginals(srv, ps, qs).reshape(6,3)
        print("\nFinal marginals:", marginals,'\n')
        any_mismatch = np.any(abs(marginals - marginals[0])>1e-1, axis=1)
        generate_uhoh(not np.any(any_mismatch))

        mutual_info, entropy = calculate_mutual_info(srv, ps,qs, srv_states)
        print("\nMutual info and entropy:", mutual_info, entropy, "\n")

        print("\nSrv:\n",srv, "\n ps, and qs, end_marginal:", repr(ps), ", ", repr(qs), ", ", repr(marginals[0]))
    return srv

# np.save('srv.npy', srv)

def generate_simple_srv(ps,qs,bs, ws, pinv, null_mat, master_mat):
    srv_states, q_states, p_states = len(bs), len(qs), len(ps)
    prev_sum = 0
    srvs = []
    for i in range(0, srv_states-1):
        #we find the solution assuming the solution['free_params'] = 0 and then solution['put_one_in'] = 1:
        b,w = bs[i], ws[i]
        solution = np.dot(pinv, b) + np.dot(null_mat, w)
        solution[abs(solution)<1e-8] = 0
        solution[abs(1-solution)<1e-8] = 1
        

        #The above step only works when the target_marginal is chosen wisely:
        assert np.all(abs(np.dot(master_mat, solution)-bs[i])<1e-7)
        

        prev_sum += solution
        srvs.append(solution.reshape(q_states,p_states))

    #calculate the last element of the srvs##
    A, B, C = impurify(srvs[0], srvs[1], (1-prev_sum).reshape(3,3), do_it = 0)
    srvs.append(C)
    ## ----------------------------------##

    #### finallyyyy ####
    srv = np.concatenate([matrix[..., np.newaxis] for matrix in srvs], axis=2)
    srv[abs(srv)< 1e-13] = 0
    return srv

def cubic_analyticalsrv_test(ps,qs, do_it = 0, search_it = 1, do_and_search=0, print_it = 0):
    #### derived variables #####
    srv_states = 3
    p_states = ps.shape[0]
    q_states = qs.shape[0]
    target_marginal = ps[::-1]
    bs = np.array([target*np.ones(p_states+q_states) for target in target_marginal])
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
    # ws= vb.ws
    ws = [1*np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]), 1*np.array([0, 0, 1, 1, 0, 0, 0, 1, 0])]

    # fixing the free parameters:
    free_params = [[5,6,7,8], [5,6,7,8]]
    put_one_in = [[8],[7]]
    norm = [0.9,   1]
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

            # w = ws[i]
            # w1 = np.dot(pinv_nm, (w - solution))
            # solution += 1*np.dot(null_mat,w1)
            solution = normalize(solution, norm[i], 0, p_states, q_states, search = 1)

        if i==1:
            b = bs[i]
            A = np.eye(9) - np.diag(prev_srvs[0])
            A = np.dot(master_mat, A)
            pinvA = np.linalg.pinv(A)
            s1 = np.dot(pinvA, b)
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
    A, B, C = impurify(srvs[0], srvs[1], (1-prev_sum).reshape(3,3), do_it = do_it)
    srvs.append(C)
    ## ----------------------------------##

    #### finallyyyy ####
    srv = np.concatenate([matrix[..., np.newaxis] for matrix in srvs], axis=2)
    srv[abs(srv)< 1e-13] = 0
    ##----------------#

    if print_it:
        marginals = calculate_marginals(srv, ps, qs).reshape(6,3)
        print("\nFinal marginals:", marginals,'\n')
        any_mismatch = np.any(abs(marginals - marginals[0])>1e-1, axis=1)
        generate_uhoh(not np.any(any_mismatch))

        mutual_info, entropy = calculate_mutual_info(srv, ps,qs, srv_states)
        print("\nMutual info and entropy:", mutual_info, entropy, "\n")

        print("\nSrv:\n",srv, "\n ps, and qs, end_marginal:", repr(ps), ", ", repr(qs), ", ", repr(marginals[0]))
    return srv

if __name__ == '__main__':
    ps,qs = np.array([0.7,0.2,0.1]), np.array([0.4,0.3,0.3])
    new_analyticalsrv_test(ps,qs)