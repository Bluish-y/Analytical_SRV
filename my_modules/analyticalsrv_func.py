import numpy as np
import warnings
from my_modules.helpers import *
import my_modules.variables as vb
np.set_printoptions(suppress=True)

def analyticalsrv_test(ps,qs, do_it = 0, search_it = 1, do_and_search=0, print_it = 0):
    """
    Compute the analytical synergistic random variables (SRV) given probability distributions ps and qs.

    Parameters:
    ps : numpy.ndarray
        Probability distribution for the first input variable.
    qs : numpy.ndarray
        Probability distribution for the second input variable.
    do_it : int, optional (default=0)
        Flag to decide if an invalid probability distribution should be corrected by making all negative values 0.
    search_it : int, optional (default=1)
        Flag to decide if you should modify the normalization parameters to increase mutual info while also keeping the SRV a valid probability distribution.
        By default, it is set to 0.9 for the first elements and 1 for the second elements of the SRV. You can change this from the `norm` parameter.
    do_and_search : int, optional (default=0)
        Flag to decide if the impurity correction should be repeated till a 100% synergistic random variable is found.
    print_it : int, optional (default=0)
        Flag to decide if the results should be printed.

    Returns:
    srv : numpy.ndarray
        The computed SRVs
    """

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
    """
    Generate a simple SRV given the inputs and intermediate variables without any optimization algorithms.

    Parameters:
    ps : numpy.ndarray
        Probability distribution for the first input variable.
    qs : numpy.ndarray
        Probability distribution for the second input variable.
    bs : list
        Target marginal distributions.
    ws : list
        Weight vectors.
    pinv : numpy.ndarray
        Pseudoinverse of the master matrix.
    null_mat : numpy.ndarray
        Null matrix derived from the master matrix.
    master_mat : numpy.ndarray
        Master matrix combining the input probability distributions.

    Returns:
    srv : numpy.ndarray
        The computed SRV
    """
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
