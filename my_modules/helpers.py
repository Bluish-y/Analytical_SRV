import numpy as np
import warnings
import my_modules.direct_srv as ds
def calculate_marginals(srv, ps, qs): #assuming only two variables
    p_states = ps.shape[0]
    q_states = qs.shape[0]
    no_of_variables = 2
    marginals = []
    shape = srv.shape
    for i in range(0,shape[0]):
        s,t = np.zeros(srv[0,0].shape), np.zeros(srv[0,0].shape)
        for j in range(0, shape[1]):
            s = s+srv[i,j]*ps[j]
            t = t+srv[j,i]*qs[j]
        marginals.append((s,t))
    return np.array(marginals)

def calculate_mutual_info(srv, ps,qs, srv_states):
    p_states = len(ps)
    q_states = len(qs)
    info = 0
    entropy = 0
    conditional_entropy = 0
    cs = np.zeros(srv_states)

    for i in range(0,q_states):
        for j in range(0,p_states):
            for k in range(0, srv_states):
                cs[k] += qs[i]*ps[j]*srv[i,j,k]
 
    for i in range(0, q_states):
        for j in range(0, p_states):
            for k in range(0,srv_states):
                if ps[i]*qs[j]*srv[i,j,k] != 0:
                    if srv[i,j,k]>0:
                        conditional_entropy += qs[i]*ps[j]*srv[i,j,k]*np.log2(1/srv[i,j,k])
                    else:
                        conditional_entropy += np.nan

    for k in range(0, srv_states):
        if cs[k] != 0:
            if cs[k]>0:
                entropy += cs[k]*np.log2(1/cs[k])
            else:
                entropy += np.nan
    info = entropy - conditional_entropy
    return info, entropy

def generate_warning(success):
    if not success:
        warning_message = (
            "\n*********************************************************************\n"
            "WARNING: Some elements in srv might not be probabilities.\n"
            "*********************************************************************\n"
        )
        warnings.warn(warning_message)
def generate_uhoh(success):
    if not success:
        warning_message = (
            "\nThe marginals didn't match! uh-oh!\n"
        )
        print(warning_message)

def my_pinv(A):
    u, s, vh = np.linalg.svd(A)
    s[s>1e-12] = 1/s[s>1e-12]
    pinv = vh.T @ np.diag(s) @ u.T
    return pinv 

def impurify(A,B,C, do_it=False):
    if do_it == True:
        for i in range(C.shape[0]):
            for j in range(C.shape[1]):
                if C[i, j] < 0:
                    # Check which matrix has the smaller corresponding element
                    if A[i, j] <= B[i, j]:
                        if A[i, j] + C[i, j] < 0:
                            # If adding the negative value makes the element in A go to zero or less
                            B[i, j] += (A[i, j] + C[i, j])
                            A[i, j] = 0
                        else:
                            A[i, j] += C[i, j]
                    else:
                        if B[i, j] + C[i, j] < 0:
                            # If adding the negative value makes the element in B go to zero or less
                            A[i, j] += (B[i, j] + C[i, j])
                            B[i, j] = 0
                        else:
                            B[i, j] += C[i, j]
        C = 1- A - B
    return A, B, C

def normalize(solution, maxi, prev_sum, p_states, q_states, search = 0):
    if min(solution)<0:
        solution -= min(solution)
    # if np.any(solution>=(maxi-prev_sum)):
    max_index = np.argmax(solution - (maxi-prev_sum))
    # k = maxi*(np.ones(p_states*q_states)-prev_sum)[max_index]
    # solution *= (k)/solution[max_index]
    k = maxi*(np.ones(p_states*q_states)-prev_sum)[np.argmax(solution)]
    solution *= (k)/max(solution)
    dx = 0.001
    ma = max(solution)
    if search:
        if np.any(abs(1-prev_sum)<1e-9):
            solution = 0*solution
        else:
            while(np.any(solution>=(1-prev_sum))):
                ma += dx
                solution /= ma

    return solution

<<<<<<< HEAD
def grad_info(solution, ps,qs, target_marginal, w, pinv, null_mat):
=======
def grad_info(solution, ps,qs, target_marginal, w, pinv, null_mat, s3):
>>>>>>> 9b40086 (Convex optimization to improve synergy values)
    srv_states = len(target_marginal)
    sol_states, p_states, q_states = len(w), len(ps), len(qs)
    b = target_marginal
    grad_b = np.zeros(srv_states)
    grad_w = np.zeros(sol_states)
    for k in range(0, srv_states):
        s1 = np.copy(solution)
<<<<<<< HEAD
        lagrange = (np.log(s1) + 1)*np.sum(pinv, axis=1)
        lagrange[abs(lagrange) == np.inf] = 1e4
=======
        lagrange = (np.log(s1/s3) + 1)*np.sum(pinv, axis=1)
        lagrange[abs(lagrange) == np.inf] = -1e4
>>>>>>> 9b40086 (Convex optimization to improve synergy values)
        sum=0
        for i in range(0, q_states):
            for j in range(0,p_states):
                sum += qs[i]*ps[j]*lagrange[3*i+j]
<<<<<<< HEAD
        grad_b[k] = -1-np.log(b[k]) + sum
    for k in range(0, sol_states):
        lagrange = (np.log(s1) + 1)*null_mat[:,k]
        lagrange[abs(lagrange) == np.inf] = 1e4
=======
        grad_b[k] = -1-np.log(b[k]/b[2]) + sum
    for k in range(0, sol_states):
        lagrange = (np.log(s1/s3) + 1)*null_mat[:,k]
        lagrange[abs(lagrange) == np.inf] = -1e4
>>>>>>> 9b40086 (Convex optimization to improve synergy values)
        sum = 0
        for i in range(0, q_states):
            for j in range(0,p_states):
                sum += qs[i]*ps[j]*lagrange[3*i+j]
        grad_w[k] = sum
<<<<<<< HEAD
=======
    grad_b[grad_b == np.inf] = 1e4
    grad_b[grad_b == -np.inf] = -1e4
    grad_w[grad_w == np.inf] = 1e4
    grad_w[grad_w == -np.inf] = -1e4
>>>>>>> 9b40086 (Convex optimization to improve synergy values)

    grad = np.concatenate((grad_b, grad_w))
    norm = np.linalg.norm(grad)
    return grad_b/norm, grad_w/norm

def print_info(srv, p, q):
    ps = [p,q]
    print(f'I(S:X) = {ds.mutual_information_srv_all_inputs(srv, ps):.2f}. ' 
      + f'H(X_i)={[f"{ds.entropy_input(xix, ps):.2f}" for xix in range(len(ps))]}'
      + f'. I(S:X_i)={[f"{ds.mutual_information_srv_given_single_input(xix, ps, srv):.5f}" for xix in range(len(ps))]}')

