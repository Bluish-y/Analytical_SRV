import my_modules.analyticalsrv_func as asrv
import my_modules.direct_srv as ds
import numpy as np
from matplotlib import pyplot as plt
from sympy import sympify

# Number of simulations
nsims = 50

# Initialize arrays to store mutual information results
ISXs_direct = np.zeros(nsims)
ISXis_direct = np.zeros(nsims)
ISXs_analytical = np.zeros(nsims)
ISXis_analytical = np.zeros(nsims)

# Loop over the number of simulations
for i in range(0, nsims):
    # Generate random probability distributions p and q using a Dirichlet distribution
    p = np.random.dirichlet([1, 1, 1])
    q = np.random.dirichlet([1, 1, 1])

    # Uncomment the following lines to test with specific values of p and q
    # q, p = np.array([0.55373569, 0.23203089, 0.21423342]), np.array([0.25712055, 0.53742108, 0.20545838])
    # q, p = np.array([0.65, 0., 0.35]), np.array([0.85, 0.15, 0])
    # q, p = np.array([0.65, 0.001, 0.349]), np.array([0.85, 0.149, 0.001])

    ps = [p, q] #notice how this is [p,q] and not [q,p] as you would expect from the report. This is due to the fact that this code uses a different definition of the vectorization operator where all the rows are stacked instead of the columns.

    # Determine the number of states for the SRV to be constructed
    kxs = tuple(len(p) for p in ps)  # Number of values for each input variable
    ks = max(kxs)  # Maximum number of states
    srv_ps_shape = kxs + (ks,)  # Shape for SRV, used in multiple places below
    on_violation = 'continue'
    
    # Compute the direct SRV assuming independence
    direct_srv_joint, totviol_joint = ds.compute_direct_srv_assuming_independence(
        ps, ks, method='joint', return_also_violation=True, on_violation=on_violation, verbose=1)

    # Compute the analytical SRV using two different methods
    analytical_srv1 = asrv.analyticalsrv_test(q, p, do_it=0, search_it=1, do_and_search=0, print_it=0)
    analytical_srv2 = asrv.cubic_analyticalsrv_test(q, p, do_it=1, search_it=1, do_and_search=1, print_it=0)

    # Calculate mutual information for direct SRV
    ISXs_direct[i] = ds.mutual_information_srv_all_inputs(direct_srv_joint, ps)
    ISXis_direct[i] = np.sum([abs(ds.mutual_information_srv_given_single_input(xix, ps, direct_srv_joint)) for xix in range(len(ps))])

    # Calculate mutual information for analytical SRV
    ISXs_analytical[i] = max(ds.mutual_information_srv_all_inputs(analytical_srv1, ps), ds.mutual_information_srv_all_inputs(analytical_srv2, ps))
    ISXis_analytical[i] = max(np.sum([abs(ds.mutual_information_srv_given_single_input(xix, ps, analytical_srv1)) for xix in range(len(ps))]),
                              np.sum([abs(ds.mutual_information_srv_given_single_input(xix, ps, analytical_srv2)) for xix in range(len(ps))]))

# Print the mutual information results for direct SRV
print(f'I(S:X) = {ds.mutual_information_srv_all_inputs(direct_srv_joint, ps):.2f}. ' 
      + f'H(X_i)={[f"{ds.entropy_input(xix, ps):.2f}" for xix in range(len(ps))]}'
      + f'. I(S:X_i)={[f"{ds.mutual_information_srv_given_single_input(xix, ps, direct_srv_joint):.5f}" for xix in range(len(ps))]}')

# Print the mutual information results for analytical SRV
print(f'I(S:X) = {ds.mutual_information_srv_all_inputs(analytical_srv2, ps):.2f}. ' 
      + f'H(X_i)={[f"{ds.entropy_input(xix, ps):.2f}" for xix in range(len(ps))]}'
      + f'. I(S:X_i)={[f"{ds.mutual_information_srv_given_single_input(xix, ps, analytical_srv2):.5f}" for xix in range(len(ps))]}')

# Data for plots
data1 = [ISXs_analytical, ISXs_direct, np.maximum(ISXs_analytical, ISXs_direct)]
data2 = [ISXis_analytical, ISXis_direct]

# Plot the results
plt.figure(figsize=(10, 8))  # Adjust the figure size

# Create the first subplot (Boxplot)
plt.subplot(2, 1, 1)  # Create a subplot (2 rows, 1 column, plot 1)
plt.boxplot(data1, labels=['Analytical Approach', 'Numerical Approach', 'max of both'], patch_artist=True)
plt.ylabel(r"$I(S:X)$")
plt.title("(a)")

# Create the second subplot (Boxplot)
plt.subplot(2, 1, 2)  # Create a subplot (2 rows, 1 column, plot 2)
plt.boxplot(data2, labels=['Analytical Approach', 'Numerical Approach'])
plt.ylabel(r"$\sum_{i=1}^{n} |I(S:X_i)|$")
plt.title("(b)")

plt.subplots_adjust(hspace=0.5)  # Adjust the space between plots
plt.show()

# Bar charts
x = np.arange(nsims)  # Generate x values for the range of simulations
width = 0.35  # Width of the bars
space = 0.15  # Additional space between groups

# Create the first subplot (Bar chart)
plt.subplot(2, 1, 1)  # Create a subplot (2 rows, 1 column, plot 1)
plt.bar(x * (1 + space) - width / 2, ISXs_analytical, width, label='Analytical Approach', color='b')
plt.bar(x * (1 + space) + width / 2, ISXs_direct, width, label='Numerical Approach', color='r')
plt.title(r"$I(S:X)$")
plt.legend()
plt.gca().set_xticklabels([])  # Remove x-axis labels

# Create the second subplot (Bar chart)
plt.subplot(2, 1, 2)  # Create a subplot (2 rows, 1 column, plot 2)
plt.bar(x * (1 + space) - width / 2, ISXis_analytical, width, label='Analytical Approach', color='g')
plt.bar(x * (1 + space) + width / 2, ISXis_direct, width, label="Numerical Approach", color='m')
plt.title(r"$\sum_{i=1}^{n} |I(S:X_i)|$")
plt.xlabel("Different values of " + r'$\vec{p}, \vec{q}$')
plt.legend()
plt.gca().set_xticklabels([])  # Remove x-axis labels

plt.subplots_adjust(hspace=0.5)  # Adjust the space between plots
plt.show()
