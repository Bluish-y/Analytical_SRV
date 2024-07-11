**Note**: The ps and qs correspond to the rows and columns instead of columns and rows as descriped in the report. This has happened since the definition of the vectorization operator is different in this code. However, all formulas remain same due to this switch.

# Analytical SRVs

## Description

This project implements and compares two approaches (analytical and numerical) for computing the mutual information between random variables under certain constraints. The core idea is to compare these two methods by running simulations and visualizing the results using plots.

## Project Structure

The project is structured as follows:

- `main.py`: A simple run of the analytical srv algorithm that is later on used as a function in `compare.py`. Use it as a testing ground for parameters.
- `compare.py`: The script that performs the simulations for both approaches and plots to compare the results.
- `my_modules/analyticalsrv_func.py`: Contains the functions for the analytical SRV computation assuming independence.
- `my_modules/direct_srv.py`: Contains the functions for the numerical SRV computation assuming independence.
- `my_modules/helpers.py`: Contains helper functions for marginal calculations, mutual information computations, and other utility functions.
- `my_modules/variables.py`: Contains predefined variables used in the computations.

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/your-repo-name.git
    cd your-repo-name
    ```
2. Install the required Python packages:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

Run the main script:
```sh
python compare.py
```

The script performs the following steps:

- Generates random probability distributions `ps` and `qs` using a Dirichlet distribution.
- Computes the SRVs using both the analytical and numerical approaches.
- Calculates the mutual information for the SRVs.
- Plots the results in two formats: boxplots and bar charts.

## Output

The script generates the following plots:

- **Boxplots**: Displays the distribution of mutual information values for both the analytical and numerical approaches.
- **Bar Charts**: Compares the mutual information values for each simulation instance.

## Function Descriptions

### `analyticalsrv_test(ps, qs, do_it=0, search_it=1, do_and_search=0, print_it=0)`

Computes the analytical SRV given input probability distributions `ps` and `qs`.

**Parameters:**

- `ps`: Probability distribution for the random variable X1.
- `qs`: Probability distribution for the random variable X2.
- `do_it`: Flag to control impurity correction. This runs the algorithm such that it produces an analytical SRV, and if the SRV has probabilities that are negative, it makes those negative values zero. The idea here is to find an SRV which might not be 100% synergistic but will definitely describe the probability distributions.
- `search_it`: Flag to control normalization search. This search performs the impurity correction described above in a loop to converge to an SRV which is 100% synergistic. However, this also reduces the mutual information, so this is just a way to test the closest SRV you can find using impurity correction.
- `do_and_search`: Flag to control both impurity correction and normalization search.
- `print_it`: Flag to print detailed computation results.

**Returns:** SRV matrix.

### `cubic_analyticalsrv_test(ps, qs, do_it=0, search_it=1, do_and_search=0, print_it=0)`

Computes the cubic analytical SRV given input probability distributions `ps` and `qs`. This is a new algorithm I was testing near the end of the project and is not mentioned in the report. It follows the same structure except the math is slightly different. Reach out to me at: yasheesinha@gmail.com for the notes. This algorithm somehow provides a better result than the previous algorithm.

**Parameters:** (Same as `analyticalsrv_test`)

**Returns:** SRV matrix.

## Dependencies

- numpy
- matplotlib
- sympy
- warnings

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

