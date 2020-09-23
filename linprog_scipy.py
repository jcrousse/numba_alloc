from scipy.optimize import linprog
import numpy as np
import time


def optimization_coefficients(weight_matrix):
    return -weight_matrix.flatten()


def constraint_col_coef(n_col, n_row):
    """
    generate rows of the inequality constraints coefficient matrix for column consraints
    """

    all_rows = []
    for i in range(n_col):
        matrix_values = np.zeros((n_row, n_col), dtype=int)
        col_offer = np.ones(n_row, dtype=int)
        matrix_values[:, i] = col_offer
        all_rows.append(matrix_values.flatten())

    cols_constraints = np.stack(all_rows)

    return cols_constraints


def constraints_row_coef(n_row, n_col):
    """
    generate rows of the inequality constraints coefficient matrix.
    """
    identities = [np.identity(n_col) for _ in range(n_row)]
    cust_constraints = np.concatenate(identities, axis=1)
    permutation = np.transpose(np.reshape(np.arange(n_col * n_row), (n_col, n_row))).flatten()
    idx = np.empty_like(permutation)
    idx[permutation] = np.arange(len(permutation))
    cust_constraints = cust_constraints[:, idx]

    return cust_constraints


def constraints_max_offer_per_cust(n_row, n_col):
    """
    Values in the resulting matrix is at most one
    """
    constraints = np.identity(n_row * n_col)
    return constraints


def generate_ineq_constraints_matrix(n_col, n_row):
    """
    generate the inequality constraints coefficient matrix.
    """
    offers_constraints = constraint_col_coef(n_col, n_row)
    customers_constraints = constraints_row_coef(n_col, n_row)
    offer_per_cust_constraints = constraints_max_offer_per_cust(n_col, n_row)
    mat_a = np.append(offers_constraints, customers_constraints, axis=0)
    mat_a = np.append(mat_a, offer_per_cust_constraints, axis=0)
    return mat_a


def generate_inquality_constraint_vector(c_max, r_max):
    n_col = len(c_max)
    n_row = len(r_max)
    b = c_max + r_max + [1] * n_row * n_col
    return b


def run_linprog(weight_matrix, c_max, r_max):
    n_cols = len(c_max)
    n_rows = len(r_max)
    c = optimization_coefficients(weight_matrix)
    mat_a = generate_ineq_constraints_matrix(n_cols, n_rows)
    b = generate_inquality_constraint_vector(c_max, r_max)
    start = time.time()
    res = linprog(c, A_ub=mat_a, b_ub=b) # noqa
    end = time.time()
    time_taken = round(end - start, 3)
    return res, time_taken


if __name__ == '__main__':
    np.random.seed(45)
    M = np.random.random((4, 2))
    R = [1, 2, 2, 1]
    C_max = [3, 3]

    opt_res, _ = run_linprog(M, C_max, R)

    print(opt_res['fun'])
