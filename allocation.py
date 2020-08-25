import numpy as np
from numba import vectorize, njit

n_customers = 1000
n_offers = 20
n_categories = 2
categories = np.array(['cat_' + str(n + 1) for n in range(n_categories)])

affinities = np.random.random((n_customers, n_offers))
allocable = affinities > 0.1
category_dtype = np.dtype({
    'names': categories,
    'formats': [np.int] * n_categories
})

@njit
def create_cat_array(n_custs, cat_dtype):
    cat_per_cust = np.zeros((n_custs), dtype=cat_dtype)
    for cust in cat_per_cust:
        cust['cat_1'] = np.random.randint(1, 4)
        # for cat in categories:
        #     cust[cat] = np.random.randint(1, 4)
    return cat_per_cust

cat_per_cust = create_cat_array(n_customers, category_dtype)

@njit
def alloc_heuristic(affinities, allocable, n_per_cat):
    for cust_affnities, cust_allocable, cust_n_cat in zip(affinities, allocable, n_per_cat):
        pass

if __name__ == '__main__':

    _ = 1