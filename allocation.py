import numpy as np
from numba import vectorize, njit

n_customers = 1000
n_offers = 20
n_categories = 2
categories = np.array(['cat_' + str(n + 1) for n in range(n_categories)])
allocation = np.zeros((n_customers, n_offers), dtype=bool)
affinities = np.random.random((n_customers, n_offers))
allocable = affinities > 0.1
# category_dtype = np.dtype({
#     'names': categories,
#     'formats': [np.int] * n_categories
# })

@njit
def create_cat_array(n_custs, n_cats):
    cat_per_cust = np.zeros((n_custs, n_cats))
    for i in range(cat_per_cust.shape[0]):
        for j in range(n_cats):
            cat_per_cust[i, j] = np.random.randint(1, 4)
    return cat_per_cust


cat_per_cust = create_cat_array(n_customers, len(categories))



@njit
def alloc_heuristic(alloc_df, aff_df, allocable_df, n_per_cat):
    for cust_alloc, cust_affnities, cust_allocable, cust_n_cat in zip(alloc_df, affinities, allocable, n_per_cat):
        # get allocable (using already allocated, cat per offer and n_per cat)
        # allocable = base_allocable AND current allocable
        affinity_vector = cust_affnities * cust_allocable
        probability_vector = affinity_vector / affinity_vector.sum()
        # go through vector and stop when sum > random number. Or cumulative sum and find first occurence above value
        # do all this while allocable returns non 0 vector.
        print(probability_vector)




if __name__ == '__main__':
    res = alloc_heuristic(allocation, affinities, allocable, cat_per_cust)
    _ = 1