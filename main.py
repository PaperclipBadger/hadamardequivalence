"""
main
----

Finds the equivalence classes for Hadamard matrices of order 4, 8 and 12.

"""
import numpy as np
from equivalence import find_equivalence_classes_parallel, \
        hadamard_candidates_by_perms_combs, monomials

def reps(n):
    print('\nRepresentatives for order {}:'.format(n))
    for m in find_equivalence_classes_parallel(
            hadamard_candidates_by_perms_combs,
            monomials,
            n):
        print(np.array(m))

if __name__ == '__main__':  # necessary for multiprocessing
    reps(4)
    reps(8)
    reps(12)
