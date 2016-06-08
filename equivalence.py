"""
equivalence.py
--------------

Finds the number of equivalence classes for Hadamard matrices of order n.

"""
# standard library
import multiprocessing
from itertools import permutations, combinations, repeat

# nonstandard library
import numpy as np


# constants
_ORDER_4_HADAMARDS = { ( (1,  1,  1,  1)
                       , (1, -1, -1,  1)
                       , (1,  1, -1, -1)
                       , (1, -1,  1, -1)
                       ) 
                     , ( (1,  1,  1,  1)
                       , (1, -1, -1,  1)
                       , (1, -1,  1, -1)
                       , (1,  1, -1, -1)
                       ) 
                     , ( (1,  1,  1,  1)
                       , (1, -1,  1, -1)
                       , (1, -1, -1,  1)
                       , (1,  1, -1, -1)
                       ) 
                     , ( (1,  1,  1,  1)
                       , (1, -1,  1, -1)
                       , (1,  1, -1, -1)
                       , (1, -1, -1,  1)
                       ) 
                     , ( (1,  1,  1,  1)
                       , (1,  1, -1, -1)
                       , (1, -1, -1,  1)
                       , (1, -1,  1, -1)
                       ) 
                     , ( (1,  1,  1,  1)
                       , (1,  1, -1, -1)
                       , (1, -1,  1, -1)
                       , (1, -1, -1,  1)
                       )
                     }


# functions
def unique_permutations(items):
    """Finds the set of unique permuations of `items`.
    
    Args:
        items (Iterable[X]): The items to find permutations of.

    Returns:
        (Set[Iterable[X]]): The set of unique permutations of `items`.
    
    Examples:
        >>> unique_permutations((1, 1, 1))
        {(1, 1, 1)}
        >>> unique_permutations((1, 1, -1))
        {(1, 1, -1), (1, -1, 1), (-1, 1, 1)}

    """
    return set(permutations(items))

def hadamard_candidates_by_permutations(order):
    """Generates candidates for normal form Hadmard matrices.

    Generates all possible normal form Hadmard matrices of a given order by
    taking the unique permutations of `order // 2` -1s and `order // 2` +1s
    as the set of possible rows, then taking permutations of `order - 1`
    possible rows.

    Args:
        order (int): the order for which to generate candidate matrices.

    Yields:
        (Tuple[Tuple[int]]): all possible normal form hadamard matrices of
        order `order`.

    Raises:
        ValueError: if `order > 2` and  `order` is not divisible by 4.

    Examples:
        Give this function a short name:
        >>> c = hadamard_candidates_by_permutations

        There is only one normal form Hadamard matrix of order 1, 2:
        >>> for m in c(1):
        ...     print(np.array(m))
        [[1]]
        >>> for m in c(2):
        ...     print(np.array(m))
        [[ 1  1]
         [ 1 -1]]

        For order 4, there are 6:
        >>> for m in c(4):
        ...     print(any(np.array_equal(m, e) for e in _ORDER_4_HADAMARDS))
        True
        True
        True
        True
        True
        True

    """
    if order > 2 and order % 4 != 0:
        raise ValueError("order must be < 2 or divisible by 4")

    seed = tuple(repeat(-1, order // 2)) + tuple(repeat(1, order // 2 - 1))
    possible_rows = tuple((1,) + perm for perm in unique_permutations(seed))

    first_row = tuple(repeat(1, order))
    for rows in permutations(possible_rows, order -1):
        yield (first_row,) + rows

def hadamard_candidates_by_perms_combs(order):
    """Generates candidates for normal form Hadmard matrices.

    Generates most of the normal form Hadamard matrices of a given order, by
    taking permutations of `order // 2` -1s and `order // 2` +1s, but only
    generating each combination of rows once.

    Args:
        order (int): the order for which to generate candidate matrices.

    Yields:
        (np.array_like): a square matrix of order `order` with first row all 1,
        first column all 1 and linearly independent rows and columns.

    Raises:
        ValueError: if `order > 2` and  `order` is not divisible by 4.

    Examples:
        Give this function a short name
        >>> c = hadamard_candidates_by_perms_combs

        Produces one candidate for orders 1, 2 and 4
        >>> for m in c(1):
        ...     print(np.array(m))
        [[1]]
        >>> for m in c(2):
        ...     print(np.array(m))
        [[ 1  1]
         [ 1 -1]]
        >>> for m in c(4):
        ...     print(any(np.array_equal(m, e) for e in _ORDER_4_HADAMARDS))
        True

        Produces many candidates for 8 and doesn't take an age to do so
        >>> from datetime import datetime, timedelta
        >>> then = datetime.now()
        >>> for m in c(8):
        ...     pass
        >>> now = datetime.now()
        >>> then - now < timedelta(seconds=2)
        True

    """
    if order > 2 and order % 4 != 0:
        raise ValueError("order must be < 2 or divisible by 4")

    seed = tuple(repeat(-1, order // 2)) + tuple(repeat(1, order // 2 - 1))
    possible_rows = tuple((1,) + perm for perm in unique_permutations(seed))

    first_row = tuple(repeat(1, order))
    for rows in combinations(possible_rows, order - 1):
        yield (first_row,) + rows

def monomials(order):
    """Generates the monomial matrices of a given order.

    Args:
        order (int): the order of the generated matrices

    Yields:
        (np.array_like): the monomial matrices of order `order`.

    Examples:
        >>> ms = monomials
        
        Two monomial matrices of order 1
        >>> for m in ms(1):
        ...     print(np.array(m))
        [[1]]
        [[-1]]
        
        More for order 2
        >>> expected = { (( 1,  0)
        ...              ,( 0,  1)
        ...              )
        ...            , ((-1,  0)
        ...              ,( 0,  1)
        ...              )
        ...            , (( 1,  0)
        ...              ,( 0, -1)
        ...              )
        ...            , ((-1,  0)
        ...              ,( 0, -1)
        ...              )
        ...            , (( 0,  1)
        ...              ,( 1,  0)
        ...              )
        ...            , (( 0, -1)
        ...              ,( 1,  0)
        ...              )
        ...            , (( 0,  1)
        ...              ,(-1,  0)
        ...              )
        ...            , (( 0, -1)
        ...              ,(-1,  0)
        ...              )
        ...            }
        >>> for m in ms(2):
        ...     print(any(np.array_equal(m, e) for e in expected))
        True
        True
        True
        True
        True
        True
        True
        True

    """
    def flip_row(rows, i):
        """Changes the sign of a row.

        Args:
            rows (Tuple[Tuple[int]]): a matrix
            i (int): the index into `rows` of the row to flip

        """
        for j in range(order):
            rows[i][j] = -rows[i][j]
    possible_rows = tuple(tuple(1 if i == n else 0 for i in range(order))
                          for n in range(order))
    
    for rows in permutations(possible_rows):
        for n in range(2**order):
            rows_ = list(list(row) for row in rows)
            for i in range(order):
                if n == 0: break
                if (n >> i) & 1: flip_row(rows_, i)
            yield rows_
    
def find_equivalence_classes(candidate_generator, monomial_generator, order):
    """Finds representative members of equivalence classes of a given order.

    Args:
        candidate_generator (Callable[[int], Iterable[np.array_like]]):
            a generator of candidate hadamard matrices, whose argument is
            the order of the generated matrices.
        monomial_generator (Callable[[int], Iterable[np.array_like]]):
            a generator of candidate hadamard matrices, whose argument is
            the order of the generated matrices.
        order (int): the order to investigate the equivalence classes of.
    
    Returns:
        (List[np.array_like]): a representative member from each equivalence
        class of hadamard matrices order n.

    Examples:
        Give this function a short name and generators
        >>> from functools import partial
        >>> f = partial(find_equivalence_classes,
        ...             hadamard_candidates_by_perms_combs,
        ...             monomials)
        
        One equivalence class for orders 1, 2 and 4
        >>> len(f(1))
        1
        >>> len(f(2))
        1
        >>> len(f(4))
        1

    """
    def equivalent_to_representative(h):
        """Determines whether h is equivalent to a representative.
        
        Args:
            h (np.array_like): a candidate matrix

        Returns:
            (bool): True iff there exist monomial matrices P and Q such that
            h = P*r*Q for some r in representatives. 

        """
        for p in monomial_generator(order):
            for q in monomial_generator(order):
                for r in representatives:
                    if np.array_equal(h, np.dot(p, r).dot(q)):
                        return True
        return False

    representatives = []
    for h in candidate_generator(order):
        if not equivalent_to_representative(h):
            representatives.append(h)
    return representatives

def find_equivalence_classes_parallel\
        (candidate_generator, monomial_generator, order):
    """Finds representative members of equivalence classes of a given order.

    Speeds up the process using multiprocessing.

    Args:
        candidate_generator (Callable[[int], Iterable[np.array_like]]):
            a generator of candidate hadamard matrices, whose argument is
            the order of the generated matrices.
        monomial_generator (Callable[[int], Iterable[np.array_like]]):
            a generator of candidate hadamard matrices, whose argument is
            the order of the generated matrices.
        order (int): the order to investigate the equivalence classes of.
    
    Returns:
        (List[np.array_like]): a representative member from each equivalence
        class of hadamard matrices order n.

    Examples:
        Give this function a short name and generators
        >>> from functools import partial
        >>> f = partial(find_equivalence_classes,
        ...             hadamard_candidates_by_perms_combs,
        ...             monomials)
        
        One equivalence class for orders 1, 2 and 4
        >>> len(f(1))
        1
        >>> len(f(2))
        1
        >>> len(f(4))
        1

    """
    def equivalent_to_representative(h):
        """Determines whether h is equivalent to a representative.
        
        Args:
            h (np.array_like): a candidate matrix

        Returns:
            (bool): True iff there exist monomial matrices P and Q such that
            h = P*r*Q for some r in representatives. 

        """
        for p in monomial_generator(order):
            for q in monomial_generator(order):
                for r in representatives:
                    if np.array_equal(h, np.dot(p, r).dot(q)):
                        return True
        return False

    representatives = []
    for h in candidate_generator(order):
        if not equivalent_to_representative(h):
            representatives.append(h)
    return representatives


if __name__ == '__main__':
    import doctest
    doctest.testmod(optionflags=doctest.NORMALIZE_WHITESPACE)

