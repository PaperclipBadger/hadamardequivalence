"""
equivalence.py
--------------

Finds the number of equivalence classes for Hadamard matrices of order n.

"""
# standard library
import sys
import multiprocessing as mp
from ctypes import c_int8, c_uint64, Structure
from itertools import permutations, combinations, repeat
from functools import partial

# nonstandard library
import numpy as np

# project
import concurrencyutils


# globals
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

def is_monomial(matrix):
    """Checks if a matrix is monomial.

    Args:
        matrix (np.array_like): a matrix

    Returns:
        (bool): True iff the matrix is monomial.

    Examples:
        >>> is_monomial([[1, 0], [0, 1]])
        True
        >>> is_monomial([[1, 0], [0, -1]])
        True
        >>> is_monomial([[1, 0], [1, 0]])
        False
        >>> is_monomial([[1, 0], [-1, 0]])
        False
        >>> all(is_monomial(m) for m in monomials(4))
        True
        >>> h1 = ((1,  1,  1,  1),
        ...       (1,  1, -1, -1),
        ...       (1, -1,  1, -1),
        ...       (1, -1, -1,  1))
        >>> h2 = ((1,  1,  1,  1),
        ...       (1, -1,  1, -1),
        ...       (1,  1, -1, -1),
        ...       (1, -1, -1,  1))
        >>> m = ((1, 0, 0, 0),
        ...      (0, 0, 1, 0),
        ...      (0, 1, 0, 0),
        ...      (0, 0, 0, 1))
        >>> is_monomial(((1/4)*np.array(h1).T).dot(np.linalg.inv(m)).dot(h2))
        True

    """
    indeces = set()
    for row in matrix:
        seen = False
        for i in range(len(row)):
            if row[i] not in {1, 0, -1}:
                return False
            if row[i]:
                if not seen and i not in indeces:
                    seen = True
                    indeces.add(i)
                else:
                    return False
    return True

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
        (candidate_generator, monomial_generator, order, 
         workers=None, limit=None, progress_meter=1<<14):
    """Finds representative members of equivalence classes of a given order.

    Speeds up the process using multiprocessing.

    Args:
        candidate_generator (Callable[[int], Iterable[np.array_like]]):
            a generator of candidate hadamard matrices, whose argument is
            the order of the generated matrices.
        monomial_generator (Callable[[int], Iterable[np.array_like]]):
            generator of candidate hadamard matrices, whose argument is
            the order of the generated matrices.
        order (int): the order to investigate the equivalence classes of.
        workers (int): the number of workers to use; defaults to one per core.
        limit (int): limit for the size of the representative list. Defaults
            to whateber the default limit for SharedStack is.
    
    Returns:
        (List[np.array_like]): a representative member from each equivalence
        class of hadamard matrices order n.

    Examples:
        Give this function a short name and generators
        >>> from functools import partial
        >>> f = partial(find_equivalence_classes_parallel,
        ...             hadamard_candidates_by_permutations,
        ...             monomials, progress_meter=False)
        
        One equivalence class for orders 1, 2 and 4
        >>> len(f(1))
        1
        >>> len(f(2))
        1
        >>> len(f(4))
        1

    """
    global _representatives
    global _lock
    global _dot_counter
    _representatives = concurrencyutils.SharedStack(c_int8, limit*order**2) if\
            limit else concurrencyutils.SharedStack(c_int8)
    _lock = mp.Lock()
    _dot_counter = mp.Value(c_uint64)
    poolgen = partial(mp.Pool, workers) if workers else mp.Pool
    with poolgen(initializer=process_initialiser, 
            initargs=[_representatives, _lock, _dot_counter]) as pool:
        pool.map(partial(work, order, monomial_generator, 
                         progress_meter=progress_meter), 
                 candidate_generator(order))

    print('')
    m = []
    for i in range(len(_representatives) // (order * order)):
        m.append(read_matrix(_representatives, i, order))
    return m

def push_matrix(stack, array_like):
    for row in array_like:
        for entry in row:
            stack.push(entry)

def read_matrix(stack, index, order):
    m = []
    for i in range(order):
        m.append([])
        for j in range(order):
            m[i].append(stack[index * order * order + i * order + j])
    return m

def process_initialiser(representatives, lock, dot_counter):
    """Initialises the globals for a subprocess."""
    global _representatives
    global _lock
    global _dot_counter
    _representatives = representatives
    _lock = lock
    _dot_counter = dot_counter

def print_dot(interval):
    """Prints a dot once every `interval` times this function is called."""
    global _dot_counter
    b = False
    with _dot_counter.get_lock():
        _dot_counter.value += 1
        b = _dot_counter.value % interval == 0
    if b:
        print(".", end='')
        sys.stdout.flush()

def equivalent(h, r, monomial_generator, progress_meter=1<<14):
    """Determines whether h is equivalent to r.

    h and r must be square matrices of the same order.
    
    Args:
        h (np.array_like): a matrix
        r (np.array_like): another matrix
        progress_meter (bool): if true, prints a . every once in a while
            so you know nothing has frozen.

    Returns:
        (bool): True iff there exist monomial matrices P and Q such that
        h = P*r*Q.

    """
    order = len(h)
    h = np.array(h)
    r = np.array(r)
    for p in monomial_generator(order):
        # check if r^-1 * p^-1 * h is monomial
        # r is hadamard, so r^-1 = 1/order r^T
        r_inv = (1/order) * r.T
        if is_monomial(r_inv.dot(np.linalg.inv(p)).dot(h)):
            return True
        if progress_meter: print_dot(progress_meter)
    return False

def work(order, monomial_generator, h, progress_meter=1<<14):
    """Determines where h is equivalent to any matrix in _representatives.
     
    Args:
        h (np.array_like): a matrix

    """
    global _representatives
    global _lock
    i = 0
    while True:
        with _lock:
            if i == len(_representatives) // (order * order):
                push_matrix(_representatives, h)
                if progress_meter:
                    print("!", end='')
                    sys.stdout.flush()
                return
        r = read_matrix(_representatives, i, order)
        if equivalent(h, r, monomial_generator, progress_meter=progress_meter):
            if progress_meter:
                print("!", end='')
                sys.stdout.flush()
            return
        else:
            i += 1


if __name__ == '__main__':
    import doctest
    doctest.testmod(optionflags=doctest.NORMALIZE_WHITESPACE)
