from multiprocessing import Array, Pool
from ctypes import Structure, c_int
from itertools import repeat

import G
from sharedlist import SharedStack


def procinit(ss):
    global s
    s = ss

class Foo(Structure):
    _fields_ = [('a', c_int)]

def horse(a):
    global s
    s.push(Foo(a))
    print(len(s))

if __name__ == '__main__':
    s = SharedStack(Foo)
    with Pool(initializer=procinit, initargs=(s,)) as p:
        p.map(horse, range(10))
    print(len(s))
    for f in s:
        print(f.a)

