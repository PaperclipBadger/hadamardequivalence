from multiprocessing import Array, Value, Condition
from ctypes import c_uint64, c_byte, pointer, POINTER, Structure

class StackLimitReached(Exception):
    pass

class StackEmpty(Exception):
    pass


class RWLock():
    """A Readers-Writer lock.
    
    Allows for multiple readers or one writer. Writers will not starve.

    Attributes:
        for_reading (RWLock.ReadLock): A lock-like object with appropriate
            `acquire`, `release`, `__enter__` and `__exit__` methods pointed
            to the *read methods of the RWLock. Chiefly for use with the 
            `with` statement.
        for_writing (RWLock.WriteLock): A lock-like object with appropriate
            `acquire`, `release`, `__enter__` and `__exit__` methods pointed
            to the *write methods of the RWLock. Chiefly for use with the 
            `with` statement.

    """
    class ReadLock():
        def __init__(self, rw):
            self._rw = rw
            self.acquire = rw.acquire_read
            self.release = rw.release_read
        def __enter__(self):
            self.acquire()
        def __exit__(self, exception_type, exception_value, traceback):
            self.release()

    class WriteLock():
        def __init__(self, rw):
            self._rw = rw
            self.acquire = rw.acquire_write
            self.release = rw.release_write
        def __enter__(self):
            self.acquire()
        def __exit__(self, exception_type, exception_value, traceback):
            self.release()

    def __init__(self):
        """Initialises the RWLock."""
        self._condition = Condition()
        self._readers = Value(c_uint64, 0, lock=False)
        self._writers_waiting = Value(c_uint64, 0, lock=False)

        self.for_reading = self.ReadLock(self)
        self.for_writing = self.WriteLock(self)

    def acquire_read(self):
        """Acquire a read lock. 
        
        Blocks if a thread has acquired the write lock or is waiting to
        acquire the write lock.
        
        """
        with self._condition:
            while self._writers_waiting.value:
                self._condition.wait()
            self._readers.value += 1

    def release_read(self):
        """Release a read lock."""
        with self._condition:
            self._readers.value -= 1
            if not self._readers.value:
                self._condition.notify_all()

    def acquire_write(self):
        """Acquire a write lock.
        
        Blocks until there are no acquired read or write locks.
        
        """
        self._condition.acquire()
        self._writers_waiting.value += 1
        while self._readers.value:
            self._condition.wait()
        self._writers_waiting.value -= 1

    def release_write(self):
        """Release a write lock."""
        self._condition.release()


class SharedStack():
    """A data race free limited stack in shared memory.

    Supports multiple readers or one writer. Expands as items are added.
    
    """
    def __init__(self, element_type, limit=1024):
        self.element_type = element_type
        self.limit = limit
        self.lock = RWLock()
        self.length = Value(c_uint64, 0)
        self.array = Array(element_type, limit, lock=False)
    
    def __len__(self):
        with self.lock.for_reading:
            return self.length.value

    def __getitem__(self, index):
        with self.lock.for_reading:
            if not (0 <= index and index < self.length.value):
                raise IndexError("SharedStack index out of range")
            return self.array[index]

    def __setitem__(self, index, value):
        with self.lock.for_writing:
            if not (0 <= index and index < self.length.value):
                raise IndexError("SharedStack index out of range")
            self.array[index] = value

    def push(self, value):
        with self.lock.for_writing:
            if self.length.value == self.limit:
                raise StackLimitReached()
            self.array[self.length.value] = value
            self.length.value += 1

    def pop(self, value):
        with self.lock.for_writing:
            if self.length.value == 0:
                raise StackEmpty()
            self.length.value -= 1
            return self.array[self.length.value]

    def peek(self, value):
        with self.lock.for_reading:
            if self.length.value == 0:
                raise StackEmpty()
            return self.array[self.length.value - 1]
