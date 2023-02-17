# 51 Pow(x,n) -- Recursion
def myPow(self, x: float, n: int) -> float:
    # we can solve this recursively by breaking it down to sub problem
    if x == 0:  # edge case
        return 0

    def power(x, n):
        if n == 0:  # base case
            return 1
        value = power(x * x, n // 2)
        return value * x if n % 2 else value  # check for odd powers

    result = power(x, abs(n))
    return result if n >= 0 else 1 / result



# 90  Print Zero Even Odd -- Threading
from threading import Lock


class ZeroEvenOdd:
    def __init__(self, n: int):
        self.n = n
        self.curr = 1
        self.zero_lock = Lock()
        self.even_lock = Lock()
        self.odd_lock = Lock()
        self.even_lock.acquire()
        self.odd_lock.acquire()

    def zero(self, printNumber: 'Callable[[int], None]') -> None:
        for i in range(self.n):
            self.zero_lock.acquire()
            printNumber(0)
            if i % 2 == 0:
                self.odd_lock.release()
            else:
                self.even_lock.release()

    def even(self, printNumber: 'Callable[[int], None]') -> None:
        for i in range(self.n // 2):
            self.even_lock.acquire()
            printNumber(self.curr)
            self.curr += 1
            self.zero_lock.release()

    def odd(self, printNumber: 'Callable[[int], None]') -> None:
        for i in range((self.n + 1) // 2):
            self.odd_lock.acquire()
            printNumber(self.curr)
            self.curr += 1
            self.zero_lock.release()
