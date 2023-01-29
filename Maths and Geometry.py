from typing import List


# Set Matrix Zeroes  -- Constant Time
def setZeroes(self, matrix: List[List[int]]) -> None:
    # figure out row and column to be set to zeroes
    rows, cols = len(matrix), len(matrix[0])
    rowZero = False
    for row in range(rows):
        for col in range(cols):
            if matrix[row][col] == 0:
                matrix[0][col] = 0
                if row > 0:
                    matrix[row][0] = 0
                else:
                    rowZero = True
    # set the rows and except the first one to zero
    for row in range(1, rows):
        for col in range(1, cols):
            if matrix[0][col] == 0 or matrix[row][0] == 0:
                matrix[row][col] = 0

    # zero out the first column in the matrix if the first element in the matrix is zero
    if matrix[0][0] == 0:
        for row in range(rows):
            matrix[row][0] = 0

    if rowZero:
        for col in range(cols):
            matrix[0][col] = 0


# Pow(x,n) -- recursion
def myPow(self, x: float, n: int) -> float:
    def power(n, x):
        if x == 0:
            return 1
        if n == 0: return 0
        value = power(n * n, x // 2)
        return value * n if x % 2 else value

    result = power(x, abs(n))
    return result if n >= 0 else 1 / result


def obtainMax(number: int, value):
    numberString = [int(i) for i in str(number)]
    result = ""
    for i in numberString:
        if value:
            pass


# Happy Number
def isHappy(self, n: int) -> bool:
    # we keep finding the sum of squares of the number using a looo
    # our exit point will be when we see a number square sum repeated (cos this will cause an infinite
    # loop) or when we actually get a 1
    # checking for a repition can be done with linked list cycle

    def sumOfSquares(n):  # helper function to find the sum of sqaure
        output = 0
        while n:
            output += (n % 10) ** 2
            n //= 10
        return output

    # checking for a repition can be done with a hashset(0(n) memory) or with fast and slow pointers(o(1) memory)
    slow, fast = n, sumOfSquares(n)
    while slow != fast:
        slow = sumOfSquares(slow)
        fast = sumOfSquares(sumOfSquares(fast))

    return fast == 1


# Plus One
def plusOne(self, digits: List[int]) -> List[int]:
    # we reverse the string first
    digits.reverse()
    index = 0
    while True:
        if index < len(digits):
            if digits[index] == 9:
                digits[index] = 0
            else:
                digits[index] += 1
                break
        else:
            digits.append(1)
            break
        index += 1
    return digits[::-1]
