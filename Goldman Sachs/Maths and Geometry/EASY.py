
# 4 Judge Robots
from typing import List


def judgeCircle(self, moves: str) -> bool:
    x = y = 0  # to represent the initial position of the robots
    for move in moves:
        if move == "U":
            y += 1
        elif move == "D":
            y -= 1
        elif move == "R":
            x += 1
        else:
            x -= 1
    return x == y == 0


# Pascals Triangle II - **
def getRow(self, rowIndex: int) -> List[int]:
    # so the first approach we can use here is to simple generate r number of rows for the pascal triangle
    # and then return the last row in our list as the answer -- rutime O(n^2)
    # but that can be optimized using some basic knowledge of maths and combinations
    # Here are few things to note
    # The kth row in a pascal triangle will have k + 1 numbers
    # to get a value at a row or column in a pascal triangle can be done using combination
    # value at nth row and kth column = nCk = n!/(n-k)!k! == n*(n-1)*...(n-k+1)/k!
    # e.g 5C4 == 5*4*3*2/1*2*3*4
    # so basically we can initiliase the numerator to the row number given and denominator to 1
    # and then basically decrease the numerator and increase the numerator as we try to figure the value at each column
    # of the rth row
    # use the link below for a better explanation maybe
    # https://leetcode.com/problems/pascals-triangle-ii/solutions/1203260/very-easy-o-n-time-0-ms-beats-100-simple-maths-all-languages/
    row = [1] * (rowIndex + 1)
    up, down = rowIndex, 1
    for i in range(1, rowIndex):
        row[i] = int(row[i - 1] * up / down)
        up -= 1
        down += 1
    return row