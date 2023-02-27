from typing import List


# 1 Cooperate Flight Bookings
def corpFlightBookings(self, bookings: List[List[int]], n: int) -> List[int]:
    prefixSum = [0] * n
    for booking in bookings:
        first, last, seats = booking
        prefixSum[first - 1] += seats
        if last < n:
            prefixSum[last] -= seats

    for i in range(1, n):
        prefixSum[i] += prefixSum[i - 1]
    return prefixSum
