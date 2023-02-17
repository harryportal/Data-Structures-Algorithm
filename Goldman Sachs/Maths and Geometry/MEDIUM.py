# 41 Robots bounded in a circle  -- Maths and Geometry
def isRobotBounded(self, instructions: str) -> bool:
    """The basic algorithm of this problem is that the robot will be bounded in a circle if either the position after
    going through the movements once is unchanged or if there is a change in direction after going through the movements
    To change the direction of a given cordinate by 90 degrees, we multiply the coordinate by rotation matrix...
    For clockwise, we multiply it by matrix [[0,1],[-1,0]] and for anti clockwise, we multiply it by [[0,-1],[1,0]"""

    dirX, dirY = 0, 1  # it initialy faces north
    x, y = 0, 0
    for movement in instructions:
        if movement == "G":
            x, y = x + dirX, y + dirY
        elif movement == "L":
            dirX, dirY = -1 * dirY, dirX
        else:
            dirX, dirY = dirY, -1 * dirX
    return (x, y) == (0, 0) or (dirX, dirY) != (0, 1)



# 30 String to Integer Atoi  -- Maths and Geomtery
def myAtoi(self, s: str) -> int:
    index, sign, n = 0, 1, len(s)
    result, maxInt, minInt = 0, pow(2, 31) - 1, -pow(2, 31)

    # let's ignore leading whitespaces
    while index < n and s[index] == " ":
        index += 1

    # let's check if it has a negative sign
    if index < n and s[index] == "-":
        sign = -1
        index += 1
    elif index < n and s[index] == "+":
        index += 1

    # keep getting the digits until a non digit is met
    while index < n and s[index].isdigit():
        # check for overflow and underflow
        value = int(s[index])
        if result > maxInt // 10 or (result == maxInt // 10 and value > maxInt % 10):
            return maxInt if sign == 1 else minInt

        # if the code get's here then there would be no overflow after adding the current value
        result = result * 10 + value
        index += 1
    return sign * result


# 11 Fraction to Recurring Decimal -- Maths and Geometry
def fractionToDecimal(self, numerator: int, denominator: int) -> str:
    """this is a actually very tricky problem but basically test our knowledge on how to code up mathematical
    division logic
    I'll try to break it down and explain each logic"""
    if numerator == 0: return "0"  # edge case

    # first take care of negative integers whether the numerator or denominator or both
    prefix = ""  # should be negative or empty(positive)
    if (numerator < 0 and denominator > 0) or (numerator > 0 and denominator < 0):
        prefix = "-"

    # make the numerators postive(this also covers the case where they are both negative
    numerator, denominator = abs(numerator), abs(denominator)

    # sort of simulate the division logic
    # One key thing to note here is that in a recurring decimal will occur whenever a remainder repeat itself
    decimals, remainders = [], []
    while True:
        decimal = numerator // denominator
        remainder = numerator % denominator
        decimals.append(str(decimal))
        remainders.append(remainder)

        numerator %= denominator
        numerator *= 10

        if numerator == 0:  # then there was no recurring decimal
            if len(decimals) == 1:
                return prefix + decimals[0]
            else:
                return prefix + decimals[0] + "." + str(decimals[1:])

        # if the remainder we just got has already occured before, then the fraction has a reccuring decimal
        if remainders.count(remainder) > 1:
            decimals.insert(remainders.index(remainder) + 1, "(")
            decimals.append(")")
            return prefix + decimals[0] + "." + str(decimals[1:])
