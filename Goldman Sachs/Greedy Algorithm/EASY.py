
# 5 Best Time to Buy and Sell Stock
def maxProfit(self, prices: List[int]) -> int:
    buy_price = prices[0]
    maxProfit = 0
    for i in range(1, len(prices)):
        sell_price = prices[i]
        if sell_price < buy_price:
            buy_price = sell_price
        else:
            maxProfit = max(maxProfit, sell_price - buy_price)
    return maxProfit
