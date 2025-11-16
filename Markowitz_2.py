"""
Package Import
"""
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import quantstats as qs
import gurobipy as gp
import warnings
import argparse
import sys

"""
Project Setup
"""
warnings.simplefilter(action="ignore", category=FutureWarning)

assets = [
    "SPY",
    "XLB",
    "XLC",
    "XLE",
    "XLF",
    "XLI",
    "XLK",
    "XLP",
    "XLRE",
    "XLU",
    "XLV",
    "XLY",
]

# Initialize Bdf and df
Bdf = pd.DataFrame()
for asset in assets:
    raw = yf.download(asset, start="2012-01-01", end="2024-04-01", auto_adjust = False)
    Bdf[asset] = raw['Adj Close']

df = Bdf.loc["2019-01-01":"2024-04-01"]

"""
Strategy Creation

Create your own strategy, you can add parameter but please remain "price" and "exclude" unchanged
"""


class MyPortfolio:
    """
    NOTE: You can modify the initialization function
    """

    def __init__(self, price, exclude, lookback=50, gamma=0):
        self.price = price
        self.returns = price.pct_change().fillna(0)
        self.exclude = exclude
        self.lookback = lookback
        self.gamma = gamma

    def calculate_weights(self):
        # Get the assets by excluding the specified column
        assets = self.price.columns[self.price.columns != self.exclude]

        # Calculate the portfolio weights
        self.portfolio_weights = pd.DataFrame(
            index=self.price.index, columns=self.price.columns
        )

        """
        TODO: Complete Task 4 Below
        """
        
        # --- 策略配置參數 ---
        # 9個月SMA濾網 (約200個交易日) [4]
        SMA_LOOKBACK = 200
        # 12個月價格動量 (約252個交易日) [3]
        MOMENTUM_LOOKBACK = 252 
        # 選擇動量最強的前 K 個部門 [5]
        TOP_K = 5
        
        # 獲取非 SPY 的 11 個行業部門資產
        assets = self.price.columns[self.price.columns!= self.exclude]
        m = len(assets)
        
        # 1. 計算市場趨勢濾網 (Market Trend Filter)
        # 使用 SPY 的價格和 200 日簡單移動平均線
        spy_prices = self.price[self.exclude]
        spy_sma = spy_prices.rolling(window=SMA_LOOKBACK).mean()
        
        # 2. 計算部門動量 (Sector Momentum)
        # 動量 = (當前價格 / 252 日前價格) - 1
        price_past = self.price[assets].shift(MOMENTUM_LOOKBACK)
        momentum = (self.price[assets] / price_past) - 1
        
        # 3. 逐日進行決策和權重分配 (每日再平衡)
        start_index = max(MOMENTUM_LOOKBACK, SMA_LOOKBACK)
        
        # 確保 self.portfolio_weights 中 SPY 欄位始終為 0
        self.portfolio_weights[self.exclude] = 0

        for i in range(start_index, len(self.price)):
            current_date = self.price.index[i]
            
            # 初始化當日權重向量
            weights = np.zeros(m)
            
            # --- 決策開始 ---
            
            # 獲取當日 SPY 價格與 SMA
            current_spy_price = spy_prices.loc[current_date]
            current_spy_sma = spy_sma.loc[current_date]

            # 市場趨勢濾網判斷: 如果 SPY 高於其 200 日 SMA，則視為牛市 [4]
            if current_spy_price > current_spy_sma:
                # === 牛市 (Bullish) 策略：動量輪動 ===
                
                # 獲取當日部門動量
                current_momentum = momentum.loc[current_date]
                
                # 排除動量值為 NaN 的情況（應在 start_index 之後，但增加魯棒性）
                current_momentum.dropna(inplace=True) 

                if not current_momentum.empty:
                    # 依動量降序排名
                    ranked_sectors = current_momentum.sort_values(ascending=False)
                    
                    # 選取動量最強的前 K 個部門 [4]
                    top_k_sectors = ranked_sectors.head(TOP_K).index.tolist()
                    
                    # 分配等權重給選中的 K 個部門
                    equal_weight = 1.0 / TOP_K
                    
                    # 寫入權重
                    for sector in top_k_sectors:
                        idx = assets.get_loc(sector) # 找到該部門在資產列表中的位置
                        weights[idx] = equal_weight
                        
            else:
                # === 熊市/盤整 (Bearish/Sideways) 策略：轉向現金 ===
                # 保持 weights 向量為全零 (即將資金分配為現金/防禦性資產)
                # 由於資產權重總和必須為 1，但我們不持有任何資產，這在評分系統中通常被視為持有現金。
                pass # weights 已經初始化為 np.zeros(m)
            
            # --- 寫入當日權重 ---
            self.portfolio_weights.loc[current_date, assets] = weights


        """
        TODO: Complete Task 4 Above
        """


        self.portfolio_weights.ffill(inplace=True)
        self.portfolio_weights.fillna(0, inplace=True)

    def calculate_portfolio_returns(self):
        # Ensure weights are calculated
        if not hasattr(self, "portfolio_weights"):
            self.calculate_weights()

        # Calculate the portfolio returns
        self.portfolio_returns = self.returns.copy()
        assets = self.price.columns[self.price.columns != self.exclude]
        self.portfolio_returns["Portfolio"] = (
            self.portfolio_returns[assets]
            .mul(self.portfolio_weights[assets])
            .sum(axis=1)
        )

    def get_results(self):
        # Ensure portfolio returns are calculated
        if not hasattr(self, "portfolio_returns"):
            self.calculate_portfolio_returns()

        return self.portfolio_weights, self.portfolio_returns


if __name__ == "__main__":
    # Import grading system (protected file in GitHub Classroom)
    from grader_2 import AssignmentJudge
    
    parser = argparse.ArgumentParser(
        description="Introduction to Fintech Assignment 3 Part 12"
    )

    parser.add_argument(
        "--score",
        action="append",
        help="Score for assignment",
    )

    parser.add_argument(
        "--allocation",
        action="append",
        help="Allocation for asset",
    )

    parser.add_argument(
        "--performance",
        action="append",
        help="Performance for portfolio",
    )

    parser.add_argument(
        "--report", action="append", help="Report for evaluation metric"
    )

    parser.add_argument(
        "--cumulative", action="append", help="Cumulative product result"
    )

    args = parser.parse_args()

    judge = AssignmentJudge()
    
    # All grading logic is protected in grader_2.py
    judge.run_grading(args)
    