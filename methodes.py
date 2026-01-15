import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd 
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.stattools import adfuller
import numpy as np
from dateutil.relativedelta import relativedelta


class Paire : 
    """Represents a trading pair with two assets in a sector."""

    def __init__(self, secteur, action1, action2):
        self.secteur = secteur
        self.action1 = action1
        self.action2 = action2 

    def __eq__(self, other):
        """Check equality: pairs are equal if they contain the same assets regardless of order."""
        if isinstance(other, Paire):
            return (self.action1 == other.action1 and self.action2 == other.action2) or (self.action1 == other.action2 and self.action2 == other.action1)
        elif isinstance(other, tuple) and len(other) == 2:
            return (self.action1 == other[0] and self.action2 == other[1]) or (self.action1 == other[1] and self.action2 == other[0])
        return False

    def in_list(self, liste_action) : 
        """Check if either asset is in the given list."""
        return (self.action1 in liste_action) or (self.action2 in liste_action)

    def __hash__(self):
        """Make pair hashable for use in sets and dicts."""
        return hash((self.action1, self.action2))

    def __repr__(self):
        return f"Paire({self.action1}, {self.action2})"

    def __str__(self):
        return f"Paire({self.action1}, {self.action2}) dans le secteur {self.secteur}"

class SectorAnalyse : 
    """Groups sector data (dataframe, pairs, etc.) and calculates best pairs for given dates."""

    def __init__(self, secteur, all_paires, date_debut_data, date_fin_trading):
        self.secteur = secteur
        self.all_paires = all_paires
        self.date_debut_data = pd.to_datetime(date_debut_data)
        self.date_fin_trading = pd.to_datetime(date_fin_trading)
        self.tickers = set([p.action1 for p in all_paires] + [p.action2 for p in all_paires])

        self.df = None 
    
    def initDf(self) : 
        """Download price data for all tickers from Yahoo Finance."""
        self.df = yf.download(list(self.tickers), start=self.date_debut_data, end=self.date_fin_trading)
 
    def get_paire_liquide(self, date, addv_threshold=20e6):
        """
        Keep only pairs where both assets have an Average Daily Dollar Volume (ADDV) above the specified threshold over the past 60 days.
        """
        date = pd.to_datetime(date)
        date_debut = date - timedelta(days=60)

        df_periode = self.df.loc[date_debut:date]

        # ADDV per action
        addv_action = {}
        for ticker in self.tickers:
            close = df_periode['Close'][ticker].dropna()
            volume = df_periode['Volume'][ticker].dropna()

            if len(close) >= 30:
                addv = (close * volume).mean()
                if addv >= addv_threshold:
                    addv_action[ticker] = addv

        # Filter pairs based on ADDV of both assets
        paires_liquides = []
        for paire in self.all_paires:
            a, b = paire.action1, paire.action2

            if a in addv_action and b in addv_action:
                # contrôle pair-level (sécurité)
                pair_addv = min(addv_action[a], addv_action[b])
                if pair_addv >= addv_threshold:
                    paires_liquides.append((paire, pair_addv))

        return paires_liquides


    def get_spread(self, df, paire) : 
        """Calculate spread using linear regression between two assets."""
        X = df[paire.action1].values.reshape(-1, 1)
        y = df[paire.action2].values
        model = LinearRegression().fit(X, y)
        beta = model.coef_[0]
        spread = df[paire.action2] - beta * df[paire.action1]
        return spread

    def get_beta(self, paire, date_start, date_end) : 
        """Compute beta coefficient of the pair over a given period."""
        X = self.df[(self.df.index >= date_start) & (self.df.index < date_end)]["Close"][paire.action1].values.reshape(-1, 1)
        y = self.df[(self.df.index >= date_start) & (self.df.index < date_end)]["Close"][paire.action2].values
        model = LinearRegression().fit(X, y)
        beta = model.coef_[0]
        return beta

    def calc_pnl_sample(self, paire, date_start, date_end) : 
        """Calculate in-sample PnL, win rate, worst trade, and max drawdown for the pair."""
        date_start = pd.to_datetime(date_start)
        date_end = pd.to_datetime(date_end)

        all_dates = pd.date_range(date_start, date_end, freq='MS').strftime('%Y-%m-%d').tolist()  

        series_pnl = []
        list_maxdrawdown = []
        for i in range(len(all_dates)-1) : 
            date_debut_periode = all_dates[i]
            date_fin_periode = all_dates[i+1]
            date_debut_spread = pd.to_datetime(date_debut_periode) - relativedelta(months=6)
            beta_month = self.get_beta(paire, date_debut_spread, date_debut_periode)

            # Calculate weights based on volatility of both assets
            action1_yields = self.df[self.df.index < date_debut_periode]["Close"][paire.action1].tail(100).pct_change().dropna().abs()
            ponderation_action1 = 1 / (action1_yields.std())

            #Action2 yield des 100 derniers jours 
            action2_yields = self.df[self.df.index < date_debut_periode]["Close"][paire.action2].tail(100).pct_change().dropna().abs()
            ponderation_action2 = abs(beta_month) / (action2_yields.std())

            ponderation_tot = ponderation_action1 + ponderation_action2 

            # Normalize weights
            weight_action1 = ponderation_action1 / ponderation_tot
            weight_action2 = ponderation_action2 / ponderation_tot

            df_tempo = self.df[(self.df.index >= date_debut_spread) & (self.df.index < date_fin_periode)]
            spread = ponderation_action1* df_tempo["Close"][paire.action1] - ponderation_action2*df_tempo["Close"][paire.action2]
            mean_day_spread = spread.rolling(window=20).mean()
            std_day_spread = spread.rolling(window=20).std()
            z_score = (spread - mean_day_spread) / std_day_spread


            prev_value_action1 = 0
            prev_value_action2 = 0
            pnl_sample = pd.Series() 
            position = 0 
            seuil = 1.5
            # Iterate through all trading dates
            for idx in spread.index[(spread.index >= date_debut_periode) & (spread.index < date_fin_periode)]:
                pnl_action1 =  position * weight_action1 * (self.df["Close"][paire.action1].loc[idx] - prev_value_action1)
                pnl_action2 = position * weight_action2 * (self.df["Close"][paire.action2].loc[idx] - prev_value_action2)
                pnl_sample.loc[idx] = pnl_action1 - pnl_action2

                if z_score.loc[idx] > seuil and position == 0:
                    position = -1  # Short spread
                elif z_score.loc[idx] < -seuil and position == 0:
                    position = 1   # Long spread
                elif abs(z_score.loc[idx]) < 0.5 and position != 0:
                    position = 0   # Close position
                prev_value_action1 = self.df["Close"][paire.action1].loc[idx]
                prev_value_action2 = self.df["Close"][paire.action2].loc[idx]


            series_pnl.append(pnl_sample)

            # Calculate max drawdown
            equity_curve = pnl_sample.cumsum()
            running_max = equity_curve.cummax()
            drawdown = equity_curve - running_max
            list_maxdrawdown.append(drawdown.min())
        
        sum_pnls = [sum(i) for i in series_pnl]
        nb_pos = len(list(filter(lambda l:l>0, sum_pnls)))
        worst = min(sum_pnls)
        return np.mean(sum_pnls), (nb_pos/len(sum_pnls)), worst, np.mean(list_maxdrawdown)
     
    def get_best_paire(self, date) : 
        """Return a dataframe with all the active pairs of the month and their characteristics."""
        
        df_paire = pd.DataFrame(columns=["weight_action1", "weight_action2", "volume", "p_value", "half_life", "volatility", 
                                         "avg_pnl", "prop_pnl_pos", "worst_pnl", "avg_max_drowdown"]) 

        date = pd.to_datetime(date)

        more_liquide_paires = self.get_paire_liquide(date)

        date_debut = date - pd.DateOffset(years=1)
        # Use 1-year of data if available, otherwise use all available data
        if pd.to_datetime(self.df.index.min()) > date_debut:
            df_date = self.df.loc[:date]["Close"]
        else:
            df_date = self.df.loc[date_debut:date]["Close"]

        for paire, vol in more_liquide_paires:
            # Check if data is available for both assets
            condition = (
    not pd.isna(df_date[paire.action1].iloc[0]) and 
    not pd.isna(df_date[paire.action2].iloc[0])
)
            if condition : 
                # Calculate beta
                beta_year = self.get_beta(paire, date_debut, date)

                # Calculate weights based on volatility
                action1_yields = df_date[paire.action1].tail(100).pct_change().dropna().abs()
                ponderation_action1 = 1 / (action1_yields.std())

                action2_yields = df_date[paire.action2].tail(100).pct_change().dropna().abs()
                ponderation_action2 = abs(beta_year) / (action2_yields.std())

                ponderation_tot = ponderation_action1 + ponderation_action2 

                # Normalize weights
                weight_action1 = ponderation_action1 / ponderation_tot
                weight_action2 = ponderation_action2 / ponderation_tot
                
                # Calculate spread
                spread = ponderation_action1 * df_date[paire.action1] - ponderation_action2*df_date[paire.action2]

                # Test for cointegration using ADF test
                adf_result = adfuller(spread.dropna())
                if adf_result[1] <= 0.05 : 
                    # Calculate half-life of mean reversion
                    spread_lag = spread.shift(1).dropna()
                    spread_ret = spread.diff().dropna()
                    spread_lag = spread_lag.loc[spread_ret.index]
                    model = LinearRegression().fit(spread_lag.values.reshape(-1, 1), spread_ret.values)
                    half_life = -np.log(2) / model.coef_[0] if model.coef_[0] != 0 else np.inf

                    # Calculate in-sample PnL
                    months = (date.year - date_debut.year) * 12 + (date.month - date_debut.month)
                    half = months // 2 
                    date_mid = date_debut + relativedelta(months=+half)
                    date_mid = date_mid.replace(day=1) # First day of month between both dates

                    avg_pnl, prop_pnl_pos, worst_pnl, avg_max_drowdown = self.calc_pnl_sample(paire, date_mid, date)
                    df_paire.loc[paire] = [weight_action1, weight_action2, vol,adf_result[1], half_life, spread.std(),avg_pnl, prop_pnl_pos,
                                        worst_pnl, avg_max_drowdown ]

        return df_paire  

    def get_backtest_value(self, df_paires, date_start, date_end) :
        """Calculate real backtest PnL for given pairs and period."""
        date_start = pd.to_datetime(date_start)
        date_end = pd.to_datetime(date_end)
        real_debut_data = date_start - timedelta(days=60)

        df_tempo = self.df[(self.df.index >= real_debut_data) & (self.df.index < date_end)]["Close"]
        
        serie_pnl = pd.Series()
        serie_pnl.name = "pnl_month"
        for paire, row in df_paires.iterrows() : # Calculate PnL for each pair
            spread = row["weight_action1"]*df_tempo[paire.action1] - row["weight_action2"]*df_tempo[paire.action2]
            mean_day_spread = spread.rolling(window=20).mean()
            std_day_spread = spread.rolling(window=20).std()
            z_score = (spread - mean_day_spread) / std_day_spread


            prev_value_action1 = 0
            prev_value_action2 = 0
            pnl_sample = pd.Series() 
            position = 0 
            seuil = 1.5
            # Iterate through all trading dates
            for idx in spread.index[(spread.index >= date_start) & (spread.index < date_end)]:
                pnl_action1 =  position * row["weight_action1"] * (df_tempo[paire.action1].loc[idx] - prev_value_action1)
                pnl_acttion2 = position * row["weight_action2"] * (df_tempo[paire.action2].loc[idx] - prev_value_action2)
                
                pnl_sample.loc[idx] = pnl_action1 - pnl_acttion2
                if z_score.loc[idx] > seuil and position == 0:
                    position = -1  # Short spread
                elif z_score.loc[idx] < -seuil and position == 0:
                    position = 1   # Long spread
                elif abs(z_score.loc[idx]) < 0.5 and position != 0:
                    position = 0   # Close position
                prev_value_action1 = df_tempo[paire.action1].loc[idx]
                prev_value_action2 = df_tempo[paire.action2].loc[idx]

            
            serie_pnl.loc[paire] = sum(pnl_sample) 
        return serie_pnl

    def __str__(self):
        return f"Secteur: {self.secteur}, Nombre de paires: {len(self.all_paires)}\nNombre de tickers: {len(self.tickers)}"
