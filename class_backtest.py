import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd 
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.stattools import adfuller
import numpy as np
import matplotlib.pyplot as plt   
import methodes as mt
from dateutil.relativedelta import relativedelta
from tqdm import tqdm
import pickle #pour la sérialisation du dictionnaire
from collections import Counter


class paire_trading : 
    """Represents a single pair trading strategy."""

    def __init__(self, paire, allocation, df, date_debut, date_fin, frais=False) : 
        """Initialize pair trading with allocation percentage and date range."""
        self.paire = paire
        self.allocation = allocation  # Percentage of total budget
        self.df = df
        self.date_debut_data = pd.to_datetime(date_debut) - pd.DateOffset(years=1)
        self.date_debut = pd.to_datetime(date_debut)
        self.date_fin = pd.to_datetime(date_fin)
        self.frais = frais

    def get_beta(self, paire, date_start, date_end) : 
        """Calculate beta coefficient for pair over given period."""
        try : 
            df_slice = self.df.loc[date_start:date_end, "Close"][
            [paire.action1, paire.action2]
            ]

            # Remove rows with NaN values
            df_slice = df_slice.dropna()

            X = df_slice[paire.action1].values.reshape(-1, 1)
            y = df_slice[paire.action2].values

            model = LinearRegression().fit(X, y)
            beta = model.coef_[0]
            return beta
        except Exception as e : 
            print(paire)
            raise e

    def sold_position(self, df, date_buy, date_sold, pos_action1, pos_action2) :    
        """Calculate portfolio value when closing position at sold date."""
        if date_buy == None : 
            return 0
        value_action1_buy = df["Close"][self.paire.action1].loc[date_buy]
        value_action1_sell = df["Close"][self.paire.action1].loc[date_sold]
        
        value_action2_buy = df["Close"][self.paire.action2].loc[date_buy]
        value_action2_sell = df["Close"][self.paire.action2].loc[date_sold]

        # Absolute position value (initial) + variation (depends on position direction)
        value_action1 = value_action1_buy*abs(pos_action1) + (value_action1_sell-value_action1_buy) *pos_action1
        value_action2 = value_action2_buy*abs(pos_action2) + (value_action2_sell-value_action2_buy) *pos_action2

        return value_action1+value_action2

    def calc_pnl_sample(self, date_start, date_end) : 
        """Calculate in-sample PnL statistics (mean, win rate, worst, max drawdown)."""
        date_start = pd.to_datetime(date_start)
        date_end = pd.to_datetime(date_end)

        all_dates = pd.date_range(date_start, date_end, freq='MS').strftime('%Y-%m-%d').tolist()  

        series_pnl = []
        list_maxdrawdown = []
        for i in range(len(all_dates)-1) : 
            date_debut_periode = all_dates[i]
            date_fin_periode = all_dates[i+1]
            date_debut_spread = pd.to_datetime(date_debut_periode) - relativedelta(months=6)
            beta_month = self.get_beta(self.paire, date_debut_spread, date_debut_periode)

            # Calculate weights based on volatility
            action1_yields = self.df[self.df.index < date_debut_periode]["Close"][self.paire.action1].tail(100).pct_change().dropna().abs()
            ponderation_action1 = 1 / (action1_yields.std())

            # Action2 yield of last 100 days
            action2_yields = self.df[self.df.index < date_debut_periode]["Close"][self.paire.action2].tail(100).pct_change().dropna().abs()
            ponderation_action2 = abs(beta_month) / (action2_yields.std())

            ponderation_tot = ponderation_action1 + ponderation_action2 

            # Normalize weights
            weight_action1 = ponderation_action1 / ponderation_tot
            weight_action2 = ponderation_action2 / ponderation_tot

            df_tempo = self.df[(self.df.index >= date_debut_spread) & (self.df.index < date_fin_periode)]
            spread = ponderation_action1* df_tempo["Close"][self.paire.action1] - ponderation_action2*df_tempo["Close"][self.paire.action2]
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
                pnl_action1 =  position * weight_action1 * (self.df["Close"][self.paire.action1].loc[idx] - prev_value_action1)
                pnl_action2 = position * weight_action2 * (self.df["Close"][self.paire.action2].loc[idx] - prev_value_action2)
                pnl_sample.loc[idx] = pnl_action1 - pnl_action2

                if z_score.loc[idx] > seuil and position == 0:
                    position = -1  # Short spread
                elif z_score.loc[idx] < -seuil and position == 0:
                    position = 1   # Long spread
                elif abs(z_score.loc[idx]) < 0.5 and position != 0:
                    position = 0   # Close position
                prev_value_action1 = self.df["Close"][self.paire.action1].loc[idx]
                prev_value_action2 = self.df["Close"][self.paire.action2].loc[idx]


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

    def calc_trade_month(self, row) : 
        """Calculate trades for the month using z-score mean reversion strategy."""
        # Calculate spread over 6 months
        df_tempo = self.df[(self.df.index >= self.date_debut_data) & (self.df.index < self.date_fin)]
        
        spread = row["weight_action1"]*df_tempo["Close"][self.paire.action1] - row["weight_action2"] * df_tempo["Close"][self.paire.action2]

        mean_day_spread = spread.rolling(window=20).mean()
        std_day_spread = spread.rolling(window=20).std()
        
        # Calculate z-score for the month
        z_score = (spread - mean_day_spread) / std_day_spread

        # Iterate through month and execute trades
        serie_value_ptf = pd.Series()
        cash_value = self.allocation
        date_buy = None 
        position = 0
        seuil = 1.5
        qte_action1 = 0
        qte_action2 = 0
        nb_trades = 0
        fee = 0.0002

        for idx in spread[(spread.index >= self.date_debut) & (spread.index < self.date_fin)].index:
            
            # Calculate portfolio values
            value_action1 = df_tempo["Close"][self.paire.action1].loc[idx]
            value_action2 = df_tempo["Close"][self.paire.action2].loc[idx]

            # Calculate daily PnL
            value_fictif_ptf = self.sold_position(df_tempo, date_buy, idx, qte_action1, qte_action2)
            serie_value_ptf.loc[idx] = cash_value + value_fictif_ptf
            
            # Open position
            if z_score.loc[idx] > seuil and position == 0:
                # Short spread
                nb_trades += 1
                date_buy = idx
                if self.frais : # Reduce value for transaction fees
                    cash_value = cash_value*(1-fee)

                qte_action1 = - cash_value* row["weight_action1"] / value_action1
                qte_action2 = cash_value*row["weight_action2"]  / value_action2 
                cash_value = 0 # Reset cash as fully invested
                position = -1 # Short the spread 

            elif z_score.loc[idx] < -seuil and position == 0:
                # Long spread
                nb_trades += 1
                date_buy = idx
                if self.frais : # Reduce value for transaction fees
                    cash_value = cash_value*(1-fee)
                
                qte_action1 = cash_value* row["weight_action1"] / value_action1
                qte_action2 = - cash_value*row["weight_action2"]  / value_action2   
                cash_value = 0 # Reset cash as fully invested
                position = 1 # Long the spread 

            elif abs(z_score.loc[idx]) < 0.5 and position != 0:
                # Close position
                nb_trades += 1

                cash_value = self.sold_position(df_tempo, date_buy, idx, qte_action1, qte_action2) 
                if self.frais : # Reduce value for transaction fees
                    cash_value = cash_value*(1-fee)
                qte_action1 = 0
                qte_action2 = 0
                date_buy = None 
                position = 0  

        if position != 0 : # If still in position, close at month end
            nb_trades += 1 
        # Return portfolio value series and trade count
        return serie_value_ptf, nb_trades

class secteur_trading : 
    """Groups all pairs within a sector and calculates best pairs for trading."""
    
    def __init__(self, secteur, liste_paires, date_debut_trading, date_fin_trading) : 
        """Initialize sector trading with pairs and date range."""
        self.secteur = secteur
        self.liste_paires = liste_paires
        self.tickers = set([p.action1 for p in liste_paires] + [p.action2 for p in liste_paires])

        self.date_debut_trading = pd.to_datetime(date_debut_trading)
        self.date_debut_data = self.date_debut_trading - pd.DateOffset(years=1)
        self.date_fin_trading = pd.to_datetime(date_fin_trading)

        self.df = None 

    def initDf(self) : 
        """Download price data for all tickers from Yahoo Finance."""
        self.df = yf.download(list(self.tickers), start=self.date_debut_data, end=self.date_fin_trading)

    def get_paire_liquide(self, date, addv_threshold=20e6):
        """Return pairs where both assets have ADDV above threshold over past 60 days."""
        date = pd.to_datetime(date)
        date_debut = date - timedelta(days=60)

        df_periode = self.df.loc[date_debut:date]

        # Calculate ADDV per asset
        addv_action = {}
        for ticker in self.tickers:
            close = df_periode['Close'][ticker].dropna()
            volume = df_periode['Volume'][ticker].dropna()

            if len(close) >= 30:
                addv = (close * volume).mean()
                if addv >= addv_threshold:
                    addv_action[ticker] = addv

        # volume filter for each pair
        paires_liquides = []
        for paire in self.liste_paires:
            a, b = paire.action1, paire.action2

            if a in addv_action and b in addv_action:
                # contrôle pair-level (sécurité)
                pair_addv = min(addv_action[a], addv_action[b])
                if pair_addv >= addv_threshold:
                    paires_liquides.append((paire, pair_addv))

        return paires_liquides
        
    def get_beta(self, paire, date_start, date_end) : 
        """Calculate beta coefficient for pair over given period."""
        try : 
            df_slice = self.df.loc[date_start:date_end, "Close"][
            [paire.action1, paire.action2]
            ]

            # Remove rows with NaN values
            df_slice = df_slice.dropna()

            X = df_slice[paire.action1].values.reshape(-1, 1)
            y = df_slice[paire.action2].values

            model = LinearRegression().fit(X, y)
            beta = model.coef_[0]
            return beta
        except Exception as e : 
            print(paire)
            raise e

    def calc_pnl_sample(self, paire, date_start, date_end) : 
        """Calculate in-sample PnL statistics (mean, win rate, worst, max drawdown)."""
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

            # Calculate weights based on volatility
            action1_yields = self.df[self.df.index < date_debut_periode]["Close"][paire.action1].tail(100).pct_change().dropna().abs()
            ponderation_action1 = 1 / (action1_yields.std())

            # Action2 yield of last 100 days
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
 
    def get_eligible_paire(self, date) : 
        """Find and rank best pairs for a given date based on cointegration, profitability metrics."""
        
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

                # Action2 yield of last 100 days
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


class BackTest : 
    """Backtesting engine for multi-sector pair trading strategy."""

    def __init__(self, df_snp):
        """Initialize backtest with SNP data containing sector information."""
        self.df_snp = df_snp
        self.all_secteurs = df_snp["GICS Sector"].unique().tolist()

        self.df_value_ptf = pd.DataFrame(columns=["Portfolio Value"])

        self.date_debut_trading=None 
        self.date_fin_trading = None 
        self.all_secteurs_trading=None
        self.budget_total = 0


    def init_dict_paire_secteur(self, df_snp_periode_trading) :
        """Generate all possible pairs for each sector."""
        dict_secteur_paires = {}

        for secteur in self.all_secteurs : 
            df_secteur = df_snp_periode_trading[df_snp_periode_trading["GICS Sector"]==secteur]
            liste_action = df_secteur["Symbol"].tolist()
            dict_secteur_paires[secteur] = []
            for i in range(len(liste_action)) : 
                for j in range(i+1, len(liste_action)) : 
                    paire = mt.Paire(secteur, liste_action[i], liste_action[j])
                    dict_secteur_paires[secteur].append(paire)
                
        return dict_secteur_paires

    def order_df(self, df_all_lignes, dico_actions_m_1, dico_actions_m_2) :
        """Rank pairs by weighted composite score and filter by volume and past usage."""

        # Normalize all columns
        df_normaliser = df_all_lignes.copy()
        df_normaliser["diff_weight"] = abs(df_normaliser["weight_action1"]-df_normaliser["weight_action2"])
        
        ranking_cols = ["avg_pnl", "half_life","diff_weight", "avg_max_drowdown"]
        abs_score = ["volatility"]

        for col in ranking_cols + abs_score :
            df_normaliser[f"zscore_{col}"] = (df_normaliser[col] - df_normaliser[col].mean())  / df_normaliser[col].std()

        df_normaliser["zscore_avg_pnl"] = df_normaliser["zscore_avg_pnl"].clip(upper=1)

        for col in abs_score :
            df_normaliser[f"zscore_{col}"] = df_normaliser[f"zscore_{col}"].abs()

        dico_ponderation = {
            "diff_weight" : 0.125,
            "volatility" : -0.234344,
            'avg_pnl' : 0.042294,
            'avg_max_drowdown' : -0.55767,
            "half_life": 0.04101
        }

        df_normaliser["score_final"] = (
            dico_ponderation["diff_weight"] * df_normaliser["zscore_diff_weight"] + dico_ponderation["volatility"] * df_normaliser["zscore_volatility"] + dico_ponderation["avg_pnl"] * df_normaliser["zscore_avg_pnl"] + 
            dico_ponderation["avg_max_drowdown"] * df_normaliser["zscore_avg_max_drowdown"] + dico_ponderation["half_life"] * df_normaliser["zscore_half_life"]
        )

        # Filter by top 20% volume
        volume_threshold = df_normaliser["volume"].quantile(0.2)
        df_normaliser = df_normaliser[df_normaliser["volume"] >= volume_threshold]

        # Adjust score by asset usage in past months
        df_normaliser["pair_cost_m_1"] = df_normaliser.index.map(
        lambda paire: dico_actions_m_1.get(paire.action1, 0) + dico_actions_m_1.get(paire.action2, 0)
    )

        df_normaliser["pair_cost_m_2"] = df_normaliser.index.map(
            lambda paire: dico_actions_m_2.get(paire.action1, 0) + dico_actions_m_2.get(paire.action2, 0)
        )

        df_normaliser["score_final"] = df_normaliser["score_final"] / (1 + 0.5 * df_normaliser["pair_cost_m_1"] + 0.3 * df_normaliser["pair_cost_m_2"])

        df_sort = df_normaliser.sort_values("score_final", ascending=False)
        return df_sort

    def count_actions_in_paires(self, liste_paires):
        """Count occurrences of each asset in selected pairs."""
        actions = []
        for paire in liste_paires:
            actions.extend([paire.action1, paire.action2])
        
        return dict(Counter(actions))

    def get_best_paire(self, df_month_ordered,nb_paires=15, aloc_secteur=0.4) : 
        """Select top pairs respecting constraints: max 2 per asset, allocation per sector."""
        selected_paires = []
        action_count = {}
        secteur_count = {}
        total_paires = 0
        max_paire = 2
        for paire, row in df_month_ordered.iterrows():
            secteur = paire.secteur
            action1 = paire.action1
            action2 = paire.action2
            
            if action_count.get(action1, 0) < max_paire and action_count.get(action2, 0) < max_paire :
                if secteur_count.get(secteur, 0) < aloc_secteur * nb_paires:
                    selected_paires.append(paire)
                    action_count[action1] = action_count.get(action1, 0) + 1
                    action_count[action2] = action_count.get(action2, 0) + 1
                    secteur_count[secteur] = secteur_count.get(secteur, 0) + 1
                    total_paires += 1

            if total_paires >= nb_paires:
                break
        return selected_paires

    def main(self, date_debut_trading, date_fin_trading, budget_total, interet_compose=False, fee = False) : 
        """Execute main backtest loop: select pairs monthly and calculate portfolio performance."""
        df_wallet = pd.DataFrame(columns=["date_debut", "date_fin", "value"])
        df_res = pd.DataFrame(columns=["date_debut", "date_fin", "paire", "z_score", "poids_init", "pnl", "nb_trade"])
        dico_actions_m_1 = {}
        dico_actions_m_2 = {}

        date_debut_trading = pd.to_datetime(date_debut_trading)
        date_fin_trading = pd.to_datetime(date_fin_trading)

        # Filter pairs available on entire period (in SNP since start date)
        df_snp_periode_trading = self.df_snp[pd.to_datetime(self.df_snp["Date added"]) < pd.to_datetime(date_debut_trading)]
        
        # Initialize pairs for each sector
        dict_paire_secteur = self.init_dict_paire_secteur(df_snp_periode_trading)

        # Initialize sector trading objects
        all_secteurs_trading = {secteur: secteur_trading(secteur, dict_paire_secteur[secteur], date_debut_trading, date_fin_trading) for secteur in dict_paire_secteur.keys()}
        
        # Download price data for each sector
        for secteur_trading_class in all_secteurs_trading.values() : 
            secteur_trading_class.initDf()

        date_monthly = pd.date_range(date_debut_trading, date_fin_trading, freq='MS').strftime('%Y-%m-%d').tolist() 

        for i in tqdm(range(len(date_monthly)-1), desc="dates") : # Iterate over each monthly period
            date_start = date_monthly[i]
            date_end = date_monthly[i+1]

            # Get eligible pairs for each sector
            secteur_month_results = {secteur : all_secteurs_trading[secteur].get_eligible_paire(date_start) for secteur in all_secteurs_trading.keys()}
            df_month_all_lignes = pd.concat(secteur_month_results.values(), axis=0)

            # Rank and filter pairs
            df_month_ordered = self.order_df(df_month_all_lignes, dico_actions_m_1, dico_actions_m_2)

            df_month_ordered = df_month_ordered[:20] # Keep only top 20 to limit losses

            # Select best pairs for the month
            best_paires = self.get_best_paire(df_month_ordered,  nb_paires=15, aloc_secteur=0.4)

            dico_actions_m_2 = dico_actions_m_1
            dico_actions_m_1 = self.count_actions_in_paires(best_paires)
            

            # Allocate budget equally among pairs
            poids_paire = budget_total / len(best_paires)

            ptf_fin_periode = 0 # Portfolio value at end of period
            for paire in best_paires : 
                paire_trade_tempo = paire_trading(paire, poids_paire, all_secteurs_trading[paire.secteur].df, date_start, date_end, frais=fee)
                serie_ptf, nb_trades = paire_trade_tempo.calc_trade_month(df_month_all_lignes.loc[paire])
                ptf_fin_periode += serie_ptf.iloc[-1] # Add portfolio value for each pair
                df_res.loc[len(df_res)] = [date_start, date_end, paire, df_month_ordered.loc[paire, "score_final"], poids_paire, serie_ptf.iloc[-1],  nb_trades]
            
            df_wallet.loc[len(df_wallet)] = [date_start, date_end, ptf_fin_periode]
            if interet_compose : # If compound interest, reinvest portfolio value
                budget_total = ptf_fin_periode
        return df_res, df_wallet
