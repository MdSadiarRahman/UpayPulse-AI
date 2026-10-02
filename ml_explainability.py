import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import shap

class RiskPredictorWithSHAP:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=50, random_state=42)
        self.explainer = None
        self.features = ['current_cash', 'predicted_demand_4h', 'e_float_balance', 'rating']

    def train_and_explain(self, agents_df: pd.DataFrame):
        """
        Trains a Random Forest model on the agent data and computes SHAP values.
        Target: expected_shortage (or risk probability proxy)
        """
        # Ensure all required features exist
        df = agents_df.copy()
        
        # We model expected_shortage to show how it depends on cash and demand
        # (This allows SHAP to discover the underlying formula rules naturally)
        if 'current_cash' not in df.columns and 'cash_balance' in df.columns:
            df['current_cash'] = df['cash_balance']
        if 'predicted_demand_4h' not in df.columns:
            np.random.seed(42)
            df['predicted_demand_4h'] = np.random.randint(10000, 90000, len(df)).astype(float)
            
        y = np.maximum(0, df['predicted_demand_4h'] - df['current_cash'])
        X = df[self.features]
        
        # Train model
        self.model.fit(X, y)
        
        # Compute SHAP values
        self.explainer = shap.TreeExplainer(self.model)
        shap_values = self.explainer.shap_values(X)
        
        return X, shap_values, self.explainer.expected_value

    def get_agent_explanation(self, X, shap_values, agent_index):
        """
        Extracts top features driving the risk for a specific agent.
        """
        sv = shap_values[agent_index]
        feature_vals = X.iloc[agent_index]
        
        # Create a DataFrame of feature impact
        impact_df = pd.DataFrame({
            'Feature': self.features,
            'Value': feature_vals.values,
            'SHAP_Value': sv
        })
        
        # Sort by absolute impact
        impact_df['Abs_Impact'] = impact_df['SHAP_Value'].abs()
        impact_df = impact_df.sort_values(by='Abs_Impact', ascending=False)
        return impact_df
