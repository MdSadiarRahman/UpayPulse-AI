import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Paths
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
MODEL_PATH = os.path.join(MODELS_DIR, "liquidity_model.pkl")

class LiquidityPredictor:
    def __init__(self):
        self.model = None
        self.feature_cols = [
            "current_cash_balance", 
            "e_float_balance", 
            "avg_hourly_cashout", 
            "daily_txn_count", 
            "peak_hour_vol", 
            "prev_cashout_trend"
        ]

    def load_data(self):
        """Loads and prepares agent and transaction data."""
        agents_df = pd.read_csv(os.path.join(DATA_DIR, "agents.csv"))
        txns_df = pd.read_csv(os.path.join(DATA_DIR, "transactions.csv"))
        
        # Convert timestamp to datetime
        txns_df["timestamp"] = pd.to_datetime(txns_df["timestamp"])
        txns_df["hour"] = txns_df["timestamp"].dt.hour
        txns_df["date"] = txns_df["timestamp"].dt.date
        
        # Extract Cash Out transactions
        cashout_df = txns_df[txns_df["transaction_type"] == "cash_out"]
        
        # Feature Engineering per agent
        features = []
        for _, agent in agents_df.iterrows():
            aid = agent["agent_id"]
            agent_txns = txns_df[txns_df["agent_id"] == aid]
            agent_cashouts = cashout_df[cashout_df["agent_id"] == aid]
            
            # daily transaction count
            daily_txn_count = len(agent_txns) / agent_txns["date"].nunique() if agent_txns["date"].nunique() > 0 else 0
            
            # avg hourly cash-out
            avg_hourly_cashout = agent_cashouts["amount"].sum() / (agent_cashouts["date"].nunique() * 24) if agent_cashouts["date"].nunique() > 0 else 0
            
            # peak hour vol
            if len(agent_cashouts) > 0:
                hourly_vol = agent_cashouts.groupby("hour")["amount"].sum()
                peak_hour_vol = hourly_vol.max()
            else:
                peak_hour_vol = 0
                
            # prev cashout trend (total cashout amount)
            prev_cashout_trend = agent_cashouts["amount"].sum()
            
            # Target logic (synthetic labels based on cash vs demand)
            # High risk if cash < (peak_hour_vol * 1.5) or cash < 20000
            if agent["cash_balance"] < max(peak_hour_vol * 1.5, 20000):
                risk = "HIGH"
            elif agent["cash_balance"] < max(peak_hour_vol * 3.0, 50000):
                risk = "MEDIUM"
            else:
                risk = "LOW"
                
            features.append({
                "agent_id": aid,
                "current_cash_balance": agent["cash_balance"],
                "e_float_balance": agent["e_float_balance"],
                "avg_hourly_cashout": avg_hourly_cashout,
                "daily_txn_count": daily_txn_count,
                "peak_hour_vol": peak_hour_vol,
                "prev_cashout_trend": prev_cashout_trend,
                "shortage_risk": risk
            })
            
        return pd.DataFrame(features)

    def train_evaluate(self):
        """Trains models, compares them, and saves the best one."""
        print("Loading and preprocessing data...")
        df = self.load_data()
        
        # Mapping target
        target_map = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
        y = df["shortage_risk"].map(target_map)
        X = df[self.feature_cols]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        print("Training Random Forest...")
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        rf_preds = rf.predict(X_test)
        
        print("Training XGBoost...")
        xgb = XGBClassifier(eval_metric="mlogloss", random_state=42)
        xgb.fit(X_train, y_train)
        xgb_preds = xgb.predict(X_test)
        
        # Evaluate
        models = {"Random Forest": rf_preds, "XGBoost": xgb_preds}
        best_f1 = -1
        best_model = None
        best_model_name = ""
        
        for name, preds in models.items():
            acc = accuracy_score(y_test, preds)
            prec = precision_score(y_test, preds, average='weighted', zero_division=0)
            rec = recall_score(y_test, preds, average='weighted', zero_division=0)
            f1 = f1_score(y_test, preds, average='weighted', zero_division=0)
            
            print(f"\\n[{name} Evaluation]")
            print(f"Accuracy:  {acc:.4f}")
            print(f"Precision: {prec:.4f}")
            print(f"Recall:    {rec:.4f}")
            print(f"F1-Score:  {f1:.4f}")
            
            if f1 > best_f1:
                best_f1 = f1
                best_model = rf if name == "Random Forest" else xgb
                best_model_name = name
                
        print(f"\\nSaving Best Model: {best_model_name}")
        os.makedirs(MODELS_DIR, exist_ok=True)
        with open(MODEL_PATH, "wb") as f:
            pickle.dump({"model": best_model, "features": self.feature_cols}, f)
            
        self.model = best_model
        return best_model_name

    def load_model(self):
        """Loads the pre-trained model."""
        if os.path.exists(MODEL_PATH):
            with open(MODEL_PATH, "rb") as f:
                data = pickle.load(f)
                self.model = data["model"]
        else:
            print("Model not found. Training new model...")
            self.train_evaluate()

    def predict_risk(self, agent_id: str):
        """Predicts liquidity risk for a specific agent and provides explanations."""
        if self.model is None:
            self.load_model()
            
        df = self.load_data()
        agent_data = df[df["agent_id"] == agent_id]
        
        if agent_data.empty:
            return {"error": "Agent not found"}
            
        X = agent_data[self.feature_cols]
        
        # Predict probability
        probs = self.model.predict_proba(X)[0]
        # Target map: {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
        pred_class = np.argmax(probs)
        risk_levels = {0: "LOW", 1: "MEDIUM", 2: "HIGH"}
        risk_level = risk_levels[pred_class]
        risk_probability = float(probs[pred_class])
        
        cash = agent_data.iloc[0]["current_cash_balance"]
        peak = agent_data.iloc[0]["peak_hour_vol"]
        
        expected_shortage = max(0, float(peak * 1.5 - cash))
        
        # Explanations
        reasons = []
        if cash < 30000:
            reasons.append("Current physical cash balance is critically low")
        if agent_data.iloc[0]["prev_cashout_trend"] > 100000:
            reasons.append("Historical cash-out withdrawal pattern is very high")
        if peak > 15000:
            reasons.append("Peak hour cash-out demand is severely increasing")
            
        if not reasons:
            reasons.append("Stable liquidity balance compared to historical demand")
            
        # Calculate SHAP explainability
        shap_reasons = []
        try:
            import shap
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(X)
            
            if isinstance(shap_values, list):
                sv = shap_values[pred_class][0]
            else:
                sv = shap_values[0]
                if len(sv.shape) > 1: # if multiclass in single array
                    sv = sv[:, pred_class]
                
            total_impact = np.sum(np.abs(sv))
            if total_impact > 0:
                feature_names = {
                    "current_cash_balance": "Current cash balance",
                    "e_float_balance": "e-Float balance",
                    "avg_hourly_cashout": "Avg hourly cash-out",
                    "daily_txn_count": "Daily transactions",
                    "peak_hour_vol": "Peak hour",
                    "prev_cashout_trend": "Cash-out demand"
                }
                
                impacts = []
                for i, val in enumerate(sv):
                    pct = (abs(val) / total_impact) * 100
                    if pct >= 5: 
                        fname = feature_names.get(self.feature_cols[i], self.feature_cols[i])
                        sign = "+" if val > 0 else "-"
                        impacts.append((fname, pct, sign))
                        
                impacts.sort(key=lambda x: x[1], reverse=True)
                for imp in impacts[:4]: # top 4 features
                    shap_reasons.append(f"{imp[0]}: {imp[2]}{int(imp[1])}%")
        except Exception as e:
            pass

        return {
            "agent_id": agent_id,
            "risk_level": risk_level,
            "risk_probability": round(risk_probability, 2),
            "expected_shortage_amount": round(expected_shortage, 2),
            "main_reasons": reasons,
            "shap_reasons": shap_reasons
        }

if __name__ == "__main__":
    # Test script functionality
    predictor = LiquidityPredictor()
    predictor.train_evaluate()
    
    # Test prediction
    print("\\nTesting Prediction:")
    res = predictor.predict_risk("AGT-0005") # Using sample agent
    print(res)
