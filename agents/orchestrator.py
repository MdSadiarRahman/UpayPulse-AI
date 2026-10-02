import sys
import os

# Ensure we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.liquidity_predictor import LiquidityPredictor
from src.merchant_engine import MerchantGrowthEngine
from src.customer_offer import CustomerOfferEngine

class OrchestratorAgent:
    """
    AI Orchestrator Agent for UpayPulse AI.
    
    Receives user queries and intelligently routes them to the correct 
    specialized agent (Risk, Customer, or Merchant) engine.
    """

    def __init__(self):
        self.risk_engine = LiquidityPredictor()
        self.merchant_engine = MerchantGrowthEngine()
        self.customer_engine = CustomerOfferEngine()
        
    def determine_intent(self, query: str) -> str:
        """
        Simple intent detection based on keywords. 
        """
        query_lower = query.lower()
        
        # Risk / Liquidity Intent
        if any(word in query_lower for word in ["cash", "shortage", "risk", "rebalance", "liquidity", "agent", "টাকা কম", "টাকা", "ঝুঁকি"]):
            return "risk"
            
        # Merchant Growth Intent
        elif any(word in query_lower for word in ["merchant", "increase sales", "business", "growth", "মার্চেন্ট", "বিক্রি", "shop", "দোকান"]):
            return "merchant"
            
        # Customer Assistance Intent
        elif any(word in query_lower for word in ["offer", "discount", "where", "buy", "customer", "অফার", "ডিসকাউন্ট"]):
            return "customer"
            
        else:
            return "unknown"

    def handle_query(self, query: str, lang: str = "en") -> str:
        import re
        intent = self.determine_intent(query)
        
        # Check if an agent ID was explicitly mentioned (e.g. AGT-0005)
        agent_match = re.search(r'\bAGT-\d{4}\b', query, re.IGNORECASE)
        target_agent = agent_match.group(0).upper() if agent_match else "AGT-0435"

        # Check if a merchant ID was explicitly mentioned (e.g. MRC-0001)
        merchant_match = re.search(r'\bMRC-\d{4}\b', query, re.IGNORECASE)
        target_merchant = merchant_match.group(0).upper() if merchant_match else "MRC-0001"
        
        if intent == "risk":
            try:
                res = self.risk_engine.predict_risk(target_agent)
                if "error" in res:
                    return f"❌ Agent `{target_agent}` not found. Please provide a valid agent ID between AGT-0001 and AGT-0500."
                prob = int(res['risk_probability'] * 100)
                level = res.get('risk_level', 'UNKNOWN')
                shortage = res.get('expected_shortage_amount', 0)
                reasons = res.get('main_reasons', [])
                reasons_str = "\n".join([f"- {r}" for r in reasons])
                
                badge_color = "🔴" if level == "HIGH" else "🟡" if level == "MEDIUM" else "🟢"
                
                if lang == "bn":
                    return f"""### 🛡️ রাউটেড টু: লিকুইডিটি রিস্ক এজেন্ট

**এজেন্ট কোড:** `{target_agent}` | **ঝুঁকির স্তর:** {badge_color} **{level}**

- **ঝুঁকির সম্ভাবনা:** `{prob}%`
- **প্রত্যাশিত ঘাটতি:** `৳{shortage:,.2f}`
- **মূল কারণসমূহ:**
{reasons_str}

> 💡 **সুপারিশ:** অবিলম্বে কাছাকাছি কোনো উদ্বৃত্ত তারল্য থাকা এজেন্ট থেকে রি-ব্যালান্সিং সহায়তা নিন।"""
                else:
                    return f"""### 🛡️ Routed to: Liquidity Risk Agent

**Agent ID:** `{target_agent}` | **Risk Level:** {badge_color} **{level}**

- **Default Risk Probability:** `{prob}%`
- **Estimated Cash Shortage:** `৳{shortage:,.2f}`
- **Key Risk Drivers:**
{reasons_str}

> 💡 **Actionable Recommendation:** Initiate urgent peer-to-peer liquidity transfer from top nearby surplus partner."""
            except Exception as e:
                return f"Risk Engine Error: {str(e)}"
            
        elif intent == "merchant":
            try:
                res = self.merchant_engine.generate_recommendation(target_merchant)
                if "error" in res:
                    return f"❌ Merchant `{target_merchant}` not found. Please provide a valid merchant ID between MRC-0001 and MRC-1000."
                rec = res['recommendation']
                cat = res.get('category', 'Retail')
                
                if lang == "bn":
                    return f"""### 🏪 রাউটেড টু: মার্চেন্ট গ্রোথ এজেন্ট

**মার্চেন্ট কোড:** `{target_merchant}` (`{cat}`)

- **প্রস্তাবিত ক্যাম্পেইন:** **{rec['offer']}**
- **সেরা সময়:** 🕒 `{rec['best_time']}`
- **টার্গেট গ্রাহক:** 🎯 `{rec['target_segment']}`
- **যৌক্তিকতা:** {rec['reason']}
- **প্রত্যাশিত প্রবৃদ্ধি:** 📈 **{rec['expected_impact']}**

> 🚀 *এই অফারটি কার্যকর করলে ডিজিটাল লেনদেন ও গ্রাহক আনুগত্য বৃদ্ধি পাবে।*"""
                else:
                    return f"""### 🏪 Routed to: Merchant Growth Agent

**Merchant ID:** `{target_merchant}` (`{cat}`)

- **Recommended Offer:** **{rec['offer']}**
- **Optimal Launch Window:** 🕒 `{rec['best_time']}`
- **Target Audience:** 🎯 `{rec['target_segment']}`
- **Strategic Rationale:** {rec['reason']}
- **Expected Lift:** 📈 **{rec['expected_impact']}**

> 🚀 *Deploying this micro-incentive converts cash shoppers into recurrent digital Upay users.*"""
            except Exception as e:
                return f"Merchant Engine Error: {str(e)}"
            
        elif intent == "customer":
            try:
                # Detect category preference from query if any
                preferred_cat = None
                for cat_key in ["pharmacy", "medicine", "restaurant", "food", "grocery", "supermarket"]:
                    if cat_key in query.lower():
                        preferred_cat = cat_key.capitalize()
                        break

                res = self.customer_engine.generate_offer("CUST-LIVE", 23.79, 90.41, preferred_category=preferred_cat)
                if "error" in res:
                    return res["error"]
                rec = res['recommended_merchant']
                
                if lang == "bn":
                    return f"""### 🎁 রাউটেড টু: কাস্টমার রিওয়ার্ড এজেন্ট

**নিকটবর্তী মার্চেন্ট:** `{rec['merchant_id']}` (`{rec['category']}`)
**দূরত্ব:** 📍 `{rec['distance']}`

- **এক্সক্লুসিভ অফার:** **{rec['offer']}**
- **অফারের কারণ:** {rec['reason']}
- **গ্রাহকের সুবিধা:** {rec.get('expected_benefit', 'উচ্চ সাশ্রয়')}

> 📱 *কাছের এই মার্চেন্টে উপায়ের মাধ্যমে পেমেন্ট করলেই সরাসরি এই ছাড় প্রযোজ্য হবে!*"""
                else:
                    return f"""### 🎁 Routed to: Customer Reward Agent

**Nearest Merchant Partner:** `{rec['merchant_id']}` (`{rec['category']}`)
**Distance:** 📍 `{rec['distance']}`

- **Exclusive Reward:** **{rec['offer']}**
- **Target Rationale:** {rec['reason']}
- **Customer Advantage:** {rec.get('expected_benefit', 'Max cashback & savings')}

> 📱 *Pay via Upay QR at this verified merchant outlet to instantly unlock your reward!*"""
            except Exception as e:
                return f"Customer Engine Error: {str(e)}"
            
        else:
            if lang == "bn":
                return """👋 আমি উপায় পালস এআই অর্কেস্ট্রেটর! আমি আপনাকে নিচের বিষয়গুলোতে সাহায্য করতে পারি:
- **এজেন্ট ক্যাশ সংকট:** যেমন *"AGT-0012 এর লিকুইডিটি কেমন?"* বা *"কোন এজেন্টের টাকা প্রয়োজন?"*
- **মার্চেন্ট সেলস বৃদ্ধি:** যেমন *"MRC-0005 এর জন্য প্রমোশন অফার দাও"*
- **গ্রাহক অফার:** যেমন *"কাছে কোনো রেস্তোরাঁ বা ফার্মেসি অফার আছে?"*"""
            else:
                return """👋 I am the UpayPulse AI Orchestrator! I can intelligently assist you with:
- **Agent Liquidity Deficit:** e.g., *"What is the risk of AGT-0012?"* or *"Which agent needs cash?"*
- **Merchant Revenue Growth:** e.g., *"Suggest a promotion for MRC-0005"*
- **Hyper-Local Customer Offers:** e.g., *"Find nearby restaurant or pharmacy offers"*"""


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding='utf-8')
    orchestrator = OrchestratorAgent()
    queries = [
        ("Which agent needs cash?", "en"),
        ("How can my shop increase sales?", "en"),
        ("Where can I get discount?", "en"),
        ("কোন এজেন্টের টাকা কম?", "bn")
    ]
    for q, l in queries:
        print(f"User Question: \"{q}\"")
        print("-" * 40)
        print(orchestrator.handle_query(q, lang=l))
        print("=" * 60, "\n")
