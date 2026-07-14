# -*- coding: utf-8 -*-
import logging
logger = logging.getLogger(__name__)

CROP_PROFILES = {
    "Rice":        {"temp":(20,35),"humidity":(70,90),"rainfall":(150,300),"ph":(5.5,7.0)},
    "Wheat":       {"temp":(10,25),"humidity":(50,75),"rainfall":(75,150), "ph":(6.0,7.5)},
    "Maize":       {"temp":(18,35),"humidity":(50,75),"rainfall":(50,150), "ph":(5.8,7.0)},
    "ChickPea":    {"temp":(10,25),"humidity":(40,70),"rainfall":(30,100), "ph":(6.0,8.0)},
    "KidneyBeans": {"temp":(18,28),"humidity":(40,70),"rainfall":(40,100), "ph":(6.0,7.5)},
    "PigeonPeas":  {"temp":(18,35),"humidity":(40,70),"rainfall":(40,100), "ph":(5.5,7.0)},
    "MothBeans":   {"temp":(25,40),"humidity":(30,65),"rainfall":(25,75),  "ph":(6.0,8.0)},
    "MungBean":    {"temp":(20,40),"humidity":(50,80),"rainfall":(40,100), "ph":(6.0,7.5)},
    "Blackgram":   {"temp":(25,40),"humidity":(55,80),"rainfall":(40,100), "ph":(5.5,7.0)},
    "Lentil":      {"temp":(10,25),"humidity":(40,65),"rainfall":(25,75),  "ph":(6.0,8.0)},
    "Sugarcane":   {"temp":(20,35),"humidity":(70,90),"rainfall":(100,250),"ph":(6.0,7.5)},
    "Cotton":      {"temp":(20,40),"humidity":(40,75),"rainfall":(50,150), "ph":(5.8,8.0)},
    "Jute":        {"temp":(24,40),"humidity":(60,90),"rainfall":(100,250),"ph":(6.0,7.5)},
    "Groundnut":   {"temp":(22,35),"humidity":(50,75),"rainfall":(50,150), "ph":(6.0,7.5)},
    "Soybean":     {"temp":(20,35),"humidity":(50,75),"rainfall":(60,150), "ph":(6.0,7.5)},
    "Sunflower":   {"temp":(18,35),"humidity":(45,70),"rainfall":(50,120), "ph":(6.0,7.5)},
    "Mustard":     {"temp":(10,25),"humidity":(40,70),"rainfall":(40,100), "ph":(6.0,7.5)},
    "Turmeric":    {"temp":(20,35),"humidity":(70,90),"rainfall":(150,300),"ph":(5.5,7.0)},
    "Ginger":      {"temp":(18,30),"humidity":(70,90),"rainfall":(120,250),"ph":(5.5,6.5)},
    "Banana":      {"temp":(20,35),"humidity":(60,85),"rainfall":(100,200),"ph":(5.5,7.0)},
    "Mango":       {"temp":(24,40),"humidity":(40,75),"rainfall":(75,200), "ph":(5.5,7.5)},
    "Coconut":     {"temp":(25,40),"humidity":(60,90),"rainfall":(100,250),"ph":(5.0,8.0)},
    "Grapes":      {"temp":(15,35),"humidity":(40,65),"rainfall":(25,75),  "ph":(5.5,7.0)},
    "Apple":       {"temp":(5,25), "humidity":(50,75),"rainfall":(50,150), "ph":(5.5,6.5)},
    "Tomato":      {"temp":(18,30),"humidity":(50,75),"rainfall":(50,120), "ph":(6.0,7.0)},
    "Potato":      {"temp":(10,25),"humidity":(60,80),"rainfall":(100,200),"ph":(5.0,6.5)},
    "Onion":       {"temp":(15,30),"humidity":(50,75),"rainfall":(50,120), "ph":(6.0,7.5)},
}

MARKET_PRICE_INR = {
    "Rice":1850,"Wheat":1975,"Maize":1870,"ChickPea":5230,"KidneyBeans":3500,
    "PigeonPeas":6300,"MothBeans":3900,"MungBean":7755,"Blackgram":6950,
    "Lentil":5500,"Sugarcane":290,"Cotton":6080,"Jute":4500,"Groundnut":5850,
    "Soybean":3880,"Sunflower":5441,"Mustard":5450,"Turmeric":7000,"Ginger":5500,
    "Banana":1200,"Mango":3000,"Coconut":1500,"Grapes":4000,"Apple":5000,
    "Tomato":2500,"Potato":1000,"Onion":2000,
}

PLANTING_CALENDAR = {
    "Rice":      {"sow":"June-July",        "harvest":"November-December","season":"Kharif"},
    "Wheat":     {"sow":"October-November", "harvest":"March-April",      "season":"Rabi"},
    "Maize":     {"sow":"June-July",        "harvest":"September-October","season":"Kharif"},
    "ChickPea":  {"sow":"October-November", "harvest":"February-March",   "season":"Rabi"},
    "Mustard":   {"sow":"September-October","harvest":"February-March",   "season":"Rabi"},
    "Cotton":    {"sow":"April-May",        "harvest":"November-January", "season":"Kharif"},
    "Sugarcane": {"sow":"October-March",    "harvest":"October-March",    "season":"Annual"},
    "Mango":     {"sow":"June-August",      "harvest":"April-June",       "season":"Annual"},
    "Banana":    {"sow":"June-July",        "harvest":"Year-round",       "season":"Annual"},
    "Potato":    {"sow":"October-November", "harvest":"January-March",    "season":"Rabi"},
    "Tomato":    {"sow":"June-July",        "harvest":"September-November","season":"Kharif"},
    "Onion":     {"sow":"October-November", "harvest":"February-April",   "season":"Rabi"},
    "Turmeric":  {"sow":"May-June",         "harvest":"January-March",    "season":"Kharif"},
    "Ginger":    {"sow":"April-May",        "harvest":"December-January", "season":"Kharif"},
    "Groundnut": {"sow":"June-July",        "harvest":"October-November", "season":"Kharif"},
    "Soybean":   {"sow":"June-July",        "harvest":"October-November", "season":"Kharif"},
    "Sunflower": {"sow":"January-February", "harvest":"April-May",        "season":"Rabi"},
}


class RecommendationSystem:
    def __init__(self, config=None):
        cfg = config or {}
        self.yield_threshold  = cfg.get("high_yield_threshold",  2000)
        self.price_threshold  = cfg.get("market_price_threshold",3000)

    def generate_recommendation(self, yield_pred, market_price):
        h_y = yield_pred    > self.yield_threshold
        h_p = market_price  > self.price_threshold
        if h_y and h_p:  return "Excellent - high yield and high price. Expand production and sell immediately."
        if h_y:          return "High yield expected - store crops for a better market window."
        if h_p:          return "Market price is high - sell existing stocks now."
        return "Average conditions - store crops and monitor market trends."

    def crop_suitability(self, crop, conditions):
        profile = CROP_PROFILES.get(crop)
        if not profile:
            return {"crop":crop,"score":0.0,"advice":["No agronomic profile available."]}
        scores, advice = [], []
        checks = [("temp","temperature","Temperature"),
                  ("humidity","humidity","Humidity"),
                  ("rainfall","rainfall","Rainfall"),
                  ("ph","ph_value","pH")]
        for key, cond_key, label in checks:
            val = conditions.get(cond_key)
            if val is None: continue
            lo, hi = profile[key]
            mid = (lo+hi)/2
            if lo <= val <= hi:
                scores.append(1.0 - abs(val-mid)/((hi-lo)/2+1e-5)*0.3)
            else:
                dev = min(abs(val-lo), abs(val-hi))
                scores.append(max(0.0, 1.0 - dev/(hi-lo+1e-5)))
                direction = "too low" if val < lo else "too high"
                advice.append(f"{label} is {direction} for {crop} (ideal: {lo}-{hi}).")
        score = round(sum(scores)/max(len(scores),1)*100, 1)
        if not advice:
            advice.append(f"Conditions are well-suited for {crop}.")
        return {"crop":crop,"score":score,"advice":advice,
                "market_price":MARKET_PRICE_INR.get(crop,0)}

    def rank_crops(self, conditions, top_n=5):
        results = [self.crop_suitability(c,conditions) for c in CROP_PROFILES]
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_n]

    def ml_recommendation(self, top_predictions, conditions):
        enhanced = []
        for pred in top_predictions:
            crop    = pred["crop"]
            ml_conf = pred["confidence"]
            rule    = self.crop_suitability(crop, conditions)
            blended = round(ml_conf*0.6 + rule["score"]/100*0.4, 4)
            enhanced.append({"crop":crop,"ml_confidence":ml_conf,
                              "rule_score":rule["score"],"blended_score":blended,
                              "advice":rule["advice"],
                              "market_price":MARKET_PRICE_INR.get(crop,0)})
        enhanced.sort(key=lambda x: x["blended_score"], reverse=True)
        return {"recommendations":enhanced,"best_crop":enhanced[0]["crop"]}

    @staticmethod
    def planting_calendar(crop):
        return PLANTING_CALENDAR.get(crop,{"sow":"Varies by region","harvest":"Varies by region","season":"N/A"})
