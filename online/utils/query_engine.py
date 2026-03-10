"""
query_engine.py - FAISS vector search + IBM Watsonx LLM with expanded KCC dataset
"""

import os
import numpy as np
import requests
from typing import Optional

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False

# ── Expanded KCC Dataset (50+ Q&A pairs) ─────────────────────────────────────
KCC_DATA = [
    # Paddy / Rice
    {"query": "paddy leaves turning yellow yellowing rice", "answer": "Yellow leaves in paddy indicate: (1) Nitrogen deficiency — apply Urea @ 25 kg/acre immediately. (2) If yellowing starts from leaf tips, Iron deficiency — spray FeSO₄ 0.5% solution. (3) Leaf blight disease — spray Carbendazim 1g/L water. Ensure proper drainage and avoid waterlogging. Test soil pH (ideal 5.5–6.5 for paddy)."},
    {"query": "brown spot disease paddy rice", "answer": "Brown spot (Helminthosporium) in paddy: Symptoms — brown oval spots on leaves and grains. Treatment: (1) Spray Mancozeb 75 WP @ 2.5 g/L or Propiconazole 25 EC @ 1 ml/L. (2) Apply balanced NPK fertilizer. (3) Use resistant varieties. (4) Treat seeds with Thiram or Captan @ 3g/kg before sowing. Spray at 7-10 day intervals if severe."},
    {"query": "rice blast neck rot paddy", "answer": "Blast disease in paddy (Pyricularia oryzae): Symptoms — diamond-shaped lesions on leaves, neck rot at flowering. Critical stage is booting to flowering. Treatment: (1) Spray Tricyclazole 75 WP @ 0.6g/L at booting stage. (2) Repeat after 10 days. (3) Avoid excess nitrogen. (4) Use resistant varieties like Samba Masuri, BPT 5204. Do NOT spray during rain."},
    {"query": "stem borer paddy dead heart rice", "answer": "Stem borer in paddy: Symptoms — 'Dead heart' in vegetative stage, 'White ear' at heading. Management: (1) Set light traps. (2) Spray Chlorpyriphos 20 EC @ 2 ml/L or Carbofuran 3G @ 17 kg/acre (granules in standing water). (3) Install pheromone traps @ 5/acre. (4) Remove and destroy dead hearts manually. (5) Flood field occasionally to kill pupae."},
    {"query": "bacterial leaf blight paddy kresek", "answer": "Bacterial Leaf Blight (Xanthomonas oryzae) in paddy: Symptoms — water-soaked lesions on leaf margins, creamy bacterial ooze. No effective cure once spread. Management: (1) Drain field immediately — do NOT flood. (2) Spray Copper Oxychloride 3g/L or Streptocycline 0.5g/L + Copper Sulphate 3g/L. (3) Avoid excess N fertilizer. (4) Use resistant varieties."},

    # Cotton
    {"query": "cotton fertilizer NPK urea", "answer": "Cotton fertilizer schedule: Basal dose (at sowing): NPK 60:30:30 kg/acre. First top dressing (30 days): Urea 25 kg/acre. Second top dressing (60 days): Urea 25 kg/acre. At boll formation: Spray 00:00:50 (MOP) @ 5g/L + Calcium Nitrate 2g/L. Micronutrients: Zinc sulphate 10 kg/acre basal + Boron spray 0.5g/L at flower initiation. Total N: 60 kg/acre, P: 30 kg/acre, K: 30 kg/acre."},
    {"query": "cotton bollworm pink bollworm", "answer": "Bollworm (Helicoverpa/Pink bollworm) in cotton: (1) Install pheromone traps @ 5/acre for monitoring. (2) Spray at 1 larva/plant: Emamectin Benzoate 5 SG @ 100g/acre OR Indoxacarb 14.5 SC @ 200ml/acre OR Spinosad 45 SC @ 75ml/acre. (3) Rotate insecticides — never use same chemical twice. (4) NPV (Helicoverpa) @ 250 LE/ha as biological option. (5) Do NOT spray during flowering 8-10 AM (bee activity)."},
    {"query": "cotton leaf curl virus disease", "answer": "Cotton Leaf Curl Virus (CLCuV) — spread by whitefly. Symptoms: upward leaf curling, vein thickening, enations. Management: (1) Control whitefly vector — spray Imidacloprid 17.8 SL @ 100ml/acre or Thiamethoxam 25 WG @ 40g/acre. (2) Remove and burn infected plants. (3) Use resistant varieties (MRC 7017, JKCH 1947). (4) Avoid late planting. (5) Yellow sticky traps @ 10/acre for monitoring."},
    {"query": "cotton boll shedding dropping", "answer": "Cotton boll shedding causes and solutions: (1) Water stress — maintain adequate soil moisture, irrigate at 50% field capacity depletion. (2) Pest damage — inspect for bollworms, spray accordingly. (3) Nutritional deficiency — spray 19:19:19 @ 5g/L + Boron 0.5g/L. (4) Hormonal imbalance — spray Planofix (NAA) @ 4.5 ml/15L water at first flowering. (5) Avoid water stagnation at boll development."},

    # Tomato
    {"query": "tomato fruit cracking splitting", "answer": "Tomato fruit cracking is caused by irregular irrigation. Fix: (1) Maintain consistent soil moisture — use drip irrigation with mulching. (2) Spray Calcium Nitrate @ 2g/L at fruit development. (3) Spray Boron 0.5g/L at flowering and fruiting stages. (4) Avoid heavy watering after dry spells. (5) Choose crack-resistant varieties: Naveen, Pusa Ruby, NS 585. (6) Apply mulch (paddy straw/plastic) to regulate soil temperature and moisture."},
    {"query": "tomato early blight late blight disease", "answer": "Tomato blight management: Early blight (Alternaria) — brown spots with concentric rings; Late blight (Phytophthora) — water-soaked lesions, white sporulation. Treatment for both: (1) Remove infected plant parts. (2) Spray Mancozeb 75 WP @ 2.5g/L OR Metalaxyl + Mancozeb @ 2.5g/L (for late blight). (3) Spray Copper Oxychloride 3g/L as preventive. (4) Ensure good air circulation — proper spacing (60x45 cm). (5) Avoid overhead irrigation."},
    {"query": "tomato flower drop not setting fruit", "answer": "Tomato flower drop causes and solutions: (1) Temperature stress (>35°C or <15°C) — provide shade net at 50% shade; spray water on plants in afternoon. (2) Low humidity — mist spray in morning. (3) Boron deficiency — spray Borax @ 1g/L. (4) Spray Planofix (4-CPA) @ 5ml/15L at 50% flowering. (5) Reduce nitrogen, increase potassium and phosphorus fertilizers. (6) Ensure pollinator activity — avoid spraying insecticides during flowering."},
    {"query": "tomato yellowing leaves pale", "answer": "Tomato leaf yellowing: (1) Nitrogen deficiency (older leaves yellow) — apply Urea 10g/L foliar spray or soil application. (2) Magnesium deficiency (interveinal yellowing) — spray Magnesium Sulphate 10g/L. (3) Iron deficiency (young leaves yellow) — spray ferrous sulphate 2g/L. (4) Root rot — improve drainage, apply Trichoderma. (5) Viral infection — control aphid vectors with Imidacloprid."},

    # Chilli
    {"query": "chilli pepper not flowering low yield", "answer": "Chilli not flowering solutions: (1) Reduce nitrogen — shift to K-rich fertilizer: spray 00:52:34 (MAP) @ 5g/L + Boric acid 1g/L. (2) Ensure adequate phosphorus — apply SSP @ 25 kg/acre. (3) Temperature control — provide 50% shade net if >38°C. (4) Spray brassinolide (plant growth regulator) @ 0.5 ml/15L. (5) Maintain soil moisture. (6) Space plants at 45x30 cm for airflow. (7) Foliar spray of 19:19:19 @ 5g/L every 15 days."},
    {"query": "chilli anthracnose fruit rot disease", "answer": "Chilli anthracnose (Colletotrichum) — sunken dark spots on fruits. Management: (1) Spray Mancozeb 2.5g/L or Carbendazim 1g/L or Propiconazole 1ml/L at 10-day intervals. (2) Harvest mature fruits promptly — avoid leaving overripe fruits on plant. (3) Treat seeds with Thiram 3g/kg. (4) Avoid overhead irrigation. (5) Remove and destroy infected fruits. (6) Use resistant varieties."},
    {"query": "chilli thrips mites pest control", "answer": "Thrips and mites in chilli: Thrips — silvery stippling on leaves, curling. Mites — fine webbing, leaf bronzing. Control: (1) Thrips: Spinosad 45 SC @ 75ml/acre or Thiamethoxam 25 WG @ 40g/acre. (2) Mites: Abamectin 1.8 EC @ 400ml/acre or Propargite 57 EC @ 400ml/acre. (3) Spray both sides of leaves (adaxial and abaxial). (4) Install blue sticky traps for thrips monitoring. (5) Avoid excessive N fertilizer."},

    # Onion/Garlic
    {"query": "onion purple blotch disease", "answer": "Onion purple blotch (Alternaria porri): Water-soaked lesions that turn purple/brown. Management: (1) Spray Iprodione 2g/L or Mancozeb 2.5g/L or Carbendazim + Mancozeb 2g/L at 10-day intervals. (2) Ensure proper plant spacing (15x10 cm). (3) Avoid overhead irrigation — use drip. (4) Remove infected leaves. (5) Apply 3-4 sprays starting at disease onset. (6) Choose tolerant varieties."},
    {"query": "onion basal rot thrips", "answer": "Onion thrips control: Spray Spinosad 75ml/acre or Fipronil 30ml/15L water. For basal rot (Fusarium): improve drainage, apply Carbendazim 1g/L drench at base. Good tips: (1) Stop irrigation 10-15 days before harvest. (2) Spray Thiourea @ 1g/L at 60 and 75 days to enhance bulb development. (3) Maintain balanced NPK (80:60:60 kg/acre). (4) Harvest when 50-75% tops have fallen naturally."},
    {"query": "increase onion yield production", "answer": "To increase onion yield: (1) Select quality transplants at 45-day age. (2) NPK @ 100:50:50 kg/ha + FYM 25 t/ha. (3) Top dress Urea in 2 splits (at 30 and 50 days). (4) Spray Thiourea @ 1g/L + 13:00:45 @ 5g/L at bulb initiation. (5) Control thrips with Spinosad. (6) Stop irrigation 10-15 days before harvest. (7) Maintain 15x10 cm spacing. Expected yield: 15-20 t/ha with good management."},

    # Groundnut
    {"query": "groundnut tikka leaf spot", "answer": "Groundnut tikka disease (Cercospora leaf spots): Early (light spot) and late (dark spot) leaf spots. Management: (1) Spray Carbendazim 1g/L or Mancozeb 2.5g/L at 40 days, repeat every 15 days. (2) Apply 3-4 sprays. (3) Use resistant varieties: ICGS 11, TAG 24, K6. (4) Maintain proper spacing (30x10 cm). (5) Apply adequate gypsum (400 kg/acre) for pod development. Spray starts at first disease symptoms."},
    {"query": "groundnut pod development filling calcium gypsum", "answer": "Groundnut pod filling improvement: (1) Apply Gypsum (Calcium Sulphate) @ 400 kg/acre at pegging stage (35-40 days) — broadcasts in standing crop. (2) Spray Calcium Nitrate @ 2g/L at pegging and pod development. (3) Boron spray @ 0.5g/L. (4) Ensure adequate soil moisture at pegging — critical stage. (5) Do not disturb soil by cultivation after pegging. (6) Top peg count should be 300-400 per m² for good yield."},
    {"query": "groundnut stem rot white mold blight", "answer": "Groundnut stem rot (Sclerotium rolfsii) — white cottony growth at base. Management: (1) Drench with Carbendazim 1g/L + Copper Oxychloride 3g/L at base. (2) Apply Trichoderma viride @ 4 kg/acre with FYM as soil treatment. (3) Remove infected plants. (4) Avoid excessive irrigation. (5) Crop rotation with non-host crops. (6) Apply neem cake @ 250 kg/acre during land preparation."},

    # Wheat
    {"query": "wheat rust yellow stripe brown", "answer": "Wheat rust diseases: Yellow rust (Puccinia striiformis) — yellow pustules on leaves; Brown rust (Puccinia recondita) — brown pustules. Management: (1) Spray Propiconazole 25 EC @ 1ml/L at first appearance. (2) Repeat after 15 days. (3) Spray Mancozeb 75 WP @ 2.5g/L as protective. (4) Use resistant varieties: HD 2781, HD 2967, PBW 550. (5) Act FAST — rust spreads rapidly; spray within 48 hours of first sighting."},
    {"query": "wheat aphid sucking pest", "answer": "Wheat aphids control: (1) Monitor from tillering stage. (2) Spray Thiamethoxam 25 WG @ 40g/acre or Imidacloprid 17.8 SL @ 100ml/acre or Dimethoate 30 EC @ 1ml/L. (3) ETL: 20-25 aphids per culm. (4) Conserve natural predators (ladybird beetles, lacewings) — avoid broad-spectrum pesticides. (5) One spray usually sufficient if done at ETL. Aphid population naturally collapses after heading."},
    {"query": "wheat fertilizer basal dose top dressing", "answer": "Wheat fertilizer schedule (per acre): Basal (at sowing): DAP @ 55 kg (provides P + N) + MOP @ 20 kg. First top dressing (21 days / CRI stage): Urea @ 30 kg. Second top dressing (45 days / tillering): Urea @ 30 kg. Total: N 60 kg, P 30 kg, K 20 kg/acre. For high-yielding varieties, increase N by 20%. Also apply Zinc Sulphate @ 10 kg/acre once in 3 seasons."},

    # Sugarcane
    {"query": "sugarcane red rot disease stem", "answer": "Red rot in sugarcane (Colletotrichum falcatum): Reddish discoloration with white patches inside stalk, foul odor. Management: (1) Use disease-free certified setts. (2) Treat setts in Carbendazim @ 1g/L for 30 minutes before planting. (3) Remove and burn infected clumps — do NOT use them as planting material. (4) Improve drainage. (5) Use resistant varieties: CoC 671, Co 86032, Co 0238. (6) Strict quarantine of infected fields."},
    {"query": "sugarcane top borer early shoot borer", "answer": "Sugarcane borers: Shoot borer (early stage) and Top borer (later stage). Management: (1) Apply Carbofuran 3G @ 10 kg/acre in soil at planting. (2) Set light traps @ 1/acre. (3) Spray Chlorpyriphos 20 EC @ 2.5ml/L in leaf whorl. (4) Release Trichogramma chilonis egg parasitoid @ 50,000/acre/week (biocontrol). (5) Remove and destroy dead hearts. (6) Intercrop with cowpea or greengram."},

    # Mango
    {"query": "mango black spots anthracnose leaves", "answer": "Mango anthracnose (Colletotrichum gloeosporioides): Black spots on leaves and fruits. Management: (1) Spray Carbendazim 1g/L or Mancozeb 2.5g/L before and during flowering. (2) Pre-harvest spray: Propiconazole 1ml/L. (3) Post-harvest treatment: Hot water dip at 50°C for 10 min. (4) Prune for air circulation. (5) Spray copper-based fungicide before flowering. (6) Apply 3 sprays: at panicle emergence, full bloom, and marble stage."},
    {"query": "mango malformation bunchy top panicle", "answer": "Mango malformation (fungal: Fusarium subglutinans): Vegetative malformation (small bushy growth) and floral malformation (compact panicles, no fruit set). Management: (1) Remove malformed panicles — cut 15-20 cm below; burn or bury. (2) Spray NAA @ 200 ppm in November. (3) Apply Carbendazim 1g/L in September-October. (4) Use rogueing — avoid propagating from diseased trees. (5) Maintain tree nutrition: K spray @ 1% in November."},
    {"query": "mango mealybug pest", "answer": "Mango mealybug (Drosicha mangiferae): Mass migration from soil to tree in January-February. Management: (1) Apply sticky bands around trunk (60 cm from ground) by December-end. (2) Dust Endosulfan 4% or Chlorpyriphos 1.5% dust at base. (3) Spray Profenofos 50 EC @ 2ml/L or Dimethoate 1.5ml/L. (4) Spray before flowering. (5) Remove trunk bark debris where they hide. (6) Introduce natural enemy Spalgis epius (biocontrol)."},

    # Soil & General
    {"query": "soil health card how to get test", "answer": "Soil Health Card Process: (1) Collect soil samples from 8-10 spots in field at 15 cm depth. Mix and take 500g composite. (2) Submit to nearest Soil Testing Lab — Agriculture Department or Krishi Vigyan Kendra (KVK). (3) Apply online at soilhealth.dac.gov.in to find nearest lab. (4) Tests done: NPK, pH, EC, Organic Carbon, micronutrients (S, Zn, Fe, Mn, Cu, B). (5) Report received within 2-3 weeks with fertilizer recommendations. (6) Card is FREE under Soil Health Card Scheme. Get tested every 3 years."},
    {"query": "soil pH alkaline acidic correction", "answer": "Soil pH correction: Acidic soil (pH<6.5): Apply Agricultural Lime (Calcitic limestone) @ 1-2 t/acre, or Dolomite @ 1 t/acre — apply 1 month before sowing. Alkaline soil (pH>7.5): Apply Gypsum (Calcium Sulphate) @ 500 kg/acre, or elemental Sulphur @ 200 kg/acre + PSB bacteria. Sandy soil needs less amendment; clayey soil needs more. Retest after 6 months. Ideal pH ranges: Paddy 5.5-6.5, Wheat 6-7, Cotton 6-8, Groundnut 6-6.5."},
    {"query": "organic farming natural fertilizer", "answer": "Organic farming practices: (1) FYM (Farm Yard Manure) @ 10-15 t/acre — apply 3-4 weeks before sowing. (2) Vermicompost @ 2-3 t/acre. (3) Green manure (Dhaincha/Sunhemp) — incorporate at 45 days growth. (4) Neem cake @ 250 kg/acre for soil health + pest control. (5) Bio-fertilizers: Rhizobium (legumes), Azotobacter (non-legumes), PSB (phosphate), VAM (mycorrhiza). (6) Jeevamruth: 200L water + 10 kg cow dung + 5-10L cow urine + 2 kg jaggery + 2 kg pulse flour — ferment 48 hours, apply 200L/acre."},
    {"query": "drip irrigation sprinkler subsidy how to apply", "answer": "Drip/Sprinkler irrigation subsidy under PMKSY: Subsidy: 55% for general farmers, 75% for SC/ST farmers on total unit cost. Application: (1) Visit State Horticulture Department or Agriculture Department. (2) Apply online at state-specific portal (search '[State name] drip irrigation subsidy'). (3) Documents: Aadhaar, land records (7/12 or patta), bank account, photo. (4) Subsidy released directly to bank after installation inspection. (5) Empanelled vendors install the system. (6) Cost range: Drip ₹30,000-1,00,000/acre; Sprinkler ₹15,000-30,000/acre. Water savings: 40-50%."},

    # Government Schemes
    {"query": "PM KISAN scheme benefit eligibility", "answer": "PM-KISAN (Pradhan Mantri Kisan Samman Nidhi): Benefit: ₹6,000/year in 3 installments of ₹2,000 each (April-July, Aug-Nov, Dec-Mar). Eligibility: Landholding farmers with cultivable land. Excluded: Institutional land holders, former/current government employees, income tax payers, >10 acres land holders. Apply: pmkisan.gov.in OR nearest CSC (Common Service Centre). Documents: Aadhaar card, land records/khasra, bank account with IFSC. Check status: pmkisan.gov.in → 'Beneficiary Status'. Call: 011-24300606."},
    {"query": "Pradhan Mantri Fasal Bima Yojana crop insurance", "answer": "PMFBY (Pradhan Mantri Fasal Bima Yojana) — Crop Insurance: Premium: 2% for Kharif crops, 1.5% for Rabi crops, 5% for commercial/horticultural crops. Government pays balance. Coverage: Natural calamities, pests, diseases, post-harvest losses (for 2 weeks). Deadline: Kharif — July 31; Rabi — December 31. Apply: pmfby.gov.in OR bank (where you have KCC) OR CSC. Documents: Aadhaar, land records, sowing certificate, bank account. Claim: Inform within 72 hours of crop loss to bank/insurance company/crop insurance app."},
    {"query": "Kisan Credit Card KCC loan apply interest", "answer": "Kisan Credit Card (KCC): Credit limit: Based on crop cultivation cost + 10% contingency + 20% post-harvest expenses. Interest: 7% p.a., with 3% subvention = effective 4% if repaid on time. Eligible: All farmers, sharecroppers, oral lessees, tenant farmers, SHG members. Apply: Visit nearest bank (SBI, PNB, Canara, Regional Rural Bank, PACS). Documents: Aadhaar, land documents/lease agreement, photo. Repayment: After harvest (12 months for short-term crops). Renewal: Annually. ATM-linked Rupay debit card provided."},
    {"query": "government scheme small farmers support", "answer": "Key schemes for small/marginal farmers: (1) PM-KISAN: ₹6,000/year income support — pmkisan.gov.in. (2) PMFBY: Crop insurance at 2% premium — pmfby.gov.in. (3) KCC: Crop loan @ 4% interest. (4) PM-KUSUM: Solar pump subsidy @ 60-90% — pmkusum.gov.in. (5) PMKSY: Drip irrigation subsidy @ 55-75%. (6) eNAM: Better market price — enam.gov.in. (7) Soil Health Card: Free soil testing. (8) PKVY: ₹50,000/ha for organic farming."},
    {"query": "enam market price sell crops online", "answer": "eNAM (National Agriculture Market) — online mandi platform: Benefits: Better price discovery, transparent auction, reduced middlemen. How to use: (1) Register at enam.gov.in or nearest APMC mandi. (2) Documents: Aadhaar, bank account, land records. (3) Bring produce to registered eNAM mandi. (4) Weighment and quality testing done. (5) Online auction — buyers bid from across India. (6) Payment within 24 hours to bank. Currently 1,260+ mandis in 22 states. Helpline: 1800-270-0224."},

    # Pest control general
    {"query": "aphid sucking pest control any crop", "answer": "Aphid control (general): (1) Monitor regularly — check undersides of leaves. (2) Spray jet of water to dislodge colonies. (3) At ETL (Economic Threshold Level): Spray Imidacloprid 17.8 SL @ 100ml/acre or Thiamethoxam 25 WG @ 40g/acre or Acetamiprid 20 SP @ 100g/acre. (4) Biological control: Conserve ladybird beetles, lacewings. (5) Yellow sticky traps @ 5-10/acre for monitoring. (6) Neem-based insecticide (NSKE 5%) as eco-friendly option. Spray strictly at ETL — avoid unnecessary spraying."},
    {"query": "whitefly pest control tomato cotton chilli", "answer": "Whitefly control: (1) Install yellow sticky traps @ 10/acre for monitoring. (2) Spray Pyriproxyfen 10 EC @ 500ml/acre or Spiromesifen 22.9 SC @ 300ml/acre or Buprofezin 25 SC @ 800ml/acre. (3) Alternate with Imidacloprid or Thiamethoxam. (4) Remove heavily infested leaves. (5) Avoid planting susceptible crops (tomato, cotton, chilli) adjacent. (6) Reflective mulch repels whitefly. (7) Do NOT use same insecticide class twice consecutively — resistance develops rapidly."},
    {"query": "fungicide pesticide spray schedule rotation", "answer": "Pesticide spray principles: (1) NEVER use the same chemical or same class twice in a row — rotate MOA (Mode of Action) groups. (2) Spray at Economic Threshold Level (ETL) — not as calendar spray. (3) Spray in early morning (6-9 AM) or evening (4-7 PM) — avoid midday heat. (4) Use correct nozzle: Flat fan for contact sprays, hollow cone for systemic. (5) Add sticker (Teepol 0.5ml/L) for better adhesion. (6) Observe pre-harvest interval (PHI) — minimum days between last spray and harvest. (7) Wear PPE (mask, gloves, goggles)."},

    # Water/irrigation
    {"query": "irrigation schedule crop water requirement", "answer": "Irrigation scheduling: Paddy: maintain 5 cm standing water; critical at tillering, panicle initiation, flowering. Cotton: once in 8-10 days; critical at squaring and boll development. Tomato: once in 4-5 days; drip preferred. Wheat: 6 critical irrigations — CRI (21 days), tillering (40 days), jointing (60 days), booting (80 days), heading (90 days), dough stage (110 days). General rule: Irrigate when soil at 50% Field Capacity. Check: Squeeze soil — if it doesn't form ball, irrigate."},
    {"query": "drip irrigation fertigation how to use", "answer": "Drip irrigation fertigation: (1) Use water-soluble fertilizers only: Urea, DAP, MOP, 19:19:19, 13:00:45 etc. (2) Filter must be clean — flush drip lines weekly. (3) Fertigation schedule: N in 5-6 splits, P in 2-3 splits, K in 3-4 splits throughout crop period. (4) Fertigate for 30-40 min/day in 2 splits (morning and afternoon). (5) First: run plain water for 15 min → inject fertilizer → flush with water for 15 min. (6) Micro-irrigation saves 40-50% water and 30-40% fertilizer vs flood irrigation."},

    # Seeds and sowing
    {"query": "seed treatment before sowing germination", "answer": "Seed treatment before sowing: (1) Fungicide treatment: Thiram @ 2g/kg + Carbendazim @ 1g/kg seed (protects from soil-borne diseases). (2) Insecticide: Imidacloprid 70 WS @ 7g/kg seed (systemic aphid/whitefly protection for 25-30 days). (3) Bio-agents: Trichoderma viride @ 4g/kg + Pseudomonas fluorescens @ 10g/kg (antagonist). (4) Bio-fertilizers: Rhizobium (legumes) @ 25g/kg or Azotobacter (non-legumes) @ 25g/kg. (5) Never mix fungicide + bio-agent — apply bio-agent last. (6) Dry treated seeds in shade for 30 min before sowing."},
    {"query": "hybrid seeds vs local varieties which better", "answer": "Hybrid vs. Local/Open-pollinated varieties: Hybrids: 20-40% higher yield, more uniform, better market quality but seeds cannot be saved for next season (buy every year), cost ₹300-1000/packet. Local/OPV varieties: Seeds can be saved, adapted to local conditions, lower input need, lower cost. Choose hybrids for: commercial vegetable farming, export crops, irrigated conditions. Choose local/OPV for: subsistence farming, rain-fed areas, traditional varieties. Recommendation: Use certified disease-resistant improved varieties from ICAR/SAU research."},

    # Animals / Farmyard
    {"query": "cattle milk production fodder feeding", "answer": "Dairy cattle management for better milk yield: (1) Feed 1 kg concentrate per 2.5 kg milk produced, plus 8-10 kg green fodder + 5 kg dry fodder per day for adult cow. (2) Concentrate: 25% maize/jowar + 20% groundnut cake + 20% wheat bran + 30% rice bran + 5% mineral mixture. (3) Grow hybrid Napier (CO-3/CO-4) or Berseem as high-quality fodder. (4) Clean drinking water: 60-80 L/day/animal. (5) Regular deworming every 6 months. (6) Vaccination: FMD, HS, BQ annually. Contact nearest veterinary hospital for schedule."},
    {"query": "poultry backyard chicken farming tips", "answer": "Backyard poultry tips: (1) Desi (country) breeds: Aseel, Kadaknath, Vanaraja — low-input, high disease resistance. (2) Feed: 50-100g/bird/day — grain + protein supplement. (3) Water: clean fresh water always. (4) Vaccination: Newcastle Disease (Ranikhet) at 7 days and 28 days, Fowl Pox at 6 weeks. (5) Deworming: Albendazole every 3 months. (6) Housing: 1 sq ft/bird, east-facing, raised floor. Government provides subsidy under NABARD schemes. Contact District Animal Husbandry Department."},

    # Fertilizers
    {"query": "urea application how much when", "answer": "Urea application guide: Urea contains 46% Nitrogen. General dosage: Rice 50-60 kg/acre, Wheat 60-70 kg/acre, Cotton 60 kg/acre, Maize 60 kg/acre, Vegetables 25-35 kg/acre. Always split into 2-3 doses — do NOT apply full dose at once. Apply when soil has moisture (not bone dry or flooded). Apply in evening to reduce volatilization losses. Mix with soil lightly after application. Signs of N deficiency: Pale/yellow older leaves, stunted growth. Never exceed recommended dose — excess N causes pest attack and lodging."},
    {"query": "DAP diammonium phosphate how to use", "answer": "DAP (Di-Ammonium Phosphate — 18% N, 46% P₂O₅) application: Standard dose: 40-55 kg/acre as basal dose (at sowing/transplanting). Apply in furrows, cover with soil before sowing — direct contact with seed can burn. For transplanted crops: mix into soil before transplanting. DAP provides both Nitrogen and Phosphorus together — economical. Can be mixed with Zinc Sulphate @ 5 kg/acre for micronutrient supplementation. Storage: Keep dry — moisture clumps DAP and reduces effectiveness. Price: ~₹1,350/50 kg bag (government subsidized MRP)."},
    {"query": "zinc deficiency white bud khaira disease", "answer": "Zinc deficiency symptoms: Paddy — 'Khaira' disease (rusty brown spots, stunted growth after transplanting). Maize — 'White bud' (white striping on young leaves). Fruit crops — small leaves, little leaf. Correction: (1) Soil application: Zinc Sulphate (ZnSO₄) @ 10-15 kg/acre before sowing — effective for 3-4 seasons. (2) Foliar spray: ZnSO₄ @ 5g/L + urea 10g/L spray on affected plants. (3) In alkaline soils (pH>7.5) zinc is more deficient — apply more frequently. (4) EDTA-chelated zinc for foliar use is more effective."},

    # Diseases by location
    {"query": "Telangana paddy cotton crops season problems", "answer": "Telangana farming calendar: Kharif (June-November): Paddy, Cotton, Maize, Soybean, Red gram. Common issues — Cotton bollworm, Paddy blast, Cotton boll shedding. Rabi (November-March): Sunflower, Groundnut, Vegetables, Maize. Government support: Rythu Bandhu @ ₹5,000/acre investment support per season. Rytu Bima crop insurance. Contact: Agricultural Officer (AO) at Mandal Agriculture Office. District-specific advice: Call Kisan Call Centre 1800-180-1551 (toll-free)."},
    {"query": "Maharashtra soybean cotton problems", "answer": "Maharashtra key crops: Kharif — Cotton, Soybean, Jowar, Bajra. Rabi — Wheat, Chickpea, Onion, Sorghum. Vidarbha region: Cotton dominant — watch for Pink bollworm in Bt cotton. Soybean: Yellow mosaic virus (whitefly-transmitted) and stem fly major issues. Marathwada: Groundnut, Jowar — drought-tolerant varieties recommended. Schemes: Gopinath Munde Shetkari Apghat Vima Yojana (accident insurance). MH Agriculture Department: mahadbt.maharashtra.gov.in for scheme applications."},

    # Miscellaneous
    {"query": "intercropping mixed cropping benefit", "answer": "Intercropping benefits and options: Popular combinations: Pigeonpea + Jowar (1:3 ratio), Cotton + Cowpea, Sugarcane + Garlic/Potato, Maize + Beans, Groundnut + Castor. Benefits: (1) Risk distribution — if one crop fails, other provides income. (2) Nutrient complementarity — legume fixes N for companion. (3) Weed suppression. (4) Better land use efficiency. (5) Year-round income. Tips: Choose crops with different root depth and canopy height. Taller crop should not shade shorter one. Maintain plant population of both crops."},
    {"query": "composting vermicompost how to make", "answer": "Vermicompost preparation: (1) Pit size: 10x3x1 ft or raised brick bed. (2) Fill with 1 foot layer: mix of crop residues, kitchen waste, cow dung (1:1). (3) Add earthworms (Eisenia fetida/Lumbricus) @ 1 kg per sq.ft. (4) Maintain moisture at 40-50% — sprinkle water every 2-3 days. (5) Cover with jute sack to maintain moisture and darkness. (6) Harvest: After 45-60 days when material becomes dark, crumbly, earthy-smelling. (7) Separate worms for next batch. Application: 2-3 t/acre/year improves soil health significantly. N-P-K: 1.5-0.5-0.8%."},
    {"query": "kisan call centre toll free number helpline", "answer": "Kisan Call Centre (KCC) National Helpline: 1800-180-1551 (Toll Free, 6 AM to 10 PM, 7 days). Language support: 22 Indian languages. For crop queries, weather, schemes, market prices. Other important agricultural helplines: PM-KISAN: 011-24300606; Soil Health Card: 1800-180-1551; PMFBY: 1800-200-7710; eNAM: 1800-270-0224. State Agriculture Departments have their own helplines — call 155260 (national farmers helpline). Online: mkisan.gov.in for crop information, weather forecasts, market prices."},
    {"query": "market price mandi rate crop selling", "answer": "Finding market/mandi prices: (1) agmarknet.gov.in — all India mandi prices daily updated. (2) enam.gov.in — real-time prices from eNAM mandis. (3) SMS service: Send SMS to 51969 with crop name for price. (4) State agriculture department apps — e.g., APFACOS (AP), Agri Market (various states). (5) Local APMC mandi notice boards. (6) WhatsApp groups of local traders. (7) MSP (Minimum Support Price) — declared by government: check cacp.gov.in. Important: Sell at proper maturity — premature or over-mature produce fetches lower price."},
]

_model = None
_index = None
_answers = []
_kcc_queries = []


def _load_model():
    global _model
    if _model is None and ST_AVAILABLE:
        try:
            _model = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception as e:
            print(f"Model load error: {e}")
    return _model


def _build_index():
    global _index, _answers, _kcc_queries
    model = _load_model()
    if model is None or not FAISS_AVAILABLE:
        return

    _kcc_queries = [item["query"] for item in KCC_DATA]
    _answers = [item["answer"] for item in KCC_DATA]

    embeddings = model.encode(_kcc_queries, convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]
    _index = faiss.IndexFlatIP(dim)
    _index.add(embeddings)


def faiss_search(query: str, top_k: int = 3) -> list:
    global _index, _answers
    if _index is None:
        _build_index()
    if _index is None:
        return _keyword_fallback(query)

    model = _load_model()
    if model is None:
        return _keyword_fallback(query)

    q_emb = model.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(q_emb)
    scores, indices = _index.search(q_emb, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < len(_answers) and score > 0.15:
            results.append({
                "answer": _answers[idx],
                "query": KCC_DATA[idx]["query"],
                "score": float(score),
            })
    return results


def _keyword_fallback(query: str) -> list:
    query_lower = query.lower()
    scored = []
    for item in KCC_DATA:
        words = [w for w in query_lower.split() if len(w) > 3]
        matches = sum(1 for w in words if w in item["query"].lower() or w in item["answer"].lower())
        if matches > 0:
            scored.append((matches, item))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [{"answer": item["answer"], "query": item["query"], "score": 0.5} for _, item in scored[:3]]


def get_watsonx_answer(query: str, context: str) -> Optional[str]:
    api_key = os.getenv("WATSONX_API_KEY", "")
    project_id = os.getenv("WATSONX_PROJECT_ID", "")
    base_url = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")

    if not api_key or not project_id:
        return None

    try:
        token_resp = requests.post(
            "https://iam.cloud.ibm.com/identity/token",
            data={"grant_type": "urn:ibm:params:oauth:grant-type:apikey", "apikey": api_key},
            timeout=10,
        )
        token_resp.raise_for_status()
        access_token = token_resp.json().get("access_token", "")
    except Exception:
        return None

    prompt = f"""You are an expert agricultural advisor for Indian farmers. Answer the farmer's query with specific, practical, actionable advice. Include exact quantities, product names, timing, and application methods. Keep language simple.

Context from agricultural database:
{context}

Farmer's Query: {query}

Expert Answer:"""

    payload = {
        "model_id": "ibm/granite-13b-instruct-v2",
        "input": prompt,
        "parameters": {
            "decoding_method": "greedy",
            "max_new_tokens": 500,
            "repetition_penalty": 1.1,
        },
        "project_id": project_id,
    }

    try:
        resp = requests.post(
            f"{base_url}/ml/v1/text/generation?version=2023-05-29",
            headers={"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"},
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["results"][0]["generated_text"].strip()
    except Exception:
        return None


def answer_query(query: str, use_llm: bool = True) -> dict:
    results = faiss_search(query)

    if not results:
        return {"answer": None, "source": "no_match", "similar_queries": [], "confidence": 0.0}

    best = results[0]
    context = "\n\n".join([r["answer"] for r in results[:2]])
    llm_answer = None

    if use_llm:
        llm_answer = get_watsonx_answer(query, context)

    final_answer = llm_answer if llm_answer else best["answer"]

    return {
        "answer": final_answer,
        "source": "llm" if llm_answer else "faiss",
        "similar_queries": [" ".join(r["query"].split()[:6]) for r in results[1:3]],
        "confidence": best["score"],
        "raw_faiss_answer": best["answer"],
    }