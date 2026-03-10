"""
schemes.py - Government scheme recommender for farmers
"""

SCHEMES = [
    {
        "name": "PM-KISAN",
        "full_name": "Pradhan Mantri Kisan Samman Nidhi",
        "benefit": "₹6,000/year in 3 installments of ₹2,000",
        "eligibility": "Small and marginal farmers with cultivable land",
        "apply": "pmkisan.gov.in or nearest CSC",
        "documents": ["Aadhaar Card", "Land Records", "Bank Account"],
        "tags": ["income", "all", "small", "marginal"],
    },
    {
        "name": "PMFBY",
        "full_name": "Pradhan Mantri Fasal Bima Yojana",
        "benefit": "Crop insurance against natural calamities",
        "eligibility": "All farmers growing notified crops",
        "apply": "pmfby.gov.in or bank/CSC",
        "documents": ["Aadhaar", "Land records", "Bank account", "Sowing certificate"],
        "tags": ["insurance", "kharif", "rabi", "all"],
    },
    {
        "name": "KCC",
        "full_name": "Kisan Credit Card",
        "benefit": "Crop loan at 4% interest rate (with subvention)",
        "eligibility": "All farmers, sharecroppers, tenant farmers",
        "apply": "Nearest Nationalized Bank / Cooperative Bank",
        "documents": ["Aadhaar", "Land records / Lease agreement", "Passport photo"],
        "tags": ["credit", "loan", "all"],
    },
    {
        "name": "PMKSY",
        "full_name": "Pradhan Mantri Krishi Sinchayee Yojana",
        "benefit": "55-75% subsidy on drip/sprinkler irrigation systems",
        "eligibility": "All farmer categories, SC/ST get higher subsidy",
        "apply": "State Agriculture Department / horticulture dept",
        "documents": ["Aadhaar", "Land records", "Bank account"],
        "tags": ["irrigation", "water", "horticulture"],
    },
    {
        "name": "Soil Health Card",
        "full_name": "Soil Health Card Scheme",
        "benefit": "Free soil testing and nutrient recommendations",
        "eligibility": "All farmers",
        "apply": "soilhealth.dac.gov.in or nearest agriculture office",
        "documents": ["Aadhaar", "Land details"],
        "tags": ["soil", "fertility", "fertilizer", "all"],
    },
    {
        "name": "eNAM",
        "full_name": "National Agriculture Market",
        "benefit": "Online market for better price discovery for produce",
        "eligibility": "All farmers with produce to sell",
        "apply": "enam.gov.in",
        "documents": ["Aadhaar", "Bank account"],
        "tags": ["market", "price", "selling", "all"],
    },
    {
        "name": "PKVY",
        "full_name": "Paramparagat Krishi Vikas Yojana",
        "benefit": "₹50,000/hectare for organic farming clusters",
        "eligibility": "Farmers willing to adopt organic farming in clusters of 50 acres",
        "apply": "State Agriculture Department",
        "documents": ["Aadhaar", "Land records", "Group registration"],
        "tags": ["organic", "sustainable", "cluster"],
    },
]

DAILY_TIPS = [
    "🌱 Test your soil every 3 years using the Soil Health Card scheme for free.",
    "💧 Drip irrigation can save up to 50% water and increase yield by 20-30%.",
    "🌿 Use green manure (Dhaincha/Sunhemp) to improve soil organic matter naturally.",
    "🐛 Install pheromone traps @ 5 per acre to monitor and control pest populations.",
    "☀️ Dry harvested produce to 12-14% moisture before storage to prevent fungal losses.",
    "🌾 Rotate crops every season to break pest and disease cycles in your field.",
    "💊 Always read pesticide labels carefully - overuse causes resistance and health risks.",
    "🌤️ Spray pesticides in early morning or evening to maximize effectiveness.",
    "🌱 Apply FYM (Farm Yard Manure) @ 5-10 tonnes/acre for better soil health.",
    "📋 Keep crop diary records - this helps get better loans and scheme benefits.",
]

STATES_DISTRICTS = {
    "Telangana": ["Hyderabad", "Warangal", "Karimnagar", "Nizamabad", "Khammam", "Nalgonda", "Adilabad", "Medak"],
    "Andhra Pradesh": ["Guntur", "Krishna", "Vijayawada", "Visakhapatnam", "Kurnool", "Kadapa", "Nellore", "Anantapur"],
    "Maharashtra": ["Pune", "Nagpur", "Nashik", "Aurangabad", "Solapur", "Kolhapur", "Amravati", "Latur"],
    "Punjab": ["Ludhiana", "Amritsar", "Jalandhar", "Patiala", "Bathinda", "Firozpur", "Gurdaspur", "Hoshiarpur"],
    "Tamil Nadu": ["Chennai", "Coimbatore", "Madurai", "Tiruchirappalli", "Salem", "Tirunelveli", "Erode", "Tiruppur"],
    "Karnataka": ["Bangalore", "Mysore", "Hubli", "Dharwad", "Belgaum", "Gulbarga", "Mangalore", "Shimoga"],
    "Uttar Pradesh": ["Lucknow", "Kanpur", "Varanasi", "Agra", "Meerut", "Allahabad", "Bareilly", "Moradabad"],
    "Madhya Pradesh": ["Bhopal", "Indore", "Jabalpur", "Gwalior", "Ujjain", "Sagar", "Rewa", "Satna"],
}

CROP_SUGGESTIONS = {
    "Telangana": {"Kharif": ["Paddy", "Cotton", "Soybean", "Maize", "Red gram"], "Rabi": ["Sunflower", "Groundnut", "Wheat", "Chickpea"]},
    "Andhra Pradesh": {"Kharif": ["Paddy", "Cotton", "Groundnut", "Maize"], "Rabi": ["Paddy", "Sunflower", "Chickpea", "Ragi"]},
    "Maharashtra": {"Kharif": ["Cotton", "Soybean", "Paddy", "Sugarcane"], "Rabi": ["Wheat", "Chickpea", "Onion", "Jowar"]},
    "Punjab": {"Kharif": ["Paddy", "Maize", "Cotton"], "Rabi": ["Wheat", "Mustard", "Potato"]},
    "Tamil Nadu": {"Kharif": ["Paddy", "Cotton", "Groundnut"], "Rabi": ["Paddy", "Sugarcane", "Banana"]},
    "Karnataka": {"Kharif": ["Paddy", "Maize", "Cotton", "Ragi"], "Rabi": ["Wheat", "Sunflower", "Chickpea"]},
}


def get_recommended_schemes(farmer_type: str = "all", crop: str = "", state: str = "") -> list:
    """Return relevant government schemes based on farmer profile."""
    recommended = []
    tags_to_match = ["all", farmer_type.lower()]
    if crop:
        tags_to_match.append(crop.lower())

    for scheme in SCHEMES:
        if any(tag in scheme["tags"] for tag in tags_to_match):
            recommended.append(scheme)

    return recommended


def get_crop_suggestions(state: str, season: str = "Kharif") -> list:
    """Return crop suggestions for the given state and season."""
    state_crops = CROP_SUGGESTIONS.get(state, {})
    return state_crops.get(season, [])


import random

def get_daily_tip() -> str:
    return random.choice(DAILY_TIPS)