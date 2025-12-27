import pandas as pd
import joblib

#model constants
MODEL_COLUMNS = [
    "Year",
    "ArtUnit",
    "Rejection101",
    "Rejection102",
    "Rejection103",
    "Rejection112",
    "Objection",
    "Cite102_GT1",
    "Cite103_GT3",
    "Cite103_EQ1",
    "Cite103_Max",
    "PriorArtScore",
    "TechCenter_Biotech & Organic Chemistry",
    "TechCenter_Business Methods & Finance",
    "TechCenter_Chemical & Materials Engineering",
    "TechCenter_Computer Architecture & Software",
    "TechCenter_Cryptography & Security",
    "TechCenter_Design Patents",
    "TechCenter_Mechanical Engineering",
    "TechCenter_Networking & Communications",
    "TechCenter_Semiconductors & Electrical Systems",
]

TECHCENTER_MAP = {
    "Biotech & Organic Chemistry (1600)": ("TechCenter_Biotech & Organic Chemistry", 1600),
    "Chemical & Materials Engineering (1700)": ("TechCenter_Chemical & Materials Engineering", 1700),
    "Computer Architecture & Software (2100)": ("TechCenter_Computer Architecture & Software", 2100),
    "Networking & Communications (2400)": ("TechCenter_Networking & Communications", 2400),
    "Cryptography & Security (2600)": ("TechCenter_Cryptography & Security", 2600),
    "Semiconductors & Electrical Systems (2800)": ("TechCenter_Semiconductors & Electrical Systems", 2800),
    "Business Methods & Finance (3600)": ("TechCenter_Business Methods & Finance", 3600),
    "Mechanical Engineering (3700)": ("TechCenter_Mechanical Engineering", 3700),
    "Design Patents (2900)": ("TechCenter_Design Patents", 2900),
}

def preprocess_inputs(tc_label, novelty, obvious, clarity, num_prior):

    # Start with zeros
    data = {col: 0 for col in MODEL_COLUMNS}

    # Year fixed or input
    data["Year"] = 2024

    # ------- ArtUnit (critical!) -------
    tc_dummy_name, art_unit = TECHCENTER_MAP[tc_label]
    data["ArtUnit"] = art_unit
    data[tc_dummy_name] = 1

    # ------- Rejection features -------
    data["Rejection102"] = 1 if novelty == "Yes" else 0
    data["Cite102_GT1"]  = data["Rejection102"]

    data["Rejection103"] = 1 if obvious == "Yes" else 0
    data["Cite103_GT3"]  = data["Rejection103"]

    data["Rejection112"] = 1 if clarity == "No" else 0
    data["Objection"]    = 1 if clarity == "Yes" else 0

    # Prior art features
    if num_prior <= 1:
        score = num_prior
    elif 2 <= num_prior <= 3:
        score = 3
    elif 4 <= num_prior <= 5:
        score = 4
    elif num_prior >= 6:
        score = 6
    else:
        score = 0

    data["PriorArtScore"] = score

    return pd.DataFrame([data])

#load model
def load_model(path="final_patent_model.pkl"):
    return joblib.load(path)