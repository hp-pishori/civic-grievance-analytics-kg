"""
preprocess.py
==============
Pune Municipal Corporation (PMC) Civic Grievance Text Preprocessing & Mock Dataset Generator.

Provides:
- Text cleaning and Hinglish/Marathi transliteration normalization.
- 50 realistic synthetic complaints reflecting Pune civic issues (roads, garbage, drainage, water, lights).
- Spatiotemporal distribution with intentional localized anomaly spikes.
- Ground truth token and character-level entity annotations for [LOCATION], [ISSUE_TYPE], [SEVERITY].
"""

import re
import json
import random
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple, Optional

try:
    import pandas as pd
except ImportError:
    pd = None


# ---------------------------------------------------------------------------
# PMC Domain Mappings & Dictionaries
# ---------------------------------------------------------------------------

PUNE_LOCATIONS = [
    "Kothrud", "Baner", "Hinjewadi", "Shivajinagar", "Aundh", "Hadapsar",
    "Viman Nagar", "Wakad", "Katraj", "Kalyani Nagar", "Yerawada",
    "Koregaon Park", "Karve Nagar", "Bavdhan", "Swargate", "Pashan",
    "Magarpatta", "Bibwewadi", "Sinhagad Road", "Dhanori"
]

WARD_MAPPING = {
    "Kothrud": "Ward 12 - Kothrud-Bavdhan",
    "Bavdhan": "Ward 12 - Kothrud-Bavdhan",
    "Karve Nagar": "Ward 13 - Karve Nagar-Warje",
    "Baner": "Ward 02 - Aundh-Baner-Balewadi",
    "Aundh": "Ward 02 - Aundh-Baner-Balewadi",
    "Pashan": "Ward 02 - Aundh-Baner-Balewadi",
    "Hinjewadi": "Ward 01 - Hinjewadi-Maan IT Corridor",
    "Wakad": "Ward 01 - Wakad-Tathawade",
    "Shivajinagar": "Ward 07 - Shivajinagar-Ghole Road",
    "Swargate": "Ward 09 - Kasba-Vishrambaug Wada",
    "Hadapsar": "Ward 19 - Hadapsar-Mundhwa",
    "Magarpatta": "Ward 19 - Hadapsar-Mundhwa",
    "Viman Nagar": "Ward 04 - Ahmednagar Road-Viman Nagar",
    "Kalyani Nagar": "Ward 04 - Ahmednagar Road-Viman Nagar",
    "Yerawada": "Ward 05 - Yerawada-Kalas-Dhanori",
    "Dhanori": "Ward 05 - Yerawada-Kalas-Dhanori",
    "Koregaon Park": "Ward 06 - Dhole Patil Road",
    "Katraj": "Ward 21 - Dhankawadi-Katraj",
    "Bibwewadi": "Ward 20 - Bibwewadi-Sahakarnagar",
    "Sinhagad Road": "Ward 14 - Sinhagad Road-Vadgaon"
}

ISSUE_DEPARTMENT_MAP = {
    "Potholes & Road Damage": "Road and Infrastructure Department",
    "Garbage Dumping": "Solid Waste Management Department",
    "Drainage & Sewerage Leak": "Drainage and Sewerage Department",
    "Water Supply Disruption": "Water Supply Department",
    "Streetlight Outage": "Electrical and Lighting Department",
    "Illegal Encroachment": "Encroachment Control Department"
}

# Slang and Marathi/Hinglish normalization dictionary
HINGLISH_LEXICON = {
    r"\bkachra\b": "garbage",
    r"\bkachara\b": "garbage",
    r"\bkhadda\b": "pothole",
    r"\bkhadde\b": "potholes",
    r"\bnal\b": "tap water pipe",
    r"\bpaani\b": "water",
    r"\bpani\b": "water",
    r"\bgutter\b": "drainage",
    r"\bgatar\b": "drainage",
    r"\bdurgandh\b": "foul smell",
    r"\bbatti\b": "streetlight",
    r"\blight\b": "streetlight",
    r"\brasta\b": "road",
    r"\bfootpath\b": "sidewalk",
    r"\btutlela\b": "broken",
    r"\bphutlela\b": "burst",
    r"\bchocked\b": "choked",
    r"\bchoked-up\b": "choked"
}


# ---------------------------------------------------------------------------
# Text Preprocessing & Normalization
# ---------------------------------------------------------------------------

def clean_text(text: str, normalize_slang: bool = False) -> str:
    """
    Cleans raw civic complaint text:
    - Removes non-standard unicode characters, handles whitespace
    - Normalizes noisy PMC mobile app formatting
    - Optionally replaces common Hinglish/Marathi civic loanwords
    """
    if not isinstance(text, str):
        return ""

    # Strip URLs or email addresses if any
    cleaned = re.sub(r"http\S+|www\S+|[\w\.-]+@[\w\.-]+", "", text)

    # Remove erratic punctuation duplication (e.g., '!!!!' -> '!')
    cleaned = re.sub(r"([!?.,])\1+", r"\1", cleaned)

    # Normalize Marathi / Hinglish loanwords if requested
    if normalize_slang:
        for pattern, replacement in HINGLISH_LEXICON.items():
            cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)

    # Normalize multiple whitespaces
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


# ---------------------------------------------------------------------------
# Synthetic PMC Grievance Dataset (50 Realistic Records with Injected Anomalies)
# ---------------------------------------------------------------------------

RAW_COMPLAINT_TEMPLATES = [
    # Normal distribution across wards and categories
    ("Major deep potholes on DP road near Kothrud depot causing severe traffic jams. Urgent repair required.", "Kothrud", "Potholes & Road Damage", "HIGH", "2024-03-01"),
    ("Garbage dumping along main canal road in Baner. Foul smell and stray dogs everywhere since 3 days.", "Baner", "Garbage Dumping", "MEDIUM", "2024-03-01"),
    ("Underground drainage pipeline chocked and leaking black sewage water on Hinjewadi Phase 1 chowk.", "Hinjewadi", "Drainage & Sewerage Leak", "HIGH", "2024-03-02"),
    ("Low pressure dirty water supply in Shivajinagar near Modern College. Water smells like sewage.", "Shivajinagar", "Water Supply Disruption", "HIGH", "2024-03-02"),
    ("Streetlight outage on IT park link road Aundh. Complete darkness for pedestrians at night.", "Aundh", "Streetlight Outage", "MEDIUM", "2024-03-03"),
    ("Huge pile of untreated garbage kachra dumped outside Magarpatta South Gate. Swarms of flies.", "Magarpatta", "Garbage Dumping", "HIGH", "2024-03-03"),
    ("Illegal vegetable vendors encroaching the entire footpath near Swargate bus terminal.", "Swargate", "Illegal Encroachment", "LOW", "2024-03-04"),
    ("Multiple potholes on bypass junction near Wakad bridge. Two-wheelers slipping daily.", "Wakad", "Potholes & Road Damage", "HIGH", "2024-03-04"),
    ("Drainage manhole cover missing on Karve Road Kothrud. Extreme danger of fatal accident.", "Kothrud", "Drainage & Sewerage Leak", "HIGH", "2024-03-05"),
    ("No PMC water supply in Kalyani Nagar lane 7 since yesterday morning. Tanker required.", "Kalyani Nagar", "Water Supply Disruption", "MEDIUM", "2024-03-05"),
    ("Flickering streetlights on main road in Koregaon Park near Lane 4. Needs bulb replacement.", "Koregaon Park", "Streetlight Outage", "LOW", "2024-03-06"),
    ("Open garbage container overflowing with domestic waste near Katraj bus stand. Terrible durgandh.", "Katraj", "Garbage Dumping", "MEDIUM", "2024-03-06"),
    ("Minor road surface chipping in Viman Nagar near Symbiosis school. Needs patching.", "Viman Nagar", "Potholes & Road Damage", "LOW", "2024-03-07"),
    ("Choked gatar drainage overflowing into basement shops near Yerawada market. Urgent assistance.", "Yerawada", "Drainage & Sewerage Leak", "HIGH", "2024-03-07"),
    ("Contaminated yellow tap water coming in pipeline at Bavdhan residential society.", "Bavdhan", "Water Supply Disruption", "HIGH", "2024-03-08"),
    ("Unauthorized tea stall stalls blocking access to public park in Pashan.", "Pashan", "Illegal Encroachment", "LOW", "2024-03-08"),
    ("Streetlight pole leaning dangerously after mini-bus collided near Hadapsar Gadital.", "Hadapsar", "Streetlight Outage", "HIGH", "2024-03-09"),
    ("Broken road divider and asphalt potholes near Sinhagad Road flyover landing.", "Sinhagad Road", "Potholes & Road Damage", "MEDIUM", "2024-03-09"),
    ("Dead animals and wet garbage rotting on open plot in Bibwewadi. Health hazard.", "Bibwewadi", "Garbage Dumping", "HIGH", "2024-03-10"),
    ("Broken water distribution pipeline wasting thousands of litres daily in Dhanori.", "Dhanori", "Water Supply Disruption", "MEDIUM", "2024-03-10"),
    ("Gutter cover broken near commercial plaza in Baner. Foul water pooling on footpath.", "Baner", "Drainage & Sewerage Leak", "MEDIUM", "2024-03-11"),
    ("Hawkers setting up illegal sheds on busy walking plaza in Aundh. Traffic choked.", "Aundh", "Illegal Encroachment", "LOW", "2024-03-11"),
    ("Continuous water supply leakage from valve chamber near Shivajinagar railway gate.", "Shivajinagar", "Water Supply Disruption", "LOW", "2024-03-12"),
    ("Three streetlights not working in row near Wakad Dutta Mandir road.", "Wakad", "Streetlight Outage", "LOW", "2024-03-12"),
    ("Uneven road leveling and tar peeling off after pipeline work in Kothrud near MIT college.", "Kothrud", "Potholes & Road Damage", "MEDIUM", "2024-03-13"),

    # --- INJECTED ANOMALY 1: Severe Drainage Burst Spike in Baner (Dates: 2024-03-14 & 2024-03-15) ---
    ("Massive sewage drainage burst in Baner near Pancard Club road. Gutter water entering homes!", "Baner", "Drainage & Sewerage Leak", "HIGH", "2024-03-14"),
    ("Drainage line completely blocked in Baner sector 3. Stagnant filthy water flood on street.", "Baner", "Drainage & Sewerage Leak", "HIGH", "2024-03-14"),
    ("Foul smell and overflowing manhole on Baner High Street. Critical health emergency for residents.", "Baner", "Drainage & Sewerage Leak", "HIGH", "2024-03-14"),
    ("Sewage water backflowing into ground floor apartments in Baner. Urgent pump team needed.", "Baner", "Drainage & Sewerage Leak", "HIGH", "2024-03-14"),
    ("Main underground sewer collapsed in Baner near highway underpass. Road submerged in sludge.", "Baner", "Drainage & Sewerage Leak", "HIGH", "2024-03-15"),
    ("Disgusting sewer leak on Baner Pashan link road. Commuters cannot cross.", "Baner", "Drainage & Sewerage Leak", "HIGH", "2024-03-15"),
    ("Choked gatar line in Baner village area overflowing into local vegetable market.", "Baner", "Drainage & Sewerage Leak", "HIGH", "2024-03-15"),

    # --- INJECTED ANOMALY 2: Massive Garbage Dumping Surge in Kothrud (Dates: 2024-03-18 & 2024-03-19) ---
    ("Sanitation strike impact: mountains of uncollected kachra garbage piled in Kothrud ward.", "Kothrud", "Garbage Dumping", "HIGH", "2024-03-18"),
    ("Garbage dumping everywhere along Paud Road Kothrud. Rotten stink and stray cattle menace.", "Kothrud", "Garbage Dumping", "HIGH", "2024-03-18"),
    ("Huge solid waste accumulation outside Kothrud vegetable market. Not cleared for 4 days.", "Kothrud", "Garbage Dumping", "HIGH", "2024-03-18"),
    ("Overflowing dumper bin with rotting medical waste near Kothrud hospital. Immediate evacuation required.", "Kothrud", "Garbage Dumping", "HIGH", "2024-03-18"),
    ("Illegal open garbage burning generating toxic smoke near Kothrud hill base.", "Kothrud", "Garbage Dumping", "HIGH", "2024-03-19"),
    ("Waste disposal vehicle not visiting Anand Nagar Kothrud. Residents throwing garbage on road.", "Kothrud", "Garbage Dumping", "MEDIUM", "2024-03-19"),
    ("Huge mound of household trash blocking entrance of society in Kothrud.", "Kothrud", "Garbage Dumping", "HIGH", "2024-03-19"),

    # Regular post-anomaly complaints
    ("Multiple heavy potholes near Hinjewadi Phase 3 circle causing 2 hour delay.", "Hinjewadi", "Potholes & Road Damage", "HIGH", "2024-03-20"),
    ("Zero water supply in Viman Nagar lane 2 since 48 hours. Tankers overcharging.", "Viman Nagar", "Water Supply Disruption", "HIGH", "2024-03-20"),
    ("Streetlights non-functional on Hadapsar railway flyover. Dark and dangerous.", "Hadapsar", "Streetlight Outage", "MEDIUM", "2024-03-21"),
    ("Illegal street food vendors blocking pedestrian walkway near Kalyani Nagar bridge.", "Kalyani Nagar", "Illegal Encroachment", "LOW", "2024-03-21"),
    ("Overflowing drainage chamber near Katraj snake park road. Smelling awful.", "Katraj", "Drainage & Sewerage Leak", "MEDIUM", "2024-03-21"),
    ("Potholes reappearing after rain on Karve Nagar main chowk.", "Karve Nagar", "Potholes & Road Damage", "MEDIUM", "2024-03-22"),
    ("Pipes ruptured during metro construction work near Shivajinagar. Clean water overflowing.", "Shivajinagar", "Water Supply Disruption", "MEDIUM", "2024-03-22"),
    ("Streetlight switchboard open with exposed live wires in Aundh. Serious safety hazard for kids.", "Aundh", "Streetlight Outage", "HIGH", "2024-03-22"),
    ("Illegal car parking garages encroaching public service road in Hinjewadi.", "Hinjewadi", "Illegal Encroachment", "LOW", "2024-03-22"),
    ("Garbage dumping in storm water nullah near Bavdhan junction.", "Bavdhan", "Garbage Dumping", "MEDIUM", "2024-03-22"),
    ("Deep pothole crater near Swargate Jedhe flyover ascent.", "Swargate", "Potholes & Road Damage", "HIGH", "2024-03-22"),
]


def find_entity_spans(text: str, loc: str, issue_cat: str, sev: str) -> List[Dict[str, Any]]:
    """
    Locates character-level entity spans for evaluation and training.
    Maps categories and terms to exact substrings in the text.
    """
    spans = []

    # 1. Location Span
    if loc in text:
        start = text.index(loc)
        spans.append({
            "text": loc,
            "start": start,
            "end": start + len(loc),
            "label": "LOCATION"
        })

    # 2. Issue Type Span Triggers
    issue_substring_candidates = {
        "Potholes & Road Damage": [
            "deep potholes", "potholes", "pothole", "asphalt potholes",
            "pothole crater", "broken asphalt", "Uneven road", "Broken road"
        ],
        "Garbage Dumping": [
            "Garbage dumping", "garbage dumping", "garbage", "kachra",
            "untreated garbage", "rotting medical waste", "solid waste", "trash", "waste"
        ],
        "Drainage & Sewerage Leak": [
            "Underground drainage pipeline", "drainage pipeline", "sewage drainage burst",
            "Drainage manhole", "Drainage line", "sewage water", "sewer leak",
            "gutter water", "drainage", "sewer", "gatar", "choked gatar"
        ],
        "Water Supply Disruption": [
            "water supply", "Water supply", "water distribution pipeline",
            "tap water", "water leakage", "water"
        ],
        "Streetlight Outage": [
            "Streetlight outage", "Streetlights", "Streetlight pole",
            "streetlights", "Streetlight", "exposed live wires"
        ],
        "Illegal Encroachment": [
            "Illegal vegetable vendors", "encroaching the entire footpath",
            "encroaching", "Unauthorized tea stall", "Illegal street food vendors",
            "Illegal car parking garages", "encroachment"
        ]
    }

    matched_issue = False
    candidates = issue_substring_candidates.get(issue_cat, [])
    for cand in candidates:
        if cand.lower() in text.lower():
            idx = text.lower().index(cand.lower())
            actual_text = text[idx:idx + len(cand)]
            spans.append({
                "text": actual_text,
                "start": idx,
                "end": idx + len(cand),
                "label": "ISSUE_TYPE"
            })
            matched_issue = True
            break

    if not matched_issue:
        # Fallback to category name if partially matched
        first_word = issue_cat.split()[0].lower()
        if first_word in text.lower():
            idx = text.lower().index(first_word)
            spans.append({
                "text": text[idx:idx + len(first_word)],
                "start": idx,
                "end": idx + len(first_word),
                "label": "ISSUE_TYPE"
            })

    # 3. Severity Span Triggers
    sev_candidates = {
        "HIGH": ["Urgent", "urgent", "Critical", "critical", "severe", "Extreme danger",
                 "Massive", "health hazard", "emergency", "fatal", "Immediately", "Immediate", "Serious safety hazard"],
        "MEDIUM": ["since 3 days", "since yesterday", "since 2 days", "terrible", "pooling", "Uneven", "wasting"],
        "LOW": ["Minor", "minor", "Needs bulb replacement", "slowly", "routine"]
    }

    matched_sev = False
    for cand in sev_candidates.get(sev, []):
        if cand.lower() in text.lower():
            idx = text.lower().index(cand.lower())
            actual_text = text[idx:idx + len(cand)]
            spans.append({
                "text": actual_text,
                "start": idx,
                "end": idx + len(cand),
                "label": "SEVERITY"
            })
            matched_sev = True
            break

    if not matched_sev:
        # Add a virtual severity label representing the default ground truth
        spans.append({
            "text": sev,
            "start": -1,
            "end": -1,
            "label": "SEVERITY"
        })

    return spans


def generate_pmc_dataset(output_path: str = None) -> pd.DataFrame:
    """
    Generates the benchmark 50-sample Pune civic dataset.
    Returns a pandas DataFrame and optionally exports to JSON/CSV.
    """
    records = []

    for i, (text, loc, issue, sev, date_str) in enumerate(RAW_COMPLAINT_TEMPLATES, start=1):
        complaint_id = f"PMC-2024-{i:04d}"
        ward = WARD_MAPPING.get(loc, f"Ward - {loc}")
        dept = ISSUE_DEPARTMENT_MAP.get(issue, "General Municipal Administration")
        entities = find_entity_spans(text, loc, issue, sev)

        record = {
            "complaint_id": complaint_id,
            "raw_text": text,
            "cleaned_text": clean_text(text),
            "ground_truth_location": loc,
            "ground_truth_issue": issue,
            "ground_truth_severity": sev,
            "ward": ward,
            "department": dept,
            "filing_date": date_str,
            "status": "Escalated" if sev == "HIGH" else "Open",
            "entities": entities
        }
        records.append(record)

    if output_path:
        if output_path.endswith(".json"):
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(records, f, indent=2)
        else:
            if pd is not None:
                pd.DataFrame(records).to_csv(output_path, index=False)
            else:
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(records, f, indent=2)
        print(f"[*] Dataset successfully exported to {output_path} ({len(records)} records)")

    if pd is not None:
        return pd.DataFrame(records)
    return records


if __name__ == "__main__":
    data = generate_pmc_dataset("pune_civic_complaints.json")
    print(f"Generated {len(data)} records. Sample record:\n")
    sample = data[0] if isinstance(data, list) else data.iloc[0].to_dict()
    print(json.dumps({k: v for k, v in sample.items() if k != "entities"}, indent=2))
