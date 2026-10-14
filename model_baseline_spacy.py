"""
model_baseline_spacy.py
========================
Baseline Rule-based & Pattern-Matching NLP Engine using spaCy.

Features:
- Tokenization & Linguistic Pattern Matching.
- EntityRuler / PhraseMatcher for Pune Municipal Corporation (PMC) entities:
  - [LOCATION]: Pune suburbs, roads, IT corridors, transit hubs.
  - [ISSUE_TYPE]: Civic complaints (Potholes, Garbage, Drainage, Water, Streetlights, Encroachment).
  - [SEVERITY]: Lexical and dependency triggers (Urgent, Critical, Moderate, Low).
- Normalization mapping to administrative Wards and PMC Departments.
- Graceful standalone fallback if spaCy linguistic packages are being initialized.
"""

import re
from typing import List, Dict, Any, Optional
from preprocess import (
    PUNE_LOCATIONS,
    WARD_MAPPING,
    ISSUE_DEPARTMENT_MAP,
    clean_text
)

# Attempt spaCy import
try:
    import spacy
    from spacy.language import Language
    from spacy.pipeline import EntityRuler
    from spacy.matcher import Matcher, PhraseMatcher
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False


# Comprehensive Location Patterns for Pune
LOCATION_PATTERNS = [
    # Suburbs and Localities
    "Kothrud", "Baner", "Hinjewadi", "Shivajinagar", "Aundh", "Hadapsar",
    "Viman Nagar", "Wakad", "Katraj", "Kalyani Nagar", "Yerawada",
    "Koregaon Park", "Karve Nagar", "Bavdhan", "Swargate", "Pashan",
    "Magarpatta", "Bibwewadi", "Sinhagad Road", "Dhanori", "Balewadi",
    "Dhole Patil Road", "Kothrud Depot", "Hadapsar Gadital", "Baner High Street",
    "Pancard Club Road", "Paud Road", "Karve Road", "Modern College",
    "Symbiosis", "Jedhe Flyover", "Hinjewadi Phase 1", "Hinjewadi Phase 2",
    "Hinjewadi Phase 3", "FC Road", "JM Road"
]

# Issue Patterns categorized by Departmental taxonomy
ISSUE_PATTERNS = {
    "Potholes & Road Damage": [
        "potholes", "pothole", "deep potholes", "khadda", "khadde", "broken road",
        "asphalt potholes", "pothole crater", "crater", "broken asphalt",
        "uneven road", "road surface chipping", "tar peeling off"
    ],
    "Garbage Dumping": [
        "garbage dumping", "garbage", "kachra", "solid waste", "untreated garbage",
        "rotting waste", "waste dumping", "dead animals", "rotting medical waste",
        "open garbage container", "mound of household trash", "trash", "waste"
    ],
    "Drainage & Sewerage Leak": [
        "underground drainage pipeline", "drainage pipeline", "sewage drainage burst",
        "drainage manhole", "drainage line", "sewage water", "sewer leak",
        "gutter water", "drainage", "sewer", "gatar", "choked gatar",
        "choked drainage", "overflowing manhole", "sewer collapsed", "drainage chamber"
    ],
    "Water Supply Disruption": [
        "water supply", "water distribution pipeline", "tap water", "water leakage",
        "dirty water", "yellow tap water", "pipeline work", "pipes ruptured",
        "zero water supply", "no water", "contaminated yellow tap water", "paani"
    ],
    "Streetlight Outage": [
        "streetlight outage", "streetlights", "streetlight pole", "streetlight",
        "flickering streetlights", "exposed live wires", "complete darkness",
        "dark and dangerous", "non-functional lamp", "batti"
    ],
    "Illegal Encroachment": [
        "illegal vegetable vendors", "encroaching the entire footpath",
        "encroaching", "unauthorized tea stall", "illegal street food vendors",
        "illegal car parking garages", "encroachment", "hawkers", "illegal sheds"
    ]
}

# Severity trigger dictionaries
SEVERITY_TRIGGERS = {
    "HIGH": [
        "urgent", "critical", "severe", "extreme danger", "massive",
        "health hazard", "emergency", "fatal", "immediately", "immediate",
        "dangerously", "hazard", "toxic smoke", "submerged", "backflowing"
    ],
    "MEDIUM": [
        "since 3 days", "since yesterday", "since 2 days", "since 4 days",
        "terrible", "pooling", "uneven", "wasting thousands", "rotting",
        "bad smell", "durgandh", "traffic jams"
    ],
    "LOW": [
        "minor", "needs bulb replacement", "slowly", "routine", "needs patching",
        "chipping", "flickering"
    ]
}


class SpacyCivicExtractor:
    """
    Baseline Rule-Based & Pattern-Matching NLP Engine for Pune Civic Grievances.
    Uses spaCy's Tokenizer, EntityRuler, and Matcher when available,
    with an internal high-performance fallback engine.
    """

    def __init__(self, use_spacy: bool = True):
        self.nlp = None
        self.ruler = None
        self.matcher = None
        self.is_spacy_loaded = False

        if use_spacy and SPACY_AVAILABLE:
            self._init_spacy_pipeline()
        else:
            self._init_fallback_pipeline()

    def _init_spacy_pipeline(self):
        """Builds a customized spaCy pipeline with EntityRuler."""
        try:
            # Try loading blank English model first to avoid downloading heavy weights
            self.nlp = spacy.blank("en")
            # Add custom EntityRuler
            self.ruler = self.nlp.add_pipe("entity_ruler", config={"overwrite_ents": True})

            patterns = []
            # 1. Location patterns
            for loc in LOCATION_PATTERNS:
                patterns.append({"label": "LOCATION", "pattern": loc})

            # 2. Issue Type patterns
            for category, phrases in ISSUE_PATTERNS.items():
                for phrase in phrases:
                    patterns.append({
                        "label": "ISSUE_TYPE",
                        "pattern": [{"LOWER": term} for term in phrase.split()]
                    })

            # 3. Severity patterns
            for sev_level, triggers in SEVERITY_TRIGGERS.items():
                for trigger in triggers:
                    patterns.append({
                        "label": "SEVERITY",
                        "pattern": [{"LOWER": term} for term in trigger.split()]
                    })

            self.ruler.add_patterns(patterns)
            self.is_spacy_loaded = True
            print("[*] spaCy baseline pipeline initialized with EntityRuler patterns.")
        except Exception as e:
            print(f"[!] spaCy initialization fallback: {e}")
            self._init_fallback_pipeline()

    def _init_fallback_pipeline(self):
        """Linguistic regex and dictionary matcher fallback engine."""
        self.is_spacy_loaded = False
        self.compiled_loc_patterns = [
            (loc, re.compile(rf"\b{re.escape(loc)}\b", re.IGNORECASE))
            for loc in LOCATION_PATTERNS
        ]
        self.compiled_issue_patterns = []
        for category, phrases in ISSUE_PATTERNS.items():
            for phrase in phrases:
                self.compiled_issue_patterns.append(
                    (category, phrase, re.compile(rf"\b{re.escape(phrase)}\b", re.IGNORECASE))
                )
        self.compiled_sev_patterns = []
        for sev_level, triggers in SEVERITY_TRIGGERS.items():
            for trigger in triggers:
                self.compiled_sev_patterns.append(
                    (sev_level, trigger, re.compile(rf"\b{re.escape(trigger)}\b", re.IGNORECASE))
                )

    def extract(self, text: str) -> Dict[str, Any]:
        """
        Extracts civic entities and returns structured dictionary with spans & normalization.
        """
        cleaned = clean_text(text)

        if self.is_spacy_loaded and self.nlp:
            return self._extract_spacy(cleaned, raw_text=text)
        else:
            return self._extract_fallback(cleaned, raw_text=text)

    def _extract_spacy(self, text: str, raw_text: str) -> Dict[str, Any]:
        """Extracts entities using spaCy EntityRuler doc."""
        doc = self.nlp(text)
        locations = []
        issue_types = []
        severities = []

        for ent in doc.ents:
            span_dict = {
                "text": ent.text,
                "start": ent.start_char,
                "end": ent.end_char,
                "label": ent.label_
            }
            if ent.label_ == "LOCATION":
                locations.append(span_dict)
            elif ent.label_ == "ISSUE_TYPE":
                issue_types.append(span_dict)
            elif ent.label_ == "SEVERITY":
                severities.append(span_dict)

        # Normalization and resolution
        return self._normalize_entities(text, raw_text, locations, issue_types, severities, "spaCy_EntityRuler_Baseline")

    def _extract_fallback(self, text: str, raw_text: str) -> Dict[str, Any]:
        """Extracts entities using regex pattern matcher."""
        locations = []
        issue_types = []
        severities = []

        # 1. Locations
        for loc_name, pattern in self.compiled_loc_patterns:
            for match in pattern.finditer(text):
                locations.append({
                    "text": match.group(0),
                    "start": match.start(),
                    "end": match.end(),
                    "label": "LOCATION"
                })

        # 2. Issues
        for category, phrase, pattern in self.compiled_issue_patterns:
            for match in pattern.finditer(text):
                issue_types.append({
                    "text": match.group(0),
                    "start": match.start(),
                    "end": match.end(),
                    "label": "ISSUE_TYPE",
                    "matched_category": category
                })

        # 3. Severity
        for sev_level, trigger, pattern in self.compiled_sev_patterns:
            for match in pattern.finditer(text):
                severities.append({
                    "text": match.group(0),
                    "start": match.start(),
                    "end": match.end(),
                    "label": "SEVERITY",
                    "matched_level": sev_level
                })

        # Deduplicate overlapping spans
        locations = self._deduplicate_spans(locations)
        issue_types = self._deduplicate_spans(issue_types)
        severities = self._deduplicate_spans(severities)

        return self._normalize_entities(text, raw_text, locations, issue_types, severities, "PatternMatcher_Baseline")

    def _deduplicate_spans(self, span_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Removes contained sub-spans to keep the longest matching phrase."""
        if not span_list:
            return []
        # Sort by span length descending
        sorted_spans = sorted(span_list, key=lambda s: (s["end"] - s["start"]), reverse=True)
        kept = []
        for span in sorted_spans:
            if not any(k["start"] <= span["start"] and span["end"] <= k["end"] for k in kept):
                kept.append(span)
        return sorted(kept, key=lambda s: s["start"])

    def _normalize_entities(self, text: str, raw_text: str,
                            locations: List[Dict[str, Any]],
                            issue_types: List[Dict[str, Any]],
                            severities: List[Dict[str, Any]],
                            model_name: str) -> Dict[str, Any]:
        """Maps extracted spans into canonical PMC Wards and Departments."""
        # Canonical location & ward
        canonical_loc = "Pune Central"
        canonical_ward = "Ward 07 - Shivajinagar-Ghole Road"
        if locations:
            canonical_loc = locations[0]["text"]
            for prime_loc, ward_val in WARD_MAPPING.items():
                if prime_loc.lower() in canonical_loc.lower() or canonical_loc.lower() in prime_loc.lower():
                    canonical_loc = prime_loc
                    canonical_ward = ward_val
                    break

        # Canonical issue & department
        canonical_issue = "General Civic Grievance"
        canonical_dept = "General Municipal Administration"
        if issue_types:
            best_issue_text = issue_types[0]["text"].lower()
            for cat, phrases in ISSUE_PATTERNS.items():
                if any(p.lower() in best_issue_text or best_issue_text in p.lower() for p in phrases):
                    canonical_issue = cat
                    canonical_dept = ISSUE_DEPARTMENT_MAP.get(cat, canonical_dept)
                    break

        # Canonical severity
        canonical_sev = "LOW"
        if severities:
            highest_rank = 0
            rank_map = {"LOW": 1, "MEDIUM": 2, "HIGH": 3}
            for s in severities:
                txt = s["text"].lower()
                for level, triggers in SEVERITY_TRIGGERS.items():
                    if any(t.lower() in txt for t in triggers):
                        if rank_map.get(level, 0) > highest_rank:
                            highest_rank = rank_map[level]
                            canonical_sev = level
        elif any(w in text.lower() for w in ["flood", "burst", "urgent", "emergency", "danger"]):
            canonical_sev = "HIGH"

        return {
            "model": model_name,
            "raw_text": raw_text,
            "locations": locations,
            "issue_types": issue_types,
            "severities": severities,
            "normalized": {
                "location": canonical_loc,
                "ward": canonical_ward,
                "issue_type": canonical_issue,
                "department": canonical_dept,
                "severity": canonical_sev
            }
        }

    def extract_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Processes a list of complaint strings."""
        return [self.extract(t) for t in texts]


if __name__ == "__main__":
    extractor = SpacyCivicExtractor()
    test_complaint = (
        "Massive sewage drainage burst in Baner near Pancard Club road. "
        "Gutter water entering homes! Urgent repair needed."
    )
    result = extractor.extract(test_complaint)
    print("Baseline Extraction Result:")
    print(f"Location Spans: {result['locations']}")
    print(f"Issue Spans:    {result['issue_types']}")
    print(f"Severity Spans: {result['severities']}")
    print(f"Normalized:     {result['normalized']}")
