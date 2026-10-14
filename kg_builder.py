"""
kg_builder.py
==============
Knowledge Graph Construction & SPARQL Query Engine using RDF/OWL & rdflib.

Ontology Design:
- Custom Namespace: http://smartcity.pune.gov.in/ontology# (pmc)
- Classes:
    - pmc:Complaint
    - pmc:IssueType
    - pmc:Ward
    - pmc:Department
    - pmc:SeverityLevel
- Object Properties:
    - pmc:has_issue (Complaint -> IssueType)
    - pmc:located_in (Complaint -> Ward)
    - pmc:handled_by (IssueType -> Department)
    - pmc:has_severity (Complaint -> SeverityLevel)
- Datatype Properties:
    - pmc:complaint_id (xsd:string)
    - pmc:raw_text (xsd:string)
    - pmc:filing_date (xsd:date)
    - pmc:status (xsd:string)
    - pmc:extraction_model (xsd:string)

Includes:
- Graph building from NLP extracted entities.
- SPARQL query execution for analytics and downstream anomaly detection.
- Export to Turtle (.ttl), N-Triples, and JSON-LD.
- Graceful standalone RDF store fallback if rdflib is pending pip installation.
"""

import json
import re
from urllib.parse import quote
from typing import List, Dict, Any, Optional, Tuple

try:
    import pandas as pd
except ImportError:
    pd = None

# RDFLib imports
try:
    import rdflib
    from rdflib import Graph, Literal, Namespace, RDF, RDFS, OWL, XSD, URIRef
    RDFLIB_AVAILABLE = True
except ImportError:
    RDFLIB_AVAILABLE = False


# ---------------------------------------------------------------------------
# Ontology Namespaces & Constants
# ---------------------------------------------------------------------------

PMC_NAMESPACE_URI = "http://smartcity.pune.gov.in/ontology#"

if RDFLIB_AVAILABLE:
    PMC = Namespace(PMC_NAMESPACE_URI)
else:
    class MockNamespace:
        def __init__(self, base):
            self.base = base
        def __getattr__(self, name):
            return f"{self.base}{name}"
        def __getitem__(self, name):
            return f"{self.base}{name}"
    PMC = MockNamespace(PMC_NAMESPACE_URI)


def clean_uri_component(text: str) -> str:
    """Creates a URL-safe identifier for RDF URIs."""
    cleaned = re.sub(r"[^\w\s-]", "", text).strip()
    return quote(cleaned.replace(" ", "_"))


class CivicKGBuilder:
    """
    Manages Knowledge Graph construction, RDF serializations, and SPARQL queries.
    """

    def __init__(self):
        self.is_rdflib_active = RDFLIB_AVAILABLE
        self.triples_list: List[Tuple[str, str, str]] = []

        if self.is_rdflib_active:
            self.graph = Graph()
            self._bind_namespaces()
            self._init_ontology_schema()
        else:
            self.graph = None
            print("[*] rdflib not installed; using built-in RDF triple store.")

    def _bind_namespaces(self):
        """Binds standard and domain prefixes to the RDF graph."""
        self.graph.bind("pmc", PMC)
        self.graph.bind("rdf", RDF)
        self.graph.bind("rdfs", RDFS)
        self.graph.bind("owl", OWL)
        self.graph.bind("xsd", XSD)

    def _init_ontology_schema(self):
        """Declares OWL classes and properties in the schema."""
        g = self.graph
        # Classes
        for cls_name in ["Complaint", "IssueType", "Ward", "Department", "SeverityLevel"]:
            cls_uri = PMC[cls_name]
            g.add((cls_uri, RDF.type, OWL.Class))
            g.add((cls_uri, RDFS.label, Literal(cls_name)))

        # Object Properties
        obj_props = [
            ("has_issue", PMC.Complaint, PMC.IssueType),
            ("located_in", PMC.Complaint, PMC.Ward),
            ("handled_by", PMC.IssueType, PMC.Department),
            ("has_severity", PMC.Complaint, PMC.SeverityLevel)
        ]
        for prop_name, domain, rng in obj_props:
            prop_uri = PMC[prop_name]
            g.add((prop_uri, RDF.type, OWL.ObjectProperty))
            g.add((prop_uri, RDFS.domain, domain))
            g.add((prop_uri, RDFS.range, rng))

    def add_complaint_record(self,
                             complaint_id: str,
                             raw_text: str,
                             filing_date: str,
                             extracted_dict: Dict[str, Any],
                             status: str = "Open") -> str:
        """
        Adds a single civic complaint and its extracted entities/relations to the KG.
        """
        norm = extracted_dict.get("normalized", {})
        loc_name = norm.get("location", "Pune")
        ward_name = norm.get("ward", f"Ward - {loc_name}")
        issue_name = norm.get("issue_type", "General Grievance")
        dept_name = norm.get("department", "Municipal Administration")
        sev_name = norm.get("severity", "LOW")
        model_name = extracted_dict.get("model", "NLP_Extractor")

        # URIs
        complaint_uri_str = f"{PMC_NAMESPACE_URI}complaint/{clean_uri_component(complaint_id)}"
        ward_uri_str = f"{PMC_NAMESPACE_URI}ward/{clean_uri_component(ward_name)}"
        issue_uri_str = f"{PMC_NAMESPACE_URI}issue/{clean_uri_component(issue_name)}"
        dept_uri_str = f"{PMC_NAMESPACE_URI}dept/{clean_uri_component(dept_name)}"
        sev_uri_str = f"{PMC_NAMESPACE_URI}severity/{clean_uri_component(sev_name)}"

        if self.is_rdflib_active and self.graph is not None:
            c_uri = URIRef(complaint_uri_str)
            w_uri = URIRef(ward_uri_str)
            i_uri = URIRef(issue_uri_str)
            d_uri = URIRef(dept_uri_str)
            s_uri = URIRef(sev_uri_str)

            g = self.graph

            # 1. Complaint Instance & Datatypes
            g.add((c_uri, RDF.type, PMC.Complaint))
            g.add((c_uri, PMC.complaint_id, Literal(complaint_id, datatype=XSD.string)))
            g.add((c_uri, PMC.raw_text, Literal(raw_text, datatype=XSD.string)))
            g.add((c_uri, PMC.filing_date, Literal(filing_date, datatype=XSD.date)))
            g.add((c_uri, PMC.status, Literal(status, datatype=XSD.string)))
            g.add((c_uri, PMC.extraction_model, Literal(model_name, datatype=XSD.string)))

            # 2. Ward Instance
            g.add((w_uri, RDF.type, PMC.Ward))
            g.add((w_uri, RDFS.label, Literal(ward_name, datatype=XSD.string)))

            # 3. Department Instance
            g.add((d_uri, RDF.type, PMC.Department))
            g.add((d_uri, RDFS.label, Literal(dept_name, datatype=XSD.string)))

            # 4. IssueType Instance & handled_by Department
            g.add((i_uri, RDF.type, PMC.IssueType))
            g.add((i_uri, RDFS.label, Literal(issue_name, datatype=XSD.string)))
            g.add((i_uri, PMC.handled_by, d_uri))

            # 5. Severity Instance
            g.add((s_uri, RDF.type, PMC.SeverityLevel))
            g.add((s_uri, RDFS.label, Literal(sev_name, datatype=XSD.string)))

            # 6. Complaint Object Relationships
            g.add((c_uri, PMC.has_issue, i_uri))
            g.add((c_uri, PMC.located_in, w_uri))
            g.add((c_uri, PMC.has_severity, s_uri))

        # Keep internal records for portable querying
        self.triples_list.append({
            "complaint_id": complaint_id,
            "complaint_uri": complaint_uri_str,
            "raw_text": raw_text,
            "filing_date": filing_date,
            "ward": ward_name,
            "ward_uri": ward_uri_str,
            "issue": issue_name,
            "issue_uri": issue_uri_str,
            "department": dept_name,
            "department_uri": dept_uri_str,
            "severity": sev_name,
            "severity_uri": sev_uri_str,
            "model": model_name,
            "status": status
        })

        return complaint_uri_str

    def build_from_dataset(self, dataset: List[Dict[str, Any]], extractor) -> int:
        """
        Processes a full dataset list or DataFrame through the specified NLP extractor,
        building all corresponding RDF triples.
        """
        count = 0
        records = dataset.to_dict(orient="records") if hasattr(dataset, "to_dict") else dataset

        for item in records:
            c_id = item.get("complaint_id", f"PMC-{count+1:04d}")
            text = item.get("raw_text") or item.get("cleaned_text", "")
            date_str = item.get("filing_date", "2024-03-01")
            status = item.get("status", "Open")

            # Extract entities using NLP model
            extracted = extractor.extract(text)
            self.add_complaint_record(c_id, text, date_str, extracted, status=status)
            count += 1

        print(f"[✓] Knowledge Graph successfully populated with {count} complaints.")
        return count

    def serialize(self, format: str = "turtle", output_path: Optional[str] = None) -> str:
        """
        Serializes the Knowledge Graph into Turtle (.ttl), N-Triples, or JSON-LD.
        """
        if self.is_rdflib_active and self.graph is not None:
            serialized_str = self.graph.serialize(format=format)
        else:
            # Fallback simple Turtle generator
            lines = [
                "@prefix pmc: <http://smartcity.pune.gov.in/ontology#> .",
                "@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .",
                "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .",
                "@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .\n"
            ]
            for t in self.triples_list:
                c_uri = f"pmc:{clean_uri_component(t['complaint_id'])}"
                w_uri = f"pmc:{clean_uri_component(t['ward'])}"
                i_uri = f"pmc:{clean_uri_component(t['issue'])}"
                d_uri = f"pmc:{clean_uri_component(t['department'])}"
                s_uri = f"pmc:{clean_uri_component(t['severity'])}"

                lines.append(f"{c_uri} a pmc:Complaint ;")
                lines.append(f'    pmc:complaint_id "{t["complaint_id"]}"^^xsd:string ;')
                lines.append(f'    pmc:filing_date "{t["filing_date"]}"^^xsd:date ;')
                lines.append(f'    pmc:raw_text "{t["raw_text"][:80]}..."^^xsd:string ;')
                lines.append(f"    pmc:located_in {w_uri} ;")
                lines.append(f"    pmc:has_issue {i_uri} ;")
                lines.append(f"    pmc:has_severity {s_uri} .\n")
                lines.append(f'{w_uri} a pmc:Ward ; rdfs:label "{t["ward"]}" .\n')
                lines.append(f'{i_uri} a pmc:IssueType ; rdfs:label "{t["issue"]}" ; pmc:handled_by {d_uri} .\n')
                lines.append(f'{d_uri} a pmc:Department ; rdfs:label "{t["department"]}" .\n')
            serialized_str = "\n".join(lines)

        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(serialized_str)
            print(f"[*] RDF Triples saved to {output_path}")

        return serialized_str

    # -----------------------------------------------------------------------
    # SPARQL Query Execution
    # -----------------------------------------------------------------------

    def execute_sparql(self, query_str: str) -> Any:
        """
        Executes a SPARQL query against the RDF graph.
        Returns a pandas DataFrame (if pandas available) or list of dicts.
        """
        if self.is_rdflib_active and self.graph is not None:
            q_res = self.graph.query(query_str)
            results = []
            for row in q_res:
                row_dict = {}
                for var in q_res.vars:
                    val = row[var]
                    row_dict[str(var)] = str(val) if val is not None else None
                results.append(row_dict)
        else:
            # Portable query parser for standard template queries
            results = self._fallback_sparql_engine(query_str)

        if pd is not None:
            return pd.DataFrame(results)
        return results

    def _fallback_sparql_engine(self, query_str: str) -> List[Dict[str, Any]]:
        """Parses and executes predefined SPARQL analytical queries over internal records."""
        q = query_str.lower()
        if "count" in q and "date" in q and "ward" in q:
            # Query 1: Daily Aggregation by Ward and Date
            counts = {}
            for t in self.triples_list:
                key = (t["filing_date"], t["ward"], t["issue"])
                counts[key] = counts.get(key, 0) + 1
            res = []
            for (date_val, ward_val, issue_val), cnt in sorted(counts.items()):
                res.append({
                    "date": date_val,
                    "ward": ward_val,
                    "issueName": issue_val,
                    "count": cnt
                })
            return res

        elif "high" in q and "deptname" in q:
            # Query 2: High Severity complaints
            res = []
            for t in self.triples_list:
                if t["severity"] == "HIGH":
                    res.append({
                        "complaintId": t["complaint_id"],
                        "date": t["filing_date"],
                        "ward": t["ward"],
                        "issueName": t["issue"],
                        "deptName": t["department"],
                        "text": t["raw_text"]
                    })
            return res

        elif "deptname" in q and "totalcomplaints" in q:
            # Query 3: Department Workload
            dept_counts = {}
            for t in self.triples_list:
                d = t["department"]
                dept_counts[d] = dept_counts.get(d, 0) + 1
            res = []
            for d, cnt in sorted(dept_counts.items(), key=lambda x: x[1], reverse=True):
                res.append({
                    "deptName": d,
                    "totalComplaints": cnt
                })
            return res

        elif "ward" in q and "complaintcount" in q:
            # Query 4: Ward Complaint Density
            ward_counts = {}
            for t in self.triples_list:
                w = t["ward"]
                ward_counts[w] = ward_counts.get(w, 0) + 1
            res = []
            for w, cnt in sorted(ward_counts.items(), key=lambda x: x[1], reverse=True):
                res.append({
                    "ward": w,
                    "complaintCount": cnt
                })
            return res

        # Default: list all complaints with metadata
        return [
            {
                "complaintId": t["complaint_id"],
                "date": t["filing_date"],
                "ward": t["ward"],
                "issue": t["issue"],
                "department": t["department"],
                "severity": t["severity"]
            }
            for t in self.triples_list
        ]

    # Pre-defined Analytical SPARQL Queries
    SPARQL_DAILY_WARD_AGGREGATION = """
    PREFIX pmc: <http://smartcity.pune.gov.in/ontology#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    SELECT ?date ?ward ?issueName (COUNT(?complaint) AS ?count)
    WHERE {
      ?complaint a pmc:Complaint ;
                 pmc:filing_date ?date ;
                 pmc:located_in ?w ;
                 pmc:has_issue ?i .
      ?w rdfs:label ?ward .
      ?i rdfs:label ?issueName .
    }
    GROUP BY ?date ?ward ?issueName
    ORDER BY ?date ?ward
    """

    SPARQL_HIGH_SEVERITY_ALERTS = """
    PREFIX pmc: <http://smartcity.pune.gov.in/ontology#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    SELECT ?complaintId ?date ?ward ?issueName ?deptName ?text
    WHERE {
      ?complaint a pmc:Complaint ;
                 pmc:complaint_id ?complaintId ;
                 pmc:filing_date ?date ;
                 pmc:located_in ?w ;
                 pmc:has_issue ?i ;
                 pmc:has_severity ?sev ;
                 pmc:raw_text ?text .
      ?w rdfs:label ?ward .
      ?i rdfs:label ?issueName ;
         pmc:handled_by ?d .
      ?sev rdfs:label ?sevLabel .
      FILTER (str(?sevLabel) = "HIGH")
    }
    ORDER BY DESC(?date)
    """

    SPARQL_DEPARTMENT_WORKLOAD = """
    PREFIX pmc: <http://smartcity.pune.gov.in/ontology#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    SELECT ?deptName (COUNT(?complaint) AS ?totalComplaints)
    WHERE {
      ?complaint a pmc:Complaint ;
                 pmc:has_issue ?i .
      ?i pmc:handled_by ?d .
      ?d rdfs:label ?deptName .
    }
    GROUP BY ?deptName
    ORDER BY DESC(?totalComplaints)
    """

    SPARQL_WARD_DENSITY = """
    PREFIX pmc: <http://smartcity.pune.gov.in/ontology#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    SELECT ?ward (COUNT(?complaint) AS ?complaintCount)
    WHERE {
      ?complaint a pmc:Complaint ;
                 pmc:located_in ?w .
      ?w rdfs:label ?ward .
    }
    GROUP BY ?ward
    ORDER BY DESC(?complaintCount)
    """


if __name__ == "__main__":
    from preprocess import generate_pmc_dataset
    from model_baseline_spacy import SpacyCivicExtractor

    print("[*] Generating mock dataset...")
    df = generate_pmc_dataset()
    extractor = SpacyCivicExtractor()

    kg = CivicKGBuilder()
    kg.build_from_dataset(df, extractor)

    # Export Turtle format
    ttl = kg.serialize(format="turtle", output_path="pune_civic_kg.ttl")
    print(f"Serialized graph contains {len(kg.triples_list)} complaint nodes.")

    print("\n--- SPARQL: High Severity Urgent Grievances ---")
    high_sev_df = kg.execute_sparql(kg.SPARQL_HIGH_SEVERITY_ALERTS)
    if hasattr(high_sev_df, "head"):
        print(high_sev_df.head(3))
    else:
        print(json.dumps(high_sev_df[:3], indent=2))
