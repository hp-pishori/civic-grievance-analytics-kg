"""
app.py
======
Streamlit Interactive Dashboard for Pune Municipal Corporation (PMC) Smart City
Civic Grievance Ontology Extraction & Anomaly Detection System.

Features:
- Tab 1: Live Complaint Extraction & Side-by-Side Model Comparison (spaCy vs Transformer)
- Tab 2: RDF Knowledge Graph Explorer & Interactive SPARQL Studio
- Tab 3: Time-Series Anomaly Detection Dashboard with Isolation Forest Alerts
- Tab 4: Comprehensive Model Benchmark & Academic Evaluation Metrics
"""

import os
import json
import time
from datetime import datetime
from typing import List, Dict, Any

import streamlit as st

# Local modules
from preprocess import (
    generate_pmc_dataset,
    clean_text,
    PUNE_LOCATIONS,
    WARD_MAPPING,
    ISSUE_DEPARTMENT_MAP
)
from model_baseline_spacy import SpacyCivicExtractor
from model_transformer_ner import TransformerCivicExtractor
from kg_builder import CivicKGBuilder
from anomaly_detection import CivicAnomalyDetector
from evaluate import evaluate_model_on_dataset

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


# ---------------------------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="PMC Civic Grievance NLP & Anomaly Graph",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
    .alert-card {
        background-color: #FEF2F2;
        border-left: 5px solid #EF4444;
        border-radius: 6px;
        padding: 14px;
        margin-bottom: 12px;
    }
    .badge-loc {
        background-color: #DBEAFE;
        color: #1E40AF;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-issue {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-sev {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Cached Resource Loaders
# ---------------------------------------------------------------------------

@st.cache_data
def load_dataset():
    """Generates and caches the 50-record Pune civic grievance dataset."""
    return generate_pmc_dataset()

@st.cache_resource
def load_extractors():
    """Initializes and caches the spaCy and Transformer NER extractors."""
    m1 = SpacyCivicExtractor()
    m2 = TransformerCivicExtractor()
    return m1, m2

@st.cache_resource
def build_default_kg(_dataset, _extractor):
    """Populates RDF Knowledge Graph using the chosen extractor."""
    builder = CivicKGBuilder()
    builder.build_from_dataset(_dataset, _extractor)
    return builder


# Load core components
dataset_raw = load_dataset()
records_list = dataset_raw.to_dict(orient="records") if hasattr(dataset_raw, "to_dict") else dataset_raw
spacy_model, transformer_model = load_extractors()


# ---------------------------------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------------------------------

st.sidebar.image("https://img.icons8.com/color/96/city-hall.png", width=70)
st.sidebar.title("PMC Smart City")
st.sidebar.caption("Civic Grievance Ontology & Anomaly System")
st.sidebar.markdown("---")

active_model_choice = st.sidebar.selectbox(
    "Active Knowledge Graph Model:",
    ["Model 2: Multilingual Transformer NER", "Model 1: spaCy EntityRuler Baseline"]
)
active_extractor = transformer_model if "Transformer" in active_model_choice else spacy_model

# Anomaly detection settings
st.sidebar.markdown("### Anomaly Settings")
contamination_rate = st.sidebar.slider("Isolation Forest Contamination:", 0.02, 0.25, 0.10, 0.02)
rolling_window = st.sidebar.slider("Rolling Window (Days):", 2, 7, 3, 1)

st.sidebar.markdown("---")
st.sidebar.info("""
**Tech Stack:**
- **NLP:** spaCy vs Hugging Face Transformers
- **Ontology:** RDF / OWL / SPARQL (rdflib)
- **Analytics:** scikit-learn Isolation Forest
- **Dataset:** 50 Pune Grievances (Hinglish/English)
""")


# ---------------------------------------------------------------------------
# Main Header
# ---------------------------------------------------------------------------

st.markdown('<div class="main-title">🏙️ Pune Smart City: Civic Grievance Ontology & Anomaly Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Comparative NLP Named Entity Extraction, RDF Knowledge Graph Construction, and Spatiotemporal Anomaly Detection</div>', unsafe_allow_html=True)

# Build Knowledge Graph
kg = build_default_kg(records_list, active_extractor)

tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Model Comparison & Extraction",
    "🕸️ Knowledge Graph & SPARQL",
    "🚨 Anomaly Detection Dashboard",
    "📊 Benchmark & Evaluation"
])


# ---------------------------------------------------------------------------
# TAB 1: Live Complaint Extraction & Side-by-Side Model Comparison
# ---------------------------------------------------------------------------

with tab1:
    st.subheader("Interactive Civic Complaint Entity Extraction")
    st.markdown("Analyze raw Pune complaints with code-switched Hindi/Marathi loanwords (`kachra`, `gatar`, `khadde`).")

    sample_presets = [
        "Massive sewage drainage burst in Baner near Pancard Club road. Gutter water entering homes! Urgent repair needed.",
        "Sanitation strike impact: mountains of uncollected kachra garbage piled in Kothrud ward. Extreme health hazard.",
        "Deep potholes on DP road near Kothrud depot causing severe traffic jams. Urgent repair required.",
        "Low pressure dirty water supply in Shivajinagar near Modern College. Paani smells like sewage.",
        "Streetlight switchboard open with exposed live wires in Aundh. Serious safety hazard for children.",
        "Illegal street food vendors blocking pedestrian walkway near Kalyani Nagar bridge. Hawkers creating nuisance."
    ]

    selected_preset = st.selectbox("Select a Sample Complaint Preset:", ["-- Custom Input --"] + sample_presets)

    if selected_preset != "-- Custom Input --":
        input_text = st.text_area("Complaint Text:", value=selected_preset, height=100)
    else:
        input_text = st.text_area(
            "Complaint Text:",
            value="Continuous water supply leakage from main valve chamber near Hinjewadi Phase 1 chowk since 3 days.",
            height=100
        )

    col_btn, _ = st.columns([1, 4])
    with col_btn:
        run_extract = st.button("Extract Entities & Compare", type="primary", use_container_width=True)

    if run_extract or input_text:
        st.markdown("---")
        col_m1, col_m2 = st.columns(2)

        # Run Model 1: spaCy
        t0 = time.perf_counter()
        m1_res = spacy_model.extract(input_text)
        m1_lat = (time.perf_counter() - t0) * 1000

        # Run Model 2: Transformer
        t0 = time.perf_counter()
        m2_res = transformer_model.extract(input_text)
        m2_lat = (time.perf_counter() - t0) * 1000

        def render_entity_chips(res):
            html = ""
            for loc in res.get("locations", []):
                html += f'<span class="badge-loc">📍 LOCATION: {loc["text"]}</span> '
            for iss in res.get("issue_types", []):
                html += f'<span class="badge-issue">⚠️ ISSUE: {iss["text"]}</span> '
            for sev in res.get("severities", []):
                html += f'<span class="badge-sev">🔥 SEVERITY: {sev["text"]}</span> '
            if not html:
                html = "<em>No explicit entity spans detected</em>"
            return html

        with col_m1:
            st.markdown("### 🧩 Model 1: spaCy EntityRuler (Baseline)")
            st.caption(f"Rule-Based Linguistic Matcher | Latency: `{m1_lat:.2f} ms`")
            st.markdown(render_entity_chips(m1_res), unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

            norm1 = m1_res["normalized"]
            st.json({
                "Ward": norm1["ward"],
                "Department": norm1["department"],
                "Categorized Issue": norm1["issue_type"],
                "Severity Level": norm1["severity"]
            })

        with col_m2:
            st.markdown("### 🤖 Model 2: Transformer Multilingual NER")
            st.caption(f"Sub-token Aggregation Classifier | Latency: `{m2_lat:.2f} ms`")
            st.markdown(render_entity_chips(m2_res), unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

            norm2 = m2_res["normalized"]
            st.json({
                "Ward": norm2["ward"],
                "Department": norm2["department"],
                "Categorized Issue": norm2["issue_type"],
                "Severity Level": norm2["severity"]
            })


# ---------------------------------------------------------------------------
# TAB 2: Knowledge Graph Explorer & Interactive SPARQL Studio
# ---------------------------------------------------------------------------

with tab2:
    st.subheader("RDF Knowledge Graph & Semantic Triple Store")
    st.markdown("Formal Semantic Web representation in RDF/OWL (`pmc:Complaint`, `pmc:Ward`, `pmc:Department`, `pmc:IssueType`).")

    col_k1, col_k2, col_k3, col_k4 = st.columns(4)
    with col_k1:
        st.metric("Total Complaint Nodes", len(kg.triples_list))
    with col_k2:
        st.metric("Pune Wards Mapped", len(set(t["ward"] for t in kg.triples_list)))
    with col_k3:
        st.metric("Municipal Departments", len(set(t["department"] for t in kg.triples_list)))
    with col_k4:
        st.metric("RDF Serialization", "Turtle (.ttl) / OWL")

    st.markdown("---")
    st.markdown("### 🌐 Interactive Subgraph Network Explorer")

    # Ward filter for visualization
    available_wards = ["All Wards"] + sorted(list(set(t["ward"] for t in kg.triples_list)))
    selected_vis_ward = st.selectbox("Filter Graph by Ward:", available_wards)

    filtered_triples = kg.triples_list
    if selected_vis_ward != "All Wards":
        filtered_triples = [t for t in kg.triples_list if t["ward"] == selected_vis_ward]

    # Render Node-Link Graph using Plotly
    if PLOTLY_AVAILABLE:
        # Build node and edge mappings
        nodes = {}
        edges = []

        def get_node_id(name, ntype):
            nid = f"{ntype}:{name}"
            if nid not in nodes:
                nodes[nid] = {"name": name, "type": ntype}
            return nid

        # Limit to 30 complaint nodes for clear visualization
        sub_triples = filtered_triples[:25]
        for t in sub_triples:
            c_nid = get_node_id(t["complaint_id"], "Complaint")
            w_nid = get_node_id(t["ward"].split(" - ")[-1], "Ward")
            i_nid = get_node_id(t["issue"], "Issue")
            d_nid = get_node_id(t["department"].replace(" Department", ""), "Dept")

            edges.append((c_nid, w_nid, "located_in"))
            edges.append((c_nid, i_nid, "has_issue"))
            edges.append((i_nid, d_nid, "handled_by"))

        # Simple radial layout positioning
        import math
        node_keys = list(nodes.keys())
        N = len(node_keys)
        node_pos = {}
        for idx, k in enumerate(node_keys):
            angle = 2 * math.pi * idx / max(N, 1)
            radius = 2.5 if nodes[k]["type"] == "Complaint" else 4.5
            node_pos[k] = (radius * math.cos(angle), radius * math.sin(angle))

        edge_x = []
        edge_y = []
        for src, dst, rel in edges:
            x0, y0 = node_pos[src]
            x1, y1 = node_pos[dst]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1.0, color="#94A3B8"),
            hoverinfo="none",
            mode="lines"
        )

        color_map = {
            "Complaint": "#3B82F6",
            "Ward": "#10B981",
            "Issue": "#EF4444",
            "Dept": "#F59E0B"
        }

        node_x = [node_pos[k][0] for k in node_keys]
        node_y = [node_pos[k][1] for k in node_keys]
        node_text = [f"<b>{nodes[k]['type']}</b>: {nodes[k]['name']}" for k in node_keys]
        node_color = [color_map.get(nodes[k]["type"], "#6B7280") for k in node_keys]

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode="markers+text",
            hoverinfo="text",
            text=[nodes[k]['name'][:14] for k in node_keys],
            textposition="top center",
            hovertext=node_text,
            marker=dict(
                size=[14 if nodes[k]["type"] == "Complaint" else 22 for k in node_keys],
                color=node_color,
                line=dict(width=1.5, color="#1E293B")
            )
        )

        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                showlegend=False,
                hovermode="closest",
                margin=dict(b=10, l=10, r=10, t=10),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                plot_bgcolor="#FFFFFF",
                height=520
            )
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("🔵 Blue: Complaint | 🟢 Green: Ward | 🔴 Red: Issue Type | 🟡 Yellow: Department")
    else:
        st.write("Plotly not available; showing sample triples table:")
        st.table(filtered_triples[:10])

    st.markdown("---")
    st.markdown("### 💻 SPARQL Query Studio")

    query_templates = {
        "1. Daily Complaint Count by Ward & Date (Anomaly Feed)": CivicKGBuilder.SPARQL_DAILY_WARD_AGGREGATION,
        "2. High Severity Urgent Grievances": CivicKGBuilder.SPARQL_HIGH_SEVERITY_ALERTS,
        "3. Departmental Workload Distribution": CivicKGBuilder.SPARQL_DEPARTMENT_WORKLOAD,
        "4. Ward Complaint Density": CivicKGBuilder.SPARQL_WARD_DENSITY
    }

    selected_query_title = st.selectbox("Pre-defined SPARQL Queries:", list(query_templates.keys()))
    sparql_code = st.text_area("SPARQL Query Editor:", value=query_templates[selected_query_title], height=160)

    if st.button("Execute SPARQL Query", type="secondary"):
        res = kg.execute_sparql(sparql_code)
        if hasattr(res, "empty") and not res.empty:
            st.dataframe(res, use_container_width=True)
        elif isinstance(res, list) and len(res) > 0:
            st.dataframe(pd.DataFrame(res) if pd else res, use_container_width=True)
        else:
            st.info("Query returned no matching triples.")


# ---------------------------------------------------------------------------
# TAB 3: Downstream Anomaly Detection Dashboard
# ---------------------------------------------------------------------------

with tab3:
    st.subheader("Time-Series Anomaly Detection (Isolation Forest)")
    st.markdown("""
    Aggregates daily complaints from RDF Knowledge Graph via SPARQL.
    Isolation Forest analyzes multi-factor velocity features ($C_t, \mu_3, \sigma_3, \Delta C_t$, surge ratio)
    to detect critical civic spikes requiring municipal dispatch.
    """)

    # 1. Fetch SPARQL aggregated data
    sparql_agg = kg.execute_sparql(CivicKGBuilder.SPARQL_DAILY_WARD_AGGREGATION)
    agg_records = sparql_agg.to_dict(orient="records") if hasattr(sparql_agg, "to_dict") else sparql_agg

    # 2. Run Anomaly Detection
    detector = CivicAnomalyDetector(contamination=contamination_rate)
    feats = detector.prepare_time_series_features(agg_records)
    ad_results = detector.fit_predict(feats)

    anomalies_detected = [r for r in ad_results if r["is_anomaly"]]

    col_a1, col_a2, col_a3 = st.columns(3)
    with col_a1:
        st.metric("Total Spatiotemporal Records", len(ad_results))
    with col_a2:
        st.metric("Anomalies Flagged", len(anomalies_detected))
    with col_a3:
        st.metric("Model Algorithm", "Scikit-Learn IsolationForest")

    st.markdown("---")
    st.markdown("### 🚨 Emergency Civic Dispatch Alerts")

    if anomalies_detected:
        for idx, a in enumerate(anomalies_detected):
            alert = a["alert"]
            st.markdown(f"""
            <div class="alert-card">
                <strong style="color: #DC2626;">[{alert['level']} SPIKE ALERT]</strong> &nbsp;
                <b>{alert['ward']}</b> &mdash; <i>{alert['date']}</i><br>
                <b>Issue:</b> {alert['issue']} | <b>Filed:</b> {alert['count']} complaints ({alert['surge_percent']} surge) |
                <b>Score:</b> <code>{alert['anomaly_score']}</code><br>
                <b>Recommended Dispatch Action:</b> {alert['recommendation']}
            </div>
            """, unsafe_allow_html=True)
    else:
        st.success("✅ No anomalous civic spikes detected for current threshold.")

    st.markdown("---")
    st.markdown("### 📈 Ward Complaint Timeline & Spikes")

    # Time series visualization
    if PLOTLY_AVAILABLE and pd is not None:
        df_ad = pd.DataFrame(ad_results)
        df_ad["date_dt"] = pd.to_datetime(df_ad["date"])
        df_ad = df_ad.sort_values("date_dt")

        fig_ts = px.line(
            df_ad,
            x="date",
            y="daily_count",
            color="ward",
            markers=True,
            title="Daily Complaint Trajectory across Pune Wards",
            labels={"daily_count": "Complaints / Day", "date": "Date"}
        )

        # Overlay red pulsing markers on anomalies
        df_spikes = df_ad[df_ad["is_anomaly"] == True]
        if not df_spikes.empty:
            fig_ts.add_trace(
                go.Scatter(
                    x=df_spikes["date"],
                    y=df_spikes["daily_count"],
                    mode="markers+text",
                    name="ANOMALY SPIKE",
                    text=[f"🚨 {w.split(' - ')[-1]}" for w in df_spikes["ward"]],
                    textposition="top center",
                    marker=dict(size=14, color="#EF4444", symbol="star")
                )
            )

        fig_ts.update_layout(height=480, hovermode="x unified")
        st.plotly_chart(fig_ts, use_container_width=True)
    else:
        st.write("Summary table of detected anomalies:")
        st.dataframe(anomalies_detected)


# ---------------------------------------------------------------------------
# TAB 4: Model Benchmark & System Evaluation
# ---------------------------------------------------------------------------

with tab4:
    st.subheader("Model Evaluation & Engineering Trade-Offs")
    st.markdown("Performance comparison on 50 annotated Pune complaints across Precision, Recall, F1, and Latency.")

    if st.button("Re-run Full Quantitative Benchmark", type="secondary"):
        with st.spinner("Evaluating models across test dataset..."):
            m1_eval = evaluate_model_on_dataset(spacy_model, records_list)
            m2_eval = evaluate_model_on_dataset(transformer_model, records_list)
            st.session_state["benchmark_results"] = (m1_eval, m2_eval)

    if "benchmark_results" not in st.session_state:
        # Load precomputed evaluation report if present
        if os.path.exists("evaluation_report.json"):
            with open("evaluation_report.json", "r") as f:
                report = json.load(f)
                m1_eval = report["model_1_baseline_spacy"]
                m2_eval = report["model_2_transformer_ner"]
        else:
            m1_eval = evaluate_model_on_dataset(spacy_model, records_list)
            m2_eval = evaluate_model_on_dataset(transformer_model, records_list)
    else:
        m1_eval, m2_eval = st.session_state["benchmark_results"]

    # Comparative Metrics Table
    m1_ent = m1_eval["entity_metrics"]
    m2_ent = m2_eval["entity_metrics"]

    benchmark_data = [
        {"Entity / Metric": "[LOCATION] F1-Score", "Model 1 (spaCy)": f"{m1_ent['LOCATION']['strict_f1']*100:.1f}%", "Model 2 (Transformer)": f"{m2_ent['LOCATION']['strict_f1']*100:.1f}%"},
        {"Entity / Metric": "[ISSUE_TYPE] F1-Score", "Model 1 (spaCy)": f"{m1_ent['ISSUE_TYPE']['strict_f1']*100:.1f}%", "Model 2 (Transformer)": f"{m2_ent['ISSUE_TYPE']['strict_f1']*100:.1f}%"},
        {"Entity / Metric": "[SEVERITY] F1-Score", "Model 1 (spaCy)": f"{m1_ent['SEVERITY']['strict_f1']*100:.1f}%", "Model 2 (Transformer)": f"{m2_ent['SEVERITY']['strict_f1']*100:.1f}%"},
        {"Entity / Metric": "Overall Micro F1-Score", "Model 1 (spaCy)": f"{m1_eval['overall_micro']['f1_score']*100:.1f}%", "Model 2 (Transformer)": f"{m2_eval['overall_micro']['f1_score']*100:.1f}%"},
        {"Entity / Metric": "Ward Mapping Accuracy", "Model 1 (spaCy)": f"{m1_eval['downstream_accuracy']['ward_accuracy']*100:.1f}%", "Model 2 (Transformer)": f"{m2_eval['downstream_accuracy']['ward_accuracy']*100:.1f}%"},
        {"Entity / Metric": "Department Assignment Acc", "Model 1 (spaCy)": f"{m1_eval['downstream_accuracy']['department_accuracy']*100:.1f}%", "Model 2 (Transformer)": f"{m2_eval['downstream_accuracy']['department_accuracy']*100:.1f}%"},
        {"Entity / Metric": "Latency per Sample", "Model 1 (spaCy)": f"{m1_eval['latency_ms_per_sample']} ms", "Model 2 (Transformer)": f"{m2_eval['latency_ms_per_sample']} ms"},
        {"Entity / Metric": "Throughput", "Model 1 (spaCy)": f"{m1_eval['throughput_samples_per_sec']} req/s", "Model 2 (Transformer)": f"{m2_eval['throughput_samples_per_sec']} req/s"}
    ]

    st.table(benchmark_data)

    st.markdown("### 🎓 Academic & Engineering Analysis for Project Report")
    st.markdown("""
    1. **Lexical Matching vs. Sub-token Representation:**
       - **spaCy EntityRuler (Model 1)** achieves high precision when exact locality gazetteers exist (e.g., *Kothrud*, *Baner*), but fails on phonetic Hinglish typos (*"khadda"*, *"tutlela pipe"*) or novel syntactic combinations.
       - **Transformer Token Classifier (Model 2)** handles sub-word morphosyntax (`Hin-je-wadi`) and contextual embeddings, recognizing implicit grievance complaints even when phrased casually.

    2. **Downstream Ontological Grounding:**
       - Both models feed into the formal **RDF/OWL Knowledge Graph**. Grounding extracted spans to canonical PMC Wards and Departments ensures uniform SPARQL querying regardless of raw grievance syntax.

    3. **Time-Series Anomaly Detection Rigor:**
       - Utilizing **Isolation Forest** over graph aggregations enables unsupervised detection of sudden localized surges (e.g. 7 drainage complaints in Baner within 48h), triggering automatic municipal emergency alerts.
    """)

# Footer
st.markdown("---")
st.caption("Pune Municipal Corporation Smart City NLP Mini-Project | Developed for Academic Research & Demonstration.")
