"""
evaluate.py
============
Evaluation & Comparative Benchmark Script: Model 1 (spaCy) vs Model 2 (Transformer NER).

Metrics Computed:
- Precision, Recall, and F1-score (Strict boundary & Relaxed span overlap).
- Per-Entity Category Performance: [LOCATION], [ISSUE_TYPE], [SEVERITY].
- Downstream Classification Accuracy: Ward Mapping, Department Assignment, Severity Level.
- System Efficiency: Latency per sample (ms) and Throughput (complaints/sec).
- Formatted tabular and JSON reporting for project presentations and reports.
"""

import time
import json
from typing import List, Dict, Any, Tuple
from preprocess import generate_pmc_dataset
from model_baseline_spacy import SpacyCivicExtractor
from model_transformer_ner import TransformerCivicExtractor


def calculate_span_overlap(pred_span: Tuple[int, int], true_span: Tuple[int, int]) -> float:
    """Computes Intersection over Union (IoU) of two character spans."""
    p_start, p_end = pred_span
    t_start, t_end = true_span

    # Check for invalid or virtual offsets
    if p_start < 0 or t_start < 0:
        return 0.0

    inter_start = max(p_start, t_start)
    inter_end = min(p_end, t_end)
    intersection = max(0, inter_end - inter_start)

    union = (p_end - p_start) + (t_end - t_start) - intersection
    return (intersection / union) if union > 0 else 0.0


def evaluate_model_on_dataset(model_extractor, dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Evaluates an NLP extraction model against annotated ground-truth test records.
    """
    records = dataset.to_dict(orient="records") if hasattr(dataset, "to_dict") else dataset

    # Tracking per entity type
    entity_stats = {
        "LOCATION": {"tp": 0, "fp": 0, "fn": 0, "relaxed_tp": 0},
        "ISSUE_TYPE": {"tp": 0, "fp": 0, "fn": 0, "relaxed_tp": 0},
        "SEVERITY": {"tp": 0, "fp": 0, "fn": 0, "relaxed_tp": 0}
    }

    # Tracking downstream normalization accuracy
    norm_acc = {
        "ward_correct": 0,
        "dept_correct": 0,
        "sev_correct": 0,
        "total": len(records)
    }

    start_time = time.perf_counter()

    for item in records:
        raw_text = item.get("raw_text", "")
        ground_truth_spans = item.get("entities", [])
        gt_ward = item.get("ward", "")
        gt_dept = item.get("department", "")
        gt_sev = item.get("ground_truth_severity", "")

        # Run model inference
        prediction = model_extractor.extract(raw_text)
        norm = prediction.get("normalized", {})

        # Check normalization accuracy
        if norm.get("ward", "").strip() == gt_ward.strip():
            norm_acc["ward_correct"] += 1
        if norm.get("department", "").strip() == gt_dept.strip():
            norm_acc["dept_correct"] += 1
        if norm.get("severity", "").strip() == gt_sev.strip():
            norm_acc["sev_correct"] += 1

        # Check entity spans
        pred_entities = []
        for loc in prediction.get("locations", []):
            pred_entities.append({"start": loc.get("start", -1), "end": loc.get("end", -1), "label": "LOCATION", "text": loc.get("text", "")})
        for iss in prediction.get("issue_types", []):
            pred_entities.append({"start": iss.get("start", -1), "end": iss.get("end", -1), "label": "ISSUE_TYPE", "text": iss.get("text", "")})
        for sev in prediction.get("severities", []):
            pred_entities.append({"start": sev.get("start", -1), "end": sev.get("end", -1), "label": "SEVERITY", "text": sev.get("text", "")})

        # Match predicted spans with ground truth
        for label in ["LOCATION", "ISSUE_TYPE", "SEVERITY"]:
            gt_for_label = [g for g in ground_truth_spans if g.get("label") == label]
            pred_for_label = [p for p in pred_entities if p.get("label") == label]

            matched_gt = set()
            for p in pred_for_label:
                p_span = (p["start"], p["end"])
                p_text = p["text"].lower()

                strict_hit = False
                relaxed_hit = False

                for g_idx, g in enumerate(gt_for_label):
                    if g_idx in matched_gt:
                        continue
                    g_span = (g["start"], g["end"])
                    g_text = g["text"].lower()

                    # Strict match: exact boundaries or text identity
                    if (p_span == g_span and p_span[0] >= 0) or (p_text == g_text):
                        strict_hit = True
                        matched_gt.add(g_idx)
                        break
                    # Relaxed match: partial substring or token overlap
                    elif calculate_span_overlap(p_span, g_span) > 0.3 or p_text in g_text or g_text in p_text:
                        relaxed_hit = True
                        matched_gt.add(g_idx)
                        break

                if strict_hit:
                    entity_stats[label]["tp"] += 1
                    entity_stats[label]["relaxed_tp"] += 1
                elif relaxed_hit:
                    entity_stats[label]["relaxed_tp"] += 1
                    entity_stats[label]["fp"] += 1
                else:
                    entity_stats[label]["fp"] += 1

            # False negatives: ground truth entities not recovered
            entity_stats[label]["fn"] += (len(gt_for_label) - len(matched_gt))

    total_latency_sec = time.perf_counter() - start_time
    ms_per_sample = round((total_latency_sec / max(len(records), 1)) * 1000, 2)
    throughput = round(len(records) / max(total_latency_sec, 0.0001), 1)

    # Compute Precision, Recall, F1 metrics
    results_by_label = {}
    total_tp, total_fp, total_fn = 0, 0, 0
    total_rel_tp = 0

    for label, counts in entity_stats.items():
        tp = counts["tp"]
        fp = counts["fp"]
        fn = counts["fn"]
        rel_tp = counts["relaxed_tp"]

        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = (2 * prec * rec) / max(prec + rec, 0.00001)

        rel_prec = rel_tp / max(rel_tp + fp, 1)
        rel_rec = rel_tp / max(rel_tp + fn, 1)
        rel_f1 = (2 * rel_prec * rel_rec) / max(rel_prec + rel_rec, 0.00001)

        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_rel_tp += rel_tp

        results_by_label[label] = {
            "strict_precision": round(prec, 4),
            "strict_recall": round(rec, 4),
            "strict_f1": round(f1, 4),
            "relaxed_f1": round(rel_f1, 4)
        }

    micro_prec = total_tp / max(total_tp + total_fp, 1)
    micro_rec = total_tp / max(total_tp + total_fn, 1)
    micro_f1 = (2 * micro_prec * micro_rec) / max(micro_prec + micro_rec, 0.00001)

    return {
        "model_name": getattr(model_extractor, "model_name", "Baseline_spaCy"),
        "entity_metrics": results_by_label,
        "overall_micro": {
            "precision": round(micro_prec, 4),
            "recall": round(micro_rec, 4),
            "f1_score": round(micro_f1, 4)
        },
        "downstream_accuracy": {
            "ward_accuracy": round(norm_acc["ward_correct"] / norm_acc["total"], 4),
            "department_accuracy": round(norm_acc["dept_correct"] / norm_acc["total"], 4),
            "severity_accuracy": round(norm_acc["sev_correct"] / norm_acc["total"], 4)
        },
        "latency_ms_per_sample": ms_per_sample,
        "throughput_samples_per_sec": throughput
    }


def run_comparative_evaluation() -> Dict[str, Any]:
    """
    Executes a side-by-side comparative evaluation of Model 1 (spaCy) and Model 2 (Transformer).
    """
    print("=====================================================================")
    print("  PMC Civic Grievance NLP Benchmark: Baseline vs. Transformer NER   ")
    print("=====================================================================")

    dataset = generate_pmc_dataset()

    print("\n[1/2] Evaluating Model 1: spaCy Pattern Matcher Baseline...")
    spacy_model = SpacyCivicExtractor()
    spacy_metrics = evaluate_model_on_dataset(spacy_model, dataset)

    print("\n[2/2] Evaluating Model 2: Transformer Multilingual Token Classifier...")
    transformer_model = TransformerCivicExtractor()
    transformer_metrics = evaluate_model_on_dataset(transformer_model, dataset)

    comparison_report = {
        "model_1_baseline_spacy": spacy_metrics,
        "model_2_transformer_ner": transformer_metrics
    }

    # Print summary table
    print("\n" + "=" * 76)
    print(f"{'METRIC':<26} | {'MODEL 1 (spaCy Baseline)':<22} | {'MODEL 2 (Transformer)':<22}")
    print("=" * 76)

    m1_ent = spacy_metrics["entity_metrics"]
    m2_ent = transformer_metrics["entity_metrics"]

    for ent_type in ["LOCATION", "ISSUE_TYPE", "SEVERITY"]:
        m1_f1 = f"{m1_ent[ent_type]['strict_f1'] * 100:.1f}%"
        m2_f1 = f"{m2_ent[ent_type]['strict_f1'] * 100:.1f}%"
        print(f"[{ent_type}] F1-Score{'':<10} | {m1_f1:<22} | {m2_f1:<22}")

    print("-" * 76)
    m1_overall = f"{spacy_metrics['overall_micro']['f1_score'] * 100:.1f}%"
    m2_overall = f"{transformer_metrics['overall_micro']['f1_score'] * 100:.1f}%"
    print(f"Overall Micro F1-Score     | {m1_overall:<22} | {m2_overall:<22}")

    print("-" * 76)
    m1_ward = f"{spacy_metrics['downstream_accuracy']['ward_accuracy'] * 100:.1f}%"
    m2_ward = f"{transformer_metrics['downstream_accuracy']['ward_accuracy'] * 100:.1f}%"
    print(f"Ward Mapping Accuracy      | {m1_ward:<22} | {m2_ward:<22}")

    m1_dept = f"{spacy_metrics['downstream_accuracy']['department_accuracy'] * 100:.1f}%"
    m2_dept = f"{transformer_metrics['downstream_accuracy']['department_accuracy'] * 100:.1f}%"
    print(f"Dept Classification Acc    | {m1_dept:<22} | {m2_dept:<22}")

    m1_sev = f"{spacy_metrics['downstream_accuracy']['severity_accuracy'] * 100:.1f}%"
    m2_sev = f"{transformer_metrics['downstream_accuracy']['severity_accuracy'] * 100:.1f}%"
    print(f"Severity Classification Acc| {m1_sev:<22} | {m2_sev:<22}")

    print("-" * 76)
    m1_lat = f"{spacy_metrics['latency_ms_per_sample']} ms"
    m2_lat = f"{transformer_metrics['latency_ms_per_sample']} ms"
    print(f"Latency per Complaint      | {m1_lat:<22} | {m2_lat:<22}")

    m1_tp = f"{spacy_metrics['throughput_samples_per_sec']} req/s"
    m2_tp = f"{transformer_metrics['throughput_samples_per_sec']} req/s"
    print(f"Inference Throughput       | {m1_tp:<22} | {m2_tp:<22}")
    print("=" * 76)

    # Save to JSON
    with open("evaluation_report.json", "w", encoding="utf-8") as f:
        json.dump(comparison_report, f, indent=2)
    print("\n[*] Full evaluation report written to evaluation_report.json\n")

    return comparison_report


if __name__ == "__main__":
    run_comparative_evaluation()
