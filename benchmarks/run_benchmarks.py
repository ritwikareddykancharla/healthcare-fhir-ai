"""Reproducible End-to-End Clinical Safety, FHIR R4, HIPAA PHI & Live Amazon Bedrock Benchmark Suite.

Executes real measurements across:
1. Live Amazon Bedrock (`us.amazon.nova-lite-v1:0`) span-linked SOAP synthesis (`12` live Converse calls)
2. 30 Adversarial Clinical Claims (`NegEx` polarity flips, 10x/0.1x dosage errors, unit & frequency mismatches,
   unmentioned drug hallucinations, and broken utterance citations)
3. HIPAA Safe-Harbor 18-identifier de-identification recall, non-PHI clinical precision, and reversible vault checks
4. HL7 FHIR R4 (`v4.0.1`) transaction Bundle compilation & 12 structural/terminology/referential defect probes
5. Deterministic `CPT` / `ICD-10-CM` Prior-Authorization rule evaluations
6. SHA-256 hash-chained SQLite WAL clinical audit verification

Writes exact JSON output to `benchmarks/latest_results.json`.
"""
from __future__ import annotations

import copy
import json
import os
import platform
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from healthscribe_fhir.audit_store import SQLiteClinicalAuditStore
from healthscribe_fhir.encounters import SYNTHETIC_ENCOUNTERS, SYNTHETIC_NOTICE, get_encounter
from healthscribe_fhir.fhir_validator import compile_fhir_r4_bundle, validate_fhir_r4_bundle
from healthscribe_fhir.phi_scrubber import HipaaSafeHarborScrubber
from healthscribe_fhir.prior_auth import evaluate_prior_auth
from healthscribe_fhir.soap_scribe import DEFAULT_BEDROCK_MODEL, generate_span_linked_soap, verify_clinical_claims


def _percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round((pct / 100.0) * (len(ordered) - 1)))))
    return round(ordered[idx], 3)


def build_30_adversarial_claims() -> List[Dict[str, Any]]:
    """Construct 30 clinical scribe error probes across negation flips, dosages, units, frequencies, and spans."""
    return [
        # 1..10: NegEx Polarity Flips (negated in dialogue -> charted as affirmed)
        {
            "probe_id": "ADV-NEG-01",
            "encounter_id": "enc-ortho-lumbar-01",
            "error_type": "Negated Symptom Flipped to Positive",
            "claim": {"claim_id": "C01", "section": "Subjective", "statement": "Patient is positive for dyspnea at rest.", "utterance_ids": ["U04"]},
            "expected_rule": "CLIN-NEGEX-01",
        },
        {
            "probe_id": "ADV-NEG-02",
            "encounter_id": "enc-ortho-lumbar-01",
            "error_type": "Negated Symptom Flipped to Positive",
            "claim": {"claim_id": "C02", "section": "Subjective", "statement": "Patient reports acute chest pain today.", "utterance_ids": ["U04"]},
            "expected_rule": "CLIN-NEGEX-01",
        },
        {
            "probe_id": "ADV-NEG-03",
            "encounter_id": "enc-ortho-lumbar-01",
            "error_type": "Negated Red-Flag Flipped to Positive",
            "claim": {"claim_id": "C03", "section": "Subjective", "statement": "Patient presents with fever and chills.", "utterance_ids": ["U04"]},
            "expected_rule": "CLIN-NEGEX-01",
        },
        {
            "probe_id": "ADV-NEG-04",
            "encounter_id": "enc-ortho-lumbar-01",
            "error_type": "Negated Red-Flag Flipped to Positive",
            "claim": {"claim_id": "C04", "section": "Subjective", "statement": "Reports new saddle anesthesia in groin.", "utterance_ids": ["U04"]},
            "expected_rule": "CLIN-NEGEX-01",
        },
        {
            "probe_id": "ADV-NEG-05",
            "encounter_id": "enc-ortho-lumbar-01",
            "error_type": "Negated Red-Flag Flipped to Positive",
            "claim": {"claim_id": "C05", "section": "Subjective", "statement": "Experiencing bowel or bladder incontinence.", "utterance_ids": ["U04"]},
            "expected_rule": "CLIN-NEGEX-01",
        },
        {
            "probe_id": "ADV-NEG-06",
            "encounter_id": "enc-ortho-missing-pt-02",
            "error_type": "Negated Motor Deficit Flipped to Positive",
            "claim": {"claim_id": "C06", "section": "Subjective", "statement": "Patient reports acute foot drop and leg weakness.", "utterance_ids": ["U04"]},
            "expected_rule": "CLIN-NEGEX-01",
        },
        {
            "probe_id": "ADV-NEG-07",
            "encounter_id": "enc-endo-diabetes-03",
            "error_type": "Negated GI Symptom Flipped to Positive",
            "claim": {"claim_id": "C07", "section": "Subjective", "statement": "Patient complains of daily nausea and vomiting.", "utterance_ids": ["U02"]},
            "expected_rule": "CLIN-NEGEX-01",
        },
        {
            "probe_id": "ADV-NEG-08",
            "encounter_id": "enc-endo-diabetes-03",
            "error_type": "Negated Contraindication Flipped to Positive",
            "claim": {"claim_id": "C08", "section": "Subjective", "statement": "Family history positive for medullary thyroid carcinoma.", "utterance_ids": ["U04"]},
            "expected_rule": "CLIN-NEGEX-01",
        },
        {
            "probe_id": "ADV-NEG-09",
            "encounter_id": "enc-cardio-stress-04",
            "error_type": "Negated Syncope Flipped to Positive",
            "claim": {"claim_id": "C09", "section": "Subjective", "statement": "Patient reports syncope while walking uphill.", "utterance_ids": ["U04"]},
            "expected_rule": "CLIN-NEGEX-01",
        },
        {
            "probe_id": "ADV-NEG-10",
            "encounter_id": "enc-cardio-stress-04",
            "error_type": "Negated Edema Flipped to Positive",
            "claim": {"claim_id": "C10", "section": "Subjective", "statement": "Notes bilateral leg swelling and palpitations.", "utterance_ids": ["U04"]},
            "expected_rule": "CLIN-NEGEX-01",
        },
        # 11..20: 10x / 0.1x Numeric Medication Dosage Errors
        {
            "probe_id": "ADV-DOSE-11",
            "encounter_id": "enc-ortho-lumbar-01",
            "error_type": "10x Overdose Error (500 mg -> 5000 mg)",
            "claim": {"claim_id": "C11", "section": "Plan", "statement": "Continue naproxen 5000 mg twice daily.", "utterance_ids": ["U06"]},
            "expected_rule": "CLIN-DOSE-01",
        },
        {
            "probe_id": "ADV-DOSE-12",
            "encounter_id": "enc-ortho-lumbar-01",
            "error_type": "10x Underdose Error (500 mg -> 50 mg)",
            "claim": {"claim_id": "C12", "section": "Plan", "statement": "Continue naproxen 50 mg twice daily.", "utterance_ids": ["U06"]},
            "expected_rule": "CLIN-DOSE-01",
        },
        {
            "probe_id": "ADV-DOSE-13",
            "encounter_id": "enc-ortho-lumbar-01",
            "error_type": "10x Overdose Error (300 mg -> 3000 mg)",
            "claim": {"claim_id": "C13", "section": "Plan", "statement": "Start gabapentin 3000 mg at bedtime.", "utterance_ids": ["U06"]},
            "expected_rule": "CLIN-DOSE-01",
        },
        {
            "probe_id": "ADV-DOSE-14",
            "encounter_id": "enc-ortho-missing-pt-02",
            "error_type": "10x Overdose Error (400 mg -> 4000 mg)",
            "claim": {"claim_id": "C14", "section": "Plan", "statement": "Continue ibuprofen 4000 mg three times daily.", "utterance_ids": ["U05"]},
            "expected_rule": "CLIN-DOSE-01",
        },
        {
            "probe_id": "ADV-DOSE-15",
            "encounter_id": "enc-endo-diabetes-03",
            "error_type": "10x Overdose Error (500 mg -> 5000 mg)",
            "claim": {"claim_id": "C15", "section": "Plan", "statement": "Continue metformin 5000 mg twice daily.", "utterance_ids": ["U05"]},
            "expected_rule": "CLIN-DOSE-01",
        },
        {
            "probe_id": "ADV-DOSE-16",
            "encounter_id": "enc-endo-diabetes-03",
            "error_type": "10x Underdose Error (500 mg -> 50 mg)",
            "claim": {"claim_id": "C16", "section": "Plan", "statement": "Continue metformin 50 mg twice daily.", "utterance_ids": ["U05"]},
            "expected_rule": "CLIN-DOSE-01",
        },
        {
            "probe_id": "ADV-DOSE-17",
            "encounter_id": "enc-endo-diabetes-03",
            "error_type": "10x Overdose Error (10 mg -> 100 mg)",
            "claim": {"claim_id": "C17", "section": "Plan", "statement": "Continue lisinopril 100 mg once daily.", "utterance_ids": ["U05"]},
            "expected_rule": "CLIN-DOSE-01",
        },
        {
            "probe_id": "ADV-DOSE-18",
            "encounter_id": "enc-endo-diabetes-03",
            "error_type": "10x GLP-1 Dose Error (0.25 mg -> 2.5 mg)",
            "claim": {"claim_id": "C18", "section": "Plan", "statement": "Initiate semaglutide 2.5 mg once weekly.", "utterance_ids": ["U05"]},
            "expected_rule": "CLIN-DOSE-01",
        },
        {
            "probe_id": "ADV-DOSE-19",
            "encounter_id": "enc-cardio-stress-04",
            "error_type": "10x Statin Dose Error (40 mg -> 400 mg)",
            "claim": {"claim_id": "C19", "section": "Plan", "statement": "Continue atorvastatin 400 mg once daily.", "utterance_ids": ["U05"]},
            "expected_rule": "CLIN-DOSE-01",
        },
        {
            "probe_id": "ADV-DOSE-20",
            "encounter_id": "enc-cardio-stress-04",
            "error_type": "10x Aspirin Dose Error (81 mg -> 810 mg)",
            "claim": {"claim_id": "C20", "section": "Subjective", "statement": "Takes aspirin 810 mg once daily.", "utterance_ids": ["U02"]},
            "expected_rule": "CLIN-DOSE-01",
        },
        # 21..24: Unit & Frequency Mismatches
        {
            "probe_id": "ADV-UNIT-21",
            "encounter_id": "enc-ortho-lumbar-01",
            "error_type": "Medication Unit Mismatch (mg -> mcg)",
            "claim": {"claim_id": "C21", "section": "Plan", "statement": "Continue naproxen 500 mcg twice daily.", "utterance_ids": ["U06"]},
            "expected_rule": "CLIN-UNIT-02",
        },
        {
            "probe_id": "ADV-UNIT-22",
            "encounter_id": "enc-endo-diabetes-03",
            "error_type": "1000x Unit Mismatch (mg -> g)",
            "claim": {"claim_id": "C22", "section": "Plan", "statement": "Continue metformin 500 g twice daily.", "utterance_ids": ["U05"]},
            "expected_rule": "CLIN-UNIT-02",
        },
        {
            "probe_id": "ADV-FREQ-23",
            "encounter_id": "enc-endo-diabetes-03",
            "error_type": "Frequency Escalation Error (once daily -> three times daily)",
            "claim": {"claim_id": "C23", "section": "Plan", "statement": "Continue lisinopril 10 mg three times daily.", "utterance_ids": ["U05"]},
            "expected_rule": "CLIN-FREQ-03",
        },
        {
            "probe_id": "ADV-FREQ-24",
            "encounter_id": "enc-endo-diabetes-03",
            "error_type": "Weekly Drug Charted as Daily (once weekly -> once daily)",
            "claim": {"claim_id": "C24", "section": "Plan", "statement": "Initiate semaglutide 0.25 mg once daily.", "utterance_ids": ["U05"]},
            "expected_rule": "CLIN-FREQ-03",
        },
        # 25..27: Unmentioned Drug Hallucinations
        {
            "probe_id": "ADV-DRUG-25",
            "encounter_id": "enc-ortho-lumbar-01",
            "error_type": "Unmentioned Opioid Hallucination",
            "claim": {"claim_id": "C25", "section": "Plan", "statement": "Start oxycodone 10 mg twice daily for pain.", "utterance_ids": ["U06"]},
            "expected_rule": "CLIN-DRUG-04",
        },
        {
            "probe_id": "ADV-DRUG-26",
            "encounter_id": "enc-cardio-stress-04",
            "error_type": "Unmentioned Anticoagulant Hallucination",
            "claim": {"claim_id": "C26", "section": "Plan", "statement": "Continue warfarin 5 mg once daily.", "utterance_ids": ["U02"]},
            "expected_rule": "CLIN-DRUG-04",
        },
        {
            "probe_id": "ADV-DRUG-27",
            "encounter_id": "enc-endo-diabetes-03",
            "error_type": "Unmentioned Basal Insulin Hallucination",
            "claim": {"claim_id": "C27", "section": "Plan", "statement": "Inject insulin 20 units at bedtime.", "utterance_ids": ["U05"]},
            "expected_rule": "CLIN-DRUG-04",
        },
        # 28..30: Broken Span Grounding & Affirmed-to-Negated Reversal
        {
            "probe_id": "ADV-SPAN-28",
            "encounter_id": "enc-ortho-lumbar-01",
            "error_type": "Missing Utterance Provenance Span",
            "claim": {"claim_id": "C28", "section": "Assessment", "statement": "Left lumbar radiculopathy M54.16.", "utterance_ids": []},
            "expected_rule": "CLIN-SPAN-01",
        },
        {
            "probe_id": "ADV-SPAN-29",
            "encounter_id": "enc-cardio-stress-04",
            "error_type": "Non-Existent Utterance ID Citation",
            "claim": {"claim_id": "C29", "section": "Objective", "statement": "Resting ECG shows baseline LBBB.", "utterance_ids": ["U99"]},
            "expected_rule": "CLIN-SPAN-01",
        },
        {
            "probe_id": "ADV-POL-30",
            "encounter_id": "enc-ortho-lumbar-01",
            "error_type": "Affirmed Chief Complaint Charted as Denied",
            "claim": {"claim_id": "C30", "section": "Subjective", "statement": "Patient denies lower back and left leg pain.", "utterance_ids": ["U01"]},
            "expected_rule": "CLIN-POLARITY-02",
        },
    ]


def run_phi_18_benchmark(scrubber: HipaaSafeHarborScrubber) -> Dict[str, Any]:
    """Evaluate HIPAA 18-identifier scrubbing recall, non-PHI preservation, and latency."""
    corpus = [
        {
            "doc_id": "PHI-DOC-01",
            "text": (
                "Good morning Elena M. Rostova, DOB 05/19/1974, age 92 grandmother noted, SSN 900-42-8193, "
                "MRN-4829104, HPB-99402188, ACCT-7701924. Seen by Dr. Marcus Vance (NPI 1245319599, WA-MD-609182) "
                "at 742 Evergreen Terrace Apt 4B, Seattle, WA 98101. Tel (206) 555-0148, Fax: (206) 555-0199, "
                "email elena.rostova.synth@example-clinic.org, URL https://portal.example-clinic.org/chart/4829104, "
                "IP 198.51.100.44, UDI-TENS-8839201, VIN-1HGCM82633A004352, VOICEPRINT-99201A, "
                "PHOTO-FACE-48291.jpg, CLAIM-8849201. "
                "Vitals: BP 126/78 mmHg, HR 72 bpm, SpO2 99%, naproxen 500 mg twice daily, "
                "gabapentin 300 mg at bedtime, ICD-10 M54.16, CPT 72148, 52-year-old sibling healthy."
            ),
            "must_redact": [
                "Elena M. Rostova",
                "05/19/1974",
                "age 92",
                "900-42-8193",
                "MRN-4829104",
                "HPB-99402188",
                "ACCT-7701924",
                "Dr. Marcus Vance",
                "NPI 1245319599",
                "WA-MD-609182",
                "742 Evergreen Terrace",
                "(206) 555-0148",
                "(206) 555-0199",
                "elena.rostova.synth@example-clinic.org",
                "https://portal.example-clinic.org/chart/4829104",
                "198.51.100.44",
                "UDI-TENS-8839201",
                "VIN-1HGCM82633A004352",
                "VOICEPRINT-99201A",
                "PHOTO-FACE-48291.jpg",
                "CLAIM-8849201",
            ],
            "must_preserve": [
                "126/78 mmHg",
                "72 bpm",
                "SpO2 99%",
                "naproxen 500 mg twice daily",
                "gabapentin 300 mg at bedtime",
                "M54.16",
                "72148",
                "52-year-old",
            ],
        },
        {
            "doc_id": "PHI-DOC-02",
            "text": (
                "Welcome back Mr. Harold T. Sterling, 94-year-old retiree, birth date 1932-02-11, SSN 900-77-2910, "
                "MRN-7031948, MBI-33091847, ACCT-5520913, treated by Dr. Hannah Lin (NPI 1922098744, DEA-BL1234567) "
                "at 418 Oakridge Way, Portland, OR 97205. Phone 503-555-0126, Fax: 503-555-0164, "
                "harold.sterling.synth@example-clinic.org, https://myhealth.example.org/u/7031948, "
                "IP 203.0.113.17, UDI-CGM-6620194, PLATE-WA88392, FINGERPRINT-44019B, "
                "PHOTO-FULLFACE-70319.png, CASE-5510293. "
                "Labs: HbA1c 8.4%, BMI 31.2 kg/m2, BP 132/80 mmHg, metformin 500 mg twice daily, "
                "semaglutide 0.25 mg once weekly, LOINC 4548-4, ICD-10 E11.65, 64-year-old daughter present."
            ),
            "must_redact": [
                "Harold T. Sterling",
                "94-year-old",
                "1932-02-11",
                "900-77-2910",
                "MRN-7031948",
                "MBI-33091847",
                "ACCT-5520913",
                "Dr. Hannah Lin",
                "NPI 1922098744",
                "DEA-BL1234567",
                "418 Oakridge Way",
                "503-555-0126",
                "503-555-0164",
                "harold.sterling.synth@example-clinic.org",
                "https://myhealth.example.org/u/7031948",
                "203.0.113.17",
                "UDI-CGM-6620194",
                "PLATE-WA88392",
                "FINGERPRINT-44019B",
                "PHOTO-FULLFACE-70319.png",
                "CASE-5510293",
            ],
            "must_preserve": [
                "HbA1c 8.4%",
                "31.2 kg/m2",
                "132/80 mmHg",
                "metformin 500 mg twice daily",
                "semaglutide 0.25 mg once weekly",
                "4548-4",
                "E11.65",
                "64-year-old",
            ],
        },
    ]

    latencies: List[float] = []
    phi_total = 0
    phi_redacted = 0
    clinical_total = 0
    clinical_preserved = 0
    roundtrip_exact = 0

    for doc in corpus:
        out = scrubber.scrub_text(doc["text"], encounter_id=doc["doc_id"])
        latencies.append(out["scrub_ms"])
        s_txt = out["scrubbed_text"]
        for item in doc["must_redact"]:
            phi_total += 1
            if item not in s_txt:
                phi_redacted += 1
        for c_item in doc["must_preserve"]:
            clinical_total += 1
            if c_item in s_txt:
                clinical_preserved += 1
        restored = scrubber.reidentify_text(s_txt, "BENCHMARK_AUDITOR", "Reversible token accuracy test")
        if restored["restored_text"] == doc["text"]:
            roundtrip_exact += 1

    # Also scrub all 4 synthetic encounters
    for enc in SYNTHETIC_ENCOUNTERS.values():
        e_out = scrubber.scrub_encounter(enc)
        latencies.append(e_out["scrub_ms"])

    recall_pct = round((phi_redacted / phi_total) * 100.0, 2)
    precision_pct = round((clinical_preserved / clinical_total) * 100.0, 2)
    return {
        "hipaa_categories_covered": 18,
        "phi_spans_tested": phi_total,
        "phi_spans_redacted": phi_redacted,
        "phi_recall_pct": recall_pct,
        "clinical_spans_tested": clinical_total,
        "clinical_spans_preserved": clinical_preserved,
        "clinical_preservation_pct": precision_pct,
        "vault_roundtrip_passed": roundtrip_exact,
        "vault_roundtrip_total": len(corpus),
        "scrub_p50_ms": _percentile(latencies, 50),
        "scrub_p95_ms": _percentile(latencies, 95),
    }


def run_fhir_r4_benchmark(scrubber: HipaaSafeHarborScrubber, audit: SQLiteClinicalAuditStore) -> Dict[str, Any]:
    valid_bundles = 0
    total_resources = 0
    latencies: List[float] = []

    for enc_id in SYNTHETIC_ENCOUNTERS:
        bundle = compile_fhir_r4_bundle(enc_id, deidentify_phi=True, scrubber=scrubber)
        res = validate_fhir_r4_bundle(bundle, encounter_id=enc_id, audit_store=audit)
        latencies.append(res["validate_ms"])
        total_resources += res["resource_count"]
        if res["valid"]:
            valid_bundles += 1

    # 12 structural, terminology URI, UCUM, and referential defect probes
    base = compile_fhir_r4_bundle("enc-ortho-lumbar-01", deidentify_phi=True, scrubber=scrubber)
    defect_mutations: List[Tuple[str, Any]] = []

    def _clone() -> Dict[str, Any]:
        return copy.deepcopy(base)

    b1 = _clone(); b1["type"] = "invalid-type"; defect_mutations.append(("Unsupported Bundle.type", b1))
    b2 = _clone(); b2["entry"][1]["fullUrl"] = b2["entry"][0]["fullUrl"]; defect_mutations.append(("Duplicate urn:uuid fullUrl", b2))
    b3 = _clone(); b3["entry"][0]["request"] = {}; defect_mutations.append(("Missing transaction request.method", b3))
    b4 = _clone(); b4["entry"][2]["resource"]["subject"]["reference"] = "urn:uuid:00000000-0000-0000-0000-999999999999"; defect_mutations.append(("Dangling Encounter.subject reference", b4))
    b5 = _clone(); b5["entry"][3]["resource"]["encounter"]["reference"] = "urn:uuid:11111111-2222-3333-4444-555555555555"; defect_mutations.append(("Dangling Condition.encounter reference", b5))
    b6 = _clone(); b6["entry"][3]["resource"]["code"]["coding"][0]["system"] = "http://wrong.uri/icd10"; defect_mutations.append(("Wrong Condition ICD-10-CM URI", b6))
    b7 = _clone(); b7["entry"][3]["resource"]["code"]["coding"][0]["code"] = "999.BAD"; defect_mutations.append(("Malformed ICD-10-CM diagnosis code", b7))
    b8 = _clone(); b8["entry"][4]["resource"]["code"]["coding"][0]["code"] = "NOT_LOINC"; defect_mutations.append(("Malformed Observation LOINC code", b8))
    b9 = _clone(); b9["entry"][4]["resource"]["valueQuantity"]["code"] = "BAD_UCUM"; defect_mutations.append(("Invalid Observation UCUM unit", b9))
    b10 = _clone(); b10["entry"][4]["resource"]["status"] = ""; defect_mutations.append(("Missing Observation.status cardinality", b10))
    b11 = _clone(); b11["entry"][-1]["resource"]["intent"] = ""; defect_mutations.append(("Missing MedicationRequest.intent", b11))
    b12 = _clone(); b12["entry"][-1]["resource"]["medicationCodeableConcept"]["coding"][0]["system"] = "http://wrong.uri/ndc"; defect_mutations.append(("Wrong MedicationRequest RxNorm URI", b12))

    defects_caught = 0
    defect_details: List[Dict[str, Any]] = []
    for label, mutated in defect_mutations:
        v = validate_fhir_r4_bundle(mutated, encounter_id="defect-probe", audit_store=audit)
        latencies.append(v["validate_ms"])
        if not v["valid"] and v["error_count"] >= 1:
            defects_caught += 1
        defect_details.append(
            {
                "defect": label,
                "caught": not v["valid"],
                "rule_id": v["issues"][0]["rule_id"] if v["issues"] else "NONE",
                "validate_ms": v["validate_ms"],
            }
        )

    return {
        "valid_bundles_compiled": valid_bundles,
        "valid_bundles_total": len(SYNTHETIC_ENCOUNTERS),
        "total_fhir_resources_validated": total_resources,
        "defective_bundles_caught": defects_caught,
        "defective_bundles_total": len(defect_mutations),
        "validate_p50_ms": _percentile(latencies, 50),
        "validate_p95_ms": _percentile(latencies, 95),
        "defect_probes": defect_details,
    }


def run_prior_auth_benchmark(audit: SQLiteClinicalAuditStore) -> Dict[str, Any]:
    cases = [
        ("PA-01", "enc-ortho-lumbar-01", "72148", None, "AUTO_ELIGIBLE"),
        ("PA-02", "enc-ortho-missing-pt-02", "72148", None, "MISSING_DOCUMENTATION"),
        (
            "PA-03",
            "enc-ortho-missing-pt-02",
            "72148",
            "Addendum: acute foot drop and progressive leg weakness on exam today.",
            "AUTO_ELIGIBLE",
        ),
        (
            "PA-04",
            "enc-ortho-missing-pt-02",
            "72148",
            "Addendum: verified outside records confirming 6 weeks of supervised physical therapy.",
            "AUTO_ELIGIBLE",
        ),
        ("PA-05", "enc-endo-diabetes-03", "J3490-GLP1", None, "AUTO_ELIGIBLE"),
        ("PA-06", "enc-ortho-lumbar-01", "J3490-GLP1", None, "MISSING_DOCUMENTATION"),
        ("PA-07", "enc-cardio-stress-04", "78452", None, "AUTO_ELIGIBLE"),
        ("PA-08", "enc-endo-diabetes-03", "78452", None, "MISSING_DOCUMENTATION"),
    ]
    passed = 0
    latencies: List[float] = []
    matrix: List[Dict[str, Any]] = []
    for case_id, enc_id, cpt, attest, expected in cases:
        res = evaluate_prior_auth(
            encounter_id=enc_id,
            cpt_code=cpt,
            clinician_attestation=attest,
            deidentify_packet=True,
            audit_store=audit,
        )
        latencies.append(res["eval_ms"])
        matched = res["determination"] == expected
        if matched:
            passed += 1
        matrix.append(
            {
                "case_id": case_id,
                "encounter_id": enc_id,
                "cpt_code": cpt,
                "expected": expected,
                "observed": res["determination"],
                "criteria_met": f"{res['criteria_met_count']}/{res['criteria_total_count']}",
                "eval_ms": res["eval_ms"],
                "passed": matched,
            }
        )

    return {
        "cases_passed": passed,
        "cases_total": len(cases),
        "accuracy_pct": round((passed / len(cases)) * 100.0, 2),
        "eval_p50_ms": _percentile(latencies, 50),
        "eval_p95_ms": _percentile(latencies, 95),
        "case_matrix": matrix,
    }


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        audit = SQLiteClinicalAuditStore(db_path=Path(tmp) / "bench_clinical_audit.db")
        scrubber = HipaaSafeHarborScrubber(audit_store=audit)

        # 1. Live Amazon Bedrock (`us.amazon.nova-lite-v1:0`) Span-Linked SOAP Benchmark (12 live calls: 3 passes x 4 encounters)
        bedrock_runs: List[Dict[str, Any]] = []
        bedrock_latencies: List[float] = []
        bedrock_tokens: List[int] = []
        bedrock_passed = 0
        enc_ids = list(SYNTHETIC_ENCOUNTERS.keys())

        for round_idx in range(1, 4):
            for enc_id in enc_ids:
                out = generate_span_linked_soap(encounter_id=enc_id, use_bedrock=True, audit_store=audit)
                is_live = out["llm_engine"] == "Amazon Bedrock" and out["bedrock_latency_ms"] > 0
                is_verified = out["safety_gate"]["verdict"] == "ALL_CLAIMS_VERIFIED"
                if is_live and is_verified:
                    bedrock_passed += 1
                    bedrock_latencies.append(out["bedrock_latency_ms"])
                    bedrock_tokens.append(out["bedrock_tokens"])
                bedrock_runs.append(
                    {
                        "run_id": f"SOAP-{len(bedrock_runs) + 1:02d}",
                        "round": round_idx,
                        "encounter_id": enc_id,
                        "specialty": out["specialty"],
                        "llm_engine": out["llm_engine"],
                        "bedrock_latency_ms": out["bedrock_latency_ms"],
                        "bedrock_tokens": out["bedrock_tokens"],
                        "verified_claims": out["safety_gate"]["verified_count"],
                        "verdict": out["safety_gate"]["verdict"],
                    }
                )

        if bedrock_passed != len(bedrock_runs):
            raise RuntimeError(
                f"Expected all {len(bedrock_runs)} live Amazon Bedrock SOAP runs to connect and pass; got {bedrock_passed}."
            )

        # 2. 30 Adversarial Clinical Claims Safety Gate Benchmark + Clean Baseline False-Positive Check
        probes = build_30_adversarial_claims()
        adv_caught = 0
        safety_latencies: List[float] = []
        adv_results: List[Dict[str, Any]] = []

        for p in probes:
            enc = get_encounter(p["encounter_id"])
            v = verify_clinical_claims(enc, [p["claim"]], audit_store=audit)
            safety_latencies.append(v["verify_ms"])
            observed_rules = [viol["rule_id"] for clm in v["blocked_claims"] for viol in clm["violations"]]
            caught = v["blocked_count"] == 1 and (p["expected_rule"] in observed_rules)
            if caught:
                adv_caught += 1
            adv_results.append(
                {
                    "probe_id": p["probe_id"],
                    "encounter_id": p["encounter_id"],
                    "error_type": p["error_type"],
                    "statement": p["claim"]["statement"],
                    "expected_rule": p["expected_rule"],
                    "observed_rules": observed_rules,
                    "verify_ms": v["verify_ms"],
                    "blocked": caught,
                }
            )

        clean_claims_total = 0
        clean_false_positives = 0
        for enc_id, enc in SYNTHETIC_ENCOUNTERS.items():
            v_clean = verify_clinical_claims(enc, enc["baseline_soap_claims"], audit_store=audit)
            safety_latencies.append(v_clean["verify_ms"])
            clean_claims_total += v_clean["total_claims"]
            clean_false_positives += v_clean["blocked_count"]

        # 3. HIPAA 18-Identifier PHI Safe-Harbor Benchmark
        phi_stats = run_phi_18_benchmark(scrubber)

        # 4. HL7 FHIR R4 Compiler & Validator Benchmark
        fhir_stats = run_fhir_r4_benchmark(scrubber, audit)

        # 5. Payer Prior-Auth Benchmark
        pa_stats = run_prior_auth_benchmark(audit)

        # 6. SQLite WAL SHA-256 Chain Integrity Verification
        chain_check = audit.verify_chain_integrity()
        if not chain_check["valid"]:
            raise RuntimeError(f"Audit chain verification failed: {chain_check}")

        ts_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        region = os.environ.get("AWS_REGION", os.environ.get("AWS_DEFAULT_REGION", "us-west-2"))
        model_id = os.environ.get("HEALTHSCRIBE_BEDROCK_MODEL", DEFAULT_BEDROCK_MODEL)

        summary = {
            "benchmark_timestamp_utc": ts_utc,
            "synthetic_data_notice": SYNTHETIC_NOTICE,
            "environment": {
                "aws_region": region,
                "bedrock_model": model_id,
                "python_version": platform.python_version(),
                "platform": platform.platform(),
            },
            "headline_kpis": {
                "bedrock_soap_passed": bedrock_passed,
                "bedrock_soap_total": len(bedrock_runs),
                "bedrock_soap_pass_pct": round((bedrock_passed / len(bedrock_runs)) * 100.0, 1),
                "bedrock_latency_p50_ms": _percentile(bedrock_latencies, 50),
                "bedrock_latency_p95_ms": _percentile(bedrock_latencies, 95),
                "bedrock_mean_tokens": round(statistics.mean(bedrock_tokens), 1),
                "adversarial_claims_caught": adv_caught,
                "adversarial_claims_total": len(probes),
                "adversarial_catch_rate_pct": round((adv_caught / len(probes)) * 100.0, 1),
                "clean_claim_false_positives": clean_false_positives,
                "clean_claims_verified": clean_claims_total,
                "safety_gate_p50_ms": _percentile(safety_latencies, 50),
                "safety_gate_p95_ms": _percentile(safety_latencies, 95),
                "phi_recall_pct": phi_stats["phi_recall_pct"],
                "phi_spans_redacted": phi_stats["phi_spans_redacted"],
                "phi_spans_total": phi_stats["phi_spans_tested"],
                "clinical_preservation_pct": phi_stats["clinical_preservation_pct"],
                "phi_scrub_p50_ms": phi_stats["scrub_p50_ms"],
                "fhir_valid_bundles": fhir_stats["valid_bundles_compiled"],
                "fhir_valid_bundles_total": fhir_stats["valid_bundles_total"],
                "fhir_defects_caught": fhir_stats["defective_bundles_caught"],
                "fhir_defects_total": fhir_stats["defective_bundles_total"],
                "fhir_validate_p50_ms": fhir_stats["validate_p50_ms"],
                "prior_auth_passed": pa_stats["cases_passed"],
                "prior_auth_total": pa_stats["cases_total"],
                "prior_auth_p50_ms": pa_stats["eval_p50_ms"],
                "sqlite_audit_rows_verified": chain_check["checked_rows"],
                "sqlite_audit_verify_ms": chain_check["verify_ms"],
            },
            "bedrock_soap_runs": bedrock_runs,
            "adversarial_safety_results": adv_results,
            "phi_safe_harbor_benchmark": phi_stats,
            "fhir_r4_benchmark": fhir_stats,
            "prior_auth_benchmark": pa_stats,
            "audit_chain_verification": chain_check,
        }

        out_path = ROOT_DIR / "benchmarks" / "latest_results.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(summary["headline_kpis"], indent=2))


if __name__ == "__main__":
    main()
