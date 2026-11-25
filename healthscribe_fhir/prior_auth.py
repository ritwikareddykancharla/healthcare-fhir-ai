"""Deterministic Payer Medical-Necessity (`CPT` / `ICD-10-CM`) Prior-Authorization Engine.

Evaluates outpatient procedure and specialty medication coverage criteria against
encounter transcript spans and compiled HL7 FHIR R4 resources, returning
`AUTO_ELIGIBLE` or `MISSING_DOCUMENTATION` with exact utterance citations so clinicians
capture missing clinical evidence before payer claim denial.
"""
from __future__ import annotations

import re
import time
from typing import Any, Dict, List, Optional, Tuple

from .audit_store import SQLiteClinicalAuditStore
from .encounters import SYNTHETIC_NOTICE, get_encounter
from .fhir_validator import compile_fhir_r4_bundle
from .phi_scrubber import HipaaSafeHarborScrubber
from .soap_scribe import detect_concept_polarities

PAYER_POLICIES: Dict[str, Dict[str, Any]] = {
    "72148": {
        "cpt_code": "72148",
        "procedure_name": "MRI Lumbar Spine Without Contrast",
        "payer_policy_id": "LCD-SPINE-MRI-2026-R4",
        "covered_icd10": ["M54.16", "M51.16", "M54.17", "G83.4"],
        "summary": (
            "Requires covered lumbar radiculopathy or cauda equina ICD-10-CM diagnosis, "
            "documented neurologic exam, AND either >= 6 weeks of conservative therapy "
            "(physical therapy + anti-inflammatory trial) OR acute progressive neurologic red flags."
        ),
    },
    "J3490-GLP1": {
        "cpt_code": "J3490-GLP1",
        "procedure_name": "Semaglutide GLP-1 Therapy Escalation (Specialty Pharmacy)",
        "payer_policy_id": "PA-ENDO-GLP1-2026-R2",
        "covered_icd10": ["E11.65", "E11.9", "E11.22"],
        "summary": (
            "Requires Type 2 Diabetes ICD-10-CM diagnosis, lab-documented HbA1c >= 7.0% (LOINC 4548-4), "
            ">= 90-day trial of metformin (or documented contraindication), and absence of medullary thyroid "
            "carcinoma / MEN2 history."
        ),
    },
    "78452": {
        "cpt_code": "78452",
        "procedure_name": "Myocardial Perfusion Imaging (Nuclear Stress Test, Multiple Studies)",
        "payer_policy_id": "AUC-CARDIO-MPI-2026-R1",
        "covered_icd10": ["I20.8", "I20.9", "R07.9", "R07.89"],
        "summary": (
            "Requires documented exertional angina or ischemic equivalent, resting 12-lead ECG abnormality "
            "(such as LBBB) preventing standard exercise ECG interpretation, and acute coronary syndrome ruled out."
        ),
    },
}


def _find_matching_utterances(
    utterances: List[Dict[str, Any]],
    pattern: re.Pattern[str],
    require_unnegated_keyword: bool = False,
) -> List[Dict[str, Any]]:
    hits: List[Dict[str, Any]] = []
    for u in utterances:
        m = pattern.search(u["text"])
        if not m:
            continue
        if require_unnegated_keyword:
            prefix = u["text"][max(0, m.start() - 45) : m.start()].lower()
            if re.search(r"\b(?:no|not|never|haven't|have\s+not|without|denies|deny)\b", prefix):
                continue
        hits.append(u)
    return hits


def evaluate_prior_auth(
    encounter_id: str,
    cpt_code: Optional[str] = None,
    clinician_attestation: Optional[str] = None,
    deidentify_packet: bool = True,
    audit_store: Optional[SQLiteClinicalAuditStore] = None,
) -> Dict[str, Any]:
    """Evaluate payer prior-authorization rules against encounter dialogue and FHIR R4 chart evidence."""
    t0 = time.perf_counter()
    encounter = get_encounter(encounter_id)
    target_cpt = (cpt_code or encounter.get("requested_cpt") or "72148").strip()
    if target_cpt not in PAYER_POLICIES:
        raise ValueError(f"Unsupported CPT / prior-auth policy code: {target_cpt}")

    policy = PAYER_POLICIES[target_cpt]
    scrubber = HipaaSafeHarborScrubber(audit_store=None)
    bundle = compile_fhir_r4_bundle(encounter_id, deidentify_phi=deidentify_packet, scrubber=scrubber)

    utterances = list(encounter.get("utterances", []))
    if clinician_attestation and clinician_attestation.strip():
        utterances.append(
            {
                "utterance_id": "U-ADDENDUM",
                "speaker": "CLINICIAN",
                "start_ms": 60000,
                "end_ms": 64000,
                "text": clinician_attestation.strip(),
            }
        )

    # Extract FHIR Condition ICD-10 codes, LOINC Observations, and RxNorm Medications
    bundle_icd10: List[Tuple[str, str]] = []
    bundle_loinc: Dict[str, Tuple[float, str]] = {}
    bundle_rxnorm: List[Tuple[str, str]] = []

    for entry in bundle.get("entry", []):
        res = entry.get("resource", {})
        rtype = res.get("resourceType")
        f_url = entry.get("fullUrl", "")
        if rtype == "Condition":
            for c in (res.get("code") or {}).get("coding", []):
                bundle_icd10.append((c.get("code", ""), f_url))
        elif rtype == "Observation":
            for c in (res.get("code") or {}).get("coding", []):
                val = float((res.get("valueQuantity") or {}).get("value", 0.0))
                bundle_loinc[c.get("code", "")] = (val, f_url)
        elif rtype == "MedicationRequest":
            for c in (res.get("medicationCodeableConcept") or {}).get("coding", []):
                bundle_rxnorm.append((c.get("code", ""), f_url))

    matched_icd = [(code, ref) for code, ref in bundle_icd10 if code in policy["covered_icd10"]]
    criteria_results: List[Dict[str, Any]] = []

    if target_cpt == "72148":
        # Criterion 1: Covered ICD-10-CM lumbar radiculopathy / disc disorder
        diag_utts = _find_matching_utterances(utterances, re.compile(r"\b(?:M54\.16|M51\.16|radiculopathy)\b", re.I))
        criteria_results.append(
            {
                "criterion_id": "PA-72148-DX",
                "label": "Covered ICD-10-CM Lumbar Radiculopathy Diagnosis (M54.16 / M51.16)",
                "met": len(matched_icd) > 0,
                "fhir_ref": matched_icd[0][1] if matched_icd else None,
                "utterance_ids": [u["utterance_id"] for u in diag_utts],
                "evidence_quote": diag_utts[0]["text"] if diag_utts else "",
                "remediation_prompt": "Document specific radicular ICD-10-CM code (e.g., M54.16) in Assessment.",
            }
        )

        # Criterion 2: >= 6 weeks conservative therapy OR progressive neurologic red flag
        pt_weeks_met = False
        pt_evidence_utt: Optional[Dict[str, Any]] = None
        for u in utterances:
            m_wk = re.search(r"(\d+)\s*weeks?\s+of\s+(?:supervised\s+)?physical\s+therapy", u["text"], re.I)
            if m_wk and int(m_wk.group(1)) >= 6:
                pt_weeks_met = True
                pt_evidence_utt = u
                break

        red_flag_met = False
        red_flag_utt: Optional[Dict[str, Any]] = None
        for u in utterances:
            pols = detect_concept_polarities(u["text"])
            if (
                pols.get("saddle_anesthesia") == "affirmed"
                or pols.get("incontinence") == "affirmed"
                or pols.get("motor_weakness") == "affirmed"
            ):
                red_flag_met = True
                red_flag_utt = u
                break

        step_therapy_ok = pt_weeks_met or red_flag_met
        chosen_utt = pt_evidence_utt or red_flag_utt
        criteria_results.append(
            {
                "criterion_id": "PA-72148-STEP",
                "label": ">= 6 Weeks Supervised Physical Therapy OR Acute Red-Flag Neurologic Deficit",
                "met": step_therapy_ok,
                "fhir_ref": bundle_rxnorm[0][1] if (step_therapy_ok and bundle_rxnorm) else None,
                "utterance_ids": [chosen_utt["utterance_id"]] if chosen_utt else [],
                "evidence_quote": chosen_utt["text"] if chosen_utt else "",
                "remediation_prompt": (
                    "Order 6 weeks of conservative physical therapy before non-contrast lumbar MRI, "
                    "or document progressive motor weakness / cauda equina red-flag findings if clinically present."
                ),
            }
        )

        # Criterion 3: Objective neurologic exam documented
        exam_utts = _find_matching_utterances(
            utterances,
            re.compile(r"\b(?:straight-leg\s+raise|reflex|strength|plantarflexion|5/5|4/5)\b", re.I),
        )
        criteria_results.append(
            {
                "criterion_id": "PA-72148-EXAM",
                "label": "Documented Lower-Extremity Neurologic Physical Examination",
                "met": len(exam_utts) > 0,
                "fhir_ref": f"urn:uuid:{bundle['entry'][2]['resource']['id']}",
                "utterance_ids": [u["utterance_id"] for u in exam_utts],
                "evidence_quote": exam_utts[0]["text"] if exam_utts else "",
                "remediation_prompt": "Record motor strength (0-5/5), deep tendon reflexes, and straight-leg raise exam.",
            }
        )

    elif target_cpt == "J3490-GLP1":
        diag_utts = _find_matching_utterances(utterances, re.compile(r"\b(?:E11\.65|E11\.9|type\s+2\s+diabetes)\b", re.I))
        criteria_results.append(
            {
                "criterion_id": "PA-GLP1-DX",
                "label": "Covered Type 2 Diabetes Mellitus Diagnosis (ICD-10-CM E11.65 / E11.9)",
                "met": len(matched_icd) > 0,
                "fhir_ref": matched_icd[0][1] if matched_icd else None,
                "utterance_ids": [u["utterance_id"] for u in diag_utts],
                "evidence_quote": diag_utts[0]["text"] if diag_utts else "",
                "remediation_prompt": "Document confirmed Type 2 Diabetes ICD-10-CM diagnosis.",
            }
        )

        a1c_val, a1c_ref = bundle_loinc.get("4548-4", (0.0, ""))
        a1c_utts = _find_matching_utterances(utterances, re.compile(r"\bHbA1c\s+\d+(?:\.\d+)?%", re.I))
        criteria_results.append(
            {
                "criterion_id": "PA-GLP1-A1C",
                "label": "Recent Hemoglobin A1c >= 7.0% (LOINC 4548-4)",
                "met": a1c_val >= 7.0,
                "fhir_ref": a1c_ref or None,
                "utterance_ids": [u["utterance_id"] for u in a1c_utts],
                "evidence_quote": a1c_utts[0]["text"] if a1c_utts else "",
                "remediation_prompt": "Attach quantitative HbA1c lab Observation (LOINC 4548-4) within past 90 days.",
            }
        )

        met_rx = [ref for code, ref in bundle_rxnorm if code == "860975"]
        met_utts = _find_matching_utterances(
            utterances,
            re.compile(r"\b(\d+)\s*months?.*?\bmetformin|\bmetformin.*?\b(\d+)\s*months?", re.I),
            require_unnegated_keyword=True,
        )
        criteria_results.append(
            {
                "criterion_id": "PA-GLP1-METFORMIN",
                "label": ">= 90-Day First-Line Metformin Trial (RxNorm 860975) or Intolerance",
                "met": bool(met_rx and met_utts),
                "fhir_ref": met_rx[0] if met_rx else None,
                "utterance_ids": [u["utterance_id"] for u in met_utts],
                "evidence_quote": met_utts[0]["text"] if met_utts else "",
                "remediation_prompt": "Document duration (>= 3 months) and daily dosage of prior metformin therapy.",
            }
        )

        mtc_ruled_out = any(
            detect_concept_polarities(u["text"]).get("medullary_thyroid_ca") == "negated" for u in utterances
        )
        mtc_utts = [
            u for u in utterances if detect_concept_polarities(u["text"]).get("medullary_thyroid_ca") == "negated"
        ]
        criteria_results.append(
            {
                "criterion_id": "PA-GLP1-SAFETY",
                "label": "Medullary Thyroid Carcinoma & MEN2 Contraindication Ruled Out",
                "met": mtc_ruled_out,
                "fhir_ref": f"urn:uuid:{bundle['entry'][0]['resource']['id']}",
                "utterance_ids": [u["utterance_id"] for u in mtc_utts],
                "evidence_quote": mtc_utts[0]["text"] if mtc_utts else "",
                "remediation_prompt": "Screen and document absence of personal/family history of MTC and MEN2.",
            }
        )

    elif target_cpt == "78452":
        angina_utts = _find_matching_utterances(
            utterances,
            re.compile(r"\b(?:exertional|walk.*?chest\s+pressure|angina|I20\.8)\b", re.I),
            require_unnegated_keyword=True,
        )
        criteria_results.append(
            {
                "criterion_id": "PA-78452-SX",
                "label": "Documented Exertional Angina or Ischemic Chest Pressure (ICD-10-CM I20.8)",
                "met": bool(matched_icd and angina_utts),
                "fhir_ref": matched_icd[0][1] if matched_icd else None,
                "utterance_ids": [u["utterance_id"] for u in angina_utts],
                "evidence_quote": angina_utts[0]["text"] if angina_utts else "",
                "remediation_prompt": "Document exertional chest symptoms and covered angina ICD-10-CM code.",
            }
        )

        ecg_utts = _find_matching_utterances(
            utterances,
            re.compile(r"\b(?:left\s+bundle\s+branch\s+block|LBBB|uninterpretable)\b", re.I),
        )
        criteria_results.append(
            {
                "criterion_id": "PA-78452-ECG",
                "label": "Baseline Resting 12-Lead ECG Uninterpretable for Exercise Treadmill (e.g., LBBB)",
                "met": len(ecg_utts) > 0,
                "fhir_ref": f"urn:uuid:{bundle['entry'][2]['resource']['id']}",
                "utterance_ids": [u["utterance_id"] for u in ecg_utts],
                "evidence_quote": ecg_utts[0]["text"] if ecg_utts else "",
                "remediation_prompt": "Document why standard exercise ECG (CPT 93015) is uninterpretable (e.g., LBBB, ventricular pacing).",
            }
        )

        acs_ruled_out = any(
            detect_concept_polarities(u["text"]).get("acute_coronary_syndrome") == "negated" for u in utterances
        )
        acs_utts = [
            u for u in utterances if detect_concept_polarities(u["text"]).get("acute_coronary_syndrome") == "negated"
        ]
        criteria_results.append(
            {
                "criterion_id": "PA-78452-ACS",
                "label": "Acute Coronary Syndrome Ruled Out Prior to Outpatient Stress Imaging",
                "met": acs_ruled_out,
                "fhir_ref": f"urn:uuid:{bundle['entry'][2]['resource']['id']}",
                "utterance_ids": [u["utterance_id"] for u in acs_utts],
                "evidence_quote": acs_utts[0]["text"] if acs_utts else "",
                "remediation_prompt": "Confirm absence of rest angina / negative troponin before outpatient stress imaging.",
            }
        )

    missing = [c for c in criteria_results if not c["met"]]
    determination = "AUTO_ELIGIBLE" if len(missing) == 0 else "MISSING_DOCUMENTATION"
    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 3)

    if deidentify_packet:
        for c in criteria_results:
            if c["evidence_quote"]:
                c["evidence_quote"] = scrubber.scrub_text(c["evidence_quote"])["scrubbed_text"]

    if audit_store:
        audit_store.append_event(
            encounter_id=encounter_id,
            action_type="PRIOR_AUTH_CHECK",
            verdict=determination,
            duration_ms=elapsed_ms,
            details={
                "cpt_code": target_cpt,
                "payer_policy_id": policy["payer_policy_id"],
                "criteria_met": len(criteria_results) - len(missing),
                "criteria_total": len(criteria_results),
            },
        )

    return {
        "encounter_id": encounter_id,
        "synthetic_label": SYNTHETIC_NOTICE,
        "cpt_code": target_cpt,
        "procedure_name": policy["procedure_name"],
        "payer_policy_id": policy["payer_policy_id"],
        "policy_summary": policy["summary"],
        "determination": determination,
        "criteria_met_count": len(criteria_results) - len(missing),
        "criteria_total_count": len(criteria_results),
        "criteria": criteria_results,
        "missing_actions": [c["remediation_prompt"] for c in missing],
        "deidentified_packet": deidentify_packet,
        "eval_ms": elapsed_ms,
    }
