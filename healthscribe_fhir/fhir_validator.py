"""Schema-Validated HL7 FHIR R4 (v4.0.1) Transaction Bundle Compiler & Validator.

Compiles verified encounter entities into HL7 FHIR R4 `Bundle` (`type: "transaction"`)
resources (`Patient`, `Practitioner`, `Encounter`, `Condition` with `ICD-10-CM`,
`Observation` with `LOINC` + `UCUM` units, and `MedicationRequest` with `RxNorm`)
and enforces structural cardinality, code-system URI registries, and internal
`urn:uuid:` referential integrity.
"""
from __future__ import annotations

import time
import uuid
import re
from typing import Any, Dict, List, Optional, Set

from .audit_store import SQLiteClinicalAuditStore
from .encounters import SYNTHETIC_NOTICE, get_encounter
from .phi_scrubber import HipaaSafeHarborScrubber

ICD10_URI = "http://hl7.org/fhir/sid/icd-10-cm"
LOINC_URI = "http://loinc.org"
RXNORM_URI = "http://www.nlm.nih.gov/research/umls/rxnorm"
UCUM_URI = "http://unitsofmeasure.org"
NPI_URI = "http://hl7.org/fhir/sid/us-npi"
ACT_CODE_URI = "http://terminology.hl7.org/CodeSystem/v3-ActCode"
CLIN_STATUS_URI = "http://terminology.hl7.org/CodeSystem/condition-clinical"
VERIF_STATUS_URI = "http://terminology.hl7.org/CodeSystem/condition-ver-status"
OBS_CAT_URI = "http://terminology.hl7.org/CodeSystem/observation-category"

ICD10_CODE_RE = re.compile(r"^[A-TV-Z][0-9][0-9AB](?:\.[0-9A-TV-Z]{1,4})?$")
LOINC_CODE_RE = re.compile(r"^\d{1,7}-\d$")
RXNORM_CODE_RE = re.compile(r"^\d{2,10}$")
URN_UUID_RE = re.compile(r"^urn:uuid:[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")

ALLOWED_UCUM_CODES: Set[str] = {
    "mm[Hg]",
    "/min",
    "%",
    "[degF]",
    "Cel",
    "kg/m2",
    "mg",
    "ug",
    "g",
    "mL",
    "[iU]",
    "L",
}


def _det_uuid(seed: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"https://healthscribefhir.internal/{seed}"))


def compile_fhir_r4_bundle(
    encounter_id: str,
    deidentify_phi: bool = True,
    scrubber: Optional[HipaaSafeHarborScrubber] = None,
) -> Dict[str, Any]:
    """Compile an outpatient encounter into an HL7 FHIR R4 transaction Bundle."""
    encounter = get_encounter(encounter_id)
    patient_src = encounter["patient"]
    prac_src = encounter["practitioner"]
    entities = encounter["structured_entities"]
    scrub = scrubber or HipaaSafeHarborScrubber()

    patient_uuid = _det_uuid(f"{encounter_id}:Patient")
    prac_uuid = _det_uuid(f"{encounter_id}:Practitioner")
    enc_uuid = _det_uuid(f"{encounter_id}:Encounter")

    patient_ref = f"urn:uuid:{patient_uuid}"
    prac_ref = f"urn:uuid:{prac_uuid}"
    enc_ref = f"urn:uuid:{enc_uuid}"

    if deidentify_phi:
        mrn_val = scrub.make_vault_token("MRN", patient_src["mrn"])
        family_name = scrub.make_vault_token("PATIENT", patient_src["full_name"])
        given_names = ["SYNTHETIC-SAFE-HARBOR"]
        # HIPAA Safe-Harbor: keep birth year only (or 1900-01-01 when age > 89)
        birth_year = int(patient_src["birth_date"].split("-")[0])
        birth_date = "1900" if patient_src.get("age_years", 2026 - birth_year) > 89 else str(birth_year)
        prac_name = scrub.make_vault_token("CLINICIAN", prac_src["full_name"])
    else:
        mrn_val = patient_src["mrn"]
        parts = patient_src["full_name"].split()
        family_name = parts[-1]
        given_names = parts[:-1]
        birth_date = patient_src["birth_date"]
        prac_name = prac_src["full_name"]

    entries: List[Dict[str, Any]] = [
        {
            "fullUrl": patient_ref,
            "resource": {
                "resourceType": "Patient",
                "id": patient_uuid,
                "meta": {"tag": [{"system": "https://healthscribefhir.internal/synthetic", "code": "SYNTHETIC"}]},
                "identifier": [{"system": "urn:oid:2.16.840.1.113883.19.5", "value": mrn_val}],
                "name": [{"family": family_name, "given": given_names}],
                "gender": patient_src["gender"],
                "birthDate": birth_date,
            },
            "request": {"method": "POST", "url": "Patient"},
        },
        {
            "fullUrl": prac_ref,
            "resource": {
                "resourceType": "Practitioner",
                "id": prac_uuid,
                "identifier": [{"system": NPI_URI, "value": prac_src["npi"]}],
                "name": [{"text": prac_name}],
            },
            "request": {"method": "POST", "url": "Practitioner"},
        },
        {
            "fullUrl": enc_ref,
            "resource": {
                "resourceType": "Encounter",
                "id": enc_uuid,
                "status": "finished",
                "class": {"system": ACT_CODE_URI, "code": "AMB", "display": "ambulatory"},
                "subject": {"reference": patient_ref},
                "participant": [{"individual": {"reference": prac_ref}}],
                "period": {"start": "2026-08-01T09:00:00Z", "end": "2026-08-01T09:30:00Z"},
            },
            "request": {"method": "POST", "url": "Encounter"},
        },
    ]

    for idx, cond in enumerate(entities.get("conditions", []), start=1):
        c_uuid = _det_uuid(f"{encounter_id}:Condition:{idx}:{cond['code']}")
        entries.append(
            {
                "fullUrl": f"urn:uuid:{c_uuid}",
                "resource": {
                    "resourceType": "Condition",
                    "id": c_uuid,
                    "clinicalStatus": {
                        "coding": [{"system": CLIN_STATUS_URI, "code": cond.get("clinical_status", "active")}]
                    },
                    "verificationStatus": {
                        "coding": [{"system": VERIF_STATUS_URI, "code": "confirmed"}]
                    },
                    "code": {
                        "coding": [
                            {
                                "system": ICD10_URI,
                                "code": cond["code"],
                                "display": cond["display"],
                            }
                        ],
                        "text": cond["display"],
                    },
                    "subject": {"reference": patient_ref},
                    "encounter": {"reference": enc_ref},
                    "note": [{"text": f"Transcript spans: {', '.join(cond.get('utterance_ids', []))}"}],
                },
                "request": {"method": "POST", "url": "Condition"},
            }
        )

    for idx, obs in enumerate(entities.get("observations", []), start=1):
        o_uuid = _det_uuid(f"{encounter_id}:Observation:{idx}:{obs['loinc']}")
        entries.append(
            {
                "fullUrl": f"urn:uuid:{o_uuid}",
                "resource": {
                    "resourceType": "Observation",
                    "id": o_uuid,
                    "status": "final",
                    "category": [
                        {
                            "coding": [
                                {
                                    "system": OBS_CAT_URI,
                                    "code": obs.get("category", "vital-signs"),
                                }
                            ]
                        }
                    ],
                    "code": {
                        "coding": [
                            {
                                "system": LOINC_URI,
                                "code": obs["loinc"],
                                "display": obs["display"],
                            }
                        ]
                    },
                    "subject": {"reference": patient_ref},
                    "encounter": {"reference": enc_ref},
                    "valueQuantity": {
                        "value": float(obs["value"]),
                        "unit": obs["unit"],
                        "system": UCUM_URI,
                        "code": obs["ucum"],
                    },
                    "note": [{"text": f"Transcript spans: {', '.join(obs.get('utterance_ids', []))}"}],
                },
                "request": {"method": "POST", "url": "Observation"},
            }
        )

    for idx, med in enumerate(entities.get("medications", []), start=1):
        m_uuid = _det_uuid(f"{encounter_id}:MedicationRequest:{idx}:{med['rxnorm']}")
        entries.append(
            {
                "fullUrl": f"urn:uuid:{m_uuid}",
                "resource": {
                    "resourceType": "MedicationRequest",
                    "id": m_uuid,
                    "status": "active",
                    "intent": "order",
                    "medicationCodeableConcept": {
                        "coding": [
                            {
                                "system": RXNORM_URI,
                                "code": med["rxnorm"],
                                "display": med["display"],
                            }
                        ],
                        "text": med["display"],
                    },
                    "subject": {"reference": patient_ref},
                    "encounter": {"reference": enc_ref},
                    "requester": {"reference": prac_ref},
                    "dosageInstruction": [
                        {
                            "text": f"{med['drug']} {med['dose_value']:g} {med['dose_unit']} {med['frequency']}",
                            "doseAndRate": [
                                {
                                    "doseQuantity": {
                                        "value": float(med["dose_value"]),
                                        "unit": med["dose_unit"],
                                        "system": UCUM_URI,
                                        "code": med["ucum"],
                                    }
                                }
                            ],
                        }
                    ],
                    "note": [{"text": f"Transcript spans: {', '.join(med.get('utterance_ids', []))}"}],
                },
                "request": {"method": "POST", "url": "MedicationRequest"},
            }
        )

    return {
        "resourceType": "Bundle",
        "id": _det_uuid(f"{encounter_id}:Bundle"),
        "meta": {
            "lastUpdated": "2026-08-01T09:35:00Z",
            "profile": ["http://hl7.org/fhir/StructureDefinition/Bundle"],
        },
        "type": "transaction",
        "entry": entries,
    }


def validate_fhir_r4_bundle(
    bundle: Dict[str, Any],
    encounter_id: str = "bundle-validation",
    audit_store: Optional[SQLiteClinicalAuditStore] = None,
) -> Dict[str, Any]:
    """Validate an HL7 FHIR R4 Bundle for cardinality, coding URIs, UCUM units, and reference integrity."""
    t0 = time.perf_counter()
    issues: List[Dict[str, str]] = []
    checks_run = 0

    def _err(path: str, rule_id: str, message: str) -> None:
        issues.append({"severity": "error", "path": path, "rule_id": rule_id, "message": message})

    checks_run += 1
    if not isinstance(bundle, dict) or bundle.get("resourceType") != "Bundle":
        _err("resourceType", "FHIR-BND-01", "Root resourceType must equal 'Bundle'.")
        return {
            "valid": False,
            "fhir_version": "4.0.1",
            "synthetic_label": SYNTHETIC_NOTICE,
            "resource_count": 0,
            "checks_executed": checks_run,
            "error_count": len(issues),
            "issues": issues,
            "validate_ms": round((time.perf_counter() - t0) * 1000.0, 3),
        }

    b_type = bundle.get("type")
    checks_run += 1
    if b_type not in ("transaction", "collection", "document"):
        _err("type", "FHIR-BND-02", f"Unsupported or missing Bundle.type '{b_type}'.")

    entries = bundle.get("entry")
    checks_run += 1
    if not isinstance(entries, list) or len(entries) == 0:
        _err("entry", "FHIR-BND-03", "Bundle.entry must be a non-empty array of resources.")
        entries = []

    full_urls: Set[str] = set()
    resource_counts: Dict[str, int] = {}

    # Pass 1: Collect fullUrl registry and validate transaction wrappers
    for i, entry in enumerate(entries):
        e_path = f"entry[{i}]"
        full_url = entry.get("fullUrl", "")
        checks_run += 1
        if not isinstance(full_url, str) or not URN_UUID_RE.match(full_url):
            _err(f"{e_path}.fullUrl", "FHIR-REF-01", f"Invalid or missing urn:uuid fullUrl '{full_url}'.")
        elif full_url in full_urls:
            _err(f"{e_path}.fullUrl", "FHIR-REF-02", f"Duplicate Bundle fullUrl '{full_url}'.")
        else:
            full_urls.add(full_url)

        if b_type == "transaction":
            req = entry.get("request") or {}
            checks_run += 1
            if req.get("method") not in ("POST", "PUT") or not req.get("url"):
                _err(f"{e_path}.request", "FHIR-TXN-01", "Transaction entry requires request.method and request.url.")

    def _check_ref(ref_obj: Any, path: str) -> None:
        nonlocal checks_run
        checks_run += 1
        ref_str = (ref_obj or {}).get("reference") if isinstance(ref_obj, dict) else None
        if not ref_str:
            _err(path, "FHIR-REF-03", "Missing required FHIR Reference.reference string.")
        elif ref_str not in full_urls:
            _err(path, "FHIR-REF-04", f"Dangling internal Bundle reference '{ref_str}' does not resolve.")

    # Pass 2: Per-resource schema, terminology URI, UCUM unit, and referential checks
    for i, entry in enumerate(entries):
        res = entry.get("resource") or {}
        r_path = f"entry[{i}].resource"
        rtype = res.get("resourceType")
        checks_run += 1
        if not rtype:
            _err(f"{r_path}.resourceType", "FHIR-RES-01", "Missing resourceType.")
            continue
        resource_counts[rtype] = resource_counts.get(rtype, 0) + 1

        if rtype == "Patient":
            checks_run += 2
            if not res.get("identifier"):
                _err(f"{r_path}.identifier", "FHIR-PAT-01", "Patient.identifier (1..*) is required.")
            if res.get("gender") not in ("male", "female", "other", "unknown"):
                _err(f"{r_path}.gender", "FHIR-PAT-02", "Patient.gender must be an HL7 AdministrativeGender code.")

        elif rtype == "Practitioner":
            checks_run += 1
            idents = res.get("identifier") or []
            if not idents or idents[0].get("system") != NPI_URI or not re.match(r"^\d{10}$", str(idents[0].get("value", ""))):
                _err(f"{r_path}.identifier[0]", "FHIR-PRAC-01", f"Practitioner requires a 10-digit NPI under {NPI_URI}.")

        elif rtype == "Encounter":
            checks_run += 2
            if res.get("status") not in ("planned", "arrived", "triaged", "in-progress", "finished"):
                _err(f"{r_path}.status", "FHIR-ENC-01", "Invalid or missing Encounter.status.")
            cls = res.get("class") or {}
            if cls.get("system") != ACT_CODE_URI or not cls.get("code"):
                _err(f"{r_path}.class", "FHIR-ENC-02", f"Encounter.class must use {ACT_CODE_URI}.")
            _check_ref(res.get("subject"), f"{r_path}.subject")
            for p_idx, part in enumerate(res.get("participant") or []):
                _check_ref(part.get("individual"), f"{r_path}.participant[{p_idx}].individual")

        elif rtype == "Condition":
            checks_run += 2
            if not (res.get("clinicalStatus") or {}).get("coding"):
                _err(f"{r_path}.clinicalStatus", "FHIR-COND-01", "Condition.clinicalStatus is required.")
            codings = ((res.get("code") or {}).get("coding")) or []
            if not codings:
                _err(f"{r_path}.code.coding", "FHIR-COND-02", "Condition.code.coding is required.")
            else:
                c0 = codings[0]
                if c0.get("system") != ICD10_URI:
                    _err(f"{r_path}.code.coding[0].system", "FHIR-COND-03", f"Condition coding system must be {ICD10_URI}.")
                if not ICD10_CODE_RE.match(str(c0.get("code") or "")):
                    _err(f"{r_path}.code.coding[0].code", "FHIR-COND-04", f"Invalid ICD-10-CM diagnosis code '{c0.get('code')}'.")
            _check_ref(res.get("subject"), f"{r_path}.subject")
            _check_ref(res.get("encounter"), f"{r_path}.encounter")

        elif rtype == "Observation":
            checks_run += 3
            if res.get("status") not in ("registered", "preliminary", "final", "amended"):
                _err(f"{r_path}.status", "FHIR-OBS-01", "Observation.status (1..1) is required.")
            codings = ((res.get("code") or {}).get("coding")) or []
            if not codings or codings[0].get("system") != LOINC_URI or not LOINC_CODE_RE.match(str(codings[0].get("code") or "")):
                _err(f"{r_path}.code.coding[0]", "FHIR-OBS-02", f"Observation.code requires a valid LOINC code ({LOINC_URI}).")
            vq = res.get("valueQuantity") or {}
            if vq.get("value") is None or vq.get("system") != UCUM_URI or vq.get("code") not in ALLOWED_UCUM_CODES:
                _err(
                    f"{r_path}.valueQuantity",
                    "FHIR-OBS-03",
                    f"Observation.valueQuantity requires numeric value and valid UCUM unit ({UCUM_URI}).",
                )
            _check_ref(res.get("subject"), f"{r_path}.subject")
            _check_ref(res.get("encounter"), f"{r_path}.encounter")

        elif rtype == "MedicationRequest":
            checks_run += 3
            if res.get("status") not in ("active", "completed", "draft", "stopped"):
                _err(f"{r_path}.status", "FHIR-MED-01", "MedicationRequest.status (1..1) is required.")
            if res.get("intent") not in ("proposal", "plan", "order", "original-order"):
                _err(f"{r_path}.intent", "FHIR-MED-02", "MedicationRequest.intent (1..1) is required.")
            codings = ((res.get("medicationCodeableConcept") or {}).get("coding")) or []
            if not codings or codings[0].get("system") != RXNORM_URI or not RXNORM_CODE_RE.match(str(codings[0].get("code") or "")):
                _err(
                    f"{r_path}.medicationCodeableConcept.coding[0]",
                    "FHIR-MED-03",
                    f"MedicationRequest requires a numeric RxNorm CUI under {RXNORM_URI}.",
                )
            dosages = res.get("dosageInstruction") or []
            if not dosages:
                _err(f"{r_path}.dosageInstruction", "FHIR-MED-04", "MedicationRequest.dosageInstruction is required.")
            else:
                dq = (((dosages[0].get("doseAndRate") or [{}])[0]).get("doseQuantity")) or {}
                if dq.get("value") is None or dq.get("system") != UCUM_URI or dq.get("code") not in ALLOWED_UCUM_CODES:
                    _err(f"{r_path}.dosageInstruction[0].doseAndRate[0].doseQuantity", "FHIR-MED-05", "Invalid UCUM doseQuantity.")
            _check_ref(res.get("subject"), f"{r_path}.subject")
            _check_ref(res.get("encounter"), f"{r_path}.encounter")
            _check_ref(res.get("requester"), f"{r_path}.requester")

        else:
            _err(f"{r_path}.resourceType", "FHIR-RES-02", f"Unsupported clinical resourceType '{rtype}'.")

    valid = len(issues) == 0
    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 3)

    if audit_store:
        audit_store.append_event(
            encounter_id=encounter_id,
            action_type="FHIR_R4_BUNDLE_VALIDATE",
            verdict="FHIR_VALID" if valid else "FHIR_SCHEMA_REJECTED",
            duration_ms=elapsed_ms,
            details={
                "resource_count": len(entries),
                "checks_executed": checks_run,
                "error_count": len(issues),
            },
        )

    return {
        "valid": valid,
        "fhir_version": "4.0.1",
        "synthetic_label": SYNTHETIC_NOTICE,
        "bundle_type": b_type,
        "resource_count": len(entries),
        "resource_breakdown": resource_counts,
        "checks_executed": checks_run,
        "error_count": len(issues),
        "issues": issues,
        "validate_ms": elapsed_ms,
    }
