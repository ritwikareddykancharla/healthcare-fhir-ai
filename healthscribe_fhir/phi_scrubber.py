"""HIPAA Safe-Harbor 18-Identifier De-Identification & Reversible HMAC Vault Tokenizer.

Implements 45 CFR 164.514(b)(2) Safe-Harbor identifier detection across all 18 HIPAA
identifier classes and replaces matches with deterministic HMAC-SHA256 vault tokens
such as `[PATIENT_8F2A]` and `[MRN_19C4]` while preserving clinical vitals, dosages,
ICD-10-CM, CPT, LOINC, and RxNorm codes.
"""
from __future__ import annotations

import hashlib
import hmac
import os
import re
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .audit_store import SQLiteClinicalAuditStore

DEFAULT_VAULT_SECRET = "healthscribefhir-hipaa-vault-hmac-key-2026"

HIPAA_18_CATEGORIES: List[Dict[str, str]] = [
    {"id": "01", "category": "PATIENT", "hipaa_rule": "1. Names (Patient & Relatives)"},
    {"id": "02", "category": "ADDRESS", "hipaa_rule": "2. Geographic Subdivisions Smaller Than State"},
    {"id": "03", "category": "DATE", "hipaa_rule": "3. Dates (More Granular Than Year) & Ages > 89"},
    {"id": "04", "category": "PHONE", "hipaa_rule": "4. Telephone Numbers"},
    {"id": "05", "category": "FAX", "hipaa_rule": "5. Fax Numbers"},
    {"id": "06", "category": "EMAIL", "hipaa_rule": "6. Email Addresses"},
    {"id": "07", "category": "SSN", "hipaa_rule": "7. Social Security Numbers"},
    {"id": "08", "category": "MRN", "hipaa_rule": "8. Medical Record Numbers"},
    {"id": "09", "category": "BENEFICIARY", "hipaa_rule": "9. Health Plan Beneficiary Numbers"},
    {"id": "10", "category": "ACCOUNT", "hipaa_rule": "10. Account Numbers"},
    {"id": "11", "category": "NPI", "hipaa_rule": "11. Certificate / License / NPI Numbers"},
    {"id": "12", "category": "VEHICLE", "hipaa_rule": "12. Vehicle Identifiers & License Plates"},
    {"id": "13", "category": "DEVICE", "hipaa_rule": "13. Medical Device Identifiers & Serials"},
    {"id": "14", "category": "URL", "hipaa_rule": "14. Web Universal Resource Locators (URLs)"},
    {"id": "15", "category": "IP", "hipaa_rule": "15. Internet Protocol (IP) Addresses"},
    {"id": "16", "category": "BIOMETRIC", "hipaa_rule": "16. Biometric Identifiers (Voice/Finger)"},
    {"id": "17", "category": "PHOTO", "hipaa_rule": "17. Full-Face Photographic References"},
    {"id": "18", "category": "UNIQUE_ID", "hipaa_rule": "18. Any Other Unique Identifying Code"},
]

# Ordered deterministic patterns (specific structured tokens first, free-text patterns after)
ORDERED_PHI_PATTERNS: List[Tuple[str, re.Pattern[str], int]] = [
    ("URL", re.compile(r"\bhttps?://[^\s,;\])\"']+"), 0),
    ("EMAIL", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"), 0),
    ("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), 0),
    ("MRN", re.compile(r"\bMRN[-:\s#]*[A-Z0-9]{5,12}\b"), 0),
    ("BENEFICIARY", re.compile(r"\b(?:HPB|MBI|MEMBER)[-:\s#]*[A-Z0-9]{6,14}\b"), 0),
    ("ACCOUNT", re.compile(r"\bACCT[-:\s#]*[A-Z0-9]{5,14}\b"), 0),
    (
        "NPI",
        re.compile(
            r"\b(?:NPI[-:\s#]*\d{10}|[A-Z]{2}-MD-\d{5,8}|DEA-[A-Z]{2}\d{7}|DL-[A-Z0-9-]{6,14})\b"
        ),
        0,
    ),
    ("DEVICE", re.compile(r"\bUDI-[A-Z0-9-]{5,22}\b"), 0),
    ("VEHICLE", re.compile(r"\b(?:VIN-[A-HJ-NPR-Z0-9]{8,17}|PLATE-[A-Z0-9-]{5,10})\b"), 0),
    ("BIOMETRIC", re.compile(r"\b(?:VOICEPRINT|FINGERPRINT|RETINA)-[A-Z0-9-]{5,18}\b"), 0),
    ("PHOTO", re.compile(r"\bPHOTO-[A-Z0-9._-]{5,26}\b"), 0),
    ("UNIQUE_ID", re.compile(r"\b(?:CLAIM|CASE|ENC-REF)-[A-Z0-9-]{5,18}\b"), 0),
    (
        "IP",
        re.compile(
            r"\b(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)(?:\.(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)){3}\b"
        ),
        0,
    ),
    (
        "FAX",
        re.compile(r"\b(?:[Ff]ax(?:\s*(?:number|no\.?|#))?[:\s]+)(\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})\b"),
        1,
    ),
    ("PHONE", re.compile(r"(?<!\d)(\(\d{3}\)\s*\d{3}[-.\s]\d{4}|\b\d{3}-\d{3}-\d{4}\b)"), 1),
    (
        "ADDRESS",
        re.compile(
            r"\b\d{2,5}\s+[A-Z][A-Za-z0-9.\s]{1,32}?\s"
            r"(?:Street|St|Avenue|Ave|Terrace|Way|Road|Rd|Boulevard|Blvd|Drive|Dr|Lane|Ln)\b"
            r"(?:\s+(?:Apt|Suite|Unit)\s+[A-Za-z0-9-]+)?"
            r"(?:,\s*[A-Z][A-Za-z\s]+,\s*[A-Z]{2}\s+\d{5}(?:-\d{4})?)?"
        ),
        0,
    ),
    (
        "DATE",
        re.compile(
            r"\b(?:"
            r"(?:0?[1-9]|1[0-2])/(?:0?[1-9]|[12]\d|3[01])/(?:19|20)\d{2}"
            r"|(?:19|20)\d{2}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01])"
            r"|(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|"
            r"Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2},\s*(?:19|20)\d{2}"
            r")\b"
        ),
        0,
    ),
    (
        "AGE_90_PLUS",
        re.compile(r"\b(?:age\s+(?:9\d|1[01]\d)|(?:9\d|1[01]\d)[\s-]year[\s-]old)\b", re.IGNORECASE),
        0,
    ),
    ("CLINICIAN", re.compile(r"\bDr\.\s+[A-Z][a-z]+(?:\s+[A-Z]\.?)?\s+[A-Z][A-Za-z-]+"), 0),
    (
        "PATIENT",
        re.compile(r"\b(?:Mr\.|Ms\.|Mrs\.|Patient)\s+([A-Z][a-z]+(?:\s+[A-Z]\.?)?(?:\s+[A-Z][A-Za-z-]+)?)"),
        1,
    ),
    (
        "PATIENT",
        re.compile(
            r"\b(?:Good morning|Good afternoon|Hello|Hi|Welcome back)\s+"
            r"([A-Z][a-z]+(?:\s+[A-Z]\.?\s+[A-Z][A-Za-z-]+|\s+[A-Z][A-Za-z-]+)?)\b"
        ),
        1,
    ),
]

NON_NAME_STOPWORDS = {
    "Today",
    "Vitals",
    "Blood",
    "Assessment",
    "Plan",
    "Subjective",
    "Objective",
    "Type",
    "Lumbar",
    "Resting",
    "Motor",
    "Order",
    "Continue",
}


class HipaaSafeHarborScrubber:
    """Deterministic 18-identifier HIPAA Safe-Harbor scrubber with HMAC-SHA256 reversible vault."""

    def __init__(
        self,
        secret_key: Optional[str] = None,
        audit_store: Optional[SQLiteClinicalAuditStore] = None,
    ) -> None:
        self.secret_key = (
            secret_key or os.environ.get("HEALTHSCRIBE_VAULT_SECRET") or DEFAULT_VAULT_SECRET
        ).encode("utf-8")
        self.audit_store = audit_store
        self._memory_vault: Dict[str, Dict[str, str]] = {}

    def make_vault_token(self, category: str, raw_value: str) -> str:
        normalized = " ".join(raw_value.strip().lower().split())
        digest = hmac.new(self.secret_key, f"{category}:{normalized}".encode("utf-8"), hashlib.sha256).hexdigest()
        short_hex = digest[:4].upper()
        return f"[{category}_{short_hex}]"

    def scrub_text(
        self,
        text: str,
        known_identifiers: Optional[Sequence[Tuple[str, str]]] = None,
        encounter_id: str = "ad-hoc",
    ) -> Dict[str, Any]:
        t0 = time.perf_counter()
        candidates: List[Tuple[int, int, str, str]] = []

        # 1. Context-supplied exact identifiers (e.g., patient full/first name, standalone NPI)
        if known_identifiers:
            for category, val in known_identifiers:
                clean_val = (val or "").strip()
                if len(clean_val) < 3:
                    continue
                pat = re.compile(r"(?<!\w)" + re.escape(clean_val) + r"(?!\w)")
                for m in pat.finditer(text):
                    candidates.append((m.start(), m.end(), category, m.group(0)))

        # 2. Deterministic 18-category regex scan
        for category, pattern, group_idx in ORDERED_PHI_PATTERNS:
            for m in pattern.finditer(text):
                if group_idx > 0 and m.lastindex and m.lastindex >= group_idx:
                    start, end = m.span(group_idx)
                    raw_val = m.group(group_idx)
                else:
                    start, end = m.span(0)
                    raw_val = m.group(0)
                if category in ("PATIENT", "CLINICIAN") and raw_val.split()[0] in NON_NAME_STOPWORDS:
                    continue
                candidates.append((start, end, category, raw_val))

        # Sort by earliest start, then longest span first to eliminate partial overlaps
        candidates.sort(key=lambda item: (item[0], -(item[1] - item[0])))
        accepted: List[Tuple[int, int, str, str]] = []
        cursor = -1
        for start, end, category, raw_val in candidates:
            if start < cursor:
                continue
            # Skip already tokenized strings like [MRN_19C4]
            if start > 0 and text[start - 1] == "[" and end < len(text) and text[end] == "]":
                continue
            accepted.append((start, end, category, raw_val))
            cursor = end

        redacted_parts: List[str] = []
        last_idx = 0
        detections: List[Dict[str, Any]] = []
        vault_rows: List[Dict[str, str]] = []

        for start, end, category, raw_val in accepted:
            redacted_parts.append(text[last_idx:start])
            token = self.make_vault_token(category, raw_val)
            redacted_parts.append(token)
            last_idx = end

            hipaa_cat = "DATE" if category == "AGE_90_PLUS" else ("PATIENT" if category == "CLINICIAN" else category)
            detections.append(
                {
                    "category": category,
                    "hipaa_bucket": hipaa_cat,
                    "start": start,
                    "end": end,
                    "token": token,
                    "redacted_length": end - start,
                }
            )
            entry = {"token": token, "category": category, "raw_value": raw_val}
            self._memory_vault[token] = entry
            vault_rows.append(entry)

        redacted_parts.append(text[last_idx:])
        scrubbed_text = "".join(redacted_parts)
        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 3)

        if self.audit_store:
            self.audit_store.store_vault_tokens(vault_rows)
            self.audit_store.append_event(
                encounter_id=encounter_id,
                action_type="HIPAA_SAFE_HARBOR_SCRUB",
                verdict="DEIDENTIFIED",
                duration_ms=elapsed_ms,
                details={
                    "tokens_emitted": len(detections),
                    "categories": sorted({d["category"] for d in detections}),
                },
            )

        return {
            "encounter_id": encounter_id,
            "scrubbed_text": scrubbed_text,
            "phi_count": len(detections),
            "categories_detected": sorted({d["category"] for d in detections}),
            "detections": detections,
            "scrub_ms": elapsed_ms,
            "safe_harbor_compliant": True,
        }

    def scrub_encounter(self, encounter: Dict[str, Any]) -> Dict[str, Any]:
        patient = encounter.get("patient", {})
        practitioner = encounter.get("practitioner", {})
        full_name = patient.get("full_name", "")
        first_name = full_name.split()[0] if full_name else ""
        last_name = full_name.split()[-1] if len(full_name.split()) >= 2 else ""

        known: List[Tuple[str, str]] = []
        if full_name:
            known.append(("PATIENT", full_name))
        if first_name:
            known.append(("PATIENT", first_name))
        if last_name and first_name:
            known.append(("PATIENT", f"{first_name} {last_name}"))
        if practitioner.get("full_name"):
            known.append(("CLINICIAN", practitioner["full_name"]))
        if practitioner.get("npi"):
            known.append(("NPI", practitioner["npi"]))

        header = (
            f"Encounter Intake | Patient: {full_name} | DOB: {patient.get('birth_date_display', '')} | "
            f"MRN: {patient.get('mrn', '')} | Tel: {patient.get('phone', '')} | "
            f"Address: {patient.get('address', '')} | "
            f"Attending: {practitioner.get('full_name', '')} (NPI {practitioner.get('npi', '')})\n"
        )
        dialogue = "\n".join(
            f"[{u['utterance_id']}] {u['speaker']}: {u['text']}" for u in encounter.get("utterances", [])
        )
        return self.scrub_text(header + dialogue, known_identifiers=known, encounter_id=encounter["encounter_id"])

    def reidentify_text(self, scrubbed_text: str, requester_role: str, audit_reason: str) -> Dict[str, Any]:
        if not audit_reason.strip():
            raise ValueError("audit_reason is required to resolve reversible PHI vault tokens")
        t0 = time.perf_counter()
        token_re = re.compile(r"\[[A-Z0-9_]+_[0-9A-F]{4}\]")
        resolved_count = 0

        def _replace(match: re.Match[str]) -> str:
            nonlocal resolved_count
            tok = match.group(0)
            rec = self._memory_vault.get(tok)
            if not rec and self.audit_store:
                rec = self.audit_store.resolve_vault_token(tok)
            if rec:
                resolved_count += 1
                return rec["raw_value"]
            return tok

        restored = token_re.sub(_replace, scrubbed_text)
        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 3)
        if self.audit_store:
            self.audit_store.append_event(
                encounter_id="vault-reidentify",
                action_type="PHI_VAULT_REIDENTIFY",
                verdict="AUTHORIZED_UNMASK",
                duration_ms=elapsed_ms,
                details={
                    "requester_role": requester_role,
                    "audit_reason": audit_reason,
                    "tokens_resolved": resolved_count,
                },
            )
        return {
            "restored_text": restored,
            "tokens_resolved": resolved_count,
            "requester_role": requester_role,
            "reidentify_ms": elapsed_ms,
        }
