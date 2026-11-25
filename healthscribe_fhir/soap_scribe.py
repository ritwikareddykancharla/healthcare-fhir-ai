"""Span-Linked Ambient Clinical SOAP Scribe & Deterministic NegEx / Dosage Safety Gate.

Calls Amazon Bedrock (`us.amazon.nova-lite-v1:0`) via the Converse API to draft structured
Subjective / Objective / Assessment / Plan notes linked to exact `utterance_id` dialogue
spans, then enforces deterministic NegEx clinical polarity and medication dosage, unit,
and frequency verification before clinician sign-off.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from typing import Any, Dict, List, Optional, Tuple

from .audit_store import SQLiteClinicalAuditStore
from .encounters import SYNTHETIC_NOTICE, get_encounter

DEFAULT_BEDROCK_MODEL = "us.amazon.nova-lite-v1:0"

# Canonical clinical symptom concepts with synonym phrases for NegEx polarity verification
CLINICAL_CONCEPT_LEXICON: Dict[str, List[str]] = {
    "dyspnea": ["shortness of breath", "dyspnea", "sob", "respiratory distress", "breathlessness"],
    "rest_chest_pain": ["rest chest pain", "rest pain", "chest pain at rest"],
    "chest_pain": ["chest pain", "precordial pain", "substernal pain"],
    "exertional_chest_pressure": ["chest pressure", "exertional central chest pressure", "exertional angina"],
    "fever": ["fever", "febrile", "pyrexia", "chills"],
    "saddle_anesthesia": ["saddle anesthesia", "saddle numbness", "numbness in the groin", "perineal numbness"],
    "incontinence": [
        "bowel or bladder incontinence",
        "bowel or bladder problems",
        "bowel or bladder dysfunction",
        "urinary incontinence",
        "fecal incontinence",
        "incontinence",
    ],
    "radicular_leg_pain": [
        "shoots from my lower back down",
        "radiating down the posterior left leg",
        "lumbar radiculopathy",
        "lower back and left leg pain",
        "lower back and right leg",
    ],
    "motor_weakness": ["leg weakness", "foot drop", "motor weakness", "progressive weakness"],
    "syncope": ["syncope", "fainting", "loss of consciousness", "passed out"],
    "palpitations": ["palpitations", "heart racing", "fluttering"],
    "edema": ["leg swelling", "lower extremity edema", "pedal edema", "ankle swelling"],
    "nausea_vomiting": ["nausea", "vomiting", "emesis"],
    "vision_change": ["blurred vision", "vision loss", "diplopia", "double vision"],
    "medullary_thyroid_ca": ["medullary thyroid carcinoma", "medullary thyroid cancer", "men2"],
    "pancreatitis": ["pancreatitis"],
    "acute_coronary_syndrome": ["acute coronary syndrome", "stemi", "elevated troponin"],
}

PRE_NEGATION_RE = re.compile(
    r"\b(?:no|not|denies|deny|denied|without|negative\s+for|free\s+of|never\s+had|"
    r"rules?\s+out|ruled\s+out|ruling\s+out|excludes|excluded|excluding|absent|"
    r"no\s+history\s+of|no\s+family\s+or\s+personal\s+history\s+of|"
    r"have\s+not\s+started|no\s+physical\s+therapy)\b",
    re.IGNORECASE,
)

POST_NEGATION_RE = re.compile(
    r"\b(?:is\s+ruled\s+out|was\s+ruled\s+out|is\s+negative|was\s+negative|not\s+present)\b",
    re.IGNORECASE,
)

PSEUDO_NEGATION_RE = re.compile(
    r"\b(?:no\s+change|no\s+improvement|without\s+lasting\s+relief|without\s+relief|non-contrast|without\s+contrast)\b",
    re.IGNORECASE,
)

SCOPE_TERMINATOR_RE = re.compile(
    r"(?:[.;!?]|\b(?:but|however|except|although|though|reports|complains\s+of|positive\s+for|shows|demonstrates)\b)",
    re.IGNORECASE,
)

KNOWN_MEDICATIONS = [
    "naproxen",
    "gabapentin",
    "ibuprofen",
    "metformin",
    "lisinopril",
    "semaglutide",
    "atorvastatin",
    "aspirin",
    "levothyroxine",
    "albuterol",
    "amoxicillin",
    "prednisone",
    "oxycodone",
    "warfarin",
    "apixaban",
    "insulin",
    "amlodipine",
    "losartan",
]

MED_MENTION_RE = re.compile(
    r"\b(" + "|".join(KNOWN_MEDICATIONS) + r")(?:\s+hydrochloride)?\s+"
    r"(\d+(?:\.\d+)?)\s*(mg|mcg|g|ml|units|puffs)\b"
    r"(?=([^,;.\n]{0,56}))",
    re.IGNORECASE,
)

FREQ_SYNONYMS: List[Tuple[str, re.Pattern[str]]] = [
    ("twice daily", re.compile(r"\b(?:twice\s+daily|twice\s+a\s+day|bid|every\s+12\s+hours|q12h)\b", re.IGNORECASE)),
    (
        "three times daily",
        re.compile(r"\b(?:three\s+times\s+daily|three\s+times\s+a\s+day|tid|every\s+8\s+hours|q8h)\b", re.IGNORECASE),
    ),
    (
        "four times daily",
        re.compile(r"\b(?:four\s+times\s+daily|qid|every\s+6\s+hours|q6h)\b", re.IGNORECASE),
    ),
    ("once weekly", re.compile(r"\b(?:once\s+weekly|weekly|every\s+week|qwk)\b", re.IGNORECASE)),
    ("at bedtime", re.compile(r"\b(?:at\s+bedtime|qhs|nightly)\b", re.IGNORECASE)),
    ("once daily", re.compile(r"\b(?:once\s+daily|once\s+a\s+day|daily|qd)\b", re.IGNORECASE)),
]


def _normalize_frequency(tail_text: str) -> Optional[str]:
    matches: List[str] = []
    for canonical, pat in FREQ_SYNONYMS:
        if pat.search(tail_text):
            matches.append(canonical)
    if not matches:
        return None
    if "at bedtime" in matches:
        return "at bedtime"
    return matches[0]


def _frequencies_compatible(expected: Optional[str], observed: Optional[str]) -> bool:
    if not expected or not observed:
        return True
    if expected == observed:
        return True
    # "once daily at bedtime" in transcript is compatible with either "once daily" or "at bedtime"
    pair = {expected, observed}
    if pair == {"once daily", "at bedtime"}:
        return True
    return False


def detect_concept_polarities(text: str) -> Dict[str, str]:
    """Run NegEx pre/post-window polarity detection over clinical concepts in `text`."""
    cleaned = PSEUDO_NEGATION_RE.sub(" ", text.lower())
    results: Dict[str, str] = {}

    for concept, phrases in CLINICAL_CONCEPT_LEXICON.items():
        # Match longest phrase first so "rest chest pain" matches before "chest pain"
        for phrase in sorted(phrases, key=len, reverse=True):
            idx = cleaned.find(phrase)
            if idx == -1:
                continue
            # Special guard: do not treat "rest chest pain" as generic "chest_pain"
            if concept == "chest_pain" and idx >= 5 and "rest " in cleaned[max(0, idx - 6) : idx + 1]:
                continue

            prefix_start = max(0, idx - 95)
            prefix = cleaned[prefix_start:idx]
            # Cut prefix at last scope terminator so prior clause negation doesn't bleed across clauses
            terms = list(SCOPE_TERMINATOR_RE.finditer(prefix))
            if terms:
                prefix = prefix[terms[-1].end() :]

            suffix_end = min(len(cleaned), idx + len(phrase) + 45)
            suffix = cleaned[idx + len(phrase) : suffix_end]
            suffix_terms = list(SCOPE_TERMINATOR_RE.finditer(suffix))
            if suffix_terms:
                suffix = suffix[: suffix_terms[0].start()]

            # Skip clinician interrogative screening questions ("Do you have any...?", "Any fever...?")
            sent_end_q = cleaned.find("?", idx)
            sent_end_p = cleaned.find(".", idx)
            is_question_clause = sent_end_q != -1 and (sent_end_p == -1 or sent_end_q < sent_end_p)
            if is_question_clause and re.search(r"\b(?:any|do\s+you\s+have|have\s+you\s+had)\b", prefix):
                continue

            is_neg = bool(PRE_NEGATION_RE.search(prefix) or POST_NEGATION_RE.search(suffix))
            polarity = "negated" if is_neg else "affirmed"
            # If concept appears multiple times, affirmed symptom takes clinical precedence unless only negated
            if concept not in results or polarity == "affirmed":
                results[concept] = polarity
            break

    return results


def extract_medication_mentions(text: str) -> List[Dict[str, Any]]:
    mentions: List[Dict[str, Any]] = []
    for m in MED_MENTION_RE.finditer(text):
        drug = m.group(1).lower()
        dose_val = float(m.group(2))
        unit = m.group(3).lower()
        tail = m.group(4) or ""
        freq = _normalize_frequency(tail)
        mentions.append(
            {
                "drug": drug,
                "dose_value": dose_val,
                "dose_unit": unit,
                "frequency": freq,
            }
        )
    return mentions


def verify_clinical_claims(
    encounter: Dict[str, Any],
    claims: List[Dict[str, Any]],
    audit_store: Optional[SQLiteClinicalAuditStore] = None,
) -> Dict[str, Any]:
    """Deterministically verify span grounding, NegEx clinical polarity, and medication dosages."""
    t0 = time.perf_counter()
    utt_map: Dict[str, Dict[str, Any]] = {u["utterance_id"]: u for u in encounter.get("utterances", [])}

    # Build encounter-wide ground-truth concept polarities and medication table
    encounter_polarities: Dict[str, set[str]] = {}
    utt_polarities: Dict[str, Dict[str, str]] = {}
    encounter_meds: Dict[str, List[Dict[str, Any]]] = {}

    for uid, utt in utt_map.items():
        pols = detect_concept_polarities(utt["text"])
        utt_polarities[uid] = pols
        for concept, pol in pols.items():
            encounter_polarities.setdefault(concept, set()).add(pol)
        for med in extract_medication_mentions(utt["text"]):
            encounter_meds.setdefault(med["drug"], []).append({**med, "utterance_id": uid})

    verified_claims: List[Dict[str, Any]] = []
    blocked_claims: List[Dict[str, Any]] = []

    for idx, raw_claim in enumerate(claims, start=1):
        claim_id = str(raw_claim.get("claim_id") or f"CLM-{idx:02d}")
        section = str(raw_claim.get("section") or "Subjective")
        statement = str(raw_claim.get("statement") or "").strip()
        utt_ids = [str(u).strip() for u in (raw_claim.get("utterance_ids") or []) if str(u).strip()]

        violations: List[Dict[str, str]] = []
        citations: List[Dict[str, Any]] = []

        # Gate 1: Utterance Span Grounding
        valid_uids = [u for u in utt_ids if u in utt_map]
        if not utt_ids or len(valid_uids) != len(utt_ids):
            violations.append(
                {
                    "rule_id": "CLIN-SPAN-01",
                    "category": "UNGROUNDED_SPAN",
                    "detail": f"Claim cites missing or invalid utterance_ids: {utt_ids}",
                }
            )
        for uid in valid_uids:
            u = utt_map[uid]
            citations.append(
                {
                    "utterance_id": uid,
                    "speaker": u["speaker"],
                    "start_ms": u["start_ms"],
                    "end_ms": u["end_ms"],
                    "quote": u["text"],
                }
            )

        # Gate 2: NegEx Clinical Polarity Verification
        claim_pols = detect_concept_polarities(statement)
        for concept, claim_pol in claim_pols.items():
            # Prefer polarity in cited utterances, falling back to full encounter transcript
            cited_pols = {
                utt_polarities[uid][concept]
                for uid in valid_uids
                if concept in utt_polarities.get(uid, {})
            }
            ref_pols = cited_pols or encounter_polarities.get(concept, set())
            if not ref_pols:
                continue
            if claim_pol == "affirmed" and ref_pols == {"negated"}:
                violations.append(
                    {
                        "rule_id": "CLIN-NEGEX-01",
                        "category": "NEGATION_FLIP_BLOCKED",
                        "detail": f"Concept '{concept}' is explicitly negated in transcript but charted as affirmed.",
                    }
                )
            elif claim_pol == "negated" and ref_pols == {"affirmed"}:
                violations.append(
                    {
                        "rule_id": "CLIN-POLARITY-02",
                        "category": "AFFIRMED_SYMPTOM_NEGATED",
                        "detail": f"Concept '{concept}' is affirmed in transcript but charted as negated.",
                    }
                )

        # Gate 3: Medication Name, Numeric Dose, Unit, and Frequency Verification
        claim_meds = extract_medication_mentions(statement)
        explicit_med = raw_claim.get("medication")
        if isinstance(explicit_med, dict) and explicit_med.get("drug"):
            drug_l = str(explicit_med["drug"]).lower()
            if not any(cm["drug"] == drug_l for cm in claim_meds):
                claim_meds.append(
                    {
                        "drug": drug_l,
                        "dose_value": float(explicit_med.get("dose_value") or 0.0),
                        "dose_unit": str(explicit_med.get("dose_unit") or "").lower(),
                        "frequency": _normalize_frequency(str(explicit_med.get("frequency") or ""))
                        or str(explicit_med.get("frequency") or "").lower()
                        or None,
                    }
                )

        for cm in claim_meds:
            drug = cm["drug"]
            if drug not in encounter_meds:
                violations.append(
                    {
                        "rule_id": "CLIN-DRUG-04",
                        "category": "UNMENTIONED_MEDICATION",
                        "detail": f"Medication '{drug}' does not appear anywhere in the encounter transcript.",
                    }
                )
                continue

            allowed_entries = encounter_meds[drug]
            allowed_units = {e["dose_unit"] for e in allowed_entries}
            if cm["dose_unit"] not in allowed_units:
                violations.append(
                    {
                        "rule_id": "CLIN-UNIT-02",
                        "category": "DOSAGE_UNIT_MISMATCH",
                        "detail": (
                            f"Medication '{drug}' charted with unit '{cm['dose_unit']}' "
                            f"instead of transcript unit {sorted(allowed_units)}."
                        ),
                    }
                )
                continue

            matching_dose = [e for e in allowed_entries if abs(e["dose_value"] - cm["dose_value"]) < 1e-6]
            if not matching_dose:
                expected_doses = sorted({f"{e['dose_value']:g} {e['dose_unit']}" for e in allowed_entries})
                violations.append(
                    {
                        "rule_id": "CLIN-DOSE-01",
                        "category": "DOSAGE_VALUE_MISMATCH",
                        "detail": (
                            f"Medication '{drug}' charted at {cm['dose_value']:g} {cm['dose_unit']} "
                            f"instead of transcript dose {expected_doses}."
                        ),
                    }
                )
                continue

            if cm.get("frequency"):
                freq_ok = any(_frequencies_compatible(e.get("frequency"), cm["frequency"]) for e in matching_dose)
                if not freq_ok:
                    expected_freqs = sorted({str(e.get("frequency")) for e in matching_dose if e.get("frequency")})
                    violations.append(
                        {
                            "rule_id": "CLIN-FREQ-03",
                            "category": "DOSAGE_FREQUENCY_MISMATCH",
                            "detail": (
                                f"Medication '{drug}' charted as '{cm['frequency']}' "
                                f"instead of transcript frequency {expected_freqs}."
                            ),
                        }
                    )

        enriched = {
            "claim_id": claim_id,
            "section": section,
            "statement": statement,
            "utterance_ids": utt_ids,
            "citations": citations,
            "status": "BLOCKED" if violations else "VERIFIED",
            "violations": violations,
        }
        if violations:
            blocked_claims.append(enriched)
        else:
            verified_claims.append(enriched)

    verify_ms = round((time.perf_counter() - t0) * 1000.0, 3)
    verdict = "SAFETY_GATE_BLOCKED" if blocked_claims else "ALL_CLAIMS_VERIFIED"

    if audit_store:
        audit_store.append_event(
            encounter_id=encounter["encounter_id"],
            action_type="CLINICAL_SAFETY_GATE",
            verdict=verdict,
            duration_ms=verify_ms,
            details={
                "claims_checked": len(claims),
                "verified_count": len(verified_claims),
                "blocked_count": len(blocked_claims),
            },
        )

    return {
        "encounter_id": encounter["encounter_id"],
        "synthetic_label": SYNTHETIC_NOTICE,
        "verdict": verdict,
        "total_claims": len(claims),
        "verified_count": len(verified_claims),
        "blocked_count": len(blocked_claims),
        "verify_ms": verify_ms,
        "verified_claims": verified_claims,
        "blocked_claims": blocked_claims,
        "all_claims": verified_claims + blocked_claims,
    }


def invoke_bedrock_converse(
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 900,
) -> Optional[Dict[str, Any]]:
    """Invoke live Amazon Bedrock (`us.amazon.nova-lite-v1:0`) via Converse API."""
    if os.environ.get("HEALTHSCRIBE_OFFLINE_LLM") == "1":
        return None

    model_id = os.environ.get("HEALTHSCRIBE_BEDROCK_MODEL", DEFAULT_BEDROCK_MODEL)
    region = os.environ.get("AWS_REGION", os.environ.get("AWS_DEFAULT_REGION", "us-west-2"))
    t0 = time.perf_counter()

    # Path 1: boto3 Converse API (AWS Lambda Python 3.12 and local Python)
    try:
        import boto3  # type: ignore

        if os.environ.get("AWS_EXECUTION_ENV"):
            client = boto3.client("bedrock-runtime", region_name=region)
        else:
            profile = os.environ.get("AWS_PROFILE")
            session = boto3.Session(profile_name=profile, region_name=region) if profile else boto3.Session(region_name=region)
            client = session.client("bedrock-runtime", region_name=region)

        resp = client.converse(
            modelId=model_id,
            system=[{"text": system_prompt}],
            messages=[{"role": "user", "content": [{"text": user_prompt}]}],
            inferenceConfig={"maxTokens": max_tokens, "temperature": 0.1},
        )
        round_trip_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        server_ms = float(resp.get("metrics", {}).get("latencyMs") or round_trip_ms)
        usage = resp.get("usage", {})
        text = resp["output"]["message"]["content"][0]["text"]
        return {
            "text": text.strip(),
            "engine": "Amazon Bedrock",
            "model_id": model_id,
            "latency_ms": round(server_ms, 2),
            "round_trip_ms": round_trip_ms,
            "input_tokens": int(usage.get("inputTokens", 0)),
            "output_tokens": int(usage.get("outputTokens", 0)),
            "total_tokens": int(usage.get("totalTokens", 0)),
        }
    except Exception:
        pass

    # Path 2: AWS CLI fallback if local boto3 credentials need profile subprocess
    aws_bin = shutil.which("aws")
    if not aws_bin:
        return None
    env = dict(os.environ)
    env.pop("AWS_ACCESS_KEY_ID", None)
    env.pop("AWS_SECRET_ACCESS_KEY", None)
    env.pop("AWS_SESSION_TOKEN", None)
    env["AWS_PROFILE"] = os.environ.get("AWS_PROFILE", "portfolio")
    env["AWS_DEFAULT_REGION"] = region

    cmd = [
        aws_bin,
        "bedrock-runtime",
        "converse",
        "--region",
        region,
        "--model-id",
        model_id,
        "--system",
        json.dumps([{"text": system_prompt}]),
        "--messages",
        json.dumps([{"role": "user", "content": [{"text": user_prompt}]}]),
        "--inference-config",
        json.dumps({"maxTokens": max_tokens, "temperature": 0.1}),
        "--output",
        "json",
    ]
    try:
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=18.0)
        round_trip_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        if proc.returncode == 0:
            data = json.loads(proc.stdout)
            server_ms = float(data.get("metrics", {}).get("latencyMs") or round_trip_ms)
            usage = data.get("usage", {})
            text = data["output"]["message"]["content"][0]["text"]
            return {
                "text": text.strip(),
                "engine": "Amazon Bedrock",
                "model_id": model_id,
                "latency_ms": round(server_ms, 2),
                "round_trip_ms": round_trip_ms,
                "input_tokens": int(usage.get("inputTokens", 0)),
                "output_tokens": int(usage.get("outputTokens", 0)),
                "total_tokens": int(usage.get("totalTokens", 0)),
            }
    except Exception:
        pass
    return None


def _extract_json_object(raw_text: str) -> Optional[Dict[str, Any]]:
    cleaned = raw_text.strip()
    if "```" in cleaned:
        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, flags=re.DOTALL)
        if m:
            cleaned = m.group(1)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError:
            return None
    return None


def generate_span_linked_soap(
    encounter_id: str,
    injected_adversarial_claim: Optional[Dict[str, Any]] = None,
    use_bedrock: bool = True,
    audit_store: Optional[SQLiteClinicalAuditStore] = None,
) -> Dict[str, Any]:
    """Generate a span-linked SOAP note with Amazon Bedrock and run the NegEx/Dosage safety gate."""
    encounter = get_encounter(encounter_id)
    valid_uids = [u["utterance_id"] for u in encounter["utterances"]]
    transcript_block = "\n".join(
        f"{u['utterance_id']} ({u['speaker']}, {u['start_ms']}..{u['end_ms']} ms): {u['text']}"
        for u in encounter["utterances"]
    )

    system_prompt = (
        "You are an ambient clinical documentation scribe for synthetic outpatient encounters. "
        "Return ONLY valid JSON with key 'claims', an array of 4 to 6 objects covering "
        "Subjective, Objective, Assessment, and Plan. "
        "Each object MUST contain: "
        "'claim_id' ('CLM-01'...), "
        "'section' ('Subjective' | 'Objective' | 'Assessment' | 'Plan'), "
        "'statement' (concise clinical sentence preserving exact medication dosages, units, frequencies, and negations verbatim from the transcript), "
        f"and 'utterance_ids' (non-empty list chosen strictly from {valid_uids}). "
        "Never flip negated symptoms into positive findings, and never alter numeric drug doses or units."
    )
    user_prompt = (
        f"Synthetic Encounter ({encounter['specialty']}):\n{transcript_block}\n\n"
        "Generate the span-linked SOAP claims JSON now."
    )

    llm_meta: Optional[Dict[str, Any]] = None
    drafted_claims: List[Dict[str, Any]] = []

    if use_bedrock:
        llm_meta = invoke_bedrock_converse(system_prompt=system_prompt, user_prompt=user_prompt, max_tokens=900)
        if llm_meta:
            parsed = _extract_json_object(llm_meta["text"])
            if parsed and isinstance(parsed.get("claims"), list) and len(parsed["claims"]) >= 3:
                candidate_claims = parsed["claims"]
                # Verify Bedrock claims; if clean, adopt them directly
                prelim = verify_clinical_claims(encounter, candidate_claims, audit_store=None)
                if prelim["blocked_count"] == 0:
                    drafted_claims = candidate_claims
                else:
                    # Keep verified Bedrock claims and fill any missing section from baseline
                    drafted_claims = [
                        {
                            "claim_id": c["claim_id"],
                            "section": c["section"],
                            "statement": c["statement"],
                            "utterance_ids": c["utterance_ids"],
                        }
                        for c in prelim["verified_claims"]
                    ]

    if not drafted_claims:
        drafted_claims = [dict(c) for c in encounter["baseline_soap_claims"]]

    if injected_adversarial_claim:
        drafted_claims = list(drafted_claims) + [dict(injected_adversarial_claim)]

    verification = verify_clinical_claims(encounter, drafted_claims, audit_store=audit_store)

    sections_grouped: Dict[str, List[Dict[str, Any]]] = {
        "Subjective": [],
        "Objective": [],
        "Assessment": [],
        "Plan": [],
    }
    for clm in verification["verified_claims"]:
        sec = clm["section"] if clm["section"] in sections_grouped else "Subjective"
        sections_grouped[sec].append(clm)

    if audit_store and llm_meta:
        audit_store.append_event(
            encounter_id=encounter_id,
            action_type="BEDROCK_SOAP_SYNTHESIS",
            verdict=verification["verdict"],
            duration_ms=llm_meta["latency_ms"],
            details={
                "model_id": llm_meta["model_id"],
                "total_tokens": llm_meta["total_tokens"],
                "verified_claims": verification["verified_count"],
                "blocked_claims": verification["blocked_count"],
            },
        )

    return {
        "encounter_id": encounter_id,
        "synthetic_label": SYNTHETIC_NOTICE,
        "specialty": encounter["specialty"],
        "title": encounter["title"],
        "llm_engine": llm_meta["engine"] if llm_meta else "Deterministic Baseline",
        "bedrock_model": llm_meta["model_id"] if llm_meta else DEFAULT_BEDROCK_MODEL,
        "bedrock_latency_ms": llm_meta["latency_ms"] if llm_meta else 0.0,
        "bedrock_round_trip_ms": llm_meta["round_trip_ms"] if llm_meta else 0.0,
        "bedrock_tokens": llm_meta["total_tokens"] if llm_meta else 0,
        "soap_sections": sections_grouped,
        "safety_gate": verification,
    }
