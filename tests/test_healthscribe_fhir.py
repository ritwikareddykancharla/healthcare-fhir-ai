"""Unit test suite for HealthScribeFHIR (12 tests covering PHI, NegEx/Dosage, FHIR R4, Prior Auth, Audit, and API)."""
from __future__ import annotations

import copy
import os
import tempfile
import unittest
from pathlib import Path

from healthscribe_fhir.audit_store import SQLiteClinicalAuditStore
from healthscribe_fhir.encounters import get_encounter
from healthscribe_fhir.fhir_validator import compile_fhir_r4_bundle, validate_fhir_r4_bundle
from healthscribe_fhir.phi_scrubber import HipaaSafeHarborScrubber
from healthscribe_fhir.prior_auth import evaluate_prior_auth
from healthscribe_fhir.server import lambda_handler, route_api_request
from healthscribe_fhir.soap_scribe import generate_span_linked_soap, verify_clinical_claims


class HealthScribeFhirTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmp_dir.name) / "test_audit.db"
        self.audit = SQLiteClinicalAuditStore(db_path=self.db_path)
        self.scrubber = HipaaSafeHarborScrubber(secret_key="unit-test-hmac-key", audit_store=self.audit)

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    def test_01_hipaa_safe_harbor_18_identifiers_and_clinical_preservation(self) -> None:
        note = (
            "Patient Elena M. Rostova (age 93, DOB 05/19/1933, SSN 900-42-8193, MRN-4829104, "
            "HPB-99402188, ACCT-7701924) seen by Dr. Marcus Vance (NPI 1245319599, WA-MD-609182) "
            "at 742 Evergreen Terrace Apt 4B, Seattle, WA 98101. Phone (206) 555-0148, "
            "Fax: (206) 555-0199, email elena.rostova.synth@example-clinic.org, "
            "URL https://portal.example-clinic.org/chart/4829104, IP 198.51.100.44, "
            "device UDI-TENS-8839201, vehicle VIN-1HGCM82633A004352, biometric VOICEPRINT-99201A, "
            "photo PHOTO-FACE-48291.jpg, ref CLAIM-8849201. "
            "Clinical preserved: BP 126/78 mmHg, HbA1c 8.4%, metformin 500 mg twice daily, "
            "52-year-old cousin healthy, ICD-10 M54.16, CPT 72148, LOINC 4548-4."
        )
        res = self.scrubber.scrub_text(note, encounter_id="enc-test")
        scrubbed = res["scrubbed_text"]
        # Raw PHI must be gone
        for raw_phi in [
            "Elena M. Rostova",
            "05/19/1933",
            "age 93",
            "900-42-8193",
            "MRN-4829104",
            "HPB-99402188",
            "ACCT-7701924",
            "Dr. Marcus Vance",
            "NPI 1245319599",
            "742 Evergreen Terrace",
            "(206) 555-0148",
            "elena.rostova.synth@example-clinic.org",
            "198.51.100.44",
            "UDI-TENS-8839201",
            "VIN-1HGCM82633A004352",
            "VOICEPRINT-99201A",
            "PHOTO-FACE-48291.jpg",
            "CLAIM-8849201",
        ]:
            self.assertNotIn(raw_phi, scrubbed)

        # Clinical values and age <= 89 must remain intact
        for clinical_token in [
            "126/78 mmHg",
            "HbA1c 8.4%",
            "metformin 500 mg twice daily",
            "52-year-old",
            "M54.16",
            "72148",
            "4548-4",
        ]:
            self.assertIn(clinical_token, scrubbed)

    def test_02_reversible_hmac_vault_roundtrip_and_audit(self) -> None:
        enc = get_encounter("enc-ortho-lumbar-01")
        scrubbed_out = self.scrubber.scrub_encounter(enc)
        self.assertGreaterEqual(scrubbed_out["phi_count"], 4)
        self.assertNotIn("MRN-4829104", scrubbed_out["scrubbed_text"])

        restored = self.scrubber.reidentify_text(
            scrubbed_out["scrubbed_text"],
            requester_role="ATTENDING_PHYSICIAN",
            audit_reason="Chart sign-off verification",
        )
        self.assertIn("MRN-4829104", restored["restored_text"])
        self.assertIn("Elena", restored["restored_text"])
        self.assertGreaterEqual(restored["tokens_resolved"], 4)

    def test_03_negex_blocks_negated_symptom_hallucination(self) -> None:
        enc = get_encounter("enc-ortho-lumbar-01")
        bad_claims = [
            {
                "claim_id": "CLM-ADV-01",
                "section": "Subjective",
                "statement": "Patient is positive for dyspnea and acute chest pain today.",
                "utterance_ids": ["U04"],
            }
        ]
        res = verify_clinical_claims(enc, bad_claims, audit_store=self.audit)
        self.assertEqual(res["verdict"], "SAFETY_GATE_BLOCKED")
        self.assertEqual(res["blocked_count"], 1)
        rule_ids = {v["rule_id"] for v in res["blocked_claims"][0]["violations"]}
        self.assertIn("CLIN-NEGEX-01", rule_ids)

    def test_04_dosage_verifier_blocks_10x_and_unit_mismatches(self) -> None:
        enc = get_encounter("enc-endo-diabetes-03")
        bad_claims = [
            {
                "claim_id": "CLM-DOSE-10X",
                "section": "Plan",
                "statement": "Continue metformin 5000 mg twice daily for glycemic control.",
                "utterance_ids": ["U05"],
            },
            {
                "claim_id": "CLM-UNIT-ERR",
                "section": "Plan",
                "statement": "Continue lisinopril 10 g once daily for hypertension.",
                "utterance_ids": ["U05"],
            },
            {
                "claim_id": "CLM-FREQ-ERR",
                "section": "Plan",
                "statement": "Initiate semaglutide 0.25 mg once daily subcutaneously.",
                "utterance_ids": ["U05"],
            },
        ]
        res = verify_clinical_claims(enc, bad_claims, audit_store=self.audit)
        self.assertEqual(res["blocked_count"], 3)
        rules = {v["rule_id"] for c in res["blocked_claims"] for v in c["violations"]}
        self.assertIn("CLIN-DOSE-01", rules)
        self.assertIn("CLIN-UNIT-02", rules)
        self.assertIn("CLIN-FREQ-03", rules)

    def test_05_clean_baseline_soap_claims_pass_100_percent(self) -> None:
        for enc_id in ("enc-ortho-lumbar-01", "enc-ortho-missing-pt-02", "enc-endo-diabetes-03", "enc-cardio-stress-04"):
            enc = get_encounter(enc_id)
            res = verify_clinical_claims(enc, enc["baseline_soap_claims"], audit_store=self.audit)
            self.assertEqual(res["verdict"], "ALL_CLAIMS_VERIFIED", msg=f"Failed on {enc_id}: {res['blocked_claims']}")
            self.assertEqual(res["blocked_count"], 0)

    def test_06_fhir_r4_bundle_compiler_passes_strict_validation(self) -> None:
        for enc_id in ("enc-ortho-lumbar-01", "enc-endo-diabetes-03", "enc-cardio-stress-04"):
            bundle = compile_fhir_r4_bundle(enc_id, deidentify_phi=True, scrubber=self.scrubber)
            val = validate_fhir_r4_bundle(bundle, encounter_id=enc_id, audit_store=self.audit)
            self.assertTrue(val["valid"], msg=f"Unexpected FHIR validation errors: {val['issues']}")
            self.assertEqual(val["error_count"], 0)
            self.assertGreaterEqual(val["resource_count"], 6)

    def test_07_fhir_r4_validator_catches_dangling_reference_and_bad_coding(self) -> None:
        bundle = compile_fhir_r4_bundle("enc-ortho-lumbar-01", deidentify_phi=True, scrubber=self.scrubber)
        broken = copy.deepcopy(bundle)
        # Break Condition.subject.reference and ICD-10-CM code
        broken["entry"][3]["resource"]["subject"]["reference"] = "urn:uuid:00000000-0000-0000-0000-000000000000"
        broken["entry"][3]["resource"]["code"]["coding"][0]["code"] = "INVALID_ICD10"
        val = validate_fhir_r4_bundle(broken, audit_store=self.audit)
        self.assertFalse(val["valid"])
        rule_ids = {iss["rule_id"] for iss in val["issues"]}
        self.assertIn("FHIR-REF-04", rule_ids)
        self.assertIn("FHIR-COND-04", rule_ids)

    def test_08_prior_auth_lumbar_mri_auto_eligible_vs_missing_pt(self) -> None:
        approved = evaluate_prior_auth("enc-ortho-lumbar-01", cpt_code="72148", audit_store=self.audit)
        self.assertEqual(approved["determination"], "AUTO_ELIGIBLE")
        self.assertEqual(approved["criteria_met_count"], 3)

        denied = evaluate_prior_auth("enc-ortho-missing-pt-02", cpt_code="72148", audit_store=self.audit)
        self.assertEqual(denied["determination"], "MISSING_DOCUMENTATION")
        self.assertGreaterEqual(len(denied["missing_actions"]), 1)

        # Adding red-flag progressive foot drop attestation satisfies step-therapy bypass
        override = evaluate_prior_auth(
            "enc-ortho-missing-pt-02",
            cpt_code="72148",
            clinician_attestation="Urgent addendum: acute foot drop and progressive motor weakness on exam.",
            audit_store=self.audit,
        )
        self.assertEqual(override["determination"], "AUTO_ELIGIBLE")

    def test_09_prior_auth_glp1_and_cardiac_nuclear_stress(self) -> None:
        glp1 = evaluate_prior_auth("enc-endo-diabetes-03", cpt_code="J3490-GLP1", audit_store=self.audit)
        self.assertEqual(glp1["determination"], "AUTO_ELIGIBLE")
        self.assertEqual(glp1["criteria_met_count"], 4)

        mpi = evaluate_prior_auth("enc-cardio-stress-04", cpt_code="78452", audit_store=self.audit)
        self.assertEqual(mpi["determination"], "AUTO_ELIGIBLE")
        self.assertEqual(mpi["criteria_met_count"], 3)

    def test_10_sqlite_wal_audit_chain_detects_single_row_tampering(self) -> None:
        self.audit.append_event("enc-ortho-lumbar-01", "TEST_OP_1", "OK", 0.4, {"step": 1})
        self.audit.append_event("enc-ortho-lumbar-01", "TEST_OP_2", "OK", 0.5, {"step": 2})
        intact = self.audit.verify_chain_integrity()
        self.assertTrue(intact["valid"])
        self.assertEqual(intact["checked_rows"], 2)

        # Directly mutate row 1 inside SQLite to simulate post-hoc tampering
        import sqlite3

        conn = sqlite3.connect(str(self.db_path))
        conn.execute("UPDATE clinical_audit_log SET verdict = 'FORGED' WHERE seq = 1;")
        conn.commit()
        conn.close()

        tampered = self.audit.verify_chain_integrity()
        self.assertFalse(tampered["valid"])
        self.assertEqual(tampered["broken_at_event_id"], "CLN-AUD-00001")

    def test_11_offline_soap_generation_and_api_router(self) -> None:
        prev = os.environ.get("HEALTHSCRIBE_OFFLINE_LLM")
        os.environ["HEALTHSCRIBE_OFFLINE_LLM"] = "1"
        try:
            soap = generate_span_linked_soap("enc-ortho-lumbar-01", use_bedrock=False, audit_store=self.audit)
            self.assertEqual(soap["safety_gate"]["verdict"], "ALL_CLAIMS_VERIFIED")

            code, health = route_api_request("GET", "/api/health", {}, {})
            self.assertEqual(code, 200)
            self.assertEqual(health["status"], "ok")

            code_pa, pa_body = route_api_request(
                "POST",
                "/api/prior-auth/check",
                {},
                {"encounter_id": "enc-ortho-lumbar-01", "cpt_code": "72148"},
            )
            self.assertEqual(code_pa, 200)
            self.assertEqual(pa_body["determination"], "AUTO_ELIGIBLE")
        finally:
            if prev is None:
                os.environ.pop("HEALTHSCRIBE_OFFLINE_LLM", None)
            else:
                os.environ["HEALTHSCRIBE_OFFLINE_LLM"] = prev

    def test_12_lambda_handler_cloudfront_api_contract(self) -> None:
        resp = lambda_handler(
            {
                "rawPath": "/api/fhir/validate",
                "requestContext": {"http": {"method": "POST"}},
                "body": '{"encounter_id": "enc-ortho-lumbar-01", "inject_defect": "dangling_reference"}',
            }
        )
        self.assertEqual(resp["statusCode"], 200)
        import json

        payload = json.loads(resp["body"])
        self.assertFalse(payload["validation"]["valid"])
        self.assertGreaterEqual(payload["validation"]["error_count"], 1)


if __name__ == "__main__":
    unittest.main()
