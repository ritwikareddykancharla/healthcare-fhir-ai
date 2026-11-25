"""HTTP Server & AWS CloudFront `/api/*` Lambda Origin Handler for HealthScribeFHIR."""
from __future__ import annotations

import base64
import json
import mimetypes
import os
import platform
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Tuple
from urllib.parse import parse_qs, urlparse

from .audit_store import SQLiteClinicalAuditStore
from .encounters import SYNTHETIC_NOTICE, get_encounter, list_encounters_summary
from .fhir_validator import compile_fhir_r4_bundle, validate_fhir_r4_bundle
from .phi_scrubber import HIPAA_18_CATEGORIES, HipaaSafeHarborScrubber
from .prior_auth import PAYER_POLICIES, evaluate_prior_auth
from .soap_scribe import DEFAULT_BEDROCK_MODEL, generate_span_linked_soap, verify_clinical_claims

ROOT_DIR = Path(__file__).resolve().parent.parent
WEB_DIR = ROOT_DIR / "web"
DOCS_DIR = ROOT_DIR / "docs"
BENCHMARK_JSON_PATH = ROOT_DIR / "benchmarks" / "latest_results.json"

AUDIT_STORE = SQLiteClinicalAuditStore()
PHI_SCRUBBER = HipaaSafeHarborScrubber(audit_store=AUDIT_STORE)

CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type, Authorization",
}


def _load_benchmarks() -> Dict[str, Any]:
    if BENCHMARK_JSON_PATH.is_file():
        try:
            return json.loads(BENCHMARK_JSON_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def route_api_request(
    method: str,
    path: str,
    query: Dict[str, str],
    body: Dict[str, Any],
    request_id: str = "local",
) -> Tuple[int, Dict[str, Any]]:
    """Pure router shared by local HTTP server and AWS Lambda `/api/*` origin behind CloudFront."""
    try:
        if method == "GET" and path == "/api/health":
            chain = AUDIT_STORE.verify_chain_integrity()
            return 200, {
                "status": "ok",
                "service": "healthscribe-fhir",
                "synthetic_data_only": True,
                "synthetic_label": SYNTHETIC_NOTICE,
                "delivery": os.environ.get("HEALTHSCRIBE_DELIVERY", "cloudfront-s3-oac + lambda-api-origin"),
                "aws_region": os.environ.get("AWS_REGION", os.environ.get("AWS_DEFAULT_REGION", "us-west-2")),
                "bedrock_model": os.environ.get("HEALTHSCRIBE_BEDROCK_MODEL", DEFAULT_BEDROCK_MODEL),
                "python_version": platform.python_version(),
                "encounters_loaded": len(list_encounters_summary()),
                "audit_chain_valid": chain["valid"],
                "audit_rows": chain["checked_rows"],
                "request_id": request_id,
            }

        if method == "GET" and path == "/api/benchmarks":
            return 200, {
                "service": "healthscribe-fhir",
                "benchmarks": _load_benchmarks(),
                "request_id": request_id,
            }

        if method == "GET" and path == "/api/state":
            chain = AUDIT_STORE.verify_chain_integrity()
            return 200, {
                "synthetic_label": SYNTHETIC_NOTICE,
                "encounters": list_encounters_summary(),
                "payer_policies": list(PAYER_POLICIES.values()),
                "hipaa_categories": HIPAA_18_CATEGORIES,
                "audit_chain": chain,
                "audit_events": AUDIT_STORE.list_recent_events(limit=25),
                "benchmarks": _load_benchmarks(),
                "request_id": request_id,
            }

        if method == "POST" and path == "/api/scribe/generate":
            encounter_id = str(body.get("encounter_id") or "enc-ortho-lumbar-01").strip()
            injected = body.get("injected_adversarial_claim")
            use_bedrock = bool(body.get("use_bedrock", True))
            out = generate_span_linked_soap(
                encounter_id=encounter_id,
                injected_adversarial_claim=injected if isinstance(injected, dict) else None,
                use_bedrock=use_bedrock,
                audit_store=AUDIT_STORE,
            )
            out["request_id"] = request_id
            return 200, out

        if method == "POST" and path == "/api/safety/verify":
            encounter_id = str(body.get("encounter_id") or "enc-ortho-lumbar-01").strip()
            encounter = get_encounter(encounter_id)
            claims = body.get("claims")
            if not isinstance(claims, list) or not claims:
                return 400, {"error": "claims array is required"}
            out = verify_clinical_claims(encounter=encounter, claims=claims, audit_store=AUDIT_STORE)
            out["request_id"] = request_id
            return 200, out

        if method == "POST" and path == "/api/fhir/validate":
            encounter_id = str(body.get("encounter_id") or "enc-ortho-lumbar-01").strip()
            deidentify = bool(body.get("deidentify_phi", True))
            custom_bundle = body.get("bundle")
            mutation = str(body.get("inject_defect") or "none").strip()

            if isinstance(custom_bundle, dict):
                bundle = custom_bundle
            else:
                bundle = compile_fhir_r4_bundle(
                    encounter_id=encounter_id,
                    deidentify_phi=deidentify,
                    scrubber=PHI_SCRUBBER,
                )
                if mutation == "dangling_reference":
                    bundle["entry"][3]["resource"]["subject"]["reference"] = "urn:uuid:00000000-dead-beef-0000-999999999999"
                elif mutation == "invalid_icd10_uri":
                    bundle["entry"][3]["resource"]["code"]["coding"][0]["system"] = "http://invalid.example/icd9"
                    bundle["entry"][3]["resource"]["code"]["coding"][0]["code"] = "999.99999"
                elif mutation == "invalid_ucum_unit":
                    for ent in bundle["entry"]:
                        if ent["resource"]["resourceType"] == "Observation":
                            ent["resource"]["valueQuantity"]["code"] = "INVALID_UNIT_XYZ"
                            break
                elif mutation == "missing_med_intent":
                    for ent in bundle["entry"]:
                        if ent["resource"]["resourceType"] == "MedicationRequest":
                            ent["resource"]["intent"] = ""
                            break

            validation = validate_fhir_r4_bundle(
                bundle=bundle,
                encounter_id=encounter_id,
                audit_store=AUDIT_STORE,
            )
            return 200, {
                "encounter_id": encounter_id,
                "inject_defect": mutation,
                "validation": validation,
                "bundle": bundle,
                "request_id": request_id,
            }

        if method == "POST" and path == "/api/prior-auth/check":
            encounter_id = str(body.get("encounter_id") or "enc-ortho-lumbar-01").strip()
            cpt_code = body.get("cpt_code")
            attestation = body.get("clinician_attestation")
            deidentify_packet = bool(body.get("deidentify_packet", True))
            out = evaluate_prior_auth(
                encounter_id=encounter_id,
                cpt_code=str(cpt_code).strip() if cpt_code else None,
                clinician_attestation=str(attestation) if attestation else None,
                deidentify_packet=deidentify_packet,
                audit_store=AUDIT_STORE,
            )
            out["request_id"] = request_id
            return 200, out

        if method == "POST" and path == "/api/phi/scrub":
            raw_text = body.get("text")
            encounter_id = str(body.get("encounter_id") or "").strip()
            reidentify_reason = str(body.get("reidentify_audit_reason") or "").strip()
            if raw_text and str(raw_text).strip():
                out = PHI_SCRUBBER.scrub_text(str(raw_text), encounter_id=encounter_id or "custom-note")
            else:
                enc = get_encounter(encounter_id or "enc-ortho-lumbar-01")
                out = PHI_SCRUBBER.scrub_encounter(enc)

            if reidentify_reason:
                out["reidentified"] = PHI_SCRUBBER.reidentify_text(
                    out["scrubbed_text"],
                    requester_role="ATTENDING_PHYSICIAN",
                    audit_reason=reidentify_reason,
                )
            out["request_id"] = request_id
            return 200, out

        if method == "POST" and path == "/api/reset":
            AUDIT_STORE.reset()
            AUDIT_STORE.append_event(
                encounter_id="system",
                action_type="AUDIT_LEDGER_INIT",
                verdict="READY",
                duration_ms=0.12,
                details={"synthetic_only": True},
            )
            return 200, {
                "status": "reset_complete",
                "service": "healthscribe-fhir",
                "request_id": request_id,
            }

        return 404, {"error": f"Unknown API route: {method} {path}"}
    except KeyError as exc:
        return 404, {"error": str(exc)}
    except ValueError as exc:
        return 400, {"error": str(exc)}
    except Exception as exc:
        return 500, {"error": str(exc)}


def lambda_handler(event: Dict[str, Any], context: Any = None) -> Dict[str, Any]:
    """AWS Lambda Function URL handler mounted behind Amazon CloudFront `/api/*` origin."""
    req_ctx = event.get("requestContext") or {}
    method = ((req_ctx.get("http") or {}).get("method") or event.get("httpMethod") or "GET").upper()
    raw_path = event.get("rawPath") or event.get("path") or "/"
    query_params = {k: str(v) for k, v in (event.get("queryStringParameters") or {}).items() if v is not None}
    req_id = getattr(context, "aws_request_id", "cloudfront-api")

    if method == "OPTIONS":
        return {"statusCode": 204, "headers": CORS_HEADERS, "body": ""}

    if raw_path.startswith("/api/"):
        body: Dict[str, Any] = {}
        raw_body = event.get("body")
        if raw_body:
            if event.get("isBase64Encoded") and isinstance(raw_body, str):
                raw_body = base64.b64decode(raw_body).decode("utf-8")
            body = json.loads(raw_body) if isinstance(raw_body, str) else dict(raw_body)
        status, payload = route_api_request(method, raw_path, query_params, body, request_id=req_id)
        return {
            "statusCode": status,
            "headers": {
                **CORS_HEADERS,
                "Content-Type": "application/json; charset=utf-8",
                "Cache-Control": "no-store",
            },
            "body": json.dumps(payload, default=str),
        }

    target = WEB_DIR / ("index.html" if raw_path in ("/", "") else raw_path.lstrip("/"))
    if not target.is_file() and raw_path.endswith(".svg"):
        target = DOCS_DIR / Path(raw_path).name
    if target.is_file():
        ctype = mimetypes.guess_type(str(target))[0] or "text/html; charset=utf-8"
        return {
            "statusCode": 200,
            "headers": {**CORS_HEADERS, "Content-Type": ctype},
            "body": target.read_text(encoding="utf-8"),
        }
    return {
        "statusCode": 404,
        "headers": {**CORS_HEADERS, "Content-Type": "application/json"},
        "body": json.dumps({"error": "Not found", "path": raw_path}),
    }


class LocalHealthScribeHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args: Any) -> None:
        return

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        for k, v in CORS_HEADERS.items():
            self.send_header(k, v)
        self.end_headers()

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        query = {k: v[0] for k, v in parse_qs(parsed.query).items() if v}

        if path.startswith("/api/"):
            status, payload = route_api_request("GET", path, query, {}, request_id="local-dev")
            raw = json.dumps(payload, indent=2, default=str).encode("utf-8")
            self.send_response(status)
            for k, v in CORS_HEADERS.items():
                self.send_header(k, v)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
            return

        rel = "index.html" if path in ("/", "") else path.lstrip("/")
        candidate = (WEB_DIR / rel).resolve()
        if not candidate.is_file() and path.endswith(".svg"):
            candidate = (DOCS_DIR / Path(path).name).resolve()
        if not candidate.is_file():
            candidate = WEB_DIR / "index.html"

        data = candidate.read_bytes()
        ctype = mimetypes.guess_type(str(candidate))[0] or "text/html"
        if ctype.startswith("text/") or ctype in ("application/javascript", "image/svg+xml"):
            ctype += "; charset=utf-8"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        length = int(self.headers.get("Content-Length") or "0")
        raw_body = self.rfile.read(length).decode("utf-8") if length > 0 else "{}"
        body = json.loads(raw_body) if raw_body.strip() else {}
        status, payload = route_api_request("POST", parsed.path, {}, body, request_id="local-dev")
        raw = json.dumps(payload, indent=2, default=str).encode("utf-8")
        self.send_response(status)
        for k, v in CORS_HEADERS.items():
            self.send_header(k, v)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


def run_local_server(port: int = 3099) -> None:
    srv = ThreadingHTTPServer(("127.0.0.1", port), LocalHealthScribeHandler)
    print(f"HealthScribeFHIR Studio running at http://localhost:{port}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    port_num = int(os.environ.get("PORT", "3099"))
    run_local_server(port=port_num)
