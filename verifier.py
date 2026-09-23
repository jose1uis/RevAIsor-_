"""Small deterministic guardrails for cited answers, not a semantic truth proof.

The agent passes the actual tool results. The two-argument public API can also
recreate scoped static evidence without an LLM for demonstrations and tests.
"""

from __future__ import annotations

import json
import re

from knowledge_graph import KG
from router import route
from tools import document_tool, role_tool


OUT_OF_SCOPE_RESPONSE = (
    "I can help only with access-review questions about role ownership, "
    "role policies, and review history."
)
INSUFFICIENT_EVIDENCE_RESPONSE = (
    "I do not have enough trusted evidence to answer this access-review question."
)
UNSUPPORTED_ACCESS_RESPONSE = (
    "I can answer questions about recorded role ownership, role policies, and "
    "review history, but I cannot change access or answer access questions "
    "outside that evidence."
)

_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_SOURCE = re.compile(r"\b(?:Role-DB|Policy-Doc-[A-Za-z0-9-]+)\b", re.I)


def _scoped_evidence(query: str, decision: dict) -> dict:
    """Compatibility path only; run_agent supplies its already executed results."""
    evidence = {}
    entities = decision["entities"]
    if "role_tool" in decision["tools_to_call"]:
        users = entities.get("users", []) + entities.get("unknown_users", [])
        evidence["role_tool"] = [role_tool(user) for user in dict.fromkeys(users)]
    if "document_tool" in decision["tools_to_call"]:
        types = decision.get("graph_context", {}).get("role_types", {})
        evidence["document_tool"] = document_tool(query + " " + " ".join(types))
    return evidence


def _plain_inline_text(text: str) -> str:
    """Unwrap paired inline Markdown markers, retaining content and punctuation."""
    pattern = re.compile(
        r"(?<!\w)(?P<marker>\*{1,3}|_{1,3}|`+)(?=\S)"
        r"(?P<content>[^\n]+?)(?<=\S)(?P=marker)(?!\w)"
    )
    previous = None
    while previous != text:
        previous = text
        text = pattern.sub(lambda match: match.group("content"), text)
    return text


def _role_ids(text: str) -> set[str]:
    """Recognize known type prefixes and other numeric IDs; ignore source IDs."""
    text = _SOURCE.sub("", _plain_inline_text(text))
    for document in KG["documents"]:
        text = re.sub(re.escape(document["source_id"]), "", text, flags=re.I)
    types = "|".join(re.escape(name) for name in KG["role_types"])
    pattern = (
        rf"\b(?:{types})[-_][A-Za-z0-9]+(?:[-_][A-Za-z0-9]+)*\b"
        r"|\b[A-Za-z][A-Za-z0-9]*[-_]\d[A-Za-z0-9]*(?:[-_][A-Za-z0-9]+)*\b"
    )
    return {match.group().upper() for match in re.finditer(pattern, text, re.I)}


def _citations(response: str, known_sources: set[str]) -> set[str]:
    """Accept source names in brackets or plain text; flag unknown bracket IDs."""
    citations = {match.group().casefold() for match in _SOURCE.finditer(response)}
    for source in known_sources:
        if re.search(rf"(?<!\w){re.escape(source)}(?!\w)", response, re.I):
            citations.add(source.casefold())
    known_roles = {role.casefold() for data in KG["users"].values() for role in data["owns"]}
    for group in re.findall(r"\[([^\[\]\n]+)\]", response):
        # Brackets may format an actual role ID; global ID/scope checks still
        # validate it. Other bracket tokens remain subject to citation checks.
        citations.update(
            part.strip().casefold() for part in re.split(r"[,;]", group)
            if part.strip().casefold() not in known_roles
        )
    return citations


def _wrong_ownership(response: str, users: dict) -> list[str]:
    """Check explicit 'Priya owns ...' claims, including two-user responses.

    This intentionally does not attempt unrestricted natural-language parsing.
    """
    response = _plain_inline_text(response)
    errors = []
    for name, data in users.items():
        pattern = (
            rf"\b{re.escape(name)}(?:['’]s\s+roles?\s+(?:is|are)|"
            r"\s+(?:owns?|holds?|has))\s+([^.!?;\n]+)"
        )
        for match in re.finditer(pattern, response, re.I):
            claim = match.group(1)
            # A sentence may continue with another person's ownership clause.
            for other in users:
                if other != name:
                    claim = re.split(rf"\b{re.escape(other)}\b", claim, flags=re.I)[0]
            # A correction can say "owns A, not B". Exclude only that explicit
            # negation here; the full response still undergoes ID/scope checks.
            for role_id in _role_ids(claim):
                claim = re.sub(rf"\bnot\s+{re.escape(role_id)}(?![\w-])", "", claim, flags=re.I)
            wrong = _role_ids(claim) - {role.upper() for role in data.get("owns", [])}
            errors.extend(f"{name}: {role}" for role in sorted(wrong))
    return errors


def verify_response(query: str, response: str, tool_outputs: dict | None = None) -> dict:
    """Check citations, scoped role IDs, explicit ownership and ISO review dates.

    These syntactic checks cannot establish the truth of every paraphrase or
    policy claim. In particular, valid citations alone do not prove entailment.
    """
    decision = route(query)
    evidence = _scoped_evidence(query, decision) if tool_outputs is None else tool_outputs
    checks = []

    def check(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    if not isinstance(response, str) or not response.strip():
        check("response_present", False, "A nonempty text response is required.")
        return {"status": "FAIL", "checks": checks}

    refusal = {
        "out_of_scope": OUT_OF_SCOPE_RESPONSE,
        "unsupported_access": UNSUPPORTED_ACCESS_RESPONSE,
    }.get(decision["intent"])
    safe_response = response.strip() == (refusal or INSUFFICIENT_EVIDENCE_RESPONSE)
    if refusal:
        check("scope_refusal", safe_response, "The deterministic scope refusal is required.")
        check("no_unnecessary_tools", not evidence, "Scope refusals must have no tool calls.")

    users = {}
    for result in evidence.get("role_tool", []):
        users.update({name: data for name, data in result.items() if isinstance(data, dict)})
    docs = evidence.get("document_tool", [])
    document_sources = {doc["source_id"].casefold() for doc in docs if "source_id" in doc}
    allowed_sources = set(document_sources)
    # A failed lookup still supports a factual statement that a user was not found.
    if evidence.get("role_tool"):
        allowed_sources.add("role-db")
    known_sources = {"Role-DB"} | {doc["source_id"] for doc in KG["documents"]}
    citations = _citations(response, known_sources)
    check(
        "citation_present",
        safe_response or bool(citations & allowed_sources),
        "No citation needed for a safe refusal." if safe_response else
        "Factual answers need at least one citation to retrieved evidence.",
    )
    unknown_citations = citations - allowed_sources
    check("citations_retrieved", not unknown_citations,
          "Unretrieved or unknown citations: " + ", ".join(sorted(unknown_citations))
          if unknown_citations else "All cited sources were retrieved for this query.")

    missing_kinds = []
    if not safe_response:
        if evidence.get("role_tool") and "role-db" not in citations:
            missing_kinds.append("Role-DB")
        if document_sources and not citations & document_sources:
            missing_kinds.append("document source_id")
    check("required_source_citations", not missing_kinds,
          "Missing citations: " + ", ".join(missing_kinds) if missing_kinds
          else "Each retrieved evidence kind has the required citation.")

    mentioned_roles = _role_ids(response)
    known_roles = {role.upper() for data in KG["users"].values() for role in data["owns"]}
    scoped_roles = {role.upper() for data in users.values() for role in data.get("owns", [])}
    invented = mentioned_roles - known_roles
    unscoped = mentioned_roles - scoped_roles
    check("no_invented_roles", not invented,
          "Unknown role IDs: " + ", ".join(sorted(invented)) if invented
          else "All mentioned role IDs exist in the knowledge graph.")
    check("roles_in_retrieved_evidence", not unscoped,
          "Role IDs outside this query's evidence: " + ", ".join(sorted(unscoped))
          if unscoped else "Role IDs are supported by this query's role lookup.")
    wrong_owners = _wrong_ownership(response, users)
    check("ownership_matches", not wrong_owners,
          "Unsupported ownership: " + ", ".join(wrong_owners) if wrong_owners
          else "Recognized explicit ownership claims match retrieved records.")

    unsupported_dates = set(_DATE.findall(response)) - set(_DATE.findall(json.dumps(evidence)))
    check("dates_in_retrieved_evidence", not unsupported_dates,
          "Unsupported ISO dates: " + ", ".join(sorted(unsupported_dates))
          if unsupported_dates else "ISO dates occur in the retrieved evidence.")
    return {"status": "SUCCESS" if all(item["passed"] for item in checks) else "FAIL",
            "checks": checks}
