"""Deterministic entity -> graph -> intent -> tool routing, before the LLM."""

import re

from knowledge_graph import KG, find_role_types, get_role_instances, graph_walk


_TOOL_SELECTION = {
    "role_ownership": ["role_tool"],
    "access_history": ["role_tool"],
    "ownership_history": ["role_tool"],
    "role_comparison": ["document_tool"],
    "role_definition": ["document_tool"],
    "combined": ["role_tool", "document_tool"],
    "unsupported_access": [],
    "out_of_scope": [],
}

_NAME = r"(?P<names>[A-Za-z][A-Za-z-]*(?:\s*(?:,|\band\b)\s*[A-Za-z][A-Za-z-]*)*)"
_UNKNOWN_NAME_PATTERNS = (
    rf"\b(?:does|do)\s+{_NAME}\s+(?:own|have|hold|possess)\b",
    rf"\b{_NAME}'s\s+(?:[A-Za-z][A-Za-z0-9-]*\s+)?"
    rf"(?:access|roles?|review(?:s|er)?|ownership)\b",
    rf"\b(?:roles?|ownership|(?:access\s+)?(?:review\s+)?history)\s+"
    rf"(?:for|of|(?:assigned|belonging)\s+to)\s+{_NAME}\b",
    rf"\b(?:compare|contrast)\s+{_NAME}\s+(?:roles?|ownership|access)\b",
)
_NOT_NAMES = {
    "a", "an", "the", "all", "any", "each", "every", "some", "other",
    "user", "users", "person", "people", "employee", "employees", "someone",
    "i", "me", "my", "we", "us", "you", "your", "he", "she", "they",
    "them", "their", "it", "its", "this", "that", "these", "those",
    "role", "roles", "access", "review", "policy", "company", "organization",
}


def _normalize(query: str) -> str:
    return query.replace("\u2019", "'").replace("\u2018", "'").strip()


def extract_entities(query: str) -> dict:
    """Recognize graph entities, retaining explicit unknown names as missing data.

    Unknown names are deliberately limited to clear ownership/history patterns;
    arbitrary capitalized words are not treated as people.
    """
    query = _normalize(query)
    users = [
        name for name in KG["users"]
        if re.search(rf"(?<!\w){re.escape(name)}(?!\w)", query, re.IGNORECASE)
    ]
    entities = {"users": users, "role_types": find_role_types(query)}
    known_names = {name.casefold() for name in KG["users"]}
    role_names = {name.casefold() for name in KG["role_types"]}
    unknown = []
    for pattern in _UNKNOWN_NAME_PATTERNS:
        for match in re.finditer(pattern, query, re.IGNORECASE):
            for name in re.split(r"\s*(?:,|\band\b)\s*", match.group("names"), flags=re.I):
                if name.casefold() not in known_names | role_names | _NOT_NAMES:
                    if name.casefold() not in {item.casefold() for item in unknown}:
                        unknown.append(name)
    if unknown:
        entities["unknown_users"] = unknown

    # IDs are distinct from generic role types. Unknown IDs are retained too.
    role_ids = re.findall(r"\b[A-Za-z][A-Za-z0-9]*-\d[A-Za-z0-9-]*\b", query)
    if role_ids:
        entities["role_ids"] = list(dict.fromkeys(value.upper() for value in role_ids))
    return entities


def classify_intent(query: str, entities: dict) -> str:
    """Classify supported read operations without treating names alone as scope."""
    query = _normalize(query).lower()
    people = bool(entities.get("users") or entities.get("unknown_users"))
    role_ids = bool(entities.get("role_ids"))
    role_types = bool(entities.get("role_types"))
    role_words = bool(re.search(r"\b(?:roles?|ownership)\b", query))
    access_words = bool(re.search(r"\b(?:access|permissions?|entitlements?|authorization)\b", query))
    policy_terms = bool(re.search(r"\b(?:mfa|jit|zero[ -]trust)\b", query))
    review_history = bool(re.search(r"\breview\s+history\b", query))
    if not (role_words or access_words or role_types or role_ids or policy_terms or review_history):
        return "out_of_scope"

    # Read-only assistant: recognize requests to act, not policy verbs such as
    # "Does GBR grant persistent access?" or historical "assigned" records.
    action = r"(?:grant|revoke|remove|delete|assign|create|change|modify|update|approve|deny|disable|enable|reset)"
    command = rf"^(?:please\s+)?{action}\b"
    request = rf"\b(?:can|could|would|will)\s+you\s+(?:please\s+)?{action}\b"
    desire = rf"\bi\s+(?:want|need|would\s+like)\s+(?:you\s+)?to\s+{action}\b"
    subsequent_action = rf"\b(?:and|then)\s+(?:then\s+)?{action}\b"
    give_access = (r"\bgive\s+[a-z][a-z-]*\s+(?:an?\s+)?"
                   r"(?:access|permissions?|(?:gbr|zgbr)(?:\s+role)?)\b")
    if any(re.search(pattern, query) for pattern in
           (command, request, desire, subsequent_action, give_access)):
        return "unsupported_access"

    comparison = bool(re.search(
        r"\b(?:compare|compared|comparison|contrast|difference|differences|different|differ|differs|distinction|versus|vs)\b",
        query,
    ))
    history = bool(re.search(r"\b(?:history|historical|reviews?|reviewed|reviewers?|approved|flagged|pending)\b", query))
    ownership = bool(re.search(r"\b(?:owns?|owned|ownership|holds?|assigned|assignments?)\b", query))
    if comparison:
        return "combined" if people or role_ids else "role_comparison"
    if history:
        return "ownership_history" if ownership else "access_history"
    if (people and (role_words or ownership)) or (role_ids and ownership):
        return "role_ownership"
    if ownership:
        return "role_ownership"
    if role_types or policy_terms:
        return "combined" if people else "role_definition"
    return "unsupported_access"


def select_tools(intent: str) -> list[str]:
    """Return a fresh list; an unrecognized intent never grants tool access."""
    return list(_TOOL_SELECTION.get(intent, []))


def route(query: str) -> dict:
    """Walk the graph before classification, then select a fixed tool plan."""
    entities = extract_entities(query)
    graph_context = graph_walk(entities)
    intent = classify_intent(query, entities)
    inferred = []
    if (intent == "role_ownership" and not entities["users"]
            and not entities.get("unknown_users") and not entities.get("role_ids")
            and entities["role_types"]):
        # Reverse the ownership edge for questions such as 'Who owns a ZGBR?'.
        for role in get_role_instances().values():
            if role["role_type"] in entities["role_types"]:
                for owner in role["owners"]:
                    if owner not in entities["users"]:
                        entities["users"].append(owner)
                        inferred.append(owner)
        graph_context = graph_walk(entities)
    if intent in {"role_ownership", "access_history", "ownership_history", "combined"}:
        for role_id in entities.get("role_ids", []):
            role = graph_context.get("role_instances", {}).get(role_id, {})
            for owner in role.get("owners", []):
                if owner not in entities["users"]:
                    entities["users"].append(owner)
                    inferred.append(owner)
    tools = select_tools(intent)
    strategy = (
        f"Detected users: {', '.join(entities['users']) or 'none'}; "
        f"role types: {', '.join(entities['role_types']) or 'none'}; "
        f"intent: {intent}; selected tools: {', '.join(tools) or 'none'}."
    )
    if entities.get("unknown_users"):
        strategy += f" No user record for: {', '.join(entities['unknown_users'])}."
    if inferred:
        strategy += f" Resolved role owners from the graph: {', '.join(inferred)}."
    return {
        "intent": intent,
        "entities": entities,
        "graph_context": graph_context,
        "tools_to_call": tools,
        "strategy": strategy,
    }
