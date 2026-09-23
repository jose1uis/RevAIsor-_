"""Deterministic evidence tools backed by the single static Knowledge Graph."""

from copy import deepcopy

from knowledge_graph import KG, find_role_types, term_pattern


def role_tool(user_id: str) -> dict:
    """Return a user's ownership and review records; cite these as [Role-DB]."""
    if not isinstance(user_id, str) or not user_id.strip():
        return {"error": "A non-empty user name is required for the Role DB."}
    for name, data in KG["users"].items():
        if name.casefold() == user_id.strip().casefold():
            return {name: deepcopy(data)}
    return {"error": f"User '{user_id}' not found in Role DB."}


def document_tool(query: str) -> list:
    """Return matching policy fragments, each with its required source_id.

    Keyword matching uses word boundaries, preventing 'GBR' from matching
    'ZGBR'. Full role names follow the same rule through find_role_types.
    Unknown or empty searches return an empty list.
    """
    if not isinstance(query, str) or not query.strip():
        return []
    role_types = find_role_types(query)
    type_aliases = {
        alias.casefold(): role_type
        for role_type, data in KG["role_types"].items()
        for alias in (role_type, data.get("full_name", role_type))
    }

    def keyword_matches(keyword: str) -> bool:
        role_type = type_aliases.get(keyword.casefold())
        if role_type is not None:
            return role_type in role_types
        return bool(term_pattern(keyword).search(query))

    return deepcopy([
        doc for doc in KG["documents"]
        if any(keyword_matches(keyword) for keyword in doc["keywords"])
    ])
