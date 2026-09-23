"""Small, static Access Review graph and a deterministic, local graph walk.

The starter's users and policy text live here once.  Role instances are derived
from ownership records, so adding a user does not require a second role table.
"""

from copy import deepcopy
import re


KG: dict = {
    "role_hierarchy": {
        "Role": {"parent": None},
        "Ownership Role": {"parent": "Role"},
    },
    "role_types": {
        "GBR": {
            "parent": "Ownership Role",
            "full_name": "Global Business Role",
            "access_model": "persistent",
            "session_verification": False,
            "description": (
                "Grants broad, persistent access across the organization. "
                "Access is validated at assignment time and does not require "
                "per-session re-verification."
            ),
            "differs_from": "ZGBR",
            "source_ids": ["Policy-Doc-GBR-001", "Policy-Doc-CMP-003"],
        },
        "ZGBR": {
            "parent": "Ownership Role",
            "full_name": "Zero-Trust Global Business Role",
            "access_model": "per-session verification",
            "session_verification": True,
            "verification_methods": ["MFA", "just-in-time (JIT) approval"],
            "description": (
                "Enforces per-session verification via MFA or just-in-time "
                "(JIT) approval. Used for privileged access to sensitive systems."
            ),
            "differs_from": "GBR",
            "source_ids": ["Policy-Doc-ZGBR-002", "Policy-Doc-CMP-003"],
        },
    },
    "documents": [
        {
            "source_id": "Policy-Doc-GBR-001",
            "title": "GBR Access Policy",
            "content": (
                "A Global Business Role (GBR) grants broad, persistent access "
                "to resources across the organization. Access is validated at "
                "the time of role assignment and does not require per-session "
                "verification."
            ),
            "keywords": ["GBR", "global business role", "broad", "persistent"],
        },
        {
            "source_id": "Policy-Doc-ZGBR-002",
            "title": "ZGBR Access Policy",
            "content": (
                "A Zero-Trust Global Business Role (ZGBR) enforces per-session "
                "verification via MFA or just-in-time (JIT) approval. Used for "
                "privileged access to sensitive systems."
            ),
            "keywords": ["ZGBR", "zero trust", "per-session", "JIT", "MFA"],
        },
        {
            "source_id": "Policy-Doc-CMP-003",
            "title": "GBR vs ZGBR Comparison",
            "content": (
                "The key distinction between a GBR and a ZGBR is access persistence. "
                "A GBR provides standing access that does not require re-validation "
                "per session. A ZGBR enforces real-time, per-session verification "
                "aligned with zero-trust principles. ZGBR is typically assigned to "
                "roles with elevated data sensitivity."
            ),
            "keywords": ["difference", "compare", "comparison", "GBR", "ZGBR", "vs"],
        },
    ],
    "users": {
        "Priya": {
            "owns": ["GBR-1234"],
            "role_type": "GBR",
            "department": "Engineering",
            "review_history": [
                {"date": "2024-01-15", "status": "APPROVED", "reviewer": "Alice"},
                {"date": "2024-07-10", "status": "APPROVED", "reviewer": "Bob"},
                {"date": "2025-01-20", "status": "PENDING", "reviewer": None},
            ],
        },
        "Carlos": {
            "owns": ["ZGBR-5678"],
            "role_type": "ZGBR",
            "department": "Finance",
            "review_history": [
                {"date": "2024-03-10", "status": "APPROVED", "reviewer": "Carol"},
                {"date": "2024-09-18", "status": "FLAGGED", "reviewer": "Carol"},
                {"date": "2025-02-05", "status": "APPROVED", "reviewer": "Dave"},
            ],
        },
        "Fatima": {
            "owns": ["GBR-9999", "ZGBR-0001"],
            "role_type": "mixed",
            "department": "Security",
            "review_history": [
                {"date": "2024-02-20", "status": "APPROVED", "reviewer": "Eve"},
                {"date": "2024-08-14", "status": "APPROVED", "reviewer": "Alice"},
                {"date": "2025-03-01", "status": "PENDING", "reviewer": None},
            ],
        },
    },
}


def term_pattern(term: str) -> re.Pattern:
    """Match a whole term; spaces and hyphens in phrases are interchangeable."""
    words = re.split(r"[\s-]+", term.strip())
    body = r"[\s-]+".join(re.escape(word) for word in words)
    return re.compile(r"(?<!\w)" + body + r"(?!\w)", re.IGNORECASE)


def find_role_types(query: str) -> list[str]:
    """Match type names without treating GBR as a substring of ZGBR.

    Match longer full names first: the phrase 'Global Business Role' inside
    'Zero-Trust Global Business Role' must not create a second type entity.
    """
    aliases = [
        (alias, role_type)
        for role_type, data in KG["role_types"].items()
        for alias in (role_type, data.get("full_name", role_type))
    ]
    occupied = []
    found = set()
    for alias, role_type in sorted(aliases, key=lambda item: len(item[0]), reverse=True):
        for match in term_pattern(alias).finditer(query):
            start, end = match.span()
            if any(start < previous_end and end > previous_start
                   for previous_start, previous_end in occupied):
                continue
            occupied.append((start, end))
            found.add(role_type)
    return [role_type for role_type in KG["role_types"] if role_type in found]


def get_role_instances() -> dict:
    """Derive owned role nodes from the current users, with Role-DB provenance."""
    instances = {}
    for user, data in KG["users"].items():
        for role_id in data["owns"]:
            prefix = role_id.rsplit("-", 1)[0]
            role_type = prefix if prefix in KG["role_types"] else data.get("role_type")
            if role_type not in KG["role_types"]:
                role_type = None
            instance = instances.setdefault(
                role_id, {"role_type": role_type, "owners": [], "source_id": "Role-DB"}
            )
            instance["owners"].append(user)
    return instances


def _canonical_names(values: list, known: dict) -> list[str]:
    lookup = {name.casefold(): name for name in known}
    return list(dict.fromkeys(
        lookup[value.strip().casefold()]
        for value in values
        if isinstance(value, str) and value.strip().casefold() in lookup
    ))


def graph_walk(entities: dict) -> dict:
    """Walk only nodes reachable from the supplied users, roles, and types.

    Paths alternate node labels and relationship names.  A type comparison does
    not pull in users; a user query follows ownership to type and policy nodes.
    The returned context is detached from KG, so callers cannot mutate evidence.
    """
    instances = get_role_instances()
    users = _canonical_names(entities.get("users", []), KG["users"])
    role_ids = _canonical_names(entities.get("role_ids", []), instances)
    role_types = _canonical_names(entities.get("role_types", []), KG["role_types"])
    missing_users = list(dict.fromkeys(
        entities.get("unknown_users", []) + [
            name for name in entities.get("users", [])
            if isinstance(name, str) and not _canonical_names([name], KG["users"])
        ]
    ))
    paths = []

    # Explicit role IDs can resolve their owners as well as their policy type.
    for role_id in list(role_ids):
        for owner in instances[role_id]["owners"]:
            if owner not in users:
                users.append(owner)
    for user in users:
        for role_id in KG["users"][user]["owns"]:
            if role_id not in role_ids:
                role_ids.append(role_id)
            paths.append([f"user:{user}", "owns", f"role:{role_id}"])

    for role_id in role_ids:
        role_type = instances[role_id]["role_type"]
        if role_type is not None:
            if role_type not in role_types:
                role_types.append(role_type)
            paths.append([f"role:{role_id}", "instance_of", f"role_type:{role_type}"])

    source_ids = ["Role-DB"] if users or role_ids else []
    hierarchy = {}
    for role_type in role_types:
        path = [f"role_type:{role_type}"]
        parent = KG["role_types"][role_type].get("parent")
        visited = {role_type}
        while parent and parent not in visited:
            visited.add(parent)
            path.extend(["subtype_of", parent])
            node = KG["role_hierarchy"].get(parent, KG["role_types"].get(parent, {}))
            hierarchy[parent] = node
            parent = node.get("parent")
        paths.append(path)
        for source_id in KG["role_types"][role_type].get("source_ids", []):
            paths.append([f"role_type:{role_type}", "documented_by", source_id])
            if source_id not in source_ids:
                source_ids.append(source_id)

    return deepcopy({
        "paths": paths,
        "users": {name: KG["users"][name] for name in users},
        "role_instances": {role_id: instances[role_id] for role_id in role_ids},
        "role_types": {name: KG["role_types"][name] for name in role_types},
        "hierarchy": hierarchy,
        "documents": [doc for doc in KG["documents"] if doc["source_id"] in source_ids],
        "source_ids": source_ids,
        "missing_users": missing_users,
    })
