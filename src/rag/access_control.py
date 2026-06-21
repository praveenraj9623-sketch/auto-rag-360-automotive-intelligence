from __future__ import annotations


ROLE_POLICIES = {
    "Executive": {
        "document_types": {
            "recall_notice",
            "consumer_complaint",
            "service_manual",
            "quality_report",
            "recall_policy",
            "supplier_notice",
        },
        "sensitivity_levels": {"public", "internal_engineering", "confidential", "supplier_visible"},
    },
    "Quality Engineer": {
        "document_types": {"recall_notice", "consumer_complaint", "quality_report", "recall_policy", "supplier_notice"},
        "sensitivity_levels": {"public", "internal_engineering", "confidential", "supplier_visible"},
    },
    "Service Engineer": {
        "document_types": {"recall_notice", "consumer_complaint", "service_manual", "supplier_notice"},
        "sensitivity_levels": {"public", "internal_engineering", "supplier_visible"},
    },
    "Supplier": {
        "document_types": {"recall_notice", "supplier_notice"},
        "sensitivity_levels": {"public", "supplier_visible"},
    },
}


def get_roles() -> list[str]:
    return list(ROLE_POLICIES.keys())


def get_policy(role: str) -> dict[str, set[str]]:
    if role not in ROLE_POLICIES:
        raise ValueError(f"Unknown role: {role}")
    return ROLE_POLICIES[role]


def can_access(metadata: dict, role: str) -> bool:
    policy = get_policy(role)
    document_type = metadata.get("document_type")
    sensitivity_level = metadata.get("sensitivity_level")
    return document_type in policy["document_types"] and sensitivity_level in policy["sensitivity_levels"]


def filter_accessible(chunks: list[dict], role: str) -> list[dict]:
    return [chunk for chunk in chunks if can_access(chunk.get("metadata", {}), role)]


def describe_policy(role: str) -> str:
    policy = get_policy(role)
    document_types = ", ".join(sorted(policy["document_types"]))
    sensitivity_levels = ", ".join(sorted(policy["sensitivity_levels"]))
    return f"{role} can retrieve document_type in [{document_types}] with sensitivity_level in [{sensitivity_levels}]."
