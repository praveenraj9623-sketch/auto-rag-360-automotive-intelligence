from __future__ import annotations

from pathlib import Path


DOCUMENT_TAGS = {
    "sample_service_manual.txt": {
        "document_type": "service_manual",
        "sensitivity_level": "internal_engineering",
    },
    "sample_quality_report.txt": {
        "document_type": "quality_report",
        "sensitivity_level": "internal_engineering",
    },
    "sample_recall_policy.txt": {
        "document_type": "recall_policy",
        "sensitivity_level": "confidential",
    },
    "sample_supplier_notice.txt": {
        "document_type": "supplier_notice",
        "sensitivity_level": "supplier_visible",
    },
}


def tag_document(path: Path) -> dict[str, str]:
    tags = DOCUMENT_TAGS.get(path.name, {})
    return {
        "document_type": tags.get("document_type", "knowledge_base"),
        "sensitivity_level": tags.get("sensitivity_level", "internal_engineering"),
    }
