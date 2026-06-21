from src.rag.access_control import can_access, filter_accessible, get_roles


def test_exactly_four_roles_are_defined() -> None:
    assert get_roles() == ["Executive", "Quality Engineer", "Service Engineer", "Supplier"]


def test_supplier_is_blocked_from_internal_engineering_docs() -> None:
    internal_engineering_doc = {
        "document_type": "service_manual",
        "sensitivity_level": "internal_engineering",
    }
    assert can_access(internal_engineering_doc, "Supplier") is False


def test_supplier_can_access_supplier_visible_notice() -> None:
    supplier_notice = {
        "document_type": "supplier_notice",
        "sensitivity_level": "supplier_visible",
    }
    assert can_access(supplier_notice, "Supplier") is True


def test_filter_accessible_removes_supplier_internal_docs() -> None:
    chunks = [
        {
            "text": "internal brake actuator diagnostic procedure",
            "metadata": {"document_type": "service_manual", "sensitivity_level": "internal_engineering"},
        },
        {
            "text": "supplier notice for brake hose bracket",
            "metadata": {"document_type": "supplier_notice", "sensitivity_level": "supplier_visible"},
        },
    ]
    allowed = filter_accessible(chunks, "Supplier")
    assert len(allowed) == 1
    assert allowed[0]["metadata"]["document_type"] == "supplier_notice"
