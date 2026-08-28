from input_orchestrator.adapters import CallableExtractorAdapter
from input_orchestrator.models import FieldState


def test_adapter_reads_requirement_fields():
    adapter = CallableExtractorAdapter(
        domain="categorical",
        call=lambda text: {
            "requirement_fields": {
                "ha_required": True,
                "access_type": "mixed",
            }
        },
        allowed_fields=["ha_required", "access_type"],
    )

    out = adapter.extract("x", message_id="M1")
    values = {o.field: o.value for o in out}

    assert values == {
        "ha_required": True,
        "access_type": "mixed",
    }


def test_adapter_reads_detailed_field_status():
    adapter = CallableExtractorAdapter(
        domain="categorical",
        call=lambda text: {
            "ha_required": {
                "value": None,
                "status": "NO_EVIDENCE",
                "source": "SEMANTIC_MODEL",
                "evidence": "HA matters",
            }
        },
        allowed_fields=["ha_required"],
    )

    out = adapter.extract("x", message_id="M1")
    assert len(out) == 1
    assert out[0].state == FieldState.MISSING
