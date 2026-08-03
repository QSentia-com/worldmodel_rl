from uuid import uuid4

from qsentia_utils.db import model_output_record, model_run_record
from qsentia_utils.models import Environment, ModelOutputCreate, ModelRunCreate


def test_model_run_create_record():
    run = ModelRunCreate(
        environment=Environment.DEV,
        producer_id="eth-futures-sentiment",
        producer_version="20260608",
        run_type="inference",
    )

    record = model_run_record(run)

    assert record["environment"] == "dev"
    assert record["producer_id"] == "eth-futures-sentiment"
    assert record["status"] == "started"
    assert "id" in record


def test_model_output_create_record():
    output = ModelOutputCreate(
        run_id=uuid4(),
        environment="dev",
        producer_id="eth-futures-sentiment",
        output={"signal": "LONG", "confidence": 0.72},
    )

    record = model_output_record(output)

    assert record["output_type"] == "prediction"
    assert record["schema_version"] == 1
    assert record["output"]["signal"] == "LONG"
