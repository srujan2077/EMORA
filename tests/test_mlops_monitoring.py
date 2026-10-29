import unittest

from monitoring.evidently_report import smoke_records
from monitoring.schema import to_dataframe, validate_records


class MonitoringSchemaTests(unittest.TestCase):
    def test_valid_records_convert(self):
        records = smoke_records(0.0)
        validate_records(records)
        self.assertEqual(len(to_dataframe(records)), 2)

    def test_empty_records_rejected(self):
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            validate_records([])

    def test_missing_field_rejected(self):
        record = smoke_records(0.0)[0]
        del record["probability_fear"]
        with self.assertRaisesRegex(ValueError, "missing fields"):
            validate_records([record])

    def test_invalid_probability_rejected(self):
        record = smoke_records(0.0)[0]
        record["probability_fear"] = 2
        with self.assertRaisesRegex(ValueError, "invalid fear"):
            validate_records([record])


if __name__ == "__main__":
    unittest.main()
