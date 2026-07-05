import unittest

import pandas as pd

from src.data.csv_normalizer import normalize_text_dataframe


class CsvNormalizerTests(unittest.TestCase):
    def test_normalizes_xquik_export_columns(self):
        result = normalize_text_dataframe(
            pd.DataFrame(
                {
                    "tweet_id": ["1", "2"],
                    "full_text": ["Great launch", "Needs better docs"],
                    "sentiment": ["positive", "negative"],
                }
            ),
            label_col="sentiment",
        )

        self.assertEqual(result["text"].tolist(), ["Great launch", "Needs better docs"])
        self.assertEqual(result["source"].tolist(), ["Xquik", "Xquik"])
        self.assertEqual(result["label_name"].tolist(), ["positive", "negative"])

    def test_drops_blank_text_rows(self):
        result = normalize_text_dataframe(
            pd.DataFrame({"comment": [" Useful ", "", None], "label": [2, 1, 0]})
        )

        self.assertEqual(result["text"].tolist(), ["Useful"])
        self.assertEqual(result["label_name"].tolist(), ["positive"])

    def test_reports_available_columns_when_text_is_missing(self):
        with self.assertRaisesRegex(ValueError, "rating, created_at"):
            normalize_text_dataframe(
                pd.DataFrame({"rating": [5], "created_at": ["2025-01-01"]})
            )


if __name__ == "__main__":
    unittest.main()
