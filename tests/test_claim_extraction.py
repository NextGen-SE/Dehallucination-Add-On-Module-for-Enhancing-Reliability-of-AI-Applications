
import unittest

from claim_extraction import (
    InputTooLongError,
    count_words,
    extract_claims,
)


class TestClaimExtraction(unittest.TestCase):

    def test_extracts_simple_factual_sentence(self):
        result = extract_claims("The Earth orbits the Sun.")

        self.assertEqual(result.status, "ok")
        self.assertEqual(len(result.claims), 1)
        self.assertEqual(
            result.claims[0].text,
            "The Earth orbits the Sun.",
        )

    def test_splits_two_atomic_claims(self):
        text = (
            "Bengaluru is in Karnataka, "
            "and Karnataka is in India."
        )

        result = extract_claims(text)

        self.assertEqual(result.status, "ok")
        self.assertEqual(len(result.claims), 2)
        self.assertEqual(
            [claim.text for claim in result.claims],
            [
                "Bengaluru is in Karnataka",
                "Karnataka is in India.",
            ],
        )

    def test_preserves_source_offsets(self):
        text = (
            "Bengaluru is in Karnataka, "
            "and Karnataka is in India."
        )

        result = extract_claims(text)

        for claim in result.claims:
            self.assertEqual(
                text[claim.start_char:claim.end_char],
                claim.text,
            )
            self.assertEqual(
                text[
                    claim.source_sentence_start:
                    claim.source_sentence_end
                ],
                claim.source_sentence,
            )

    def test_preserves_original_sentence_for_split_claims(self):
        text = (
            "Bengaluru is in Karnataka, "
            "and Karnataka is in India."
        )

        result = extract_claims(text)

        self.assertTrue(
            all(
                claim.source_sentence == text
                for claim in result.claims
            )
        )

    def test_extracts_multiple_sentences(self):
        text = (
            "The Earth orbits the Sun. "
            "Water contains hydrogen and oxygen."
        )

        result = extract_claims(text)

        self.assertEqual(result.status, "ok")
        self.assertGreaterEqual(len(result.claims), 2)

    def test_filters_questions(self):
        result = extract_claims(
            "Is the Moon made of cheese?"
        )

        self.assertEqual(
            result.status,
            "no_verifiable_claims",
        )
        self.assertEqual(result.claims, [])

    def test_filters_opinion_statements(self):
        result = extract_claims(
            "I think this movie is wonderful."
        )

        self.assertEqual(
            result.status,
            "no_verifiable_claims",
        )

    def test_empty_input(self):
        result = extract_claims("   ")

        self.assertEqual(
            result.status,
            "no_verifiable_claims",
        )
        self.assertEqual(result.claims, [])
        self.assertEqual(
            result.message,
            "No verifiable claims found",
        )

    def test_rejects_oversized_input(self):
        text = "word " * 2001

        with self.assertRaises(InputTooLongError):
            extract_claims(text)

    def test_rejects_non_string_input(self):
        with self.assertRaises(TypeError):
            extract_claims(None)

    def test_rejects_invalid_max_words(self):
        with self.assertRaises(TypeError):
            extract_claims("The Earth orbits the Sun.", 0)

    def test_counts_words(self):
        self.assertEqual(
            count_words("The Earth orbits the Sun."),
            5,
        )


if __name__ == "__main__":
    unittest.main()
