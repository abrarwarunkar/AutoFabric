"""
AutoFabric Governance — PII Detector
Uses Microsoft Presidio to detect and redact sensitive entities before
any LLM call, ensuring privacy-safe operation.

Detected entities: PERSON, EMAIL_ADDRESS, PHONE_NUMBER, CREDIT_CARD, US_SSN
"""
import logging
from typing import Dict

logger = logging.getLogger(__name__)

# Presidio entities to scan for
PII_ENTITIES = [
    "PERSON",
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "CREDIT_CARD",
    "US_SSN",
]


def _build_engines():
    """Build Presidio analyzer + anonymizer. Lazy to avoid slow import at module load."""
    try:
        from presidio_analyzer import AnalyzerEngine
        from presidio_anonymizer import AnonymizerEngine
        analyzer = AnalyzerEngine()
        anonymizer = AnonymizerEngine()
        return analyzer, anonymizer
    except Exception as e:
        logger.error(f"Presidio engines failed to initialize: {e}")
        return None, None


_analyzer = None
_anonymizer = None


def _get_engines():
    global _analyzer, _anonymizer
    if _analyzer is None:
        _analyzer, _anonymizer = _build_engines()
    return _analyzer, _anonymizer


def detect_and_redact(text: str) -> Dict:
    """
    Detect PII entities in text and redact them with [REDACTED].

    Returns:
        {
            detected: bool,
            entities: [{"type": str, "start": int, "end": int, "score": float}],
            redacted_text: str,
        }
    """
    analyzer, anonymizer = _get_engines()

    if analyzer is None:
        logger.warning("Presidio not available — skipping PII detection")
        return {
            "detected": False,
            "entities": [],
            "redacted_text": text,
        }

    try:
        results = analyzer.analyze(
            text=text,
            entities=PII_ENTITIES,
            language="en",
        )

        if not results:
            return {
                "detected": False,
                "entities": [],
                "redacted_text": text,
            }

        # Anonymize detected entities
        anonymized = anonymizer.anonymize(
            text=text,
            analyzer_results=results,
        )

        entities_list = [
            {
                "type": r.entity_type,
                "start": r.start,
                "end": r.end,
                "score": round(r.score, 3),
            }
            for r in results
        ]

        logger.info(f"PII detected: {[e['type'] for e in entities_list]}")

        return {
            "detected": True,
            "entities": entities_list,
            "redacted_text": anonymized.text,
        }

    except Exception as e:
        logger.error(f"PII detection error: {e}")
        return {
            "detected": False,
            "entities": [],
            "redacted_text": text,
        }
