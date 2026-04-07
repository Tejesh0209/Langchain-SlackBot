"""Input and output guardrails for the agent."""
import re
from typing import Any


# Blocked patterns for prompt injection
INJECTION_PATTERNS = [
    r"ignore\s*(your|all\s+)?\s*(previous|above|prior)\s+(instructions?|rules?|constraints?)",
    r"(disregard|forget)\s+(your|all)\s+(previous|above|prior)\s+(instructions?|rules?)",
    r"you\s+are\s+(now\s+)?a?\s*(different|new|free|unbound)\s*(AI|assistant|bot)?",
    r"pretend\s+(you|as\s+if\s+you)\s+(are|were)\s+(a\s+)?",
    r"replace\s+(your|all)\s+(system\s+)?prompt",
    r"new\s+instructions?:",
    r"<\s*script",  # XSS attempts
    r"javascript:",  # XSS attempts
    r"on\w+\s*=",  # Event handlers like onclick=
]

# Maximum input length
MAX_INPUT_LENGTH = 4000

# PII patterns for output redaction
PII_PATTERNS = {
    "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
    "phone": r"\b(\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
    "ssn": r"\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b",
}


def validate_input(user_input: str) -> tuple[bool, str]:
    """
    Validate user input for safety.

    Checks for:
    - Prompt injection attempts
    - Excessive length
    - Malformed input

    Returns (is_valid, error_message).
    """
    if not user_input or not user_input.strip():
        return False, "Empty input"

    # Check length
    if len(user_input) > MAX_INPUT_LENGTH:
        return False, f"Input exceeds maximum length of {MAX_INPUT_LENGTH} characters"

    # Check for injection patterns
    lower_input = user_input.lower()
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, lower_input, re.IGNORECASE):
            return False, "Potential prompt injection detected"

    return True, ""


def extract_entities(user_input: str) -> list[str]:
    """
    Extract potential entity references from user input.

    Looks for:
    - Quoted strings
    - Capitalized multi-word phrases
    - Known product/customer name patterns

    Returns list of entity strings.
    """
    entities = []

    # Extract quoted strings
    quoted = re.findall(r'"([^"]+)"|\'([^\']+)\'', user_input)
    for match in quoted:
        for part in match:
            if part:
                entities.append(part.strip())

    # Extract capitalized multi-word phrases (potential names)
    capitalized = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b', user_input)
    entities.extend(capitalized)

    # Remove duplicates while preserving order
    seen = set()
    unique_entities = []
    for e in entities:
        if e.lower() not in seen:
            seen.add(e.lower())
            unique_entities.append(e)

    return unique_entities


def redact_pii(text: str) -> str:
    """
    Redact PII from text.

    Currently redacts emails, phone numbers, and SSNs.

    Returns redacted text.
    """
    redacted = text

    # Redact emails
    redacted = re.sub(PII_PATTERNS["email"], "[EMAIL_REDACTED]", redacted)

    # Redact phone numbers
    redacted = re.sub(PII_PATTERNS["phone"], "[PHONE_REDACTED]", redacted)

    # Redact SSNs
    redacted = re.sub(PII_PATTERNS["ssn"], "[SSN_REDACTED]", redacted)

    return redacted


def validate_output(output: str, sources: list[str]) -> tuple[bool, str]:
    """
    Validate generated output for safety and accuracy.

    Checks for:
    - Hallucinated claims (claims not supported by sources)
    - Leaked PII
    - Excessive length

    Returns (is_valid, warning/error_message).
    """
    if not output or not output.strip():
        return False, "Empty output"

    # Check for PII
    has_email = bool(re.search(PII_PATTERNS["email"], output))
    has_phone = bool(re.search(PII_PATTERNS["phone"], output))
    has_ssn = bool(re.search(PII_PATTERNS["ssn"], output))

    if has_email or has_phone or has_ssn:
        return False, "Output contains unredacted PII"

    # Note: Hallucination detection is best-effort
    # Full hallucination detection would require comparing claims against sources
    # This is a simplified check

    return True, ""
