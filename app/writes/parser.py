"""
Parse purchase sentences such as "Uday bought iPhone for 80k".

This is plain code, not an LLM: the amount, date and names are extracted with
fixed rules so the result is predictable and testable.
"""
import re
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal, InvalidOperation

MAX_AMOUNT = Decimal("1000000000000")
MAX_PRODUCT_LENGTH = 100
MAX_NAME_WORDS = 4

EXAMPLE = "Uday bought iPhone for 80k"

VERBS = r"(?:bought|purchased|ordered)"

# Words that start a question or a command, never a purchase statement.
NON_STATEMENT_STARTS = frozenset(
    {
        "who", "what", "when", "where", "which", "why", "how", "did", "does",
        "do", "is", "are", "was", "were", "can", "could", "would", "should",
        "has", "have", "will", "show", "list", "tell", "give", "find", "count",
        "get", "display", "print",
    }
)

PRONOUNS = frozenset(
    {"i", "he", "she", "they", "we", "you", "it", "someone", "somebody", "who"}
)

MULTIPLIERS = {
    "k": Decimal(1_000),
    "thousand": Decimal(1_000),
    "l": Decimal(100_000),
    "lac": Decimal(100_000),
    "lacs": Decimal(100_000),
    "lakh": Decimal(100_000),
    "lakhs": Decimal(100_000),
    "mn": Decimal(1_000_000),
    "million": Decimal(1_000_000),
    "cr": Decimal(10_000_000),
    "crore": Decimal(10_000_000),
    "crores": Decimal(10_000_000),
}

_LETTER = r"[^\W\d_]"
_NAME = rf"(?P<name>{_LETTER}(?:{_LETTER}|[.'\- ]){{0,58}}?)"

_AMOUNT = (
    r"(?:(?:rs\.?|inr|usd|rupees?)\s*|[\u20b9$])?"
    r"(?P<number>\d[\d,]*(?:\.\d+)?)"
    r"\s*(?P<unit>k|thousand|lakhs?|lacs?|l|crores?|cr|million|mn)?(?!\w)"
    r"\s*(?:rupees|rs\.?|inr)?"
)

_FULL = re.compile(
    rf"^\s*{_NAME}\s+{VERBS}\s+(?P<product>.+?)\s+"
    rf"(?:for|at|worth|costing)\s+{_AMOUNT}(?P<tail>.*)$",
    re.IGNORECASE | re.DOTALL,
)

_PARTIAL = re.compile(
    rf"^\s*{_NAME}\s+{VERBS}\s+(?P<product>.+)$",
    re.IGNORECASE | re.DOTALL,
)

_AMOUNT_ONLY = re.compile(rf"^\s*{_AMOUNT}\s*$", re.IGNORECASE)
_TRAILING_CONNECTOR = re.compile(r"\s+(?:for|at|worth|costing)\s*$", re.IGNORECASE)
_CONNECTOR_THEN_DIGIT = re.compile(
    r"\b(?:for|at|worth|costing)\s+\D{0,3}\d", re.IGNORECASE
)
_ISO_DATE = re.compile(r"\b(?:on\s+)?(\d{4})-(\d{1,2})-(\d{1,2})\b", re.I)
_DMY_DATE = re.compile(r"\b(?:on\s+)?(\d{1,2})[/-](\d{1,2})[/-](\d{4})\b", re.I)
_RELATIVE_DATE = re.compile(r"\b(today|yesterday)\b", re.I)
_LEADING_ARTICLE = re.compile(r"^(?:a|an|the|some)\s+", re.I)


class WriteRequestError(ValueError):
    """A recognized data-entry request that cannot be accepted as written."""


@dataclass(frozen=True)
class PartialPurchase:
    """A purchase statement that is complete except for the amount."""

    customer_name: str
    product: str
    order_date: date


class MissingAmountError(WriteRequestError):
    """The sentence names a customer and product but gives no amount."""

    def __init__(self, message: str, partial: PartialPurchase):
        super().__init__(message)
        self.partial = partial


@dataclass(frozen=True)
class PurchaseRequest:
    customer_name: str
    product: str
    amount: Decimal
    order_date: date
    ignored_text: str = ""


def _is_statement(text: str) -> bool:
    stripped = text.strip()

    if not stripped or stripped.endswith("?"):
        return False

    first_word = re.split(r"\s+", stripped, maxsplit=1)[0].casefold()

    return first_word.strip(".,:;!'\"") not in NON_STATEMENT_STARTS


def looks_like_purchase(text: str) -> bool:
    """True if the text is a 'NAME bought PRODUCT ...' statement."""
    return _is_statement(text) and _PARTIAL.match(text) is not None


def _tidy_name(name: str) -> str:
    name = " ".join(name.split()).strip(" .-'")

    if name.islower() or name.isupper():
        name = name.title()

    return name


def _parse_date(tail: str, today: date) -> tuple[date, str]:
    """Return (order date, leftover text) from the text after the amount."""
    order_date = today
    leftover = tail

    relative = _RELATIVE_DATE.search(leftover)
    if relative:
        if relative.group(1).casefold() == "yesterday":
            order_date = today - timedelta(days=1)
        leftover = leftover[: relative.start()] + leftover[relative.end():]

    iso = _ISO_DATE.search(leftover)
    dmy = None if iso else _DMY_DATE.search(leftover)

    try:
        if iso:
            year, month, day = (int(part) for part in iso.groups())
            order_date = date(year, month, day)
            leftover = leftover[: iso.start()] + leftover[iso.end():]
        elif dmy:
            day, month, year = (int(part) for part in dmy.groups())
            order_date = date(year, month, day)
            leftover = leftover[: dmy.start()] + leftover[dmy.end():]
    except ValueError:
        raise WriteRequestError(
            "I couldn't read that date. Use a real date such as 2026-10-25."
        )

    if order_date > today:
        raise WriteRequestError(
            "Order dates in the future aren't accepted. "
            "Use today's date or an earlier one."
        )

    leftover = re.sub(r"\b(?:please|thanks|thank you)\b", " ", leftover, flags=re.I)
    leftover = re.sub(r"[\s.,!;]+", " ", leftover).strip()

    return order_date, leftover


def _parse_amount(number: str, unit: str | None) -> Decimal:
    try:
        value = Decimal(number.replace(",", ""))
    except InvalidOperation:
        raise WriteRequestError("I couldn't read that amount.")

    if unit:
        value *= MULTIPLIERS[unit.casefold()]

    value = value.quantize(Decimal("0.01"))

    if value <= 0:
        raise WriteRequestError("The amount must be greater than zero.")
    if value > MAX_AMOUNT:
        raise WriteRequestError("That amount is too large to accept.")

    return value


def _check_name(raw_name: str) -> str:
    name = _tidy_name(raw_name)
    words = name.split()

    if not words or len(words) > MAX_NAME_WORDS or name.casefold() in PRONOUNS:
        raise WriteRequestError(
            "Please use the customer's name, for example: " + EXAMPLE
        )

    return name


def _check_product(raw_product: str) -> str:
    product = _LEADING_ARTICLE.sub("", " ".join(raw_product.split()))
    product = product.strip(" .,;:!")

    if not product:
        raise WriteRequestError(
            "I couldn't find the product. Try: " + EXAMPLE
        )
    if len(product) > MAX_PRODUCT_LENGTH:
        raise WriteRequestError(
            f"The product name is too long (limit {MAX_PRODUCT_LENGTH} characters)."
        )

    return product


def looks_like_amount(text: str) -> bool:
    """True if the whole message is just an amount such as '80k'."""
    return _AMOUNT_ONLY.match(text) is not None


def parse_amount_only(text: str) -> Decimal:
    """Parse a message that is only an amount; raises WriteRequestError."""
    match = _AMOUNT_ONLY.match(text)

    if match is None:
        raise WriteRequestError("I couldn't read that amount. Try: 80k")

    return _parse_amount(match.group("number"), match.group("unit"))


def parse_purchase(text: str, today: date | None = None) -> PurchaseRequest:
    """
    Parse a purchase statement.

    Raises WriteRequestError with a user-friendly message when the text is a
    purchase statement but is missing something or has an invalid value.
    """
    today = today or date.today()

    if not _is_statement(text):
        raise WriteRequestError("That doesn't look like a purchase statement.")

    match = _FULL.match(text)

    if match is None:
        partial = _PARTIAL.match(text)

        if partial is None:
            raise WriteRequestError(
                "That doesn't look like a purchase statement."
            )

        name = _check_name(partial.group("name"))
        raw_product = partial.group("product")
        message = "I need an amount to save an order. Try: " + EXAMPLE

        # "rice for 80kg" has an amount-like word that could not be read;
        # that is a bad amount, not a missing one.
        if _CONNECTOR_THEN_DIGIT.search(raw_product):
            raise WriteRequestError(message)

        product = _check_product(_TRAILING_CONNECTOR.sub("", raw_product))

        raise MissingAmountError(message, PartialPurchase(name, product, today))

    name = _check_name(match.group("name"))
    product = _check_product(match.group("product"))
    amount = _parse_amount(match.group("number"), match.group("unit"))
    order_date, leftover = _parse_date(match.group("tail"), today)

    return PurchaseRequest(
        customer_name=name,
        product=product,
        amount=amount,
        order_date=order_date,
        ignored_text=leftover,
    )