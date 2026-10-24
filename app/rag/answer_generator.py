import json
import logging
import re
from dataclasses import dataclass, field

from langchain_core.documents import Document

from app.llm.ollama_client import OllamaClient


logger = logging.getLogger(__name__)


@dataclass
class AnswerResult:
    answer: str
    sources: list[dict] = field(default_factory=list)
    citation_ids: list[str] = field(default_factory=list)


class GroundedAnswerGenerator:
    """Generate answers and verify every factual claim against evidence."""

    CITATION_PATTERN = re.compile(r"\[S(\d+)\]")
    SOURCE_BLOCK_PATTERN = re.compile(r"(?=^\[S\d+\]\n)", re.MULTILINE)

    # "According to the evidence," is framing, not a factual claim.
    PREAMBLE_PATTERN = re.compile(
        r"^\s*according to (?:the )?"
        r"(?:evidence|sources?|context|documents?)\s*[,:]?\s*",
        re.IGNORECASE,
    )
    SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+|\n+")
    LIST_MARKER_PATTERN = re.compile(r"^\s*(?:[-*\u2022]|\d+[.)])\s+")

    # Verdict shapes the verifier may return. Anything else fails closed.
    ALLOWED_VERDICT_KEYS = ({"supported"}, {"reason", "supported"})

    ABSTENTION = (
        "The available sources do not provide enough verified "
        "information to answer that question."
    )

    def __init__(self, llm=None, verifier=None) -> None:
        self.llm = llm or OllamaClient()
        self.verifier = verifier or OllamaClient(response_format="json")

    @staticmethod
    def _build_sources(
        documents: list[Document],
    ) -> tuple[list[dict], str]:
        """Build numbered evidence blocks and user-visible source metadata."""
        sources = []
        context_blocks = []

        for index, document in enumerate(documents, start=1):
            source_id = f"S{index}"
            metadata = document.metadata or {}

            source = {
                "id": source_id,
                "source": str(metadata.get("source", "Unknown source")),
                "page": metadata.get("page"),
                "content": document.page_content,
            }
            sources.append(source)

            location = source["source"]
            if source["page"] is not None:
                location += f", page {source['page']}"

            context_blocks.append(
                f"[{source_id}]\n"
                f"Source location: {location}\n"
                f"Evidence:\n{document.page_content}"
            )

        return sources, "\n\n".join(context_blocks)

    @classmethod
    def _context_for_citations(
        cls,
        context: str,
        citation_ids: list[str],
    ) -> str:
        """
        Select evidence blocks referenced by the answer's citations.

        Citation selection only narrows the evidence considered by the
        verifier. It does not establish that any claim is supported.
        """
        cited_ids = set(citation_ids)
        blocks = cls.SOURCE_BLOCK_PATTERN.split(context)
        selected_blocks = []

        for block in blocks:
            block = block.strip()
            if not block:
                continue

            first_line = block.splitlines()[0].strip()
            match = re.fullmatch(r"\[(S\d+)\]", first_line)

            if match and match.group(1) in cited_ids:
                selected_blocks.append(block)

        return "\n\n".join(selected_blocks)

    @classmethod
    def _split_claims(cls, answer: str) -> list[str]:
        """
        Split an answer into sentence-level claims.

        Citation-only fragments (e.g. "Hyderabad. [S1]") are re-attached to
        the preceding claim. Leading framing such as "According to the
        evidence," is removed. Limitation: abbreviations such as "Dr. X"
        can split a sentence; the resulting fragment is then verified on
        its own and, if unsupported, fails closed.
        """
        claims: list[str] = []

        for fragment in cls.SENTENCE_SPLIT_PATTERN.split(answer):
            fragment = cls.LIST_MARKER_PATTERN.sub("", fragment).strip()
            if not fragment:
                continue

            residue = cls.CITATION_PATTERN.sub("", fragment)
            if not re.search(r"\w", residue):
                if claims and cls.CITATION_PATTERN.search(fragment):
                    claims[-1] += " " + fragment
                continue

            claims.append(fragment)

        if claims:
            stripped = cls.PREAMBLE_PATTERN.sub("", claims[0]).strip()
            if re.search(r"\w", cls.CITATION_PATTERN.sub("", stripped)):
                claims[0] = stripped
            else:
                claims.pop(0)

        return claims

    def _claim_supported(self, claim: str, evidence: str) -> bool:
        """Ask the verifier about ONE claim. Fail closed on any problem."""
        prompt = (
            f"EVIDENCE:\n{evidence}\n\n"
            f"CLAIM: {claim}\n\n"
            "Is the claim supported by the evidence? Ignore citation "
            "markers like [S1]. Return ONLY JSON: "
            '{"reason": "<one short sentence>", "supported": true or false}'
        )

        try:
            raw = self.verifier.invoke(prompt)

            if not isinstance(raw, str):
                logger.warning(
                    "Grounding verification rejected a non-text response."
                )
                return False

            verdict = json.loads(raw.strip())

        except Exception:
            logger.exception("Grounding verification failed.")
            return False

        if (
            not isinstance(verdict, dict)
            or set(verdict.keys()) not in self.ALLOWED_VERDICT_KEYS
            or type(verdict["supported"]) is not bool
            or ("reason" in verdict and not isinstance(verdict["reason"], str))
        ):
            logger.warning(
                "Grounding verification rejected an invalid verdict."
            )
            return False

        if not verdict["supported"]:
            logger.warning(
                "Claim rejected by verifier: claim=%r reason=%r",
                claim[:200],
                str(verdict.get("reason", ""))[:200],
            )

        return verdict["supported"]

    def _verify_answer(self, answer: str, context: str) -> bool:
        """
        Approve an answer only if EVERY claim is accepted by the verifier.

        Each claim is checked separately against the evidence blocks it
        cites (or all supplied evidence if it carries no citation).
        """
        claims = self._split_claims(answer)

        if not claims:
            logger.warning("Grounding verification found no claims to check.")
            return False

        for claim in claims:
            cited_ids = list(
                dict.fromkeys(
                    f"S{match}"
                    for match in self.CITATION_PATTERN.findall(claim)
                )
            )
            evidence = (
                self._context_for_citations(context, cited_ids)
                if cited_ids
                else ""
            )

            if not self._claim_supported(claim, evidence or context):
                logger.info(
                    "Grounding verification completed: supported=False"
                )
                return False

        logger.info(
            "Grounding verification completed: supported=True, claims=%d",
            len(claims),
        )
        return True

    def generate(
        self,
        query: str,
        documents: list[Document],
    ) -> AnswerResult:
        """Generate a cited answer and verify it against cited evidence."""
        if not query.strip():
            raise ValueError("Query must not be empty.")

        if not documents:
            return AnswerResult(
                answer=(
                    "I don't have enough information in the indexed "
                    "sources to answer that question."
                )
            )

        sources, context = self._build_sources(documents)

        prompt = f"""
You are an enterprise knowledge assistant.

Answer the user's question using ONLY the evidence below.

Rules:
1. Use only facts explicitly supported by the evidence.
2. Do not add assumptions, outside knowledge, or unsupported details.
3. Cite every factual claim using the exact source IDs provided,
   such as [S1] or [S2].
4. Put citations immediately after the claims they support.
5. Never cite a filename or page number instead of a source ID.
6. Use only supplied source IDs. Never invent citation IDs.
7. If the evidence is insufficient, say so explicitly.
8. Be concise and precise.

SOURCE CONTEXT:
{context}

USER QUESTION:
{query}

GROUNDED ANSWER:
"""

        try:
            raw_answer = self.llm.invoke(prompt)
        except Exception:
            logger.exception("Answer generation failed.")
            return AnswerResult(
                answer=(
                    "I couldn't generate an answer from the available "
                    "evidence. Please try again."
                ),
                sources=sources,
            )

        if not isinstance(raw_answer, str) or not raw_answer.strip():
            return AnswerResult(
                answer=(
                    "I couldn't generate an answer from the available "
                    "evidence. Please try again."
                ),
                sources=sources,
            )

        answer = raw_answer.strip()
        valid_ids = {source["id"] for source in sources}

        def validate_citation(match: re.Match) -> str:
            citation_id = f"S{match.group(1)}"

            if citation_id in valid_ids:
                return match.group(0)

            logger.warning("Removed invalid citation: %s", citation_id)
            return ""

        answer = self.CITATION_PATTERN.sub(validate_citation, answer)

        citation_ids = list(
            dict.fromkeys(
                f"S{match}"
                for match in self.CITATION_PATTERN.findall(answer)
            )
        )

        if not citation_ids:
            logger.warning(
                "RAG rejected answer because it contains no valid citations."
            )
            return AnswerResult(
                answer=self.ABSTENTION,
                sources=sources,
            )

        verification_context = self._context_for_citations(
            context,
            citation_ids,
        )

        if not verification_context:
            logger.warning(
                "RAG rejected answer because cited evidence was not found."
            )
            return AnswerResult(
                answer=self.ABSTENTION,
                sources=sources,
            )

        if not self._verify_answer(answer, verification_context):
            logger.warning(
                "RAG rejected answer because grounding verification failed."
            )
            return AnswerResult(
                answer=self.ABSTENTION,
                sources=sources,
            )

        return AnswerResult(
            answer=answer,
            sources=sources,
            citation_ids=citation_ids,
        )