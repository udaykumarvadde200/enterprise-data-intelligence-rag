from langchain_core.documents import Document


class CrossEncoderReranker:
    """Rerank documents using a local cross-encoder classification model."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        model=None,
    ) -> None:
        self.model_name = model_name
        self.model = model
        self.tokenizer = None
        self.torch = None
        self.device = None

        # An injected fake model keeps unit tests independent of downloads.
        if model is None:
            try:
                import torch
                from transformers import (
                    AutoModelForSequenceClassification,
                    AutoTokenizer,
                )
            except ImportError as exc:
                raise RuntimeError(
                    "Install torch and transformers to use cross-encoder reranking."
                ) from exc

            self.torch = torch
            self.device = (
                "cuda" if torch.cuda.is_available() else "cpu"
            )

            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name
            )
            self.model.to(self.device)
            self.model.eval()

    def _predict_scores(
        self,
        query: str,
        documents: list[Document],
    ) -> list[float]:
        pairs = [
            (query, document.page_content)
            for document in documents
        ]

        # Supports injected fake models used in unit tests.
        if self.tokenizer is None:
            return [
                float(score)
                for score in self.model.predict(pairs)
            ]

        scores = []

        for start in range(0, len(pairs), 8):
            batch = pairs[start:start + 8]
            encoded = self.tokenizer(
                [pair[0] for pair in batch],
                [pair[1] for pair in batch],
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            encoded = {
                key: value.to(self.device)
                for key, value in encoded.items()
            }

            with self.torch.inference_mode():
                output = self.model(**encoded)
                logits = output.logits.squeeze(-1)

            scores.extend(
                logits.detach().float().cpu().tolist()
            )

        return scores

    def rerank(
        self,
        query: str,
        documents: list[Document],
        top_k: int = 5,
    ) -> list[Document]:
        if not query.strip():
            raise ValueError("Query must not be empty.")
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")
        if not documents:
            return []

        scores = self._predict_scores(query, documents)

        if len(scores) != len(documents):
            raise ValueError(
                "The reranker returned an unexpected number of scores."
            )

        ranked = sorted(
            zip(documents, scores),
            key=lambda item: item[1],
            reverse=True,
        )

        results = []

        for document, score in ranked[:top_k]:
            metadata = dict(document.metadata)
            metadata["rerank_score"] = float(score)

            results.append(
                Document(
                    page_content=document.page_content,
                    metadata=metadata,
                )
            )

        return results