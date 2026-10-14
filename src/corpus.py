from dataclasses import dataclass


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    text: str
    source_id: str | None = None
    section: str | None = None


def load_default_corpus() -> list[Document]:
    """Return a small default corpus for local RAG practice.

    Students can replace or extend this corpus with course-approved sources.
    """
    return [
        Document(
            doc_id="doc-1",
            title="RAG Fundamentals",
            text=(
                "Retrieval augmented generation combines retrieval with text generation. "
                "Before answering, the retriever searches an indexed knowledge base and selects passages "
                "that are relevant to the user's question. The generator then uses those passages as its "
                "context. This design lets an application use current or domain-specific information without "
                "retraining the language model. The model should answer using retrieved evidence rather than "
                "unsupported assumptions, and each important claim should be traceable to a source."
            ),
        ),
        Document(
            doc_id="doc-2",
            title="Chunking and Metadata",
            text=(
                "Chunking splits long documents into smaller passages that can be searched independently. "
                "A chunk that is too small may separate a claim from its explanation, while a chunk that is "
                "too large may mix unrelated topics and dilute similarity. Overlap repeats a small amount of "
                "text at adjacent boundaries so a sentence is less likely to lose its context. Good chunk size "
                "and overlap therefore improve retrieval quality. Metadata such as document ID, title, and "
                "section supports filtering, citations, and source traceability."
            ),
        ),
        Document(
            doc_id="doc-3",
            title="Refusal Behavior",
            text=(
                "A grounded assistant must compare the question with the evidence returned by retrieval. If "
                "the retrieved context is empty, irrelevant, or does not support a requested claim, the system "
                "should refuse and ask for better evidence. A consistent refusal explains that the available "
                "corpus is insufficient instead of inventing an answer from general model knowledge. This "
                "reduces hallucinations and increases trust because users can distinguish supported answers "
                "from missing information."
            ),
        ),
        Document(
            doc_id="doc-4",
            title="Retrieval Evaluation",
            text=(
                "Retrieval evaluation starts with questions whose relevant source documents are known. Hit rate "
                "checks whether a relevant result appears in the returned set. Mean reciprocal rank gives more "
                "credit when the first relevant result appears near the top. A top-k pipeline must return no more "
                "than k chunks and order them from highest to lowest similarity. Comparing several representative "
                "queries is more reliable than selecting settings from a single example."
            ),
        ),
        Document(
            doc_id="doc-5",
            title="Citation Quality",
            text=(
                "Citations connect generated statements to retrieved chunks. Citation identifiers should match "
                "the chunk IDs supplied to the answer generator, and the cited text should actually support the "
                "claim. Clear references let a reader inspect the evidence and find the original document. A "
                "citation is not sufficient by itself: correctness and completeness must also be checked."
            ),
        ),
    ]


def load_training_rag_corpus() -> list[Document]:
    """Return a corpus profile inspired by training_rag notebook examples.

    This profile emphasizes citations, source metadata, and evaluation framing.
    """
    return [
        Document(
            doc_id="tr-doc-1",
            title="RAG with Citations",
            text=(
                "RAG with citations should retrieve relevant documents with source metadata and include references in the answer. "
                "Citations improve transparency and trust because users can verify evidence."
            ),
        ),
        Document(
            doc_id="tr-doc-2",
            title="Structured Citation Output",
            text=(
                "A cited answer can include answer text and list of source titles. "
                "A common format is an answer body followed by a References section."
            ),
        ),
        Document(
            doc_id="tr-doc-3",
            title="LLM-as-a-Judge Evaluation",
            text=(
                "LLM-as-a-judge can evaluate answer quality, citation correctness, and grounding behavior at scale. "
                "It is useful but introduces bias, so metrics should be interpreted carefully."
            ),
        ),
        Document(
            doc_id="tr-doc-4",
            title="Refusal and Grounding",
            text=(
                "When retrieved evidence is missing or irrelevant, the system should refuse unsupported claims. "
                "This behavior reduces hallucinations and aligns with safe RAG practice."
            ),
        ),
    ]


def list_corpus_profiles() -> list[str]:
    return ["default", "training-rag"]


def load_corpus(profile: str = "default") -> list[Document]:
    if profile == "default":
        return load_default_corpus()
    if profile == "training-rag":
        return load_training_rag_corpus()
    raise ValueError(f"Unknown corpus profile: {profile}")
