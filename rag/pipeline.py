import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from rag.retriever import Retriever
from rag.generator import Generator
from config import TOP_K


class FinRAGPipeline:
    """
    Lazily initializes retriever/generator once (they load models / open
    the index), then answer() can be called cheaply many times.
    """
    def __init__(self):
        self.retriever = Retriever()
        self.generator = Generator()

    def answer(self, question: str, top_k: int = TOP_K, company_filter: str = None):
        chunks, retrieval_ms = self.retriever.retrieve(
            question, top_k=top_k, company_filter=company_filter
        )
        if not chunks:
            return {
                "answer": "No relevant context was found in the indexed documents "
                          "for this question.",
                "contexts": [],
                "retrieval_latency_ms": retrieval_ms,
                "generation_latency_ms": 0.0,
            }

        answer_text, generation_ms = self.generator.generate(question, chunks)

        return {
            "answer": answer_text,
            "contexts": chunks,
            "retrieval_latency_ms": retrieval_ms,
            "generation_latency_ms": generation_ms,
        }


if __name__ == "__main__":
    pipeline = FinRAGPipeline()
    q = "What was the company's total revenue in the most recent fiscal year?"
    result = pipeline.answer(q)
    print("ANSWER:\n", result["answer"])
    print(f"\nRetrieval latency: {result['retrieval_latency_ms']:.1f} ms")
    print(f"Generation latency: {result['generation_latency_ms']:.1f} ms")
