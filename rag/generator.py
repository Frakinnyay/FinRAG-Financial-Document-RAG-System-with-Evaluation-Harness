import sys
import time
from pathlib import Path

from groq import Groq

sys.path.append(str(Path(__file__).parent.parent))
from config import GROQ_API_KEY, GROQ_MODEL

SYSTEM_PROMPT = """You are a financial analyst assistant. You answer questions \
strictly using the provided context excerpts from company annual reports/filings.

Rules:
1. Only use information present in the context. Never use outside knowledge.
2. If the context does not contain enough information to answer, say so clearly \
instead of guessing.
3. For every factual claim or number you state, cite the source in the format \
[Company, FY<year>, p.<page>] immediately after the claim.
4. When numbers come from a table, quote them exactly as they appear.
5. Be concise and precise. Do not pad with generic commentary.
"""

USER_TEMPLATE = """Context excerpts:
{context}

Question: {question}

Answer the question using only the context above, with inline citations \
[Company, FY<year>, p.<page>] after each factual claim."""


def _format_context(chunks):
    blocks = []
    for c in chunks:
        tag = f"[{c['company']}, FY{c['fiscal_year']}, p.{c['page_number']}, {c['chunk_type']}]"
        blocks.append(f"{tag}\n{c['text']}")
    return "\n\n---\n\n".join(blocks)


class Generator:
    def __init__(self):
        if not GROQ_API_KEY:
            raise RuntimeError(
                "GROQ_API_KEY not set. Copy .env.example to .env and add your key "
                "from https://console.groq.com/keys"
            )
        self.client = Groq(api_key=GROQ_API_KEY)

    def generate(self, question: str, retrieved_chunks: list):
        """Returns (answer_text, generation_latency_ms)."""
        context = _format_context(retrieved_chunks)
        user_msg = USER_TEMPLATE.format(context=context, question=question)

        t0 = time.perf_counter()
        response = self.client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.1,
            max_tokens=800,
        )
        latency_ms = (time.perf_counter() - t0) * 1000

        answer = response.choices[0].message.content
        return answer, latency_ms
