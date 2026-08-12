"""
Gradio demo app for FinRAG. Deployable directly to Hugging Face Spaces
(free CPU tier) — just push this repo to a Space with SDK = Gradio.

Locally:
    python app.py
"""
import gradio as gr

from rag.pipeline import FinRAGPipeline

print("Loading FinRAG pipeline (index + embedding model)... this happens once.")
pipeline = FinRAGPipeline()
print("Ready.")

COMPANIES = ["All", "TCS", "RELIANCE", "INFOSYS", "TATAMOTORS", "HDFC"]


def format_contexts(contexts):
    if not contexts:
        return "_No context retrieved._"
    blocks = []
    for i, c in enumerate(contexts, 1):
        blocks.append(
            f"**[{i}] {c['company']} · FY{c['fiscal_year']} · p.{c['page_number']} "
            f"· {c['chunk_type']} · score={c['score']:.3f}**\n\n"
            f"```\n{c['text'][:600]}{'...' if len(c['text']) > 600 else ''}\n```"
        )
    return "\n\n---\n\n".join(blocks)


def ask(question, company):
    if not question.strip():
        return "Please enter a question.", "", ""

    company_filter = None if company == "All" else company
    result = pipeline.answer(question, company_filter=company_filter)

    latency_note = (
        f"Retrieval: {result['retrieval_latency_ms']:.0f} ms  |  "
        f"Generation (Groq): {result['generation_latency_ms']:.0f} ms"
    )
    return result["answer"], format_contexts(result["contexts"]), latency_note


with gr.Blocks(title="FinRAG — Financial Document Q&A") as demo:
    gr.Markdown(
        "# 📊 FinRAG — Annual Report Q&A with Citations\n"
        "Ask analyst-style questions over indexed annual reports "
        "(TCS, Reliance, Infosys, Tata Motors, HDFC Bank). "
        "Answers are grounded in retrieved excerpts and cite "
        "`[Company, FY, page]` for every claim."
    )
    with gr.Row():
        with gr.Column(scale=3):
            question = gr.Textbox(
                label="Your question",
                placeholder="e.g. What was TCS's net profit in the most recent fiscal year?",
                lines=2,
            )
        with gr.Column(scale=1):
            company = gr.Dropdown(COMPANIES, value="All", label="Filter by company")
    submit = gr.Button("Ask", variant="primary")

    answer_box = gr.Markdown(label="Answer")
    latency_box = gr.Markdown()
    with gr.Accordion("Retrieved context (click to expand)", open=False):
        context_box = gr.Markdown()

    submit.click(ask, inputs=[question, company], outputs=[answer_box, context_box, latency_box])
    question.submit(ask, inputs=[question, company], outputs=[answer_box, context_box, latency_box])

    gr.Examples(
        examples=[
            ["What was TCS's total revenue for the most recent fiscal year?", "TCS"],
            ["What are Reliance Industries' main business segments?", "RELIANCE"],
            ["What was Infosys's attrition rate for the year?", "INFOSYS"],
            ["What was Tata Motors' consolidated net profit or loss?", "TATAMOTORS"],
            ["What was HDFC Bank's gross NPA ratio?", "HDFC"],
        ],
        inputs=[question, company],
    )

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)