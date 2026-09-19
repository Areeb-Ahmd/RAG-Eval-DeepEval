"""
src/generator.py — the GENERATOR component.

Given a query and context (retrieved chunks), produce an answer that is
grounded in the context. The prompt is faithfulness-first: answer ONLY from
the context, and abstain when the context doesn't contain the answer.

    from src.generator import generate
    answer = generate("what is drift?", ["chunk text 1", "chunk text 2"])
"""

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0)

# faithfulness-first prompt: ground every claim in the context, abstain if unsure
prompt = ChatPromptTemplate.from_template(
    """You are a helpful teaching assistant for a course on LLM evaluations.
Answer the student's question using ONLY the context provided below.

Rules:
- Use ONLY information present in the context. Do not add outside knowledge.
- Do not strengthen or overstate claims. If the context says two things are
  "different," do not upgrade that to "separate methods" or stronger wording.
- The context is an informal lecture transcript. Synthesize and rephrase what
  IS there — do not require the question's exact wording to appear.
- Only abstain if the context contains NOTHING relevant to the question. If it
  contains partial information, answer with what is present.
- If you must abstain, say exactly:
  "I don't have enough information in the course material to answer that."
- Answer the question DIRECTLY in the first sentence. Keep supporting detail
  tight — include at most one or two examples, and only if they serve the
  question being asked.

Context:
{context}

Question: {question}

Answer:"""
)


chain = prompt | llm | StrOutputParser()


def generate(query: str, context: list[str]) -> str:
    """Generate a grounded answer from the query and context chunks."""
    context_text = "\n\n".join(context)
    return chain.invoke({"question": query, "context": context_text})


# quick manual test: python src/generator.py
if __name__ == "__main__":
    ctx = [
        "Online eval means evaluating your system on live production traffic "
        "after deployment. It works without an answer key, unlike offline eval."
    ]
    print(generate("what is online eval?", ctx))