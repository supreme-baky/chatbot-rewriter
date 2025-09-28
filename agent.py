# agent.py
import os
import re
import torch
from typing import Optional, List

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from functions import load_vectorstore  # expects your functions.py to provide this

# -------------------------
# Configuration
# -------------------------
# Model: set to a small offline-capable model; change to local path if you downloaded it manually
MODEL_ID = os.getenv("MODEL_ID", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")

# Force transformers offline behavior (if you cached model)
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["BITSANDBYTES_NOWELCOME"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

# -------------------------
# Load model & tokenizer (CPU-safe)
# -------------------------
# Note: this will try to load from cache. Make sure you've run download script or cached model offline.
device_map = "cpu"
torch_dtype = torch.float32

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map=device_map,
    torch_dtype=torch_dtype,
    low_cpu_mem_usage=True,
)

gen_pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    temperature=0.25,     # medium-strength conservative creativity
    top_p=0.9,
    do_sample=True,
    repetition_penalty=1.05,
)

# -------------------------
# Retriever (Chroma)
# -------------------------
_vectordb = None
_retriever = None
try:
    _vectordb = load_vectorstore("./chroma_db")
    _retriever = _vectordb.as_retriever(search_kwargs={"k": 4})
except Exception:
    # If chroma isn't available, RAG will be disabled gracefully
    _vectordb = None
    _retriever = None

# -------------------------
# Helpers
# -------------------------
REWRITE_WORD_THRESHOLD = 50  # trigger rewrite at >= 50 words

RAG_KEYWORDS = [
    r"\baccording to\b",
    r"\bstrunk\b",
    r"\bwhite\b",
    r"\bwilliams\b",
    r"\bbizup\b",
    r"\bstyle\b",
    r"\blesson\b",
    r"\bchapter\b",
    r"\bsection\b",
    r"\bclarity\b",
    r"\bgrace\b",
    r"\bcoherence\b",
]
_rag_regex = re.compile("|".join(RAG_KEYWORDS), flags=re.IGNORECASE)


def _should_rewrite(text: str) -> bool:
    words = len(text.split())
    return words >= REWRITE_WORD_THRESHOLD


def _auto_detect_tone(text: str) -> str:
    t = text.lower()
    # very simple heuristics for tone detection
    if any(w in t for w in ["please", "thanks", "thank you", "kindly"]):
        return "friendly"
    if any(w in t for w in ["dear", "sir", "madam", "sincerely"]):
        return "formal"
    if any(w in t for w in ["research", "study", "analysis", "therefore", "however"]):
        return "academic"
    return "professional"


def _fetch_context(question: str) -> str:
    if _retriever is None:
        return ""
    docs = _retriever.get_relevant_documents(question)
    return "\n\n".join(d.page_content for d in docs)


def _preserve_paragraphs_reconstruct(original_text: str, rewritten_text: str) -> str:
    """
    We ask model to preserve paragraphs, but sometimes pipeline returns a single block.
    Keep simple: if original has multiple paragraphs, try to split rewritten by double-newline.
    If counts don't match, return rewritten as-is.
    """
    orig_pars = [p.strip() for p in re.split(r"\n\s*\n", original_text) if p.strip()]
    rew_pars = [p.strip() for p in re.split(r"\n\s*\n", rewritten_text) if p.strip()]

    if len(orig_pars) == len(rew_pars) and len(rew_pars) > 1:
        # return reconstructed with original paragraph count
        return "\n\n".join(rew_pars)
    # otherwise return rewritten text as-is
    return rewritten_text.strip()


# -------------------------
# Prompt templates
# -------------------------
def _build_rewrite_prompt(original: str, tone: str, context: Optional[str]) -> str:
    # Medium-strength rewrite instructions + preserve paragraphs
    instructions = (
        "Rewrite the text to improve clarity, coherence, and concision (MEDIUM strength). "
        "Preserve the original meaning and paragraph breaks. "
        "Keep sentences readable and natural. "
        "Prefer active voice, remove needless words, and improve flow. "
        "Do NOT invent facts. If context is provided, prefer wording that aligns with the given style rules."
    )

    prompt_parts = [
        f"[Task] {instructions}",
        f"[Tone] {tone}",
    ]

    if context:
        prompt_parts.append("[Reference context — use only for style guidance]:")
        prompt_parts.append(context.strip())

    prompt_parts.append("[Original Text]:")
    prompt_parts.append(original.strip())
    prompt_parts.append("\nRewritten text:")

    return "\n\n".join(prompt_parts)


def _build_qa_prompt(question: str, context: Optional[str]) -> str:
    prompt = []
    if context:
        prompt.append("You are a writing coach. Use the context below (from authoritative style sources) to answer the question precisely.")
        prompt.append("Context:\n" + context.strip())
    else:
        prompt.append("You are a writing coach. Answer concisely using general writing knowledge.")
    prompt.append("Question:\n" + question.strip())
    prompt.append("Answer:")
    return "\n\n".join(prompt)


def _postprocess_rewrite(original: str, output_text: str) -> str:
    # Try to preserve paragraph distribution if possible
    reconstructed = _preserve_paragraphs_reconstruct(original, output_text)
    return reconstructed.strip()

# -------------------------
# Public function
# -------------------------
def chatbot(user_input: str) -> str:
    """
    Public API for app. Returns labeled revised text (minimal label on its own line).
    - If user_input has >= REWRITE_WORD_THRESHOLD words -> rewrite mode
    - Else -> RAG Q&A mode (if chroma available), otherwise general answer
    """

    if not user_input or not user_input.strip():
        return "Revised Version:\n\n"  # nothing to do

    # decide mode
    if _should_rewrite(user_input):
        # rewrite mode
        tone = _auto_detect_tone(user_input)
        # fetch context only if it looks relevant
        use_context = bool(_rag_regex.search(user_input)) and _retriever is not None
        context = _fetch_context(user_input) if use_context else None

        prompt = _build_rewrite_prompt(user_input, tone, context)
        out = gen_pipe(prompt, return_full_text=False)
        # pipeline returns list of dicts with "generated_text"
        text = out[0]["generated_text"] if isinstance(out, list) and out else str(out)
        rewritten = _postprocess_rewrite(user_input, text)
        return f"Revised Version:\n\n{rewritten}"

    else:
        # Q&A mode (short input treated as question)
        tone = _auto_detect_tone(user_input)
        context = _fetch_context(user_input) if _retriever is not None else None
        prompt = _build_qa_prompt(user_input, context)
        out = gen_pipe(prompt, return_full_text=False)
        text = out[0]["generated_text"] if isinstance(out, list) and out else str(out)
        return text.strip()
