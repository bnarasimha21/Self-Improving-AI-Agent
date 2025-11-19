"""
Self-improving agent using DigitalOcean Gradient SDK for both task LLM and meta-LLM.

Requirements:
  pip install gradient
  export GRADIENT_MODEL_ACCESS_KEY="..."     # required
  optionally: export GRADIENT_AGENT_ACCESS_KEY and GRADIENT_AGENT_ENDPOINT

Docs reference: DigitalOcean gradient-python README (usage examples).
"""
import os
from typing import Dict, Tuple
from gradient import Gradient
from dotenv import load_dotenv

load_dotenv()

# === CONFIG ===
MODEL_TASK = os.getenv("MODEL_TASK", "llama3.3-70b-instruct")   # model for market research
MODEL_META = os.getenv("MODEL_META", "llama3.3-70b-instruct")   # model used as meta-LLM to rewrite prompts
MAX_ITERS = 4

# === Init clients ===
MODEL_ACCESS_KEY = os.getenv("GRADIENT_MODEL_ACCESS_KEY")
if not MODEL_ACCESS_KEY:
    raise RuntimeError("Set GRADIENT_MODEL_ACCESS_KEY environment variable")

# Use the Gradient client for inference
inference_client = Gradient(model_access_key=MODEL_ACCESS_KEY)

# === Quality rubric ===
REQUIRED_SECTIONS = [
    "Top 3 trends",
    "Competitor analysis",
    "Actionable insight",
    "Sources",
]

# === Task LLM call (Gradient) ===
def task_llm_call_gradient(prompt: str, model: str = MODEL_TASK, temperature: float = 0.2, max_tokens: int = 700) -> str:
    """
    Calls Gradient's serverless inference chat completion API.
    Returns text content from the first choice.
    """
    # The SDK exposes: inference_client.chat.completions.create(...)
    resp = inference_client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a helpful market-research assistant."},
            {"role": "user", "content": prompt},
        ],
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    try:
        return resp.choices[0].message.content
    except Exception:
        if hasattr(resp, "choices") and len(resp.choices) > 0:
            c = resp.choices[0]
            if hasattr(c, "message") and hasattr(c.message, "content"):
                return c.message.content
            elif hasattr(c, "text"):
                return c.text
        return str(resp)

# === Meta LLM: use Gradient to rewrite prompt automatically ===
def meta_rewrite_prompt_gradient(original_prompt: str, response: str, checks: Dict[str, bool], model: str = MODEL_META) -> str:
    """
    Use a model to rewrite the prompt. We instruct the model to return only the improved prompt.
    """
    failed = [k for k, ok in checks.items() if not ok]
    instruction = (
        "You are a prompt-writing assistant. The ORIGINAL_PROMPT and RESPONSE are below.\n\n"
        "Produce ONLY an improved prompt (<= 120 words) that will make the task model include the "
        "missing sections. Keep it concise and preserve the user's original intent.\n\n"
        "ORIGINAL_PROMPT:\n" + original_prompt + "\n\n"
        "RESPONSE:\n" + response + "\n\n"
        "MISSING_OR_INCOMPLETE_SECTIONS:\n" + (", ".join(failed) if failed else "none") + "\n\n"
        "Guidance:\n"
        "- If Top 3 trends missing, ask: 'Give the top 3 trends as numbered bullets.'\n"
        "- If Competitor analysis missing, ask: 'Include competitor analysis (3 competitors, 1-line each).'\n"
        "- If Actionable insight missing, ask: 'Provide a single clear actionable recommendation for product or GTM.'\n"
        "- If Sources missing, ask: 'List the types of sources used (e.g., industry reports, product pages).'\n"
        "Return ONLY the improved prompt text (no commentary)."
    )

    resp = inference_client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You rewrite prompts to improve LLM output quality."},
            {"role": "user", "content": instruction},
        ],
        model=model,
        temperature=0.0,
        max_tokens=300,
    )

    try:
        return resp.choices[0].message.content.strip()
    except Exception:
        # fallback
        return str(resp).strip()

# === Quality checker ===
def quality_check(response: str) -> Tuple[bool, Dict[str, bool]]:
    lower = response.lower()
    checks = {}
    checks["Top 3 trends"] = ("top 3 trends" in lower) or ("top 3" in lower)
    checks["Competitor analysis"] = ("competitor analysis" in lower) or ("competitor" in lower)
    checks["Actionable insight"] = ("actionable insight" in lower) or ("actionable" in lower)
    checks["Sources"] = ("source" in lower) or ("sources" in lower)
    success = all(checks.values())
    return success, checks

# === Main loop ===
def self_improving_loop(initial_prompt: str, max_iters: int = MAX_ITERS):
    prompt = initial_prompt
    last_response = ""
    for i in range(1, max_iters + 1):
        print(f"\n--- Iteration {i} — sending prompt to task LLM ---\n")
        print(prompt)
        response = task_llm_call_gradient(prompt)
        print(f"\n--- Task LLM response (iteration {i}) ---\n")
        print(response[:3000])

        ok, checks = quality_check(response)
        print(f"\nQuality checks: {checks}\n")
        last_response = response
        if ok:
            print("Response accepted by quality checker.")
            return {"response": response, "iterations": i, "checks": checks}

        # Not ok -> use meta model to rewrite prompt
        print("Calling meta model to rewrite prompt based on failed checks...")
        improved = meta_rewrite_prompt_gradient(prompt, response, checks)
        print("\n--- Meta-improved prompt ---\n")
        print(improved)
        # Use the improved prompt next iteration
        prompt = improved

    print("Max iterations reached; returning last response (may be incomplete).")
    return {"response": last_response, "iterations": max_iters, "checks": checks}

# === Demo run ===
if __name__ == "__main__":
    starting_prompt = (
        "Market Research: Write a market research brief for entering the North American smart-wearables market.\n"
        "Make it useful for a Product Manager."
    )
    result = self_improving_loop(starting_prompt, max_iters=4)
    print("\n\n===== FINAL OUTPUT =====\n")
    print(result["response"])
    print("\n===== METADATA =====")
    print("Iterations used:", result["iterations"])
    print("Final checks:", result["checks"])
