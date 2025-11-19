import os
from typing import Dict, Tuple, TypedDict, Any
from gradient import Gradient
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END

load_dotenv()

MODEL_TASK = os.getenv("MODEL_TASK", "llama3.3-70b-instruct")   # model for market research
MODEL_META = os.getenv("MODEL_META", "llama3.3-70b-instruct")   # model used as meta-LLM to rewrite prompts
MAX_ITERS = 4

MODEL_ACCESS_KEY = os.getenv("GRADIENT_MODEL_ACCESS_KEY")
if not MODEL_ACCESS_KEY:
    raise RuntimeError("Set GRADIENT_MODEL_ACCESS_KEY environment variable")

inference_client = Gradient(model_access_key=MODEL_ACCESS_KEY)

REQUIRED_SECTIONS = [
    "Top 3 trends",
    "Competitor analysis",
    "Actionable insight",
    "Sources",
]

def task_llm_call_gradient(prompt: str, model: str = MODEL_TASK, temperature: float = 0.2, max_tokens: int = 700) -> str:
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

def meta_rewrite_prompt_gradient(original_prompt: str, response: str, checks: Dict[str, bool], model: str = MODEL_META) -> str:
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

def quality_check(response: str) -> Tuple[bool, Dict[str, bool]]:
    lower = response.lower()
    checks = {}
    checks["Top 3 trends"] = ("top 3 trends" in lower) or ("top 3" in lower)
    checks["Competitor analysis"] = ("competitor analysis" in lower) or ("competitor" in lower)
    checks["Actionable insight"] = ("actionable insight" in lower) or ("actionable" in lower)
    checks["Sources"] = ("source" in lower) or ("sources" in lower)
    success = all(checks.values())
    return success, checks

class AgentState(TypedDict):
    prompt: str
    response: str
    checks: Dict[str, bool]
    iterations: int

def generate_node(state: AgentState) -> AgentState:
    response = task_llm_call_gradient(state['prompt'])
    ok, checks = quality_check(response)
    
    return {
        "response": response,
        "checks": checks,
        "iterations": state['iterations'] + 1
    }

def rewrite_node(state: AgentState) -> AgentState:
    improved = meta_rewrite_prompt_gradient(state['prompt'], state['response'], state['checks'])
    return {"prompt": improved}

def check_quality_edge(state: AgentState) -> str:
    if all(state['checks'].values()):
        return END
    if state['iterations'] >= MAX_ITERS:
        return END
    return "rewrite"

workflow = StateGraph(AgentState)
workflow.add_node("generate", generate_node)
workflow.add_node("rewrite", rewrite_node)

workflow.set_entry_point("generate")
workflow.add_conditional_edges("generate", check_quality_edge)
workflow.add_edge("rewrite", "generate")

app = workflow.compile()

if __name__ == "__main__":
    starting_prompt = (
        "Market Research: Write a market research brief for entering the North American smart-wearables market.\n"
        "Make it useful for a Product Manager."
    )
    
    initial_state = {
        "prompt": starting_prompt,
        "iterations": 0,
        "response": "",
        "checks": {}
    }
    
    result = app.invoke(initial_state)
    
    print(result["response"])
