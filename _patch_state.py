"""Patch script: inject explain_agent_evidence() into dashboard/state.py"""
import re

src = open("dashboard/state.py", "r", encoding="utf-8").read()

NEW_FN = '''

def explain_agent_evidence(agent, evidence, bundle):
    """Focused LLM explanation for one agent's extracted evidence metrics."""
    import os
    from langchain_groq import ChatGroq
    from langchain_core.messages import SystemMessage, HumanMessage
    from config.config_loader import get

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return "GROQ_API_KEY not set."

    is_blamed = bundle and (bundle.primary_agent or "").lower() == agent.lower()
    if bundle:
        arbiter_note = (
            f"Arbiter verdict: '{bundle.primary_agent}' caused a {bundle.primary_cause.value} "
            f"failure ({bundle.priority_level.value}). "
            + ("THIS is the blamed agent." if is_blamed else "This agent is not blamed.")
        )
    else:
        arbiter_note = "No arbiter verdict available yet."

    prompt = "\\n".join([
        f"AGENT: {agent}",
        f"SOURCES CITED: {evidence.source_count}",
        f"NAMED ENTITIES EXTRACTED: {evidence.entity_count}",
        f"TOOL CALLS MADE: {len(evidence.tool_calls)}",
        "",
        f"ARBITER CONTEXT: {arbiter_note}",
        "",
        f"In 3-4 sentences explain:",
        f"1. What do these numbers reveal about what the {agent} agent did in the pipeline?",
        f"2. Are these source/entity counts high, low, or normal for this agent role?",
        f"3. How does this agent relate to the overall pipeline verdict?",
        f"4. What should an engineer inspect first when debugging this agent?",
        "",
        "Be specific and technical. Use hedged language (this is heuristic analysis). No bullet points.",
    ])

    try:
        llm = ChatGroq(
            model=get("llm", "model"),
            temperature=0.0,
            max_tokens=512,
            api_key=api_key,
        )
        resp = llm.invoke([
            SystemMessage(content=(
                "You are an AI observability analyst. Explain extracted evidence metrics "
                "for a single agent in a multi-agent research pipeline. "
                "Be concise, technical, and actionable. Write in flowing prose — no bullet points."
            )),
            HumanMessage(content=prompt),
        ])
        return str(resp.content)
    except Exception as exc:
        return f"LLM error: {exc}"
'''

# Insert after run_explanation
marker = "def run_explanation(bundle: AnalysisBundle) -> AnalysisBundle:\n    return LLMExplainer().explain(bundle)"
if marker not in src:
    print("ERROR: marker not found — check state.py")
else:
    new_src = src.replace(marker, marker + NEW_FN, 1)
    open("dashboard/state.py", "w", encoding="utf-8").write(new_src)
    print("state.py patched OK")
