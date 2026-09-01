from typing import Any

def build_prompt(profile: dict[str, Any], question: str, context_chunks: list[dict[str, Any]]) -> str:
    context = "\n\n".join(
        f"[Source: {chunk.get('source', 'unknown')}]\n{chunk.get('content', '').strip()}"
        for chunk in context_chunks
    )

    holdings = ", ".join(profile.get("current_holdings", [])) or "Not specified"
    profile_text = (
        "Primary goal: {primary_goal}\n"
        "Horizon: {horizon_years} years\n"
        "Risk score: {risk_score}/10\n"
        "Investor profile: {investor_profile}\n"
        "Current holdings: {holdings}"
    ).format(
        primary_goal=profile.get("primary_goal", "Not specified"),
        horizon_years=profile.get("horizon_years", "Not specified"),
        risk_score=profile.get("risk_score", "Not specified"),
        investor_profile=profile.get("investor_profile", "Not specified"),
        holdings=holdings,
    )

    return f"""You are a financial intelligence assistant operating locally. Use the retrieved research context as the primary source of truth. Distinguish clearly between retrieved facts and general reasoning.

RETRIEVED RESEARCH CONTEXT
--------------------------
{context or 'No retrieved research context was found for this question.'}

CLIENT PROFILE
--------------
{profile_text}

USER QUESTION
-------------
Question: {question}

INSTRUCTIONS
------------
- Answer in a research-grounded, educational, and personalized manner.
- Prioritize the retrieved research context over general assumptions.
- If the research does not contain enough information, say so clearly.
- Do not invent sources, numbers, regulations, or performance claims.
- Do not present speculation as guaranteed financial advice.
- Ground your answer in the supplied profile and the retrieved context.
"""