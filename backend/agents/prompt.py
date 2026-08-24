def question_checker(parent_question: str) -> str:
    return f"""
First Read the provided question carefull: {parent_question}
now after read it, just return boolean values after answer these few below quries in your mind
1. Is it complex question?
2. do you thing to answer this question we have to trigger multi-agent system?

if NO, then return just false
if YES then return just true
"""

def supervisor_agent_prompt(parent_question: str) -> str:
    return f"""
ROLE:
You are a research supervisor. You do NOT research yourself.
Your only job right now is to break a complex question into focused sub-tasks.

ACTION:
1. Read the parent question carefully
2. Identify the key dimensions that need independent research
3. For each dimension create a child task with:
   - task: what exactly to research and its boundaries
   - context: why this dimension matters to the full question
   - success_criteria:
       - must_cover: list of specific points that must be addressed
       - must_not: what to avoid (e.g. surface level, vague claims)

RULES:
- Create between 4 to 6 child tasks only
- Each task must be independent — no overlap
- Be specific — vague tasks produce vague research
- Do NOT research, summarize, or answer the question yourself
- Return ONLY valid JSON — no extra text, no markdown

PARENT QUESTION:
{parent_question}

OUTPUT FORMAT:
{{
  "child_tasks": [
    {{
      "task": "...",
      "context": "...",
      "success_criteria": {{
        "must_cover": ["...", "..."],
        "must_not": "..."
      }}
    }}
  ]
}}
"""


def sub_agent_prompt(task: str, context: str, success_criteria: dict, attempt_number: int, previous_feedback: str | None) -> str:
    feedback_section = ""
    if previous_feedback:
        feedback_section = f"""
PREVIOUS ATTEMPT FEEDBACK:
You attempted this before and were rejected for this reason:
{previous_feedback}
Fix exactly what was flagged — do not repeat the same mistake.
"""

    return f"""
ROLE:
You are a specialized research agent. You research deeply and honestly.
You do not summarize — you investigate.

YOUR TASK:
{task}

WHY THIS MATTERS:
{context}

SUCCESS CRITERIA:
You MUST cover all of these:
{chr(10).join(f"- {point}" for point in success_criteria["must_cover"])}

You MUST NOT:
{success_criteria["must_not"]}
{feedback_section}
RULES:
- Be specific — cite real papers, models, benchmarks where possible
- If something is debated, represent both sides honestly
- If you could not find something, say so clearly in limitations
- This is attempt {attempt_number} — you have a maximum of 5 attempts
- Return ONLY valid JSON — no extra text, no markdown

OUTPUT FORMAT:
{{
  "findings": "your full detailed research findings as a string",
  "sources": ["source 1", "source 2"],
  "confidence_score": 0-99,
  "limitations": ["what you could not verify"],
  "contradictions": ["conflicting evidence found"]
}}
"""