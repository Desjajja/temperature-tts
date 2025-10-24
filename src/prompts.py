
AIME_PROMPT = """Please reason step by step, and put your final answer within \\boxed{{}}.

{question}
"""

MATH500_PROMPT = """Answer the following math question step by step, given in LaTeX
format, clearly and concisely, and present the final answer as \\boxed{{x}}.

{question}
"""

def build_prompt(template_name: str, question: str) -> str:
    if template_name.upper() == "AIME":
        return AIME_PROMPT.format(question=question)
    elif template_name.upper() == "MATH500":
        return MATH500_PROMPT.format(question=question)
    else:
        return question
