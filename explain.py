import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()  # reads .env file and loads GROQ_API_KEY into environment

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def get_explanation(text, verdict, confidence):
    """
    Takes the original text, BERT's verdict, and confidence score.
    Returns a 2-3 sentence plain-English explanation from Groq.
    """
    if verdict == "FAKE":
        framing = "why the wording resembles common misinformation patterns (vague sourcing, absolute claims, emotionally loaded language, conspiratorial framing)"
    else:
        framing = "why the wording resembles typical credible reporting (specific attribution, measured language, verifiable details like numbers/dates/sources)"

    prompt = f"""This statement was flagged as {verdict} ({confidence:.1f}% confidence) by a misinformation-pattern classifier:

"{text}"

Write a 2-sentence explanation focused on {framing}. Write directly — no preamble, no "this statement was classified as," no restating the task. Focus only on wording/style, not fact-checking. If confidence is below 60%, acknowledge briefly that the signal is mixed/weak rather than overstating certainty.
"""

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,   # lower = more focused/consistent, less random
            max_tokens=150
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"(Explanation unavailable: {str(e)})"