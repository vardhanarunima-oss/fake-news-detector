from explain import get_explanation

result = get_explanation(
    text="Government secretly putting mind control chips in vaccines",
    verdict="FAKE",
    confidence=86.0
)
print(result)