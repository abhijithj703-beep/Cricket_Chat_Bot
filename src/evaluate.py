from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualRelevancyMetric
)

from generate import ask_question


# ============================================================
# 1. Define the normal RAG test cases
# ============================================================

test_cases_data = [
    {
        "id": "TC01",
        "question": "What happens if a batter hits the ball twice?",
        "expected_behavior": """
        The answer should explain Law 34 (Hit the ball twice).
        The striker can be dismissed if they wilfully strike the ball again
        before it has been touched by a fielder, unless the second strike
        is solely to guard the wicket.
        """
    },

    {
        "id": "TC02",
        "question": "What is a no ball?",
        "expected_behavior": """
        The answer should explain the meaning of a No ball and relevant
        consequences under Law 21, including the one-run penalty and that
        the delivery does not count as one of the six balls.
        """
    },

    {
        "id": "TC03",
        "question": "Can a batter strike the ball twice?",
        "expected_behavior": """
        The answer should explain that a second strike can be permitted
        in specific circumstances, such as guarding the wicket, while
        wilfully striking the ball again before a fielder touches it can
        result in dismissal under Law 34.
        """
    },

    {
        "id": "TC04",
        "question": "When is a batter out for hitting the ball twice?",
        "expected_behavior": """
        The answer should explain the conditions for dismissal under Law 34,
        including wilfully striking the ball again before it has been touched
        by a fielder and the exception for solely guarding the wicket.
        """
    },

    {
        "id": "TC05",
        "question": (
            "If the batter deliberately makes contact with the ball again "
            "before a fielder touches it, are they dismissed?"
        ),
        "expected_behavior": """
        The answer should give a conditional response. The striker is out
        under Law 34 if they wilfully strike the ball again before a fielder
        touches it, unless the second strike is solely for guarding the wicket.
        """
    },

    {
        "id": "TC06",
        "question": "What happens when a batter obstructs the field?",
        "expected_behavior": """
        The answer should explain dismissal for Obstructing the field under
        Law 37 and may include consequences for runs and the fact that the
        bowler does not receive credit for the wicket.
        """
    },

    {
        "id": "TC09",
        "question": "What happens when a ball goes outside the boundary?",
        "expected_behavior": """
        The answer should explain the relevant boundary rules, including
        when a boundary four or six is scored and relevant consequences
        for runs and the ball becoming dead.
        """
    },

    {
        "id": "TC10",
        "question": (
            "Can a batter be out for hitting the ball twice if the ball "
            "has already been touched by a fielder?"
        ),
        "expected_behavior": """
        The answer should state that the striker is not out under Law 34
        merely because they wilfully strike the ball after it has touched
        a fielder, subject to the relevant Law 37 provisions.
        """
    }
]


# ============================================================
# 2. Run the RAG and create DeepEval test cases
# ============================================================

deepeval_test_cases = []

for test in test_cases_data:

    print(f"\nRunning {test['id']}: {test['question']}")

    result = ask_question(test["question"])

    test_case = LLMTestCase(
        input=result["question"],
        actual_output=result["answer"],
        retrieval_context=result["retrieval_context"]
    )

    deepeval_test_cases.append(test_case)

    print("Answer:")
    print(result["answer"])


# ============================================================
# 3. Define RAG metrics
# ============================================================

faithfulness_metric = FaithfulnessMetric(
    threshold=0.7
)

answer_relevancy_metric = AnswerRelevancyMetric(
    threshold=0.7
)

contextual_relevancy_metric = ContextualRelevancyMetric(
    threshold=0.7
)


# ============================================================
# 4. Run normal RAG evaluation
# ============================================================

print("\n========================================")
print("RAG BASELINE EVALUATION")
print("========================================")

evaluate(
    test_cases=deepeval_test_cases,
    metrics=[
        faithfulness_metric,
        answer_relevancy_metric,
        contextual_relevancy_metric
    ]
)


# ============================================================
# 5. Refusal tests — TC07 and TC08
# ============================================================

refusal_tests = [
    {
        "id": "TC07",
        "question": "Who won the 2019 Cricket World Cup?"
    },
    {
        "id": "TC08",
        "question": "Who is India's current cricket captain?"
    }
]


print("\n========================================")
print("REFUSAL TESTS")
print("========================================")


for test in refusal_tests:

    print(f"\nRunning {test['id']}: {test['question']}")

    result = ask_question(test["question"])

    answer = result["answer"]

    print("\nAnswer:")
    print(answer)

    # Simple baseline refusal check
    expected_refusal = (
        "I could not find the answer in the provided Laws of Cricket."
    )

    if expected_refusal.lower() in answer.lower():
        print("Result: PASS - Correct refusal")
    else:
        print("Result: FAIL - Possible hallucination")