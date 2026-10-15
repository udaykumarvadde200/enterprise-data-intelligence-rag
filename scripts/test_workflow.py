from app.graph.workflow import EnterpriseWorkflow


workflow = EnterpriseWorkflow()

thread_id = "customer-clarification-test"


def ask(question: str):
    print("\n" + "=" * 60)
    print(f"QUESTION: {question}")
    print("=" * 60)

    response = workflow.ask(
        question,
        thread_id=thread_id,
    )

    print("ROUTE:", response.get("route"))
    print("ANSWER:")
    print(response.get("answer"))

    if response.get("error"):
        print("ERROR:", response["error"])

    return response


# Test 1: SQL question
first = ask("Show me Rahul Sharma's orders.")

assert first.get("route") == "sql"
assert first.get("result", {}).get("status") == "success"

# Test 2: Ambiguous customer
second = ask("Show me John's orders.")

assert second.get("result", {}).get("status") == (
    "clarification_required"
)

# Test 3: Clarification must continue in the same conversation
third = ask("The Bangalore one.")

assert third.get("route") == "sql"
assert third.get("result", {}).get("status") == "success"

sql = third["result"]["sql"].lower()

assert "customer_id = 4" in sql, (
    f"Expected SQL to enforce customer_id = 4, got: {sql}"
)

products = {row[2] for row in third["result"]["results"]}
assert products == {
    "Dell XPS 15",
    "Logitech MX Master 3S",
}, f"Unexpected products returned: {products}"

assert len(third["result"]["results"]) == 2

print("\nALL WORKFLOW TESTS PASSED")