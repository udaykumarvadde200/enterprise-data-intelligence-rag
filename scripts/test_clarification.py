from app.sql.sql_pipeline import SQLPipeline


pipeline = SQLPipeline()


print("=" * 60)
print("TURN 1")
print("=" * 60)

result = pipeline.run(
    "Show me John's orders."
)

print("STATUS:")
print(result["status"])

print("MESSAGE:")
print(result["message"])


assert result["status"] == "clarification_required"


print("\n" + "=" * 60)
print("TURN 2")
print("=" * 60)

result = pipeline.run(
    "The Bangalore one."
)

print("STATUS:")
print(result["status"])

assert result["status"] == "success"


print("\nGENERATED SQL:")
print(result["sql"])


print("\nQUERY RESULT:")

for row in result["results"]:
    print(row)


# ---------------------------------------------------------
# Verify that only customer_id = 4 was returned
# ---------------------------------------------------------

expected_results = [
    (
        6,
        4,
        "Dell XPS 15",
        120000.00,
        row[4],
    )
    for row in result["results"]
]


customer_ids = {
    row[1]
    for row in result["results"]
}


assert customer_ids == {4}, (
    f"Expected only customer_id=4, "
    f"but got customer IDs: {customer_ids}"
)


assert len(result["results"]) == 2, (
    f"Expected 2 orders, "
    f"but got {len(result['results'])}"
)


print("\n" + "=" * 60)
print("TEST PASSED")
print("=" * 60)