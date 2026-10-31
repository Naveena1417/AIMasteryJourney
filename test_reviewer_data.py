from agents.reviewer import review_result


question = "What is the highest salary?"

agent_name = "data"

# Intentionally wrong result
result = "35000"


print("\n📝 Data Agent Result")
print("====================")
print(result)


review = review_result(
    question,
    agent_name,
    result
)


print("\n🧐 Reviewer")
print("===========")
print("Approved:", review["approved"])
print("Feedback:", review["feedback"])


if review["approved"]:
    print("\n✅ Reviewer approved the result.")
else:
    print("\n❌ Reviewer rejected the result.")