import pandas as pd


# Load the actual dataset
data = pd.read_csv("data/sample.csv")

question = "What is the highest salary?"

# Simulate a WRONG Data Agent result
agent_result = 35000

# Calculate the correct answer directly from the dataset
expected_result = data["Salary"].max()

print("\n📝 Reviewer Rejection Test")
print("==========================")
print("Question:", question)
print("Data Agent Result:", agent_result)
print("Expected Result:", expected_result)

if agent_result == expected_result:
    print("\n✅ Reviewer would approve the result.")
else:
    print("\n❌ Reviewer would reject the result.")
    print(
        f"Feedback: Expected {expected_result}, "
        f"but Data Agent returned {agent_result}."
    )