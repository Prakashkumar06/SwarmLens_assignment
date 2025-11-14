from src.agent_graph import run_agent
from src.evaluate import evaluate_all

question = "What is renewable energy?"
result = run_agent(question)

answer = result["answer"]

reference = "Renewable energy comes from natural sources that replenish faster than consumed."

results = evaluate_all(question, answer, reference)
print(results)
