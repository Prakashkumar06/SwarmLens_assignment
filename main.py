from src.agent_graph import run_agent
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--q", "-q", type=str, required=True, help="Question")
    args = parser.parse_args()
    res = run_agent(args.q)
    print("=== ANSWER ===")
    print(res.get("answer"))
    print("\n=== REFLECTION ===")
    print(res.get("reflection"))

if __name__ == "__main__":
    main()
