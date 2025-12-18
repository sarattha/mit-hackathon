import asyncio

from agents import Agent, Runner


async def main():
    # Try GPT-5
    print("--- Testing gpt-5 ---")
    try:
        agent = Agent(name="DebugAgent", instructions="You are a helpful assistant.", model="gpt-5")
        result = await Runner.run(agent, "Hello")
        print("Success GPT-5:", result.final_output)
    except Exception as e:
        print("Error GPT-5:", e)

    # Try default
    print("\n--- Testing Default ---")
    try:
        agent = Agent(name="DebugAgent", instructions="You are a helpful assistant.")
        result = await Runner.run(agent, "Hello")
        print("Success Default:", result.final_output)
    except Exception as e:
        print("Error Default:", e)


if __name__ == "__main__":
    asyncio.run(main())
