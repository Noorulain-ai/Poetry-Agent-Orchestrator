from agents import Agent, Runner, OpenAIChatCompletionsModel, set_tracing_disabled, AsyncOpenAI
from dotenv import load_dotenv
import os

# --- Load environment & disable tracing
load_dotenv()
set_tracing_disabled(disabled=True)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env file")

# --- Gemini client setup
client = AsyncOpenAI(
    api_key=GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# --- Lyric Poetry Agent
lyric_poetry_agent = Agent(
    name="lyric_poetry_agent",
    model=OpenAIChatCompletionsModel(model="gemini-2.5-flash", openai_client=client),
    instructions=(
        "You are an expert in lyric poetry. Focus on emotional, personal reflections with vivid imagery. "
        "Respond in under 50 words."
    )
)

# --- Narrative Poetry Agent
narrative_poetry_agent = Agent(
    name="narrative_poetry_agent",
    model=OpenAIChatCompletionsModel(model="gemini-2.5-flash", openai_client=client),
    instructions=(
        "You are an expert in narrative poetry. Tell a short story through verse, with a clear plot and characters. "
        "Keep it under 50 words."
    )
)

# --- Dramatic Poetry Agent
dramatic_poetry_agent = Agent(
    name="dramatic_poetry_agent",
    model=OpenAIChatCompletionsModel(model="gemini-2.5-flash", openai_client=client),
    instructions=(
        "You are an expert in dramatic poetry. Write lines that resemble dramatic monologue or scene. "
        "Use conflict and emotion. Stay under 50 words."
    )
)

# --- Triage Agent
triage_agent = Agent(
    name="triage_agent",
    model=OpenAIChatCompletionsModel(model="gemini-2.5-flash", openai_client=client),
    instructions=(
        "You are a poetry classifier. Given a poem or prompt, decide if it's lyric, narrative, or dramatic, "
        "and hand it off to the correct expert agent. Response must be under 50 words."
    ),
    handoffs=[lyric_poetry_agent, narrative_poetry_agent, dramatic_poetry_agent]
)

# --- Poem input
input_text = "Write a poem about a lonely traveler recounting his memories."

# --- Run analysis
lyric_result = Runner.run_sync(starting_agent=lyric_poetry_agent, input=input_text)
narrative_result = Runner.run_sync(starting_agent=narrative_poetry_agent, input=input_text)
dramatic_result = Runner.run_sync(starting_agent=dramatic_poetry_agent, input=input_text)
triage_result = Runner.run_sync(starting_agent=triage_agent, input=input_text)

# --- Print results
print("\n🌸❤ LYRIC POETRY AGENT OUTPUT:")
print(lyric_result.final_output)

print("\n📜🌹 NARRATIVE POETRY AGENT OUTPUT:")
print(narrative_result.final_output)

print("\n🎭👩🏻‍🤝‍🧑🏻 DRAMATIC POETRY AGENT OUTPUT:")
print(dramatic_result.final_output)

print("\n🔍🕵️‍♀️ TRIAGE AGENT OUTPUT:")
print(triage_result.final_output)
