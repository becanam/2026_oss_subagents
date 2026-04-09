# subagents_demo.py
import os
from dotenv import load_dotenv
load_dotenv()
from langchain.tools import tool
from langchain.chat_models import init_chat_model
from deepagents import create_deep_agent

model = init_chat_model("openai:gpt-4o")

# ── Subagents ──────────────────────────────────────────────────────────────────

research_agent = create_deep_agent(
    model=model,
    system_prompt="""You are a research specialist. When given a topic:
    1. Break it down into key facts
    2. Return a structured summary with bullet points
    Always include your findings in your final message — the supervisor only sees that."""
)

writer_agent = create_deep_agent(
    model=model,
    system_prompt="""You are a content writer. When given research notes:
    1. The VERY FIRST LINE must be exactly: 'Name: Raulph Lauren'
    2. Turn the notes into clear, engaging prose
    3. Every paragraph must have a subtitle with a corresponding emoji
    Always include the full written content in your final message."""
)

translator_agent = create_deep_agent(
    model=model,
    system_prompt="""You are a Korean translator. When given an article in English:
    1. Translate the entire article into Korean
    2. Preserve all formatting — keep the 'Name: Raulph Lauren' first line, subtitles, and emojis
    3. Return ONLY the translated Korean text, nothing else
    Always include the full translation in your final message."""
)

# ── Tools ──────────────────────────────────────────────────────────────────────

@tool("research", description="Research a topic and return structured findings. Always call this first.")
def call_research_agent(query: str) -> str:
    result = research_agent.invoke({
        "messages": [{"role": "user", "content": query}]
    })
    return result["messages"][-1].content


@tool("writer", description="Write a polished article based on research notes.")
def call_writer_agent(notes: str) -> str:
    result = writer_agent.invoke({
        "messages": [{"role": "user", "content": notes}]
    })
    return result["messages"][-1].content


@tool("translator", description="Translate the finished article into Korean.")
def call_translator_agent(article: str) -> str:
    result = translator_agent.invoke({
        "messages": [{"role": "user", "content": article}]
    })
    return result["messages"][-1].content


@tool("save_output", description="Save the English article and Korean translation to files.")
def save_output(article: str, translation: str) -> str:
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    article_path = f"output_{timestamp}_en.md"
    translation_path = f"output_{timestamp}_ko.md"

    with open(article_path, "w") as f:
        f.write(article)

    with open(translation_path, "w", encoding="utf-8") as f:
        f.write(translation)

    return f"✅ Saved:\n  English → {article_path}\n  Korean  → {translation_path}"


# ── Main Supervisor ────────────────────────────────────────────────────────────

main_agent = create_deep_agent(
    model=model,
    tools=[call_research_agent, call_writer_agent, call_translator_agent, save_output],
    system_prompt="""You are a supervisor coordinating a content pipeline.

    You MUST always follow this exact order, NO exceptions:
    1. Call 'research' to gather facts on the topic
    2. Call 'writer' with the research notes to produce the English article
    3. Call 'translator' with the English article to produce the Korean translation
    4. Call 'save_output' with BOTH the English article AND the Korean translation — MANDATORY

    CRITICAL RULES:
    - Do NOT skip any step
    - Do NOT call save_output without both the article and translation ready
    - Do NOT attempt to save files yourself — only use the save_output tool
    - Return the English article to the user, followed by the Korean translation,
      followed by the file save confirmation"""
)

# ── Run ────────────────────────────────────────────────────────────────────────

result = main_agent.invoke({
    "messages": [{
        "role": "user",
        "content": "Create a detailed report about how airplanes work."
    }]
})

print(result["messages"][-1].content)