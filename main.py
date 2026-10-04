"""Bedtime story agent: draft, judge, and rewrite with a LangGraph loop."""

import json
import os
from typing import Annotated

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from typing_extensions import TypedDict

load_dotenv()

from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

console = Console()

STORYTELLER_MODEL = os.getenv("STORYTELLER_MODEL", "qwen2.5:1.5b")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gemma4:cloud")
REWRITER_MODEL = os.getenv("REWRITER_MODEL", "gemma4:cloud")
PASS_SCORE = 7
MAX_REWRITES = 2
GRAPH_IMAGE_PATH = "graph.png"


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    user_request: str
    first_story: str
    original_story: str
    score: int
    problems: str
    suggestions: str
    rewritten_story: str
    iterations: int


storyteller_llm = ChatOllama(model=STORYTELLER_MODEL, temperature=0.2)
judge_llm = ChatOllama(model=JUDGE_MODEL, temperature=0.1)
rewriter_llm = ChatOllama(model=REWRITER_MODEL, temperature=0.3)

storyteller_prompt = ChatPromptTemplate.from_template("""
You are a kind, imaginative children's storyteller.
You write bedtime stories for children ages 5–10.
User's story request: {story_request}

Rules:
- Use simple, clear language.
- Avoid anything scary, violent, or upsetting.
- Keep a gentle bedtime tone.
- Use a clear beginning, middle, and happy ending.
- Include a soft lesson about kindness, courage, or friendship.
""")

judge_prompt = ChatPromptTemplate.from_template("""
You are a children's book reviewer.

Review the following story:
{original_story}

Evaluate the story for:
- Age appropriateness (5–10)
- Language clarity
- Story structure
- Bedtime tone
- Positive lesson

Return your response as JSON with:
- score (1–10)
- problems (list)
- suggestions (string)
""")

rewriter_prompt = ChatPromptTemplate.from_template("""
Story:
{original_story}

Rewrite the above story to fix below issues:
{problems}
Incorporate below suggestions:
{suggestions}

Keep it gentle, age-appropriate, and suitable for bedtime.
""")


def storyteller(state: AgentState):
    result = (storyteller_prompt | storyteller_llm).invoke(
        {"story_request": state["user_request"]}
    )
    return {"original_story": result.content, "first_story": result.content}


def _parse_judge_response(content: str) -> dict:
    text = content.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()
    return json.loads(text)


def judge(state: AgentState):
    content = (judge_prompt | judge_llm).invoke(
        {"original_story": state["original_story"]}
    ).content

    parsed = _parse_judge_response(content)
    problems = parsed["problems"]
    if isinstance(problems, list):
        problems = "\n".join(f"- {item}" for item in problems)

    return {
        "problems": problems,
        "suggestions": parsed["suggestions"],
        "score": int(parsed["score"]),
    }


def rewriter(state: AgentState):
    result = (rewriter_prompt | rewriter_llm).invoke(
        {
            "original_story": state["original_story"],
            "problems": state["problems"],
            "suggestions": state["suggestions"],
        }
    )
    return {
        "rewritten_story": result.content,
        "original_story": result.content,
        "iterations": state.get("iterations", 0) + 1,
    }


def route_after_judge(state: AgentState):
    """Rewrite while the score is at or below PASS_SCORE, up to MAX_REWRITES."""
    if state["score"] <= PASS_SCORE and state.get("iterations", 0) < MAX_REWRITES:
        return "rewriter"
    return END


def route_after_rewriter(state: AgentState):
    """Send the rewrite back to the judge until MAX_REWRITES is reached."""
    if state.get("iterations", 0) < MAX_REWRITES:
        return "judge"
    return END


def build_graph():
    builder = StateGraph(AgentState)
    builder.add_node("storyteller", storyteller)
    builder.add_node("judge", judge)
    builder.add_node("rewriter", rewriter)
    builder.add_edge(START, "storyteller")
    builder.add_edge("storyteller", "judge")
    builder.add_conditional_edges(
        "judge", route_after_judge, {"rewriter": "rewriter", END: END}
    )
    builder.add_conditional_edges(
        "rewriter", route_after_rewriter, {"judge": "judge", END: END}
    )
    return builder.compile()


def save_graph_image(graph) -> None:
    if os.path.exists(GRAPH_IMAGE_PATH):
        return
    try:
        graph.get_graph().draw_mermaid_png(output_file_path=GRAPH_IMAGE_PATH)
        console.print(f"[green]Graph diagram saved to {GRAPH_IMAGE_PATH}[/green]")
    except Exception as exc:
        console.print(f"[yellow]Could not generate graph diagram: {exc}[/yellow]")


def print_result(result: dict) -> None:
    console.print("\n[bold cyan]User request[/bold cyan]")
    console.print(result["user_request"])

    console.print(f"\n[bold yellow]Score:[/bold yellow] {result['score']}/10")
    console.print("\n[bold red]Judge notes[/bold red]")
    console.print(result["problems"] or "(none)")
    console.print(result["suggestions"] or "(none)")

    console.print("\n[bold magenta]First draft[/bold magenta]")
    console.print(result["first_story"])

    if result["rewritten_story"]:
        console.print("\n[bold green]Rewritten story[/bold green]")
        console.print(result["rewritten_story"])
    else:
        console.print(
            "\n[bold green]The first draft scored above the threshold, so it was kept.[/bold green]"
        )
    console.print("=" * 70)


def run() -> None:
    graph = build_graph()
    save_graph_image(graph)
    console.print(Panel.fit("Story Planning Agent", style="bold green"))

    while True:
        task = console.input("[bold yellow]Enter a story topic:[/bold yellow] ").strip()
        if not task:
            continue
        if task.lower() in {"exit", "quit"}:
            break

        try:
            result = graph.invoke(
                {
                    "messages": [HumanMessage(content=task)],
                    "user_request": task,
                    "first_story": "",
                    "original_story": "",
                    "problems": "",
                    "suggestions": "",
                    "rewritten_story": "",
                    "score": 0,
                    "iterations": 0,
                }
            )
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            console.print(f"[red]The story run failed: {exc}[/red]")
            continue

        print_result(result)


if __name__ == "__main__":
    run()
