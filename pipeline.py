import re
import time

from langchain_core.messages import ToolMessage

from agent import build_search_agent, writer_chain, critic_chain
from tools import scrape_url


def get_tool_outputs(result) -> str:
    """Raw verbatim tool output (real text + real URLs)."""
    outputs = [
        m.content for m in result.get("messages", [])
        if isinstance(m, ToolMessage)
    ]
    return "\n\n".join(outputs) if outputs else ""


def get_final_text(result) -> str:
    messages = result.get("messages", [])
    if not messages:
        return ""
    content = messages[-1].content
    if isinstance(content, list):
        return "\n".join(
            item.get("text", str(item)) if isinstance(item, dict) else str(item)
            for item in content
        )
    return str(content)


def extract_urls(text: str, max_urls: int = 3) -> list:
    """Find up to 3 real, non-PDF URLs in the text."""
    urls = re.findall(r"https?://[^\s\"')\],]+", text)
    clean = []
    for u in urls:
        if not u.lower().endswith(".pdf") and u not in clean:
            clean.append(u)
        if len(clean) == max_urls:
            break
    return clean


def safe_run(func, step_name: str, wait: int = 65):
    """If rate-limited, wait and retry once automatically."""
    try:
        return func()
    except Exception:
        print(f"\n⚠ {step_name} hit the free API limit. Waiting {wait}s, then retrying...")
        time.sleep(wait)
        return func()


def run_research_pipeline(topic: str) -> dict:
    state = {}

    # ============================================
    # STEP 1 - SEARCH
    # ============================================
    print("\n" + "=" * 50)
    print("STEP 1 - Search agent is working...")
    print("=" * 50)

    search_agent = build_search_agent()

    search_result = safe_run(
        lambda: search_agent.invoke({
            "messages": [(
                "user",
                f"Use the web_search tool to find recent, reliable information about: {topic}"
            )]
        }),
        "Step 1"
    )

    raw = get_tool_outputs(search_result)
    state["search_results"] = raw if raw else get_final_text(search_result)

    print("\nSearch Results:\n")
    print(state["search_results"][:3000])

    print("\n(Pausing 60s to respect free API limits...)")
    time.sleep(60)

    # ============================================
    # STEP 2 - SCRAPE (tries up to 3 URLs)
    # ============================================
    print("\n" + "=" * 50)
    print("STEP 2 - Reader is scraping top resource...")
    print("=" * 50)

    urls = extract_urls(state["search_results"])
    state["scraped_content"] = "No additional page was scraped."

    if not urls:
        print("  → no scrapable URL found, using search data only.")
    else:
        for url in urls:
            print("  → trying:", url)
            result = scrape_url.invoke(url)

            if result.startswith("Could not scrape") or result.startswith("Skipped"):
                print("    ✗ failed, trying next URL...")
                continue

            state["scraped_content"] = result
            print("    ✓ scraped successfully!")
            break

    print("\nScraped Content:\n")
    print(state["scraped_content"][:1500])

    print("\n(Pausing 60s to respect free API limits...)")
    time.sleep(60)

    # ============================================
    # STEP 3 - WRITER
    # ============================================
    print("\n" + "=" * 50)
    print("STEP 3 - Writer is drafting the report...")
    print("=" * 50)

    research_combined = (
        f"SEARCH RESULTS:\n{state['search_results']}\n\n"
        f"DETAILED SCRAPED CONTENT:\n{state['scraped_content']}"
    )

    state["report"] = safe_run(
        lambda: writer_chain.invoke({"topic": topic, "research": research_combined}),
        "Step 3"
    )

    print("\nFinal Report:\n")
    print(state["report"])

    print("\n(Pausing 60s to respect free API limits...)")
    time.sleep(60)

    # ============================================
    # STEP 4 - CRITIC
    # ============================================
    print("\n" + "=" * 50)
    print("STEP 4 - Critic is reviewing the report...")
    print("=" * 50)

    state["feedback"] = safe_run(
        lambda: critic_chain.invoke({"report": state["report"]}),
        "Step 4"
    )

    print("\nCritic Report:\n")
    print(state["feedback"])

    return state


if __name__ == "__main__":
    topic = input("\nEnter a research topic: ").strip()
    if not topic:
        print("Please enter a valid topic.")
    else:
        run_research_pipeline(topic)