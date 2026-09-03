import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# Pick up key from environment without passing an empty string
api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")

# Fast model: Gemini 1.5 Flash
fast_llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    api_key=api_key,
    temperature=0.5,
    timeout=60,
    max_retries=2,
)

# Writer model: Gemini 1.5 Flash (reliable and avoids free-tier rate limits)
writer_llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    api_key=api_key,
    temperature=0.4,
    timeout=90,
    max_retries=2,
)

# Critic model: Gemini 1.5 Flash
critic_llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    api_key=api_key,
    temperature=0.2,
    timeout=60,
    max_retries=2,
)

fast_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a concise, factual AI research assistant. "
        "Answer clearly using markdown formatting. "
        "Use headers, bullet points, and bold text where helpful. "
        "Be direct, structured, and informative.",
    ),
    ("human", "Question or topic: {topic}"),
])

writer_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an expert research writer. "
        "Create clear, structured, factual reports using markdown. "
        "Always cite sources when available.",
    ),
    (
        "human",
        """Write a detailed research report on the topic below 
using the provided research data.

Topic: {topic}

Research Gathered:
{research}

Use this exact structure:

# {topic}

## Introduction
(2-3 sentences overview)

## Key Findings
- Finding 1
- Finding 2
- Finding 3
- Finding 4
- Finding 5

## Detailed Analysis
(3-4 paragraphs of in-depth analysis)

## Conclusion
(2-3 sentences summary)

## Sources
(List only valid URLs found in the research data)
""",
    ),
])

critic_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a strict but fair research critic and editor. "
        "Provide honest, constructive, specific feedback.",
    ),
    (
        "human",
        """Review the research report below and provide structured feedback.

Report:
{report}

Use this exact format:

## Critic Review

**Score: X/10**

### Strengths
- Strength 1
- Strength 2
- Strength 3

### Areas to Improve
- Issue 1
- Issue 2
- Issue 3

### Verdict
(One clear sentence summarizing overall quality)
""",
    ),
])

parser = StrOutputParser()

fast_chain = fast_prompt | fast_llm | parser
writer_chain = writer_prompt | writer_llm | parser
critic_chain = critic_prompt | critic_llm | parser
