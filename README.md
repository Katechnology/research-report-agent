# Research Report Agent

This mini project turns a research-workflow assignment into a standalone notebook. A planner chooses search queries and an outline; search tools gather source excerpts; a research agent creates an evidence brief; a writer drafts; a reviewer checks the draft; and an editor revises it. The default target is about 1,200 words. The notebook is a walkthrough, while `research_workflow.py` contains reusable code.

## Setup

Use Python 3.10 or newer. Check with `python3 --version`; if that command points to an older Python, substitute your newer Python executable in the first command below.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

Add your **OpenAI API key** to the local `.env` file. `TAVILY_API_KEY` is optional: with it, the research agent can also search the wider web. Without it, academic paper search and Wikipedia remain available. Paper search uses OpenAlex, with Crossref as a fallback when OpenAlex is unavailable. Never commit `.env`.

Start Jupyter with `jupyter notebook`, open `research_workflow.ipynb`, and run the cells in order. Change `topic` or `ReportConfig` in the second code cell to adjust the report. Model and search API calls may incur charges.

Run the offline workflow checks with:

```bash
python -m unittest discover -s tests -v
```

The default models are `gpt-4.1-mini` for planning and `gpt-4.1` for research, writing, review, and editing. Each can be changed in `ReportConfig`. These models support the Chat Completions interface used by `aisuite`. The previous `gpt-4o-mini` model remains an option if lower cost matters more than report quality.

## Files

- `research_workflow.ipynb`: runnable demonstration and report display
- `research_workflow.py`: planner, research, writer, editor, and orchestration code
- `research_tools.py`: OpenAlex/Crossref paper search, Wikipedia, and optional Tavily search functions
- `.env.example`: names of the local environment variables

## How it works

The planner returns 3–5 distinct search queries and a 4–7 section outline. Python executes those searches, deduplicates the results, and gives each source an ID. The research agent summarizes only retrieved excerpts. The writer uses those notes to draft a report. A separate reviewer flags unsupported claims and missing coverage; the editor revises the draft. The application turns `[S1]`-style citations into links from the source catalog and appends a source list. It reports the final word count, source count, citation count, and any length warning.

Research summaries and model-generated claims can still contain mistakes. Search result excerpts are often incomplete, so check linked sources before sharing the final report. The workflow prioritizes accuracy over meeting its word target when evidence is thin. Generated Markdown files in `reports/` are ignored by Git.

## Project origin

Adapted from my course assignment. The original grader instructions and course tests are not part of this project.
