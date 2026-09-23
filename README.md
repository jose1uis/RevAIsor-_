# RevAIsor Access Review Agent

A runnable Python CLI for the internship assessment. It reads a static access-review
snapshot, selects tools deterministically, synthesizes a cited answer with an LLM,
and verifies the answer before displaying it. It never changes access.

## Setup and run

Requires Python 3.10 or newer.

```bash
git clone https://github.com/jose1uis/RevAIsor-_.git
cd RevAIsor-_
python -m venv .venv
```

Activate with `.venv\Scripts\Activate.ps1` in PowerShell, or
`source .venv/bin/activate` on macOS/Linux, then:

```bash
python -m pip install -r requirements.txt
```

For live synthesis, set an API key in your shell. PowerShell:

```powershell
$env:OPENAI_API_KEY = "<your API key>"
$env:OPENAI_MODEL = "gpt-4.1-mini"
```

On macOS/Linux, use `export OPENAI_API_KEY="<your API key>"` and
`export OPENAI_MODEL="gpt-4.1-mini"`. The model is optional and defaults to
`gpt-4.1-mini`; choose a Responses API model available to your account.
`.env.example` documents the variables; the program does **not** automatically
load a `.env` file. Never commit a real key.

```bash
python agent.py "What role does Priya own?"
python agent.py "What is the difference between GBR and ZGBR?"
python agent.py "Show me Priya's access review history."
python agent.py "Order a pizza for the office"
python agent.py "What is the difference between Priya's GBR role and a ZGBR?"
```

Each run shows entities, intent, graph paths/context, tool selection, raw outputs,
the final answer, and verification checks. This is deterministic execution
metadata, not private model reasoning. Missing credentials produce a clear error
and exit code 2; scope refusals need neither an API key nor tool calls.
An exhausted API credit balance produces a billing-specific error; successful
live generation requires an API key with available API credits.

## Offline testing

Add `--mock` to any agent command for an explicitly labeled offline evidence
renderer. It exercises routing, real local tools, and verification without an
LLM or network. It is a testing aid, not a replacement for live LLM synthesis.

```bash
python agent.py "What role does Priya own?" --mock
python agent.py "What is the difference between GBR and ZGBR?" --mock
python agent.py "Show me Priya's access review history." --mock
python agent.py "Order a pizza for the office" --mock
python agent.py "What is the difference between Priya's GBR role and a ZGBR?" --mock
python agent.py "what roles does carlos own?" --mock
python agent.py "What role does Zoe own?" --mock
python agent.py "What is our VPN policy?" --mock
python -m unittest discover -s tests -v
```

Tests include case/word boundaries, missing users, graph extension, both-tools
routing, fabricated/unrelated roles and citations, refusals with no tool/LLM calls,
CLI exit codes, and SDK request/response handling with a local mock HTTP transport.
That transport tests integration; it does not establish live model quality.
Exit codes: **0** verified answer/refusal, **1** failed verification, **2**
configuration/input/service error. A refused or missing-data question returning
0 means it was handled safely, not that the requested facts exist.

## Technical write-up

### Knowledge graph schema

A dictionary stores `Role -> Ownership Role -> {GBR, ZGBR}`, type characteristics,
three users, their ownership/review records, and three sourced policy documents.
Role-instance nodes are derived from ownership, including Fatima's mixed roles.
The graph walk follows user ownership, role type, parent hierarchy, and policy
source edges; its paths are visible in the CLI. Tools return copies from this
single data source. A dictionary is sufficient for this small static dataset;
a larger production system could use a dedicated graph database.

### Routing strategy

Case-insensitive term boundaries identify known users, role IDs, and role types.
A graph walk precedes intent classification and tool selection. Ownership/history
uses `role_tool`, definitions/comparison uses `document_tool`, and user-specific
comparisons use both. Explicit unknown names are retained in supported question
patterns. Unsupported access actions and unrelated requests are handled locally.
Deterministic routing is predictable, inexpensive, fast, and easy to test.

### System prompt design

The prompt restricts the model to access-review evidence, requires concise
answers, forbids fabricated records and access changes, and requires it to
acknowledge missing evidence or correct mistaken premises. Ownership, history,
and lookup claims cite **[Role-DB]**; policy claims cite the returned
**[source_id]** next to the claim. It describes records as a static snapshot.

### How the verifier prevents hallucinations

`trusted KG/tools -> LLM synthesis -> deterministic verification -> display`.
The LLM is not the source of truth. Checks require retrieved citations and source
kinds, reject invented or unrelated role IDs, check recognized ownership clauses,
and reject ISO dates absent from evidence. Refusals need no fabricated citation.
Failed answers are withheld and the CLI exits nonzero. These are syntactic
guardrails: they cannot prove arbitrary policy paraphrases, reviewer/status
pairings, or every ownership phrasing. The optional LLM judge adds a separate
assessment, not a correctness guarantee.

### Trade-offs and design decisions

One direct OpenAI SDK keeps the repository small; no database or agent framework
is needed. English routing patterns intentionally cover this bounded dataset,
not unrestricted language understanding. Data is static and historical, with no
assumed current access status. Add users in `KG["users"]`; add role types under
`KG["role_types"]` with parent/source links and matching documents. Derived role
instances and graph walks pick up those additions without a duplicate table.
The offline renderer favors faithful evidence display over conversational nuance.

## Independent evaluator (bonus)

`evaluator.py` makes a separate LLM call after generation, judging grounding,
citation accuracy, and relevance. Its validated JSON verdict contains the three
booleans, PASS/FAIL, and a brief assessment; it never influences the original
answer. It requires an API key even when evaluating an offline agent answer.
Optionally set `OPENAI_EVALUATOR_MODEL`; otherwise it uses `OPENAI_MODEL`.

```bash
python evaluator.py "What role does Priya own?"
python evaluator.py "What role does Priya own?" --mock-agent
python agent.py "What role does Priya own?" --mock --json > run.json
python evaluator.py --result-file run.json
```

## Files

```text
RevIAsor-_/
|-- README.md
|-- agent.py             # CLI, synthesis, verification gate
|-- router.py            # deterministic routing
|-- knowledge_graph.py   # source data and graph traversal
|-- tools.py             # structured evidence lookups
|-- verifier.py          # deterministic output checks
|-- llm_client.py        # environment-configured OpenAI adapter
|-- evaluator.py         # independent bonus judge
|-- requirements.txt
|-- .env.example
|-- .gitignore
`-- tests/
    |-- test_agent.py
    |-- test_router.py
    |-- test_graph_tools.py
    |-- test_verifier.py
    |-- test_llm_client.py
    `-- test_evaluator.py
```

The starter APIs remain available: `run_agent(query) -> str` and
`verify_response(query, response) -> dict` from `agent`. For raw evidence and
diagnostics use `run_agent_detailed(query)` or `--json`. During the agent loop the
verifier receives actual tool outputs; its two-argument compatibility form
recreates scoped static evidence deterministically.

## Source reconciliation and assumptions

- Based on the [read-only starter](https://github.com/RevAIsor/Mathematical-Engineer-Test-AI-Agents/tree/622c41f9237d2744f34aa7a92194a1810340d747).
  Only this personal submission repository is modified.
- The starter KG omitted Carlos, Fatima, ZGBR, and two policies already present in
  its tools. Those exact records/texts are consolidated here; ZGBR characteristics
  come from those policies. No new people, IDs, or review events were invented.
- The PDF's `Policy-Doc-X` is illustrative; answers use the real three source IDs.
  Its four listed queries plus separate combined example cover the stated five.
- The PDF allows notebooks, but the accompanying instructions require this normal
  Python repository. Its optional-verifier wording is superseded by the explicit
  deterministic-verifier requirement.
- The PDF mentions a mock endpoint, but none was provided in the starter. The
  explicit `--mock` renderer is local. Successful live synthesis and evaluator
  quality validation remain pending; SDK integration and failure handling are
  covered by the offline test suite.
- The starter README mentions five query functions, but supplies two tool
  functions. Both concepts are preserved; unused Gemini imports are removed.

SDK usage follows the [official OpenAI Python documentation](https://developers.openai.com/api/docs/libraries)
and [Responses API reference](https://developers.openai.com/api/reference/python/resources/responses/methods/create).
