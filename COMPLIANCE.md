# EU AI Act compliance — VOIDSEED demo (LLMwebsite / voidseed-api)

Regulation (EU) 2024/1689 ("the AI Act"), as amended by Regulation (EU)
2026/1744. Article 50 has applied since 2 August 2026. This note covers the
public chat demo at `llmwebsite-pink.vercel.app` (repo: `LLMwebsite`) and its
inference backend (repo: `voidseed-api`, deployed by hand via `scp`, not a
git repository).

Written 2026-08-12. Not legal advice — see "Requires human/legal review"
below for the open items.

## What the system is

A retrieval-augmented chat demo over a 155,410,513-parameter GPT trained
from scratch (`tinyllm2`/VOIDSEED). The deployed checkpoint was trained on a
security-documentation corpus (hacktricks, hacktricks-cloud,
PayloadsAllTheThings, SecLists, GTFOBins.github.io, CheatSheetSeries, wstg —
not TinyStories, which was an earlier, separate 24M-parameter smoke-test
model with no public deployment). A visitor types a question, the backend
retrieves relevant passages from an FTS5 index over that corpus, and the
model generates an answer conditioned on the retrieved text.

## Art. 3(1) — is this an "AI system"?

Yes. It is machine-based, generates output (text) from input for an
explicit objective (answering the visitor's question) after training, and
operates with the varying autonomy characteristic of Art. 3(1)'s
definition. Not disputed.

## Roles — Art. 3(3)/(4)

The operator of this deployment is both:
- **Provider (Art. 3(3))** — trained the model and put the demo into
  service under its own name.
- **Deployer (Art. 3(4))** — operates it for end users in the course of
  this activity.

## Why not Art. 5 (prohibited practices)

Art. 5 lists eight prohibited practices. None apply to a Q&A chat demo:

- (a) subliminal/manipulative/deceptive techniques materially distorting
  behaviour — the system answers questions; it does not attempt to
  manipulate visitor behaviour.
- (b) exploiting vulnerabilities of age, disability or social/economic
  situation — not targeted at, or aware of, any such characteristic.
- (c) social scoring leading to detrimental treatment in unrelated
  contexts — no scoring of any kind.
- (d) criminal-risk assessment based solely on profiling or personality
  traits — not a risk-assessment system.
- (e) untargeted scraping of facial images from the internet or CCTV — no
  biometric or facial data involved anywhere in this system.
- (f) emotion inference in workplaces and education — no emotion
  inference; the demo is a public web page, not a workplace/education
  deployment.
- (g) biometric categorisation inferring sensitive characteristics — no
  biometric input of any kind.
- (h) real-time remote biometric identification in public spaces for law
  enforcement — not applicable; text-only, no biometric identification,
  no law-enforcement use.

## Why not Annex III (high-risk)

Annex III lists eight high-risk domains: (1) biometrics, (2) critical
infrastructure, (3) education/vocational training, (4) employment/worker
management, (5) access to essential private and public services
(credit, insurance, emergency services, etc.), (6) law enforcement,
(7) migration/asylum/border control, (8) administration of justice and
democratic processes. This system does not operate in any of them — it is
a standalone public demo answering security-documentation questions, with
no decision-making role in any of the eight domains.

## Why not Annex I

Annex I lists the EU harmonisation legislation whose scope defines
"safety component" high-risk AI (machinery, toys, lifts, medical devices,
etc.). This system is software with no physical safety-component role
under any Annex I instrument.

## Why not GPAI (Art. 51-56)

The deployed model: 155,410,513 parameters, 1,960,108,296 training tokens.

Compute estimate (6ND, the standard training-FLOP approximation):
6 × 155,410,513 × 1,960,108,296 ≈ **1.83 × 10^18 FLOP**.

- The Commission's guidelines use 10^23 FLOP as the presumption criterion
  for "general-purpose AI model" under Art. 51.
- Art. 51(2)'s systemic-risk threshold is 10^25 FLOP.

1.83e18 is roughly five orders of magnitude below the GPAI presumption
threshold and seven below the systemic-risk threshold. **Art. 53 GPAI
obligations do not apply** to this model.

## Art. 50(1) — visible disclosure to natural persons

Implemented:
- `LLMwebsite/index.html:10` — a `<p id="ai-disclosure">` above the chat
  container, visible on page load: "You're talking to an AI system.
  Answers are machine-generated and may be wrong."
- `LLMwebsite/index.html:5` — `<meta name="ai-generated" content="true">`
  in `<head>`.
- `LLMwebsite/style.css` — `#ai-disclosure` rule (muted `#5c6370`, 12px,
  centred, capped at the same 640px as the chat container) plus the
  `body` flex-direction change needed to stack it above `#chat-container`
  without breaking the existing centred layout.

## Art. 50(2) — machine-readable marking of AI-generated output

Implemented on every surface that carries a generated reply:
- `LLMwebsite/main.js` (`addMessage()`) — `data-ai-generated="true"` set
  on bot message `<div>`s only, not user messages.
- `voidseed-api/main.py` — `GenerateResponse.ai_generated: bool = True`
  on the JSON body, and `response.headers["X-AI-Generated"] = "true"` on
  `POST /generate` (the production backend `main.js` actually calls).
- `LLMwebsite/backend.py` — `'ai_generated': True` added to the `POST
  /chat` JSON response, and an `after_request` hook sets
  `X-AI-Generated: true` on every response from this local-dev Flask
  server (its contract differs from the production API: `{"response":
  ...}` vs `{"answer": ...}` — see `README.md`).

## Requires human/legal review

- **Corpus licensing.** The training corpus (hacktricks, hacktricks-cloud,
  PayloadsAllTheThings, SecLists, GTFOBins.github.io, CheatSheetSeries,
  wstg) — each carries its own licence; not verified in this audit —
  requires human/legal review.
- **Art. 2(10).** Art. 2(10) excludes AI used by a natural person in a
  purely personal, non-professional activity from parts of the
  Regulation. Whether this public student-competition demo (Stardance
  Challenge) falls inside or outside that exclusion is an open question —
  not resolved here, requires human/legal review.
- **Art. 50(2) marking deadline.** If this system was first placed on the
  market or put into service before 2 August 2026, the Omnibus shifts the
  Art. 50(2) marking deadline for that placement to 2 December 2026
  rather than 2 August 2026. Whether that applies here (exact
  first-placement date) is not established in this audit — requires
  human/legal review.

## GDPR (Regulation (EU) 2016/679)

Written 2026-08-12, same audit as the AI Act notes above. Not legal advice —
see "Requires human/legal review" below.

### What's collected

No accounts, no prompt/answer storage, no cookies. The chat itself is
stateless: `voidseed-api/main.py`'s `/generate` endpoint takes a prompt,
returns an answer, and keeps nothing about the exchange in application
state or a database.

**Correction (verified on the VM, 2026-08-12):** an earlier draft of this
section said Caddy logs client IPs. That was wrong on inspection — checked
`voidseed-api/Caddyfile` directly (two bare `reverse_proxy` blocks, no `log`
directive at all, so Caddy emits no access log). The only thing persisted
is uvicorn's own stdout, captured by journald under `voidseed-api.service`.
Read those lines directly (`journalctl -u voidseed-api.service`): every one
shows `10.60.1.2:0` as the client address — a fixed internal Nest network
address (Caddy's own hop), not the visitor's real IP, and it never varies
across requests. uvicorn isn't configured with `--proxy-headers`, so it logs
the TCP peer it sees (Caddy, on the private network) rather than whatever
`X-Forwarded-For` says. The visitor's real IP is read from
`X-Forwarded-For` by `main.py`'s own rate-limiter, but that value lives only
in an in-memory dict, never written to disk, cleared on process restart —
already covered above. **No visitor-identifying data is persisted by this
system anywhere**, in application state or in infrastructure logs.

### Legal basis

Not applicable in the way GDPR envisions "legal basis" — there is no
personal data of the visitor being processed for a lasting purpose to have
one. The one thing that does exist (uvicorn's operational log of internal
proxy traffic — an internal network address, timestamp, method/path) isn't
personal data about the visitor, since it identifies the reverse proxy, not
them.

### Retention

No visitor data to retain. The internal-traffic log lines described above
persist under journald's ordinary disk-space-based rotation, same as any
service log on this VM — not a GDPR-relevant retention question, since
nothing in that stream identifies a visitor.

### Rights (Art. 12-22)

Not actionable and not needed: there is no personal data of any visitor
held anywhere in this system to access, correct, or erase.

### Requires human/legal review

- **Reconfirm periodically.** This finding depends on uvicorn's current
  logging config (no `--proxy-headers`) and Caddy's current Caddyfile (no
  `log` directive) staying as they are. If either changes — e.g. someone
  adds `--forwarded-allow-ips` to see real client IPs in the app log for
  debugging, or adds a `log` block to Caddy — this section needs re-checking
  against the new config, not assumed to still hold.
- **Art. 2(10).** GDPR doesn't have an Art. 2(10) personal-activity
  exclusion (that's an AI Act concept — see the AI Act section above for
  the equivalent question under Art. 2(10) AI Act). Whether GDPR's own
  household-activity exemption (Art. 2(2)(c)) could apply to this specific
  student-competition demo is a separate open question, not resolved
  here — requires human/legal review.
