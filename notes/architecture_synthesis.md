# Architecture Synthesis: The Log, Events, and Boundaries

**Sources:** The Log (Jay Kreps), DDIA Ch11 — Stream Processing (Kleppmann), Hexagonal Architecture (Cockburn), Three Books One Idea (synthesis)

---

## The Single Core Idea

All four sources arrive at one insight, stated at different levels of abstraction:

> **Separate what happened (the log) from what is currently true (derived state), and hide both behind clean interfaces.**

| Source | How They Say It |
|---|---|
| Cockburn | "Code pertaining to the inside must not leak into the outside." |
| Kreps | "The log is the authoritative source of truth; every table or index is a derived projection." |
| Kleppmann | "The mutable state is just a cached view of the log." |
| APoSD | "Deep modules: small interface, large implementation." |

---

## Three Orthogonal Concerns

The four sources address three distinct but complementary concerns. Understanding them separately prevents conflating them.

### 1 — Temporal Integrity: "What happened, in order" (Kreps + Kleppmann)

The log is append-only, totally ordered, and time-indexed by entry number. It is the only authoritative thing. Every other data structure — tables, indexes, caches, dashboards — is a derived view that can be reconstructed by replaying the log.

**Key implications:**
- `current_state = fold(log_events, initial_state)` — state is a computation over history
- A replica is fully described by a single number: its log cursor position. `cursor + log = entire replica state`
- **N×M → N+M**: a central log eliminates point-to-point connections. N producers write to the log; M consumers subscribe. Adding a new consumer is one subscription, not N new connections.
- **Log compaction**: for keyed records, discard entries whose key has a more recent update. The log becomes a complete backup of current state without retaining full history.

**Event sourcing** (Kleppmann's refinement): application logic is *explicitly* built around immutable events.
- A request arrives as a **command** → validated → if accepted, becomes an **event** (durable, immutable, appended to the log)
- Multiple different read views (projections) can be derived from the same log
- Debugging = replaying the log to any point in time

---

### 2 — Boundary Integrity: "What knows about what" (Cockburn + Clean Architecture + APWP)

The application's core business logic defines **ports** (technology-agnostic contracts). **Adapters** implement those contracts and live outside the core. The inside never knows what is plugged in.

**Key implications:**
- Dependency direction: everything points inward; the core depends on nothing external
- Swap adapters freely: real DB ↔ FakeRepository, live broker ↔ in-memory queue, REST ↔ CLI
- **Fully isolated mode**: run the entire application against test doubles with no infrastructure

```
[ External World ]
       ↕  Adapter (translator, lives outside)
     [ Port ]  ← contract defined inside
 [ Core Business Logic ]
     [ Port ]  ← contract defined inside
       ↕  Adapter (translator, lives outside)
[ External World ]
```

The hexagonal shape is intentional — all sides are equal. Every external dependency, regardless of which "side" it's on, is handled symmetrically through a port/adapter pair.

**Primary actors** drive the application (user, HTTP client, test driver).  
**Secondary actors** are driven by it (database, email, message broker, audit system).

---

### 3 — Interface Integrity: "What callers see vs. what hides behind" (APoSD + Cockburn)

The best modules are **deep**: a small, simple interface hiding a large, complex implementation. Complexity is pulled downward, not pushed outward.

**Key implications:**
- Interface size directly determines cognitive load for every caller
- **Classitis** — too many small, thin classes — is a failure mode: complexity leaks upward into callers
- If you cannot describe an interface simply in a comment, the design is wrong
- The core's port definitions should be as small as possible; all complexity lives inside the adapter implementations

---

## How They Build on Each Other

```
Cockburn (2005) ── identified the problem: entanglement of business logic and external entities
    └─► Clean Architecture ── where to draw the boundary: around business rules
        └─► APWP ── how to implement it in Python: Repository, UoW, Message Bus
            └─► APoSD ── what the boundary should look like: deep modules, small interfaces

Kreps (The Log) ── identified the data primitive: the append-only ordered log
    └─► Kleppmann (DDIA Ch11) ── how to build systems on it: CDC, event sourcing, stream processors
        └─► Kafka ── the production implementation: partitioned, replicated, compacted
```

The two trees **converge at the Message Bus** (APWP):
- The message bus is an approximation of a log
- It enforces the Cockburn boundary: producers publish events without knowing consumers
- Adding a consumer is one `subscribe()` call — zero changes to any producer

---

## The Unified Pattern

```
┌───────────────────────────────────────────────────────────┐
│  PRIMARY ACTORS (HTTP, CLI, test driver)                  │
└────────────────────────┬──────────────────────────────────┘
                         │  Primary Adapter (translates protocol)
                         ▼
                    [ Primary Port ]
                         │
                         ▼
┌───────────────────────────────────────────────────────────┐
│  CORE BUSINESS LOGIC                                      │
│  Commands → validate → emit Events                        │
│  Queries  → read from derived state (fold of log)         │
└────────────────────────┬──────────────────────────────────┘
                         │  publishes to
                         ▼
┌───────────────────────────────────────────────────────────┐
│  EVENT LOG  (append-only, ordered, entry number = clock)  │
│  Source of truth. Every write yields a log entry number.  │
└────────┬─────────────────────────────────────┬────────────┘
         │  subscribe                          │  subscribe
         ▼                                     ▼
┌─────────────────┐                  ┌──────────────────────┐
│  Consumer A     │                  │  Consumer B           │
│  (Adapter)      │                  │  (Adapter)            │
│  DB / index     │                  │  Stream processor     │
│  derived state  │                  │  real-time output     │
└─────────────────┘                  └──────────────────────┘
```

---

## The Key Failure Modes (and Their Fixes)

| Failure Mode | Origin | Fix |
|---|---|---|
| Business logic entangled with HTTP/DB | Cockburn | Ports & Adapters: core defines contracts, adapters implement them |
| Mutable dict as source of truth — restart wipes state | Kreps | Event log → table: dict is the derived view, log is authoritative |
| N×M coupling: new consumer touches every producer | Kreps | Central log/bus: write once, consumers subscribe independently |
| State is only as durable as memory | Kleppmann | Commands → immutable events → log → derive state on restart |
| Crash = restart from zero | Kleppmann | Checkpoint + cursor: resume from last processed log entry |
| Thin classes everywhere, complexity leaks to callers | APoSD | Deep modules: pull complexity down, minimize interface surface |

---

## Event Sourcing vs. Traditional State

| Aspect | Traditional (State-Based) | Event-Sourced (Log-Based) |
|---|---|---|
| Source of truth | Current state in mutable store | Append-only event log |
| Reads | Query current state directly | Derive from log (or pre-built projection) |
| Auditability | Requires a separate audit table | Free: the log is the audit trail |
| Time-travel | Requires point-in-time backups | Free: replay log to any cursor offset |
| Crash recovery | Restore last snapshot | Replay log from last checkpoint offset |
| Adding a consumer | Wire into state mutation code paths | Subscribe to the log — zero producer changes |
| Debugging | "What is the current value?" | "What sequence of events produced this state?" |

---

## Stream Processing (Kleppmann — the operational view)

A **stream processor** reads from one or more logs and writes to one or more logs. It keeps local state (a local table or index). It journals a changelog of its local state — this changelog is itself a log that other processors can subscribe to.

**Three processing options:**
1. **Store** — write events to DB/cache/index for later querying
2. **Push** — send events to users; stream to real-time dashboards
3. **Transform** — input stream(s) → output stream(s) (the pipeline model)

**Fault tolerance via microbatching**: break the infinite stream into small blocks (~1 second); treat each like a mini-batch. Failure = re-run that block. Creates an implicit tumbling window.

**Window types:**
- Tumbling — fixed-size, non-overlapping
- Sliding — fixed-size, moves forward continuously
- Session — groups events until a gap of inactivity

---

## For FLARE

From the source notes, the direct FLARE mappings:

| Concept | FLARE Application |
|---|---|
| Log = source of truth | Telemetry packet stream is append-only; CCSDS sequence counter = log entry number |
| State = fold(log) | Current sensor state = latest value per channel derived from packet stream |
| Replica = cursor position | FLARE ingest offset = full replay position; one number = entire state |
| N×M → N+M | Telemetry log → detector, storer, alerter, auditor subscribe independently |
| Event sourcing | Incident lifecycle: `Detected → Investigated → Mitigated → Resolved` = immutable event log |
| CDC | FLARE incident DB → ASTR-O audit trail: same change log, same order, guaranteed consistency |
| Log compaction | Per-sensor: keep latest value, discard intermediate readings within retention window |
| 3-timestamp correction | `corrected_time = t1 + (t3 - t2)` — compensates for spacecraft clock drift without trusting device clock |
| Stream processing | Raw CCSDS packets → Isolation Forest → `AnomalyDetected` → LLM → `RecommendationAudited` |
| CEP | "3 threshold breaches from same sensor in 30s" = state machine match over event stream |
| Microbatching | 1-second packet batches; each independently replayable on failure |
| Ports & Adapters | Core anomaly detection logic never touches Kafka, ASTR-O API, or Slack directly |
| Deep modules | Anomaly detector exposes a small interface (`score(packet) → AnomalyEvent | None`); internal windowing/model state is hidden |

---

## Decision Guide

**Reach for a log (event sourcing / append-only) when:**
- Audit trail matters (compliance, debugging, incident review)
- Crash recovery without data loss is required
- Multiple downstream consumers need the same events
- State needs to be reconstructable from scratch
- You want time-travel or replay

**Reach for ports & adapters when:**
- Business logic needs to be testable in isolation
- External dependencies change or are unreliable
- Multiple interaction modes exist (HTTP + CLI + background job + test)
- You want to defer infrastructure choices

**Reach for deep modules (APoSD) when:**
- Designing any public-facing interface
- The implementation is complex but callers should not feel it
- Choosing between many thin classes vs. fewer, richer ones

---

*Last updated: 2026-05-31*
