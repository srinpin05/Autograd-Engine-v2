# Architecture Plan

> A high-level template for planning software architecture. Fill in each section as you make decisions; leave the rest as `TODO` until you need them.

---

## 1. Overview

**Project name:** Autograd Engine v2

**One-line description:** C++ implementation of an autograd engine to understand backpropagation.



### What is this?
This project is a from scratch C++ implementation of an autodifferentiation engine. 

### Why does it exist?
This project's purpose is to understand the underlying code and math behind backpropagation of neural networks. 

---

## 2. Goals & Non-Goals

### Goals

- Given a series of functions applied to inputs, engine should be able to return gradients of outputs with respect to the inputs. 
- Reverse mode implementation.
- Implement around 20+ operations.

### Out of Scope
- Visualization of differentiation through graph (for now)
- Using engine for real backprop of neural network (for now)

---

## 3. High-Level Architecture

> Describe the big picture. A diagram beats prose here.

### Diagram
```text
┌─────────────┐      ┌──────────────┐      ┌─────────────┐
│   Client    │ ───> │   Service    │ ───> │  Datastore  │
└─────────────┘      └──────────────┘      └─────────────┘
                            │
                            v
                     ┌──────────────┐
                     │  External    │
                     │   API        │
                     └──────────────┘
```

**How to read this:** *(one sentence per arrow/box describing its job)*

### Style / Pattern
`Monolith` | `Modular Monolith` | `Microservices` | `Layered` | `Event-driven` | `Pipeline` | `Peer-to-Peer` | `Other: __`

Why this style? *(1–2 sentences)*

---

## 4. Components / Modules

> Break the system into named pieces. Each component should have a single, clear responsibility.

| Component | Responsibility | Talks to | Owns |
|-----------|---------------|----------|------|
| `Example` | Does X | Y, Z | Data: A; State: B |
| | | | |
| | | | |
| | | | |

### Component details

#### `ComponentName`
- **Purpose:**
- **Inputs:**
- **Outputs:**
- **Key dependencies:**
- **Failure modes / what happens when it breaks:**

---

## 5. Data Flow

> Trace the path of a typical request/operation end-to-end.

1. User does **X**.
2. `ComponentA` receives it and …
3. `ComponentB` …
4. Data is persisted to …
5. Response returns via …

### Data shape (high level)
What are the main data entities? *(name + one-line description)*

| Entity | Description |
|--------|-------------|
| `User` | The actor using the system. |
| | |

---

## 6. Key Design Decisions

> For each significant decision: what you chose, what you rejected, and why.

### Decision 1: `Title`
- **Context:** What problem triggered this?
- **Choice:** What we picked.
- **Alternatives considered:** A, B, C.
- **Why:** Trade-offs that tipped the scale.
- **Consequences:** What this locks in / costs us.

### Decision 2: `Title`
- ...

---

## 7. Cross-Cutting Concerns

How does the system handle these? (Tick or describe; leave blank if N/A.)

| Concern | Approach |
|---------|----------|
| **Authentication / Authorization** | |
| **Error handling** | |
| **Logging / Observability** | |
| **Configuration** | |
| **Concurrency / Threading model** | |
| **Caching** | |
| **Security (secrets, input validation)** | |
| **Performance targets** | |
| **Deployment / Runtime** | |

---

## 8. Dependencies

### External (libraries, services, APIs)
- `library/service` — *why we need it; what we use from it*

### Internal (other systems, teams, repos)
- `team/system` — *contract / interface we depend on*

---

## 9. Risks & Trade-offs

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| | Low/Med/High | Low/Med/High | |
| | | | |

### Known trade-offs
Things we are deliberately accepting as the lesser evil:
- *e.g., "We chose eventual consistency over strong consistency for write throughput."*

---

## 10. Future Considerations

> Things we are **not** doing now but want to keep the door open for.

- [ ]
- [ ]

What would have to be true to revisit the current architecture? *(e.g., 10x users, new regulatory requirement, etc.)*

---

## 11. Open Questions

- [ ] Question 1?
- [ ] Question 2?

---

## 12. Glossary

| Term | Meaning |
|------|---------|
| | |

---

## Appendix: Change Log

| Date | Author | Change |
|------|--------|--------|
| | | Initial draft |
