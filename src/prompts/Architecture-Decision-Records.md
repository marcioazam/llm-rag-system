# Architecture Decision Records (ADR) Prompt

## Objetivo
Gerar um rascunho padronizado de ADR (Architecture Decision Record) baseado no contexto fornecido.

## Estrutura
1. **Título**
   - Breve e descritivo.
2. **Contexto**
   - Situação atual, restrições, requisitos.
3. **Decisão**
   - O que foi decidido (inclua alternativas rejeitadas).
4. **Consequências**
   - Positivas e negativas, impacto em longo prazo.
5. **Estado**
   - Proposed | Accepted | Deprecated | Superseded.
6. **Referências**
   - Links para tickets, docs, POCs.

## Placeholders
- {{problem}}
- {{alternatives}}
- {{constraints}}
- {{decision}}

---
Use quando a equipe precisa registrar escolhas arquiteturais importantes.

## Overview
This prompt helps structure architectural decisions systematically using the ADR (Architecture Decision Record) format, ensuring comprehensive documentation of technical choices.

## Instructions
When analyzing or making significant architectural decisions, apply this structured format for documentation:

```
# ADR-[NUMBER]: [TITLE]

## Status
[PROPOSED | ACCEPTED | SUPERSEDED | DEPRECATED]
Date: YYYY-MM-DD

## Context
[Describe the forces at play, including technological, business, and team constraints. These forces are probably in tension, and should be called out as such. The language used should be value-neutral. Simply describe facts and constraints.]

## Decision
[Describe the decision that was made in full sentences, with active voice: "We will..."]

## Consequences
[Describe the resulting context after applying the decision, with emphasis on trade-offs and impacts. All consequences should be listed - positive, negative, and neutral.]

## Alternatives Considered
[List alternatives that were considered, and briefly explain why they were not chosen]

## Compliance Verification
[Describe how adherence to this decision can be verified]

## Related Decisions
[List related architectural decisions, with links]

## References
[List relevant references, papers, patterns, or resources that influenced this decision]
```

## When to Apply This Prompt
- When making significant design/architecture choices that impact multiple components
- When introducing new technologies, frameworks or patterns
- When changing established patterns or approaches
- When making decisions that have long-term maintenance implications
- Before implementing complex integrations between systems

## Example Application

```
# ADR-042: Adopt Event Sourcing for Transaction Processing

## Status
ACCEPTED
Date: 2023-09-15

## Context
Our transaction processing system currently uses a traditional CRUD approach with direct database updates. We're experiencing issues with:
- Audit trail requirements from regulatory bodies
- Concurrency conflicts during high-volume periods
- Difficulty reconstructing historical states for debugging
- Need for better scalability as transaction volume grows 20% annually

## Decision
We will implement an event sourcing pattern for the transaction processing subsystem:
- All state changes will be recorded as immutable events
- Current state will be derived from event stream
- Commands and queries will be separated (CQRS)
- Events will be stored in an append-only event store (using EventStoreDB)

## Consequences

### Positive
- Complete audit trail available by design
- Improved concurrency handling through event-based model
- Ability to reconstruct system state at any point in time
- Better scalability through event-driven architecture
- Easier debugging of complex transaction sequences

### Negative
- Increased complexity in system architecture
- Learning curve for development team
- Additional infrastructure components to maintain
- Potential performance overhead for simple queries
- More complex testing procedures

### Neutral
- Need to adapt deployment and monitoring processes
- Events become the source of truth instead of current state

## Alternatives Considered
1. Enhanced CRUD with audit tables - Rejected due to performance impact and incomplete state capture
2. Database CDC (Change Data Capture) - Rejected due to coupling with database vendor and incomplete transaction context
3. Command pattern without event sourcing - Rejected due to lack of temporal query capabilities

## Compliance Verification
- Code reviews must verify use of event store for state changes
- Automated tests must verify event sequence correctness
- Static analysis tools will verify architecture boundaries

## Related Decisions
- ADR-038: Microservice Boundaries
- ADR-040: Database Per Service Pattern

## References
- Vernon, V. (2016). Domain-Driven Design Distilled. Addison-Wesley
- Fowler, M. "Event Sourcing" https://martinfowler.com/eaaDev/EventSourcing.html
- CQRS Pattern: https://docs.microsoft.com/en-us/azure/architecture/patterns/cqrs
``` 