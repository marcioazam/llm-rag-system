# System Boundary Analysis Prompt

## Overview
This prompt guides the systematic identification and analysis of system boundaries, integration points, and dependencies, helping to clarify responsibilities, minimize coupling, and establish appropriate interfaces between components.

## Instructions
When analyzing system boundaries or designing integration points, follow this structured approach:

```
# System Boundary Analysis: [SYSTEM/COMPONENT NAME]

## System Context Diagram
[Create a high-level diagram showing the system and its external dependencies]

## Boundary Identification

### Internal Components
- [List major internal components/modules]

### External Dependencies
| Dependency | Type | Direction | Protocol/Format | Criticality | Owner |
|------------|------|-----------|----------------|-------------|-------|
| [Name] | [System/Service/Library/API] | [Inbound/Outbound/Bidirectional] | [REST/gRPC/File/etc.] | [High/Medium/Low] | [Team/Organization] |

## Boundary Interfaces

### Interface 1: [Name]
- **Purpose:** [Description]
- **Type:** [Synchronous/Asynchronous/Batch/etc.]
- **Flow Direction:** [Inbound/Outbound/Bidirectional]
- **Protocol Details:** [Specifics about the communication protocol]
- **Data Formats:** [JSON/XML/Binary/etc.]
- **Rate/Volume:** [Expected transaction rates or data volumes]
- **SLA Requirements:** [Performance/Availability expectations]
- **Error Handling:** [How errors are communicated and managed]
- **Authentication/Authorization:** [Security mechanisms]
- **Resilience Patterns:** [Circuit breakers, retries, fallbacks, etc.]

### Interface 2: [Name]
[...repeat for each interface...]

## Boundary Concerns

### Data Ownership & Consistency
- [Identify master data sources]
- [Define data synchronization mechanisms]
- [Document consistency models and trade-offs]

### Security Considerations
- [Authentication requirements across boundaries]
- [Authorization controls]
- [Data protection needs]
- [Encryption requirements]

### Failure Modes
- [How system responds to dependency failures]
- [Graceful degradation strategies]
- [Recovery mechanisms]

### Monitoring & Observability
- [How interactions are logged/traced]
- [Key metrics to monitor at boundaries]
- [Alerting thresholds]

### Deployment & Versioning
- [Versioning strategy for interfaces]
- [Deployment dependencies]
- [Migration/upgrade considerations]
```

## When to Apply This Prompt
- When designing new system integrations
- During architectural decomposition of monoliths
- When clarifying team boundaries in a microservices environment
- When assessing impacts of architectural changes
- For identifying potential failure points in distributed systems
- When defining SLAs between services/components
- During system documentation and onboarding

## Example Application

```
# System Boundary Analysis: Customer Profile Service

## System Context Diagram
[Diagram showing Customer Profile Service connected to: Authentication Service, Payment Service, Marketing Database, Event Bus, and Customer Portal UI]

## Boundary Identification

### Internal Components
- Profile Storage Module
- Profile Validation Engine
- Customer Preferences Manager
- Notification Handler
- API Gateway

### External Dependencies
| Dependency | Type | Direction | Protocol/Format | Criticality | Owner |
|------------|------|-----------|----------------|-------------|-------|
| Authentication Service | Service | Inbound/Outbound | REST/OAuth2 | High | Security Team |
| Payment Service | Service | Outbound | gRPC | Medium | Payments Team |
| Marketing Database | Database | Outbound | JDBC | Low | Marketing Team |
| Event Bus | Message System | Outbound | Kafka/JSON | High | Platform Team |
| Customer Portal UI | Frontend | Inbound | REST/GraphQL | High | Web Team |

## Boundary Interfaces

### Interface 1: Authentication Integration
- **Purpose:** Verify user identity and retrieve authentication context
- **Type:** Synchronous
- **Flow Direction:** Outbound
- **Protocol Details:** REST API calls with OAuth2 tokens
- **Data Formats:** JSON requests/responses
- **Rate/Volume:** ~500 calls/second peak
- **SLA Requirements:** <100ms response time, 99.99% availability
- **Error Handling:** Cache last known good auth state, fail closed on errors
- **Authentication/Authorization:** Mutual TLS, service account credentials
- **Resilience Patterns:** Circuit breaker with 5s timeout, exponential backoff retry

### Interface 2: Event Publication
- **Purpose:** Publish customer profile change events
- **Type:** Asynchronous
- **Flow Direction:** Outbound
- **Protocol Details:** Kafka topics with exactly-once semantics
- **Data Formats:** JSON events with envelope metadata
- **Rate/Volume:** ~100 events/second average, 1000/second peak
- **SLA Requirements:** <500ms max latency to event publication
- **Error Handling:** Dead letter queue for failed publications
- **Authentication/Authorization:** SASL/SCRAM authentication with ACLs
- **Resilience Patterns:** Local buffering, batch publications

## Boundary Concerns

### Data Ownership & Consistency
- Customer Profile Service is the system of record for core profile data
- Marketing preferences sync to Marketing Database with eventual consistency
- Payment information is tokenized, with actual payment data owned by Payment Service
- Cache invalidation messages published on profile updates

### Security Considerations
- PII data is encrypted at rest with field-level encryption
- All boundary crossings require authentication
- GDPR compliance requires data access logging
- Sensitive data removed from logs and events

### Failure Modes
- Authentication Service failures: Use cached credentials for up to 5 minutes
- Payment Service failures: Queue payment verification for later processing
- Event Bus failures: Buffer events locally, apply backpressure if buffer exceeds 10,000 events
- Marketing Database unavailable: Operate in degraded mode with sync happening via batch process

### Monitoring & Observability
- OpenTelemetry tracing across all service boundaries
- Latency histograms for all synchronous interfaces
- Event processing lag monitored via Kafka consumer group offsets
- Structured logging with correlation IDs passed between services

### Deployment & Versioning
- REST APIs versioned in URL path (/v1/, /v2/)
- Event schema evolution follows compatibility rules with registry enforcement
- Changes to interfaces require advance notice and grace period for consumers
- Canary deployments with traffic shifting used for major version changes
``` 