# Risk Assessment Framework Prompt

## Overview
This prompt provides a structured approach to identifying, assessing, and mitigating technical risks in software systems or architecture proposals, enabling better-informed engineering decisions.

## Instructions
When evaluating technical approaches or identifying risks in a system, apply this systematic risk assessment framework:

```
# Technical Risk Assessment: [SYSTEM/COMPONENT NAME]

## Risk Identification & Analysis

| ID | Risk Description | Risk Category | Likelihood (1-5) | Impact (1-5) | Risk Score | Mitigation Strategy | Owner | Status |
|----|-----------------|---------------|-----------------|-------------|------------|---------------------|-------|--------|
| R1 | [Description] | [Category] | [1-5] | [1-5] | [L*I] | [Strategy] | [Owner] | [Open/Monitoring/Closed] |

## Risk Categories
- Performance/Scalability
- Security/Data Protection
- Reliability/Availability
- Maintainability/Technical Debt
- Integration/Compatibility
- Operational/DevOps
- Regulatory/Compliance
- Resource/Skills Gap

## Likelihood Scale
1. Very Unlikely (0-20% probability)
2. Unlikely (21-40% probability)
3. Possible (41-60% probability)
4. Likely (61-80% probability)
5. Very Likely (81-100% probability)

## Impact Scale
1. Minimal (negligible effect on system/business)
2. Minor (system degrades but functions)
3. Moderate (significant performance/functionality issues)
4. Major (critical functionality unavailable)
5. Severe (complete system failure/security breach)

## Critical Risks (Score ≥ 15)
[List and provide detailed analysis of all high-priority risks]

## Monitoring & Review Strategy
[Describe how risks will be monitored over time and review frequency]

## Risk Response Plan
[Outline the decision-making process for responding to risks if/when they materialize]
```

## When to Apply This Prompt
- Before beginning major architectural changes
- When evaluating new technology adoption
- During system design reviews
- Before critical deployments
- When troubleshooting recurring issues
- For periodic system health assessments
- When planning for scale or increased load

## Example Application

```
# Technical Risk Assessment: Payment Processing Microservice

## Risk Identification & Analysis

| ID | Risk Description | Risk Category | Likelihood (1-5) | Impact (1-5) | Risk Score | Mitigation Strategy | Owner | Status |
|----|-----------------|---------------|-----------------|-------------|------------|---------------------|-------|--------|
| R1 | Third-party payment gateway unavailability | Reliability | 3 | 5 | 15 | Implement circuit breaker pattern with fallback to secondary provider | Team Alpha | Open |
| R2 | Database connection pool exhaustion during peak loads | Performance | 4 | 4 | 16 | Increase pool size, add connection timeout handling, implement request throttling | Team Alpha | In Progress |
| R3 | PCI compliance violations in logs | Compliance | 2 | 5 | 10 | Implement PII filtering in logs, regular compliance scanning | Security Team | Monitoring |
| R4 | Transaction record inconsistency during partial failures | Data Integrity | 3 | 4 | 12 | Implement distributed transactions with saga pattern | Team Alpha | Open |
| R5 | Expertise gap in Kafka stream processing | Resource | 4 | 3 | 12 | Training program for team, hire consultant for initial setup | Management | In Progress |

## Risk Categories
- Performance/Scalability
- Security/Data Protection
- Reliability/Availability
- Maintainability/Technical Debt
- Integration/Compatibility
- Operational/DevOps
- Regulatory/Compliance
- Resource/Skills Gap

## Likelihood Scale
1. Very Unlikely (0-20% probability)
2. Unlikely (21-40% probability)
3. Possible (41-60% probability)
4. Likely (61-80% probability)
5. Very Likely (81-100% probability)

## Impact Scale
1. Minimal (negligible effect on system/business)
2. Minor (system degrades but functions)
3. Moderate (significant performance/functionality issues)
4. Major (critical functionality unavailable)
5. Severe (complete system failure/security breach)

## Critical Risks (Score ≥ 15)

### R1: Third-party payment gateway unavailability
The payment gateway has had 99.95% uptime historically, but any downtime directly prevents customer purchases. We'll implement a circuit breaker pattern with automatic fallback to a secondary provider. We'll also cache payment token validation where possible to reduce dependency on external calls.

### R2: Database connection pool exhaustion during peak loads
Our load tests show we may exhaust connections during Black Friday traffic levels. We're increasing the connection pool size from 50 to 100, adding better timeout handling, and implementing adaptive request throttling based on system load.

## Monitoring & Review Strategy
- Daily automated monitoring of risk indicators via dashboard
- Weekly review of emerging risks in team standups
- Monthly comprehensive risk reassessment
- Post-incident analysis to identify previously unknown risks

## Risk Response Plan
- Risks with score 15+ require immediate mitigation planning
- Risks that materialize trigger incident response protocol
- Risk materialization causes automatic reassessment of similar risks
- Post-mitigation verification testing required before closing risks 