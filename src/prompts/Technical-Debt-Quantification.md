# Technical Debt Quantification Prompt

## Overview
This prompt provides a structured framework for identifying, quantifying, and prioritizing technical debt in software systems, enabling data-driven decisions about remediation efforts.

## Instructions
When assessing technical debt in a codebase or system, follow this systematic approach:

```
# Technical Debt Assessment: [SYSTEM/COMPONENT NAME]

## Debt Inventory

| ID | Description | Category | Location | Estimated Size | Impact | Urgency | Remediation Cost | Technical Risk | Business Risk |
|----|------------|----------|----------|---------------|--------|---------|-----------------|---------------|--------------|
| TD1 | [Brief description] | [Category] | [Files/Components] | [S/M/L] | [1-5] | [1-5] | [Person-days] | [1-5] | [1-5] |

## Categories of Technical Debt
- **Architectural Debt**: Structural issues affecting system design
- **Code Quality Debt**: Issues with code readability, complexity, or duplication
- **Test Debt**: Missing or inadequate tests
- **Documentation Debt**: Missing, outdated, or inadequate documentation
- **Dependency Debt**: Outdated libraries or problematic dependencies
- **Infrastructure Debt**: Issues with deployment, CI/CD, or operational tools
- **Knowledge Debt**: Reliance on specialized knowledge held by few team members
- **Process Debt**: Inefficient development or operational processes

## Quantification Method

### Metrics Used
- [List objective metrics collected (code coverage, complexity, etc.)]
- [Describe how each metric is calculated and interpreted]

### Scoring Criteria

**Impact (1-5):**
1. Minimal - Almost no effect on development or operations
2. Minor - Slightly slows down development or creates occasional issues
3. Moderate - Regularly impedes development or creates recurring issues
4. Significant - Substantially impedes development or reliability
5. Critical - Severely impairs development productivity or system stability

**Urgency (1-5):**
1. Low - Can be addressed whenever convenient
2. Normal - Should be addressed in the next few months
3. High - Should be addressed in the coming weeks
4. Very High - Should be addressed in current sprint/cycle
5. Critical - Requires immediate attention

**Technical Risk (1-5):**
1. Minimal - Unlikely to cause technical issues
2. Low - May cause minor, isolated issues
3. Moderate - Could lead to significant localized problems
4. High - Likely to cause system-wide issues
5. Critical - Expected to cause catastrophic system failure

**Business Risk (1-5):**
1. Minimal - No meaningful business impact
2. Low - Minimal impact on business metrics
3. Moderate - Noticeable impact on some business metrics
4. High - Significant impact on key business metrics
5. Critical - Threatens business viability or key objectives

### Debt Score Calculation
Debt Score = (Impact × 0.3) + (Urgency × 0.2) + (Technical Risk × 0.25) + (Business Risk × 0.25)

## High-Priority Debt Items (Score > 3.5)

### TD1: [Item Name]
- **Detailed Description**: [Comprehensive explanation of the debt]
- **Root Causes**: [What led to this debt accumulation]
- **Consequences**: [Current and projected impacts if not addressed]
- **Affected Stakeholders**: [Teams/processes impacted]
- **Remediation Plan**:
  1. [Step 1]
  2. [Step 2]
  3. [Step 3]
- **Required Resources**: [People, time, tools needed]
- **Expected Benefits**: [Quantifiable improvements after remediation]
- **Verification Approach**: [How to confirm successful remediation]

### TD2: [Item Name]
[...repeat for each high-priority item...]

## Overall Technical Debt Assessment

### Debt Distribution
[Graph or table showing debt distribution by category]

### Trend Analysis
[Description of how technical debt has changed over time]

### Capacity Allocation
- Current remediation capacity: [X person-days/sprint]
- Recommended allocation: [Y% of development capacity]

## Remediation Strategy

### Short-term Actions (Next Sprint)
1. [Action 1]
2. [Action 2]

### Medium-term Actions (Next Quarter)
1. [Action 1]
2. [Action 2]

### Long-term Actions (Next Year)
1. [Action 1]
2. [Action 2]

### Prevention Measures
1. [Process change 1]
2. [Process change 2]
```

## When to Apply This Prompt
- During technical debt-focused sprints or review cycles
- Before planning major system rework or upgrades
- When onboarding new team members to understand system challenges
- As part of regular system health assessments
- When experiencing increasing development friction
- During team retrospectives focused on technical challenges
- When preparing budget requests for technical improvements

## Example Application

```
# Technical Debt Assessment: Order Processing Service

## Debt Inventory

| ID | Description | Category | Location | Estimated Size | Impact | Urgency | Remediation Cost | Technical Risk | Business Risk |
|----|------------|----------|----------|---------------|--------|---------|-----------------|---------------|--------------|
| TD1 | Inconsistent error handling patterns | Code Quality | src/services/* | M | 4 | 3 | 10 person-days | 4 | 3 |
| TD2 | MongoDB still using deprecated driver (v3.2) | Dependency | src/data/mongo-client.js | S | 3 | 5 | 5 person-days | 5 | 4 |
| TD3 | Missing integration tests for payment processor | Test | tests/integration/ | M | 3 | 4 | 8 person-days | 4 | 5 |
| TD4 | Hardcoded configuration values | Code Quality | src/config/* | S | 2 | 3 | 3 person-days | 3 | 2 |
| TD5 | No documentation for disaster recovery procedures | Documentation | N/A | M | 3 | 4 | 7 person-days | 2 | 5 |
| TD6 | Monolithic architecture limiting scaling | Architectural | Overall system | XL | 5 | 3 | 90 person-days | 4 | 4 |

## Categories of Technical Debt
- **Architectural Debt**: Structural issues affecting system design
- **Code Quality Debt**: Issues with code readability, complexity, or duplication
- **Test Debt**: Missing or inadequate tests
- **Documentation Debt**: Missing, outdated, or inadequate documentation
- **Dependency Debt**: Outdated libraries or problematic dependencies
- **Infrastructure Debt**: Issues with deployment, CI/CD, or operational tools
- **Knowledge Debt**: Reliance on specialized knowledge held by few team members
- **Process Debt**: Inefficient development or operational processes

## Quantification Method

### Metrics Used
- **Code Complexity**: Measured using Cyclomatic Complexity (CC) with SonarQube
- **Code Duplication**: Percentage of duplicated lines measured by SonarQube
- **Test Coverage**: Statement and branch coverage using Istanbul
- **Dependency Freshness**: Average version lag behind current releases
- **Issue Tracking**: Number of technical debt items in backlog
- **Developer Survey**: Quarterly assessment of perceived technical constraints

### Scoring Criteria
[Standard criteria applied as described in template]

### Debt Score Calculation
Debt Score = (Impact × 0.3) + (Urgency × 0.2) + (Technical Risk × 0.25) + (Business Risk × 0.25)

## High-Priority Debt Items (Score > 3.5)

### TD2: MongoDB Deprecated Driver
- **Detailed Description**: The system uses MongoDB driver v3.2 which reached end-of-life 18 months ago. It lacks support for newer MongoDB features, has known security vulnerabilities, and will not receive any updates.
- **Root Causes**: Driver update was postponed multiple times due to feature prioritization and concerns about API changes.
- **Consequences**: 
  - Security vulnerabilities in the driver expose us to potential data breaches
  - Unable to use MongoDB 4.x features like transactions
  - Performance optimizations in newer drivers unavailable
  - Increasing incompatibility with other updated dependencies
- **Affected Stakeholders**: Development team, Security team, Operations
- **Remediation Plan**:
  1. Audit current driver usage patterns across codebase
  2. Document API differences between v3.2 and current version
  3. Create adapter layer to minimize code changes
  4. Update driver and adapter implementation
  5. Comprehensive testing of all database operations
  6. Staged rollout starting with development environment
- **Required Resources**: 5 person-days (2 senior, 3 mid-level)
- **Expected Benefits**: 
  - Elimination of security vulnerabilities
  - 15% performance improvement based on benchmarks
  - Ability to use MongoDB transactions in future features
- **Verification Approach**: 
  - Complete test suite passing
  - Load testing showing equal or better performance
  - Security scan showing vulnerability resolved

### TD3: Missing Integration Tests for Payment Processor
- **Detailed Description**: The payment processing module has only unit tests that mock the payment gateway. We lack true integration tests that validate the complete payment flow with the actual gateway test environment.
- **Root Causes**: Initial implementation was rushed to meet a deadline, and the development team changed shortly after launch.
- **Consequences**: 
  - Multiple production incidents related to payment gateway interface changes
  - High manual testing burden before releases
  - Reluctance to refactor payment code due to fear of regression
  - Slow incident resolution due to complex debugging
- **Affected Stakeholders**: QA team, On-call engineers, Customers experiencing payment issues
- **Remediation Plan**:
  1. Document current payment flows and edge cases
  2. Set up isolated test environment with gateway test instance
  3. Implement integration test suite covering main flows
  4. Add test cases for known edge cases and past incidents
  5. Configure tests to run in CI/CD pipeline
- **Required Resources**: 8 person-days, gateway test environment costs
- **Expected Benefits**: 
  - 60% reduction in payment-related production incidents
  - 40% reduction in manual testing time before releases
  - Faster identification of gateway-related issues
- **Verification Approach**: 
  - Test suite detecting intentionally introduced issues
  - Regression testing of past incident scenarios

## Overall Technical Debt Assessment

### Debt Distribution
Code Quality: 30%
Test Coverage: 25%
Dependency: 20%
Architecture: 15%
Documentation: 10%

### Trend Analysis
Technical debt has increased approximately 15% over the past six months, primarily in test coverage and documentation categories. The architectural debt score has remained stable but represents the largest single item by remediation cost.

### Capacity Allocation
- Current remediation capacity: 10 person-days/sprint (5% of capacity)
- Recommended allocation: 30 person-days/sprint (15% of capacity)

## Remediation Strategy

### Short-term Actions (Next Sprint)
1. Update MongoDB driver (TD2) - highest urgency + risk score
2. Document disaster recovery procedures (TD5) - critical business risk

### Medium-term Actions (Next Quarter)
1. Implement payment integration tests (TD3)
2. Refactor error handling patterns (TD1)
3. Externalize configuration values (TD4)

### Long-term Actions (Next Year)
1. Begin incremental decomposition of monolithic architecture (TD6)
   - Start with payment processing module as first independent service
   - Define service boundaries and interfaces
   - Implement API gateway pattern

### Prevention Measures
1. Implement "tech debt budget" alongside feature planning
2. Add automated dependency scanning to CI/CD pipeline with alerts
3. Require integration tests for all new features interacting with external systems
4. Monthly tech debt review meeting with rotating ownership
``` 