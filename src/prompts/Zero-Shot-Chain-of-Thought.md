---
description: 
globs: 
alwaysApply: false
---
# Advanced Zero-Shot Chain-of-Thought Framework

## Core Methodology
This framework enables systematic problem decomposition and step-by-step reasoning without requiring prior examples. It produces transparent, traceable solution paths across any technical domain by explicitly articulating each cognitive step from problem parsing to implementation.

## Implementation Process

### Phase 1: Problem Architecture Analysis
- Deconstruct the problem statement into its fundamental components
- Identify explicit and implicit requirements
- Extract key variables, constraints, and expected outcomes
- Map the problem to known theoretical domains and patterns
- Formulate a precise technical specification from the original description
- Identify verification criteria for solution correctness

### Phase 2: Explicit Reasoning Pathway
Articulate a sequential reasoning path that includes:

#### Conceptual Mapping
- Transform problem domain concepts to computational constructs
- Identify appropriate abstractions and representations
- Connect problem terminology to technical implementation elements
- Establish relationships between domain objects
- Map constraints to validation requirements

#### Algorithm & Data Structure Selection
- Evaluate multiple algorithmic approaches systematically
- Analyze space/time tradeoffs for each approach
- Consider implementation complexity vs. performance benefits
- Select data structures with appropriate characteristics for the operations needed
- Justify each selection with explicit technical reasoning
- Identify potential optimization opportunities

#### Edge Case Identification
- Systematically enumerate boundary conditions
- Identify input ranges and special cases
- Analyze potential failure modes and exception scenarios
- Consider resource constraints and limiting behavior
- Document assumptions that might affect solution validity

#### Implementation Strategy
- Outline modular solution components and their interfaces
- Define critical functions and their specifications
- Plan the solution architecture with clear separation of concerns
- Consider error handling strategies
- Establish validation points throughout the solution

### Phase 3: Structured Implementation
Develop the solution with explicit reasoning embedded:

#### Pseudocode Outline
- Create a language-agnostic solution outline
- Highlight key algorithms and their operation
- Document state transitions and transformations
- Include error handling and validation logic
- Provide complexity analysis for critical operations

#### Executable Implementation
- Translate pseudocode to working code in appropriate language
- Apply idiomatic patterns for the target language
- Implement robust error handling and validation
- Include context-explaining comments connecting code to reasoning
- Structure code for readability and maintainability

#### Verification Framework
- Include test cases covering normal operation
- Add validation for identified edge cases
- Implement assertions for critical assumptions
- Provide examples demonstrating solution correctness
- Include runtime complexity analysis

## Response Structure

```
# Problem Analysis
[Precise decomposition of problem statement and requirements]

# Reasoning Pathway
## Step 1: [Specific reasoning step]
[Detailed explanation with technical justification]

## Step 2: [Specific reasoning step]
[Detailed explanation with technical justification]

[Continue with additional steps as needed]

# Solution Architecture
[High-level description of solution approach and components]

# Implementation
```code
// Implementation with embedded reasoning comments
// explaining critical design decisions
```

# Complexity Analysis
- Time Complexity: O(...) because [explanation]
- Space Complexity: O(...) because [explanation]
- Edge Case Handling: [Discussion of special cases]

# Verification
- Test Case 1: [Example with expected output]
- Test Case 2: [Example with expected output]
- [Additional test cases for edge conditions]
```

## Application Guidelines
- Scale reasoning depth to problem complexity
- Externalize all critical thinking steps, even when they seem obvious
- Document decision points where multiple approaches were considered
- Connect theoretical analysis directly to implementation choices
- Explain not just what the solution does, but why specific approaches were chosen

This framework enables rigorous problem-solving with complete transparency of reasoning, supporting both solution correctness and knowledge transfer through explicit articulation of the full cognitive pathway from problem to implementation.

Now, here is my programming task:
[I will describe my specific task here]

