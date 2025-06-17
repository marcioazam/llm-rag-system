# Constraint-Solver Protocol Framework

When responding to my programming problems that involve complex constraints, optimization challenges, or logical requirements, I want you to implement a Constraint-Solver Protocol framework. This approach formalizes problem-solving through systematic constraint modeling and solution validation.

## Process to follow:

1. **Problem Formalization**:
   - Translate my programming problem into a formal constraint satisfaction problem (CSP)
   - Explicitly identify:
     - Variables and their domains
     - Constraints (hard requirements that must be satisfied)
     - Objective function (if optimization is needed)
   - Note any assumptions made during this formalization

2. **Constraint Modeling**:
   - Express each constraint in precise mathematical or logical notation
   - Categorize constraints as:
     - Equality constraints (x = y)
     - Inequality constraints (x ≤ y)
     - Logical constraints (if A then B)
     - Domain constraints (x ∈ {1, 2, 3})
   - Identify constraint types (linear, non-linear, boolean, etc.)

3. **Solution Strategy Selection**:
   - Select an appropriate algorithmic approach based on constraint types:
     - SAT (Boolean Satisfiability) for boolean constraints
     - SMT (Satisfiability Modulo Theories) for mixed constraint types
     - Linear Programming for linear constraints and objective functions
     - Dynamic Programming for optimal substructure problems
     - Backtracking/Search for general CSPs
   - Justify your choice of solution strategy

4. **Solver Implementation**:
   - Implement the selected solution strategy in code
   - Include any necessary helper functions or data structures
   - Apply optimization techniques (pruning, propagation, heuristics)
   - Document complexity analysis (time and space)

5. **Solution Verification**:
   - Verify that all constraints are satisfied by your solution
   - Trace through concrete examples/test cases
   - Prove correctness or optimality where possible
   - Identify any relaxations or approximations made

6. **Final Solution and Analysis**:
   - Present the complete solution with well-commented code
   - Explain how the solution satisfies each constraint
   - Analyze limitations, edge cases, and scaling properties
   - Suggest alternative approaches or refinements if applicable

This framework promotes rigorous problem-solving through formal modeling of constraints, systematic solution development, and thorough verification. It's especially valuable for complex logical problems, scheduling, resource allocation, optimization challenges, and other constraint-driven scenarios.