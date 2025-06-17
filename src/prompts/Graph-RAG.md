# Graph-RAG Programming Prompt

When solving programming tasks, please utilize a Graph-based Retrieval-Augmented Generation approach. This method extends traditional RAG by structuring retrieved information as a knowledge graph that captures relationships between code entities, dependencies, and architectural components before generating solutions.

## Graph-RAG Methodology

### 1. Entity and Relationship Extraction
- Identify key entities in the codebase: classes, functions, modules, APIs, data structures
- Map relationships between these entities: calls, inherits from, imports, depends on, creates
- Note architectural patterns: MVC components, services, repositories, event handlers
- Recognize cross-cutting concerns: authentication, logging, error handling, configuration

### 2. Knowledge Graph Construction
- Mentally construct a graph representation of the codebase's structure
- Place identified entities as nodes in the graph
- Connect related entities with labeled edges describing their relationships
- Identify clusters or subgraphs representing functional components or domains
- Recognize central/important nodes with many connections

### 3. Contextual Traversal and Analysis
- Locate where the current task fits within this knowledge graph
- Identify closely connected entities that may be affected by changes
- Traverse the graph to understand dependencies and impact pathways
- Recognize patterns in how similar features are implemented across the graph
- Determine which existing parts of the graph your solution should connect to

### 4. Graph-Aware Solution Generation
- Generate solutions that respect the existing graph structure
- Maintain consistency with established entity relationships
- Follow the same architectural patterns evident in the knowledge graph
- Consider how your solution creates new nodes or edges in the graph
- Ensure new components integrate properly with the existing structure

### 5. Relationship-Based Documentation
- Document how your solution relates to existing entities in the knowledge graph
- Explain changes or additions to the graph structure
- Highlight important connections and dependencies
- Provide context about where your solution fits in the overall architecture

## Implementation Guidelines

- Frame your solution in terms of its position and relationships within the larger system
- Consider both direct and indirect dependencies when designing your solution
- Respect architectural boundaries evident in the graph structure
- Explicitly note when your solution crosses architectural boundaries
- Follow established patterns for similar relationships elsewhere in the graph
- If creating new entity types, explain how they relate to the existing graph structure

This approach results in solutions that are more holistically integrated with the existing codebase, respecting architectural boundaries and established patterns while creating properly connected new components.