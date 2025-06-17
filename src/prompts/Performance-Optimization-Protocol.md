# Performance Optimization Protocol Prompt

## Objetivo
Orientar a análise de desempenho e propor otimizações mensuráveis em código, consultas ou infraestrutura.

## Passos
1. **Coleta de Evidências**
   - Analise métricas (CPU, memória, I/O, tempo de resposta).
   - Identifique hotspots (funções lentas, queries pesadas).
2. **Classificação do Gargalo**
   - CPU-bound, I/O-bound, Network, Garbage Collector, Lock Contention.
3. **Hipóteses de Otimização**
   - Liste 2–3 abordagens (ex.: caching, paralelismo, index, batch).
4. **Experimento Controlado**
   - Defina métrica alvo (ms/op, throughput, p95 latency).
   - Plano de benchmark antes/depois.
5. **Implementação Sugerida**
   - Mostre código ou config modificado.
6. **Riscos & Trade-offs**
   - Impacto em legibilidade, custo infra, complexidade.

## Placeholders
- {{profile_report}}
- {{hotspots}}
- {{language}}
- {{framework}}

---
Use quando o usuário pedir "otimizar esta função/SQL" ou apresentar profiler/flamegraph.

## Overview
This prompt provides a systematic approach to identifying, diagnosing, measuring, and resolving performance issues in software systems, using a data-driven methodology applicable across different platforms and languages.

## Instructions
When addressing performance concerns in a system, follow this structured performance optimization protocol:

```
# Performance Optimization Protocol: [SYSTEM/COMPONENT NAME]

## 1. Problem Definition & Goals

### Current Performance Metrics
- [Baseline metrics showing current performance]
- [User-perceived performance measurements]
- [System resource utilization]

### Performance Goals
- [Specific, measurable targets]
- [Percentile requirements (e.g., p95, p99)]
- [Business justification for improvements]

### Constraints
- [Resource limitations]
- [Compatibility requirements]
- [Implementation timeline]
- [Maintenance considerations]

## 2. Performance Profiling

### Test Environment
- [Environment specifications]
- [Test data characteristics]
- [Load generation approach]
- [Monitoring tools deployed]

### Profiling Results
- [CPU profiling results]
- [Memory allocation patterns]
- [I/O operations analysis]
- [Network communication patterns]
- [Database query performance]
- [Rendering/UI performance (if applicable)]
- [Thread/async execution analysis]

### Critical Path Analysis
- [Identification of bottlenecks]
- [Dependencies affecting performance]
- [Hot spots in code execution]
- [Resource contention issues]

## 3. Root Cause Analysis

### Identified Issues
| ID | Description | Category | Impact | Effort to Fix | Priority |
|----|------------|----------|--------|--------------|----------|
| P1 | [Issue description] | [Category] | [High/Med/Low] | [High/Med/Low] | [1-5] |

### Performance Anti-patterns Detected
- [List specific anti-patterns found]
- [Evidence for each pattern]
- [Impact assessment]

## 4. Optimization Strategy

### Quick Wins (Low Effort, High Impact)
1. [Optimization 1]
   - Expected improvement: [Metric]
   - Implementation approach: [Brief description]
2. [Optimization 2]
   - Expected improvement: [Metric]
   - Implementation approach: [Brief description]

### Strategic Improvements (Higher Effort)
1. [Optimization 1]
   - Expected improvement: [Metric]
   - Implementation approach: [Brief description]
   - Prerequisites: [Dependencies or prior work needed]
2. [Optimization 2]
   - Expected improvement: [Metric]
   - Implementation approach: [Brief description]
   - Prerequisites: [Dependencies or prior work needed]

### Experimental Approaches
- [Approaches to test with uncertainty about outcomes]
- [A/B testing strategy]
- [Iterative measurement plan]

## 5. Implementation Plan

### Phase 1: [Timeline]
- [Actions]
- [Expected outcomes]
- [Verification methods]

### Phase 2: [Timeline]
- [Actions]
- [Expected outcomes]
- [Verification methods]

### Rollback Plan
- [Criteria for rollback decision]
- [Rollback procedure]
- [Monitoring during rollout]

## 6. Metrics & Verification

### Key Performance Indicators
- [Primary metrics to track]
- [Instrumentation needs]
- [Success criteria]

### Testing Protocol
- [Load testing approach]
- [Performance regression testing]
- [User experience validation]

## 7. Knowledge Transfer & Documentation

### Code Patterns
- [Document optimized patterns for reuse]
- [Anti-patterns to avoid]

### Performance Monitoring
- [Ongoing monitoring strategy]
- [Alert thresholds]
- [Dashboards]

### Learnings
- [Key insights gained]
- [Unexpected discoveries]
- [Areas for future exploration]
```

## When to Apply This Prompt
- When users report performance issues
- During scaling phases or increasing user load
- Before major releases requiring performance verification
- After significant architectural changes
- During regular performance review cycles
- When adding features with potential performance impact
- When optimizing for new deployment environments

## Example Application

```
# Performance Optimization Protocol: User Dashboard UI Rendering

## 1. Problem Definition & Goals

### Current Performance Metrics
- Initial page load: 4.2s average (3.8s - 5.1s range)
- Time to interactive: 6.3s average
- First contentful paint: 2.1s average
- CPU utilization during rendering: 86% peak
- Memory usage: 230MB average, spikes to 450MB
- Network requests: 48 separate requests, 2.8MB total transfer

### Performance Goals
- Initial page load: <2.5s (50% improvement)
- Time to interactive: <3.0s (55% improvement)
- First contentful paint: <1.2s (40% improvement)
- Reduce CPU utilization to <60% peak
- Memory usage stable at <200MB
- Reduce network requests to <20 with total transfer <1.5MB

### Constraints
- Must maintain support for IE11
- Cannot require users to change settings
- Must maintain all existing functionality
- Implementation must be completed within next sprint (2 weeks)
- No additional server-side infrastructure changes allowed

## 2. Performance Profiling

### Test Environment
- Test Device: Dell XPS 13 (i5, 8GB RAM) - representing median user device
- Network: Throttled to 5Mbps download, 1Mbps upload, 100ms latency
- Browser: Chrome 96, Firefox 95, Safari 15, Edge 96, IE11
- Test Data: Production-equivalent dataset (10k items in dashboard)
- Load Generation: Puppeteer scripts simulating user interactions
- Monitoring: Chrome DevTools, Lighthouse, custom instrumentation

### Profiling Results
- **CPU Profiling**:
  - 42% time spent in JavaScript execution
  - 28% time spent in rendering
  - 18% time spent in parsing JSON responses
  - 12% time spent in other browser activities

- **Memory Analysis**:
  - 120MB in JavaScript objects
  - 45MB in DOM nodes
  - 65MB in images and other media
  - Memory growth of ~5MB/minute during active use indicating potential leak

- **Network Analysis**:
  - 1.2s spent in network requests
  - Largest contentful paint blocked by non-critical CSS
  - 18 render-blocking resources identified
  - Unnecessary data fetched upfront (only 15% used on initial render)

- **Rendering Analysis**:
  - Layout thrashing detected during table updates
  - 245ms spent in unnecessary style recalculations
  - 12 long tasks (>50ms) blocking main thread
  - 8 instances of forced synchronous layout

### Critical Path Analysis
- Dashboard initial render blocked by:
  1. Large vendor.js bundle (1.2MB)
  2. Complete data fetch before any UI rendering
  3. Complex initial state calculations
  4. Non-optimized images
  5. Render-blocking CSS for non-visible components

## 3. Root Cause Analysis

### Identified Issues
| ID | Description | Category | Impact | Effort to Fix | Priority |
|----|------------|----------|--------|--------------|----------|
| P1 | No code splitting for vendor bundles | Bundle Size | High | Medium | 1 |
| P2 | Data fetched eagerly instead of lazily | Network | High | Low | 1 |
| P3 | Images not optimized or responsive | Assets | Medium | Low | 2 |
| P4 | Layout thrashing in table updates | Rendering | High | Medium | 1 |
| P5 | Excessive DOM nodes (>5000 elements) | Structure | Medium | High | 3 |
| P6 | Unoptimized Redux state updates | State Management | Medium | Medium | 2 |
| P7 | Memory leak in chart components | Memory | Medium | Medium | 2 |
| P8 | Critical CSS not inlined | Rendering | Medium | Low | 2 |
| P9 | No component-level memoization | Computation | Low | Medium | 3 |

### Performance Anti-patterns Detected
- **Loads Everything Upfront**: Fetches all data before showing UI
- **Monolithic Bundles**: No code-splitting or lazy-loading
- **Chatty API Pattern**: Multiple small API requests instead of consolidated calls
- **Repeated Computation**: Expensive calculations not memoized
- **Large Component Trees**: Deeply nested component hierarchy causing excess reconciliation
- **Prop Drilling**: Excessive props passed through many levels
- **Unnecessary Renders**: Components re-rendering when data hasn't changed

## 4. Optimization Strategy

### Quick Wins (Low Effort, High Impact)
1. **Implement code splitting and lazy loading**
   - Expected improvement: 40% reduction in initial bundle size
   - Implementation approach: Use dynamic imports for routes and large components
   
2. **Implement lazy data fetching with skeleton UI**
   - Expected improvement: 60% reduction in time to first contentful paint
   - Implementation approach: Show UI immediately, fetch data in background with loading indicators
   
3. **Optimize and compress images**
   - Expected improvement: 500KB reduction in page size
   - Implementation approach: Convert to WebP with PNG fallback, implement responsive sizes
   
4. **Inline critical CSS**
   - Expected improvement: 300ms reduction in render blocking time
   - Implementation approach: Extract and inline styles needed for above-the-fold content

### Strategic Improvements (Higher Effort)
1. **Virtual scrolling for large data tables**
   - Expected improvement: 70% reduction in DOM nodes
   - Implementation approach: Replace static tables with virtual scroll library
   - Prerequisites: Component refactoring, event handler adjustments
   
2. **Implement proper memoization strategy**
   - Expected improvement: 30% reduction in re-renders and calculations
   - Implementation approach: Use React.memo, useMemo, and useCallback strategically
   - Prerequisites: Component profiling to identify unnecessary re-renders
   
3. **Fix memory leaks in chart components**
   - Expected improvement: Stable memory usage, no growth over time
   - Implementation approach: Proper cleanup in useEffect hooks, dispose D3 resources
   - Prerequisites: Detailed memory profiling

### Experimental Approaches
- **Server-side rendering for initial state**
  - A/B test with 10% of users to measure actual improvement
  - Monitor for any negative impacts on server load
  
- **GraphQL adoption to reduce over-fetching**
  - Build proof-of-concept implementation for dashboard data
  - Measure reduction in data transfer and parsing time

## 5. Implementation Plan

### Phase 1: Foundation (Week 1)
- Implement code splitting and lazy loading
- Optimize images and implement responsive loading
- Inline critical CSS
- Add performance monitoring instrumentation
- Expected outcome: 30-40% improvement in initial load metrics

### Phase 2: Enhanced Optimizations (Week 2)
- Implement virtual scrolling for tables
- Add strategic memoization to prevent unnecessary renders
- Fix identified memory leaks
- Expected outcome: Additional 20-30% improvement and stable memory usage

### Phase 3: Experimental (Post-sprint)
- A/B test server-side rendering approach
- Evaluate GraphQL implementation
- Expected outcome: Data to inform next optimization cycle

### Rollback Plan
- Separate feature flags for each optimization
- Automated performance testing before deployment
- Automated rollback if performance degrades by >10% on key metrics
- Preserve old bundle for quick reversion if needed

## 6. Metrics & Verification

### Key Performance Indicators
- Core Web Vitals (LCP, FID, CLS)
- Custom metrics:
  - Time to interactive dashboard
  - Interaction responsiveness (time to update after click)
  - Memory usage stability
  - CPU utilization during interactions

### Testing Protocol
- Automated Lighthouse tests on CI/CD pipeline
- Synthetic testing with realistic user flows
- Real User Monitoring (RUM) with percentile analysis
- A/B performance comparisons using feature flags

## 7. Knowledge Transfer & Documentation

### Code Patterns
- Document virtualization implementation for reuse
- Create memoization decision tree for developers
- Update style guide with performance best practices
- Document bundle size budgets and enforcement

### Performance Monitoring
- New Grafana dashboard for frontend performance
- Alert thresholds for regression detection
- Weekly performance review in team standups

### Learnings
- Identified critical rendering path bottlenecks
- Discovered that eager data loading was more impactful than initially thought
- Virtual DOM reconciliation was not the bottleneck as assumed
- Further research needed on WebWorkers for complex calculations
``` 