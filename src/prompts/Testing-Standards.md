# Testing Standards for Portal Segurado

## Overview
This document defines the testing standards and best practices for the Portal Segurado project, ensuring consistent test implementation and comprehensive test coverage across the application.

## Test Directory Structure
All tests should be organized in the `src/Tests` directory following this structure:

```
src/Tests/
├── unit/                 # Unit tests
│   ├── components/       # Tests for Vue components
│   ├── services/         # Tests for service modules
│   └── store/            # Tests for Vuex store modules
├── integration/          # Integration tests
└── e2e/                  # End-to-end tests
```

## Testing Frameworks
- **Unit Testing**: Jest with Vue Test Utils
- **E2E Testing**: Cypress

## Naming Conventions
- Test files should be named with the `.spec.js` extension
- Test filenames should match the name of the module being tested
- Example: `PolicyComponent.vue` → `PolicyComponent.spec.js`

## Test Structure Guidelines

### Component Tests
Component tests should verify:
1. Correct rendering with different props
2. Event emission
3. User interaction handling
4. Conditional rendering
5. Error states and loading states

Example pattern:
```javascript
describe('ComponentName.vue', () => {
  // Setup
  let wrapper;
  
  beforeEach(() => {
    // Mount component with standard props
    wrapper = shallowMount(ComponentName, {
      propsData: { /* standard props */ }
    });
  });
  
  it('renders correctly with default props', () => {
    // Assertions
  });
  
  it('emits expected events when triggered', async () => {
    // Trigger event
    // Assert event was emitted
  });
  
  // Additional tests...
});
```

### Store Tests
Store tests should verify:
1. Initial state
2. Mutations
3. Actions (including API calls)
4. Getters

Example pattern:
```javascript
describe('Store Module', () => {
  // Mock dependencies
  
  describe('Mutations', () => {
    // Test mutations
  });
  
  describe('Actions', () => {
    // Test actions
  });
  
  describe('Getters', () => {
    // Test getters
  });
});
```

### Service Tests
Service tests should verify:
1. Correct API call parameters
2. Response handling
3. Error handling
4. Data transformation

Example pattern:
```javascript
describe('Service Module', () => {
  // Mock dependencies
  
  beforeEach(() => {
    // Reset mocks
  });
  
  describe('methodName', () => {
    it('calls API with correct parameters', async () => {
      // Arrange
      // Act
      // Assert
    });
    
    it('handles errors appropriately', async () => {
      // Arrange
      // Act
      // Assert
    });
  });
});
```

## Best Practices

### General Testing Principles
1. Tests should be independent of each other
2. Use descriptive test names (Following "when/should" or "given/when/then" patterns)
3. Keep tests focused and small
4. Follow the AAA pattern (Arrange, Act, Assert)
5. Mock external dependencies
6. Don't test implementation details, test behavior
7. Don't modify global state

### Mocking
1. Mock external dependencies and services
2. Use Jest's mock functions (`jest.fn()`, `jest.mock()`)
3. Reset mocks between tests (`jest.clearAllMocks()`)
4. Be explicit about mock return values

### Testing API Calls
1. Mock API calls, don't make real requests in tests
2. Test both success and error scenarios
3. Verify correct parameters are passed to API methods
4. Test response handling logic

### Test Coverage Requirements
1. Minimum 80% code coverage for critical components
2. All business logic should have test coverage
3. All user-facing components should have tests for rendering and interaction
4. Critical user flows should have integration or E2E tests

## Running Tests
```bash
# Run all tests
npm run test

# Run unit tests
npm run test:unit

# Run E2E tests
npm run test:e2e
```

## Continuous Integration
Tests should be run:
1. Before merging changes
2. In the CI/CD pipeline
3. Tests must pass before deployment

## Additional Resources
- [Jest Documentation](https://jestjs.io/docs/getting-started)
- [Vue Test Utils Documentation](https://vue-test-utils.vuejs.org/)
- [Cypress Documentation](https://docs.cypress.io/)

# Testing Standards Prompt

## Objetivo
Definir guidelines de teste para o projeto (níveis, nomenclatura, cobertura mínima, boas práticas).

## Conteúdo
1. **Piramide de Testes**
   - Unitário (rápido, isolado) – 70%
   - Integração (serviços, banco) – 20%
   - End-to-End/UI – 10%
2. **Nomenclatura**
   - `test_<unit>_<scenario>_<expected>.py`
3. **Cobertura**
   - Alvo mínimo 85% branches; blocos críticos 100%.
4. **Boas Práticas**
   - Fixtures reutilizáveis
   - Mock externo apenas se necessário
   - Sem dependência de ordem
5. **Ferramentas**
   - Python → PyTest + coverage
   - JS → Jest + ts-jest
6. **CI Requirements**
   - Falhar build se cobertura < threshold.

## Placeholders
- {{language}}
- {{framework}}

---
Use este template quando definindo "padrões de teste" ou criando guia QA. 