# Test Directory Standard for Portal Segurado

## IMPORTANT RULE
All tests in the Portal Segurado project **MUST** be stored in the `src/Tests` directory.

## Purpose
This rule standardizes the location of all test files in the Portal Segurado project, ensuring consistency and facilitating test discovery, execution, and maintenance.

## Directory Structure
```
src/Tests/
├── unit/                 # Unit tests
│   ├── components/       # Tests for Vue components
│   ├── services/         # Tests for service modules
│   └── store/            # Tests for Vuex store modules
├── integration/          # Integration tests
└── e2e/                  # End-to-end tests
```

## File Naming
- All test files must use the `.spec.js` extension
- Test filenames should match the name of the file being tested
- Examples:
  - `PolicyCard.vue` → `PolicyCard.spec.js`
  - `policyService.js` → `policyService.spec.js`
  - `policy.module.js` → `policy.module.spec.js`

## Implementation
When creating a new test, always place it in the appropriate subdirectory within the `src/Tests` directory. This structure helps maintain organization as the codebase grows and makes it easier to run specific types of tests.

## Benefits
- Consistent test organization
- Simplified test discovery
- Clear separation of test types
- Easier CI/CD integration
- Improved test maintenance

## Test Execution
Test scripts in `package.json` should be configured to target this directory structure:
```json
{
  "scripts": {
    "test": "jest src/Tests",
    "test:unit": "jest src/Tests/unit",
    "test:integration": "jest src/Tests/integration",
    "test:e2e": "cypress run"
  }
}
```

## Migration Guide
If you have tests located outside the `src/Tests` directory:

1. Create the appropriate subdirectory in `src/Tests`
2. Move the test files to the new location
3. Update any import paths in the test files
4. Verify tests still pass after migration

## CI/CD Integration
Continuous Integration pipelines should be configured to run tests from the `src/Tests` directory. 