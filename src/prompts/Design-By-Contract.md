# Design By Contract Prompt

## Overview
This prompt helps apply Design by Contract (DbC) principles to software interfaces, functions, or modules by explicitly defining preconditions, postconditions, and invariants, enhancing reliability regardless of programming language.

## Instructions
When designing critical interfaces or functions, use this structured approach to define contracts:

```
# Design By Contract: [INTERFACE/FUNCTION/MODULE NAME]

## Interface Definition
```[language]
[Interface signature/function declaration]
```

## Contract Specification

### Preconditions
- Conditions that must be true before the function/method is called
- Responsibility of the caller

1. [Precondition 1]
2. [Precondition 2]
...

### Postconditions
- Conditions that will be true after the function/method completes successfully
- Responsibility of the function/method implementation

1. [Postcondition 1]
2. [Postcondition 2]
...

### Invariants
- Conditions that remain true throughout execution
- Must be maintained before and after execution

1. [Invariant 1]
2. [Invariant 2]
...

## Error Handling & Contract Violations
- [Describe how precondition violations will be handled]
- [Describe how postcondition violations will be detected]
- [Describe how invariant violations will be monitored]

## Contract Verification Strategy
- [Static verification methods]
- [Dynamic verification methods]
- [Testing approaches]

## Performance Implications
- [Document any performance considerations related to contract enforcement]
```

## When to Apply This Prompt
- When designing new public APIs or interfaces
- For critical functions with complex input/output requirements
- For components with strict reliability requirements
- When documenting existing interfaces with implicit contracts
- For interfaces between different teams or systems
- When replacing or refactoring safety-critical code
- For defining module boundaries in large systems

## Example Application

```
# Design By Contract: AccountTransferService.transferFunds()

## Interface Definition
```java
/**
 * Transfers funds between two accounts
 * @param sourceAccountId ID of the source account
 * @param targetAccountId ID of the target account
 * @param amount Amount to transfer
 * @param currency Currency code of the transfer
 * @return Transaction reference ID
 * @throws InsufficientFundsException If source account has insufficient funds
 * @throws AccountNotFoundException If either account doesn't exist
 * @throws InvalidAmountException If amount is negative or zero
 */
public String transferFunds(
    String sourceAccountId, 
    String targetAccountId, 
    BigDecimal amount, 
    String currency
) throws InsufficientFundsException, AccountNotFoundException, InvalidAmountException;
```

## Contract Specification

### Preconditions
1. Both sourceAccountId and targetAccountId must be non-null and valid account identifiers
2. sourceAccountId and targetAccountId must refer to different accounts
3. amount must be greater than zero
4. currency must be a valid ISO 4217 currency code
5. The source account must have sufficient available balance >= amount
6. Both accounts must be in active status (not blocked, closed, or suspended)
7. The transaction must not exceed daily/transaction limits for either account

### Postconditions
1. Source account balance decreased by exactly the transfer amount
2. Target account balance increased by exactly the transfer amount
3. A transaction record is created with status "COMPLETED"
4. A valid transaction reference ID is returned
5. No other account balances are affected
6. Both accounts remain in active status

### Invariants
1. Total sum of all account balances in the system remains constant
2. All transactions have corresponding balanced debit and credit entries
3. Account status transitions follow the defined state machine
4. All monetary values maintain exact decimal precision without rounding errors

## Error Handling & Contract Violations
- Precondition violations throw specific typed exceptions (documented in interface)
- Postcondition violations trigger system alerts and automatic transaction reversal
- Invariant violations trigger system shutdown in critical scenarios and alert operations
- All contract violations are logged with full context for audit purposes

## Contract Verification Strategy
- Static verification: Custom static analysis rules verify null checks and amount validation
- Dynamic verification: AspectJ aspects verify pre/postconditions in non-production environments
- Testing: Specialized test suite verifies each contract condition with positive and negative tests
- Production monitoring: Balance reconciliation jobs verify system-wide invariants

## Performance Implications
- Precondition checks add approximately 0.5ms to each transaction
- Database constraints enforce key invariants without additional application code
- Contract verification is conditionally compiled in production builds for performance
- Critical path postconditions are always enforced, secondary conditions can be configured
``` 