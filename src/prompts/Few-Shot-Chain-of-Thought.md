# Advanced Few-Shot Chain-of-Thought Framework

## Core Methodology
This framework leverages carefully selected examples to establish effective reasoning patterns before tackling new problems. By demonstrating high-quality analytical processes through representative examples, it enables consistent, structured problem-solving across any technical domain with enhanced precision and completeness.

## Implementation Structure

### Calibration Phase: Example-Based Reasoning Patterns
The following examples demonstrate effective reasoning pathways for technical problems. Each illustrates a systematic approach that:
1. Decomposes the problem into core components
2. Exposes the complete cognitive process
3. Connects analytical steps directly to implementation decisions
4. Handles edge cases and complexity considerations explicitly
5. Verifies solution correctness methodically

Pay special attention to:
- The explicit articulation of each reasoning step
- The connection between theoretical analysis and practical implementation
- The systematic identification and handling of edge cases
- The verification procedures to establish solution correctness
- The integration of domain-specific knowledge with general problem-solving approaches

These patterns should be applied and adapted to new problems with similar methodical rigor.

### Example 1: Finding the Maximum Subarray Sum

**Problem:** Given an array of integers, find the contiguous subarray with the largest sum.

**Reasoning Process:**
1. First, I need to understand what makes this problem challenging. The presence of negative numbers means we can't simply include all elements.
   
2. Let me consider possible approaches:
   - Brute force: Check all possible subarrays (O(n²) time)
   - Dynamic programming: Build solution from previous results
   - Kadane's algorithm: A specialized DP approach for this problem (O(n) time)
   
3. Kadane's algorithm is optimal here. It works by:
   - Tracking current_sum (sum of current subarray)
   - Tracking max_sum (maximum sum found so far)
   - For each element, deciding whether to extend previous subarray or start a new one
   
4. Edge cases to consider:
   - Empty array (return 0 or error depending on requirements)
   - All negative numbers (return the largest single element)
   - All zeros (return 0)
   
5. Let me trace through the algorithm with a sample input:
   Input: [-2, 1, -3, 4, -1, 2, 1, -5, 4]
   
   Initialization: current_sum = 0, max_sum = -infinity
   
   Element -2: current_sum = max(0 + (-2), 0) = 0, max_sum = max(-infinity, 0) = 0
   Element 1: current_sum = max(0 + 1, 0) = 1, max_sum = max(0, 1) = 1
   Element -3: current_sum = max(1 + (-3), 0) = 0, max_sum = 1
   Element 4: current_sum = max(0 + 4, 0) = 4, max_sum = max(1, 4) = 4
   Element -1: current_sum = max(4 + (-1), 0) = 3, max_sum = 4
   Element 2: current_sum = max(3 + 2, 0) = 5, max_sum = max(4, 5) = 5
   Element 1: current_sum = max(5 + 1, 0) = 6, max_sum = max(5, 6) = 6
   Element -5: current_sum = max(6 + (-5), 0) = 1, max_sum = 6
   Element 4: current_sum = max(1 + 4, 0) = 5, max_sum = 6
   
   Final result: 6 (subarray [4, -1, 2, 1])

**Implementation:**
```python
def max_subarray_sum(nums):
    if not nums:
        return 0
        
    current_sum = 0
    max_sum = float('-inf')
    
    for num in nums:
        current_sum = max(current_sum + num, 0)
        max_sum = max(max_sum, current_sum)
    
    # Handle all-negative case
    if max_sum == 0 and all(n < 0 for n in nums):
        return max(nums)
        
    return max_sum
```

**Complexity Analysis:**
- Time Complexity: O(n) - we process each element exactly once
- Space Complexity: O(1) - we use only two variables regardless of input size

**Verification:**
- Test with example: [-2, 1, -3, 4, -1, 2, 1, -5, 4] → 6
- Test all negatives: [-1, -2, -3, -4] → -1
- Test empty array: [] → 0

Now, please describe your specific programming task, and I'll help you solve it with a similar step-by-step approach.