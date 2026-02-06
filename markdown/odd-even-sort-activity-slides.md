---
marp: true
theme: default
paginate: true
---

# Odd-Even Sort
## Classroom Warm-Up Activity

A hands-on algorithm demonstration for ~45 students.

---

## The Algorithm

**Odd-Even Transposition Sort** is a parallel variant of Bubble Sort where adjacent pairs compare and swap simultaneously.

---

## How to Run It

1. Students line up in a row, numbered 1–45
2. Instructor shouts **"Odd!"** → pairs (1,2), (3,4), (5,6), ... compare names alphabetically and swap if out of order
3. Instructor shouts **"Even!"** → pairs (2,3), (4,5), (6,7), ... compare and swap
4. Alternate odd/even rounds until a full cycle has no swaps

---

## Completion Check

Have students raise their hand if they swapped.

When a full odd+even cycle has zero hands raised, the line is sorted.

---

## Expected Duration

With 45 students, expect roughly **20–25 rounds** in the worst case.

The algorithm is guaranteed to complete in at most *n* phases.

---

## Why This Works for a Classroom

| Advantage | Explanation |
|-----------|-------------|
| Everyone participates | No standing around waiting for sequential comparisons |
| Physically intuitive | Students experience the algorithm, not just see pseudocode |
| Demonstrates parallelism | O(n) parallel steps vs O(n²) sequential |
| Natural icebreaker | Students interact with neighbors |

---

## Teaching Moments

- **Parallel computing:** This is how sorting networks operate
- **GPU sorting:** Similar principles apply to parallel GPU implementations
- **Complexity:** Parallelism reduces time complexity dramatically
- **Comparison to Bubble Sort:** Same swaps, but concurrent execution

---

## Algorithm Properties

- **Time complexity:** O(n) parallel phases, O(n²) total comparisons
- **Space complexity:** O(1) — in-place sorting
- **Stable:** Yes — equal elements maintain relative order
- **Parallel efficiency:** Each phase can be executed in O(1) with n/2 processors
