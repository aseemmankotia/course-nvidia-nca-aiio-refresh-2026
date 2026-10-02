# NCA-AIIO Exam Simulation & Test-Taking Strategy Cheat Sheet

## Exam Format Essentials
| Metric | Value | Strategy |
|--------|-------|----------|
| Total Questions | 50 | Budget 15 questions margin for hard ones |
| Time Limit | 90 minutes | 1.8 min/question average |
| Passing Score | ~70% (35/50) | You can miss 15 questions - don't panic |
| Delivery | Remote-proctored online | Test environment setup beforehand |
| Format | Single-answer only | No multiple-response questions |

## Domain Weights & Question Distribution
| Domain | Weight | Est. Questions | Study Priority |
|--------|--------|----------------|----------------|
| Infrastructure | 40% | ~20 | HIGH - most questions |
| Essential AI | 38% | ~19 | HIGH - second most |
| Platform | 22% | ~11 | MEDIUM - don't skip |

## Time Management Checkpoints
| Questions Completed | Time Elapsed | Status |
|---------------------|--------------|--------|
| 10 | 18 min | On pace |
| 25 | 45 min | On pace |
| 40 | 72 min | On pace (18 min for review) |
| 45 | 81 min | Behind - fast review only |

## Question Type Distribution & Approach
| Type | % of Exam | Time Budget | Technique |
|------|-----------|-------------|----------|
| Scenario-based | 40% | 2-2.5 min | List requirements → eliminate violations → apply preferences |
| Service selection | 30% | 1.5-2 min | Eliminate wrong service category → compare remaining on specific needs |
| Configuration | 20% | 1-1.5 min | Know specific values/syntax - no partial credit |
| Conceptual | 10% | 1 min | Quick recall - flag if uncertain |

## Elimination Technique (Scenario Questions)
**STEP 1:** Extract ALL hard requirements
- Multi-environment? (hybrid/cloud/on-prem)
- Performance specs? (latency, throughput)
- Compliance needs? (NVAIE certified, regulatory)
- Scale requirements? (GPU count, node count)

**STEP 2:** Eliminate answers missing ANY requirement
- Single-node answer when 64 GPUs needed → eliminate
- No HA feature when HA required → eliminate
- Wrong service category (training GPU for inference) → eliminate

**STEP 3:** Apply soft preferences to remaining options
- Cost optimization → choose simpler/cheaper if equal functionality
- Minimize complexity → choose managed over self-managed
- Future-proof → choose current tech over deprecated

## Common Wrong Answer Patterns to Avoid
| Pattern | Example | Why It's Wrong | How to Spot |
|---------|---------|----------------|-------------|
| Over-engineering | Adding multi-region when not required | Adds cost/complexity without benefit | Answer has features not in requirements |
| Cost-insensitive | Choosing H100 for simple inference | Wasteful when cheaper options work | 'Most expensive' ≠ 'best' unless justified |
| Outdated tech | Recommending V100 for new projects | Deprecated or superseded options | Check if current NVIDIA portfolio |
| Always-latest | Picking H100 because it's newest | Newer ≠ better for all use cases | Cost/inference scenarios favor L40S |

## nvidia-smi Troubleshooting Quick Reference
```
GPU: 15%, Memory: 95% → MEMORY BOTTLENECK (batch size too large, data loading slow)
GPU: 98%, Memory: 30% → COMPUTE BOUND (model complexity, need faster GPU)
GPU: 12%, Memory: 15% → UNDERUTILIZED (scheduling issue, not GPU problem)
Temp: >85C + Power: maxed → THERMAL THROTTLING (cooling insufficient)
Temp: normal + Power: <50% of limit → LOW UTILIZATION (not a problem, expected for light load)
```

## Capacity Planning Formula
**With Overhead/Buffer:**
```
Total Capacity Needed = Base Requirement × (1 + Overhead %)
Units Required = Total Capacity Needed / Per-Unit Capacity
```
Example: 60k req/hr, 20% overhead, 2k req/hr per GPU
- Total needed: 60,000 × 1.2 = 72,000
- GPUs: 72,000 / 2,000 = 36 GPUs

**DON'T** calculate base first then add overhead with rounding - compounds errors.

## Review Strategy (Final 10 Minutes)
1. **Prioritize by question type** - scenario/service selection over conceptual
2. **Use elimination** - remove 2 obviously wrong answers, improve 25%→50% odds
3. **Change answers strategically** - when you can apply technique, not gut feeling
4. **Never leave blank** - no penalty for wrong answers, always guess

## Common Exam Keywords & Their Meaning
| Phrase | Actual Requirement | Watch Out For |
|--------|-------------------|---------------|
| "Cost-effective" | Minimize cost while meeting requirements | Still must meet ALL requirements |
| "High availability" | Redundancy, failover, uptime SLA | Not same as "disaster recovery" |
| "Scalable" | Can grow (usually GPU count) | May need specific architecture (multi-node) |
| "Minimize complexity" | Prefer managed/simpler when equal | Tie-breaker, not primary requirement |
| "Budget not a concern" | Don't eliminate on cost | Doesn't mean pick most expensive |

## Calculator Usage
Provided for:
- Capacity planning (throughput, GPU count)
- Cost estimation (when numbers provided)
- Percentage calculations

NOT needed for:
- Service comparisons (qualitative)
- Architecture decisions (scenario-based)

## Final Tips
- **Flag liberally** - review is fast with elimination technique
- **Scenario questions**: requirements → eliminate → preferences
- **Service selection**: wrong category → wrong tier → right answer
- **Troubleshooting**: analyze GIVEN data, don't pattern-match from practice
- **Capacity**: overhead multiplier BEFORE division
- **Never assume** - only include what question explicitly states
