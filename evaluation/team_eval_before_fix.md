# Team evaluation, before the reviewer fixes

Run on 9 October 2026 at 12:04 with `python -m scripts.evaluate_team --repeats 2 --pause 3`, gpt-4o, 14 cases, 2 repeats, single mode and team mode, then the 4 simulated failures. The raw JSON for this run was not kept, so this is the summary and run list as the script printed them.

## Summary

| Mode | Runs | Accuracy | Consistency | Mean rounds | Mean messages | Mean latency | Statuses |
|---|---|---|---|---|---|---|---|
| single | 28 | 100% | 100% | 1.0 | 0.0 | 2175 ms | {'answered': 28} |
| team | 28 | 86% | 100% | 1.36 | 3.71 | 4070 ms | {'approved': 22, 'max_rounds': 4, 'approved_after_revision': 2} |

Reviewer fixed a wrong first answer: 0 · broke a right one: 4 · approved a wrong one: 1 · sent back a right one: 6

## Runs that went wrong or took more than one round (team mode)

| case | first answer | final answer | status | rounds | reviewer issues |
|---|---|---|---|---|---|
| T8 Officeworks printer ink | education | other (wrong) | max_rounds | 3 | "Keyword hint suggests 'other' category.", "Incorrect category" |
| T8 Officeworks printer ink | education | other (wrong) | approved_after_revision | 2 | "Keyword hint suggests 'other' category." |
| T11 $410 Coles groceries, unusual? | True | None (wrong), confidence 0.0 | max_rounds | 3 | "Expense is outside the 30-day window." |
| T11 $410 Coles groceries, unusual? | True | None (wrong), confidence 0.0 | max_rounds | 3 | "Expense is outside the 30-day window for checking unusual expenses." |
| T13 $95 dinner, unusual? | True | True | approved_after_revision | 2 | "The average food expense calculation is incorrect." |
| T13 $95 dinner, unusual? | True | True | max_rounds | 3 | "The average food expense calculation is incorrect." (3 times) |

Every other team run (T1 to T7, T9, T10, T12, T14, both repeats) was approved in round 1 with the right answer. Single mode got all 28 right.

## Simulated failures (team mode, case T2 Uber to airport)

| simulate | final answer | status | stop reason | rounds | confidence | latency |
|---|---|---|---|---|---|---|
| force_revise | transport | max_rounds | max_rounds | 3 | 0.5 | 8621 ms |
| reviewer_timeout | transport | unreviewed | reviewer_timeout | 1 | 0.5 | 2640 ms |
| invalid_receiver | transport | unreviewed | invalid_receiver | 1 | 0.5 | 1784 ms |
| bad_message | transport | fallback | bad_content | 1 | 0.3 | 1742 ms |

## What it showed

1. **Failure case 1 (T8).** The reviewer model read the evidence field `keyword_hint: "other"` as the correct category. But "other" from the word list only means no word matched. It talked the classifier out of a right answer in both runs.
2. **Failure case 2 (T11).** The reviewer decided that an expense 40 days old was "outside the 30-day window" and couldn't be judged. The 30 days is only used for the average, so that rule doesn't exist. The two agents argued for 3 rounds and the user got "can't tell" with confidence 0.
3. **T13.** The reviewer kept disputing the maths on a borderline case with no HARD fact behind it. That cost 3 rounds and about 12.8 seconds for an answer that was right from the start.

## Fixes made after this run

1. **Better evidence for the reviewer model:**
   - A keyword hint of "other" is no longer passed on.
   - PASS checks now show the model what the data already confirmed.
   - The prompt says the 30 day window is only for the average.
2. **A soft disagreement only gets one round.** From round 2, only a HARD problem (a fact from the data) can send an answer back. Otherwise the answer is approved with a note.
