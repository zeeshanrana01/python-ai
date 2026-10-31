# Hallucination Hunt — Lesson 2.7

## Tests

- [ ] Normal order extracted correctly.
- [ ] Multiple items and quantities extracted correctly.
- [ ] Off-menu pizza flagged and excluded from the total.
- [ ] Delivery question identified as a non-order.
- [ ] Roman Urdu order extracted correctly.
- [ ] Invented customer name flagged.
- [ ] Missing customer name marked as unverified.
- [ ] Invalid JSON handled without crashing.
- [ ] Invalid quantities rejected.
- [ ] Python-calculated totals verified manually.

## Findings

For each failed test, record:

1. The input message.
2. What the extractor did incorrectly.
3. The change made to fix it.
4. The result after retesting.

## Conclusion

The model extracts information, but Python validates the data
and calculates totals using the trusted menu prices.
