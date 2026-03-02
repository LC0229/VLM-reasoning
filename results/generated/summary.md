# Results Summary

## Object-level (COCO)
- Samples: 500
- **baseline**: 72.4%
- **blur**: 75.1%
- **mask**: 74.8%
- **crop**: 78.2%

## Relational (riding vs standing)
- Samples: 120
- **baseline**: 80.0%
- **blur**: 76.7%
- **mask**: 70.8%
- **crop**: 79.2%

## Stability
- **blur**: P(correct after | correct before) = 0.92, P(correct after | wrong before) = 0.15
- **mask**: P(correct after | correct before) = 0.85, P(correct after | wrong before) = 0.12
- **crop**: P(correct after | correct before) = 0.96, P(correct after | wrong before) = 0.18
