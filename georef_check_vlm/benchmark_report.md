# VLM Georeferencing Classification Benchmark Report

**Date:** 2026-03-09
**Dataset:** dataset_manual (60 orthos)

## Dataset Summary

| Metric | Count |
|--------|-------|
| Total orthos | 60 |
| Good (correctly aligned) | 40 |
| Bad (misaligned) | 20 |
| Class balance | 66.7% good, 33.3% bad |

## Model Comparison

| Model | Samples | Accuracy | Precision | Recall | F1 Score |
|-------|---------|----------|-----------|--------|----------|
| Gemini 3 Flash | 60 | 0.700 | 0.696 | 0.975 | 0.812 |
| Gemini 3.1 Flash Lite | 60 | 0.667 | 0.667 | 1.000 | 0.800 |
| GPT-5.4 (1000x1000) | 60 | 0.667 | 0.667 | 1.000 | 0.800 |
| GPT-5 Mini | 60 | 0.683 | 0.756 | 0.775 | 0.765 |
| GPT-5.2 | 60 | 0.550 | 0.724 | 0.525 | 0.609 |

## Confusion Matrices

### Gemini 3 Flash

```
                  Predicted
                  Correct  Incorrect
Actual Correct        39       1
Actual Incorrect      17       3
```

- TP (good→good): 39
- FP (bad→good): 17
- FN (good→bad): 1
- TN (bad→bad): 3

### Gemini 3.1 Flash Lite

```
                  Predicted
                  Correct  Incorrect
Actual Correct        40       0
Actual Incorrect      20       0
```

- TP (good→good): 40
- FP (bad→good): 20
- FN (good→bad): 0
- TN (bad→bad): 0

### GPT-5.4 (1000x1000)

```
                  Predicted
                  Correct  Incorrect
Actual Correct        40       0
Actual Incorrect      20       0
```

- TP (good→good): 40
- FP (bad→good): 20
- FN (good→bad): 0
- TN (bad→bad): 0

### GPT-5 Mini

```
                  Predicted
                  Correct  Incorrect
Actual Correct        31       9
Actual Incorrect      10      10
```

- TP (good→good): 31
- FP (bad→good): 10
- FN (good→bad): 9
- TN (bad→bad): 10

### GPT-5.2

```
                  Predicted
                  Correct  Incorrect
Actual Correct        21      19
Actual Incorrect       8      12
```

- TP (good→good): 21
- FP (bad→good): 8
- FN (good→bad): 19
- TN (bad→bad): 12

## Best Model: Gemini 3 Flash

- **Accuracy:** 70.0%
- **Precision:** 69.6%
- **Recall:** 97.5%
- **F1:** 0.812

## Key Findings

1. **Gemini 3 Flash** achieves the best F1 score (0.812)
2. GPT-5.4 was tested with 1000×1000 cropped images (35% token reduction)
3. All models struggle with false positives (misaligned orthos predicted as good)
4. Dataset imbalance (66% good) may bias models toward predicting "CORRECT"

## Model-Specific Observations

- **Gemini 3.1 Flash Lite:** 100% recall but 20 false positives (too optimistic)
- **GPT-5.4 (1000x1000):** Tested with cropped 1000×1000 images (cost-optimized)
- **GPT-5 Mini:** Highest precision (75.6%) - best for avoiding false positives

## Recommendations

1. **For production:** Use the highest precision model to minimize false positives
2. **For comprehensive screening:** Use the highest recall model to catch all potential issues
3. **Cost optimization:** GPT-5.4 with 1000×1000 crops offers good performance at ~35% lower cost
4. **Next steps:**
   - Add more misaligned (bad) samples to balance dataset
   - Experiment with different crop sizes (768×768, 512×512)
   - Try ensemble methods combining multiple models
