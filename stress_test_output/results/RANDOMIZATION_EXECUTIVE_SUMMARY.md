# Randomization Quality Assessment - Executive Summary

## 🚨 Critical Finding: Experiment Randomization Failed

**Grade: F (7.7%)** - Only 1 out of 13 balance tests passed

---

## What This Means

Your A/B test of 27 insurance parameter variants **was not properly randomized**. Customers were not randomly assigned to treatment variants - instead, there are systematic patterns where certain types of customers got certain variants more often.

### The Problem in Plain English

Imagine you're testing whether red or blue pills work better for headaches. But all the young, healthy people got red pills while all the older people with chronic conditions got blue pills. If red pills show better results, is it because:
1. Red pills actually work better? (treatment effect)
2. Young healthy people naturally recover faster? (confounding)

**We can't tell without proper randomization.**

Same issue here: Top-performing variants may owe their success to getting better customers (existing customers, higher income) rather than having better parameter values.

---

## Key Evidence

### 1. Sample Sizes Are Unbalanced ❌
- Expected: ~2,136 per variant
- Actual: 2,026 - 2,247 per variant
- **Statistical test**: χ²=44.73, p=0.013 (significant)

### 2. Customer Characteristics Differ Across Variants ❌

**All 5 continuous features failed balance:**

| Feature | Why It Matters | Imbalance |
|---------|---------------|-----------|
| Customer Age | Older customers may have different risk tolerance | Variants range 38.9 - 41.0 years (p<0.0001) |
| Customer Income | Income is strongest predictor (top 25% convert 1.6× better) | Variants range $82K - $88K (p<0.0001) |
| Pet Age | Affects insurance urgency | Variants range 3.1 - 3.5 years (p<0.0001) |
| Premium | Price sensitivity differs by customer | Variants range $441 - $581 (p<0.0001, **6% CV**) |

**5 of 6 categorical features failed balance:**

| Feature | Why It's Critical | Imbalance |
|---------|------------------|-----------|
| **Has Multiple Pet Discount** ⚠️ | These customers convert at **60%** vs 14% baseline (4× higher!) | χ²=52, p=0.002 |
| **Has Debit Card** ⚠️ | Existing customers - much higher conversion | χ²=79, p<0.0001 |
| **Strongly Connected Users** ⚠️ | Existing customers via referral network | χ²=87, p<0.0001 |
| **State** ⚠️⚠️ | Massive geographic clustering | χ²=2,704, p<0.0001 |
| Designer Breed | Demographic/price signal | χ²=70, p=0.046 |

### 3. Variants Rolled Out Differently Over Time ❌

- Monthly imbalance: χ²=573, p<0.0001
- Weekly imbalance: χ²=2,004, p<0.0001

Some variants were heavily used early, others late. Captures different customer behavior patterns (seasonal effects, campaign momentum).

---

## Why This Happened

**Most Likely Root Cause**: The assignment algorithm used customer features (like income, customer ID, state) to determine which variant to show, rather than true random assignment.

Evidence:
- Premium clustering (6% CV) → premium-based assignment
- Existing customer clustering → customer-type hashing
- Geographic clustering → state-based assignment
- Temporal patterns → sequential/adaptive deployment

---

## Impact on Your Analysis

### Stage 2: Variant Performance Rankings ⚠️ **POTENTIALLY BIASED**

Remember the key finding that 70_100_50000 was the best variant with $111.90 per quote (+13.7% lift)?

**This may be wrong (or right for the wrong reasons).**

If 70_100_50000 got:
- More existing customers (60% conversion vs 14% baseline)
- Higher income customers (top quartile converts 1.6× better)
- Deployed during favorable time periods

...then its strong performance could be entirely due to customer composition, not the actual parameters (70% coinsurance, $100 deductible, $50K limit).

### Stage 3-4: Feature Analysis ✅ **STILL VALID**

Good news: Feature importance, correlation analysis, and engineered features are not affected. We correctly identified:
- Existing customers as dominant predictors
- Income as strongest numerical predictor
- Interaction effects and composite scores

### Stage 5: Modeling Strategy ⚠️ **REQUIRES ADJUSTMENT**

Cannot just predict "which variant performs best" without accounting for the fact that variants got different customer mixes.

Need **causal inference techniques**:
- Propensity score adjustment
- Regression with covariates
- Segment-specific analysis

---

## What Needs to Happen Now

### Immediate (Before Production Decision):

1. **Re-analyze variants with covariate adjustment** ⏰ 1 week
   - Fit model: Sales ~ Variant + Age + Income + MultiPet + DebitCard + State + Month
   - Get adjusted treatment effects (controlling for customer differences)
   - Compare adjusted vs unadjusted rankings

2. **Quantify bias magnitude** ⏰ 2 days
   - How much do rankings change after adjustment?
   - Is 70_100_50000 still the best after controlling for customer composition?

3. **Segment analysis** ⏰ 3 days
   - Analyze existing customers separately from new customers
   - Check if top variant is consistent across segments

### Short-term (For Modeling):

4. **Use causal inference methods** ⏰ 2 weeks
   - Propensity score weighting
   - Doubly robust estimation
   - Compare multiple adjustment methods for robustness

### Long-term (Future Experiments):

5. **Fix randomization** ⏰ 1 day design + testing
   - Use cryptographically secure random assignment
   - Do NOT hash customer IDs or features
   - Pre-launch balance checks

---

## Business Impact

### The Good News ✅
- Your data is high quality (only randomization is flawed)
- Feature analysis is valid → you know what drives conversion
- Customer segmentation insights are actionable
- Engineered features can be used in models
- **This is fixable with proper statistical adjustment**

### The Bad News ⚠️
- Cannot trust simple variant comparisons
- Best variant may not actually be best
- $1.35M revenue lift estimate may be biased (could be higher or lower)
- Need 1-2 weeks additional analysis before production decision

### The Bottom Line 💰

**With proper adjustment, you can still:**
- Identify truly best-performing variants
- Build accurate recommendation system
- Deliver significant business value ($1M+ annual revenue)

**Timeline impact:**
- Add 1-2 weeks for covariate-adjusted analysis
- No change to model development timeline
- Much higher confidence in final recommendations

---

## Recommended Decision Path

```
┌──────────────────────────────────────┐
│ Week 1: Covariate-Adjusted Analysis  │
│  • Run ANCOVA with all covariates    │
│  • Get adjusted variant rankings     │
│  • Compare to unadjusted rankings    │
└──────────────────┬───────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│ Week 2: Validation & Sensitivity     │
│  • Segment analysis (existing vs new)│
│  • Propensity score validation       │
│  • Check robustness across methods   │
└──────────────────┬───────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│ Decision Point: Top Variant Confirmed?│
├──────────────────────────────────────┤
│ IF YES: Proceed with production      │
│         deployment and modeling      │
│                                      │
│ IF NO:  Investigate further,         │
│         consider new experiment      │
└──────────────────────────────────────┘
```

---

## FAQ

**Q: Should we throw out all the previous analysis?**
A: No. Feature analysis (Stages 3-4) is valid. Only variant comparisons (Stage 2) need adjustment.

**Q: Is the $1.35M revenue opportunity estimate wrong?**
A: It's potentially biased. Could be higher or lower. Need adjustment to get accurate estimate.

**Q: How common is this problem?**
A: Very common in real-world A/B tests, especially when dealing with legacy systems. Standard practice is to adjust for observed imbalances.

**Q: Can we still build the recommendation system?**
A: Yes, but use causal inference methods that adjust for customer composition, not simple predictions.

**Q: Will this delay the project significantly?**
A: 1-2 weeks for proper adjustment. Small price to pay for accurate, trustworthy results.

---

## One-Sentence Summary

**Your A/B test has severe randomization problems where high-value customers (existing customers, high income) were unevenly distributed across variants, so you must use covariate adjustment methods (ANCOVA, propensity scores) to isolate true treatment effects before making production decisions.**

---

*For detailed technical analysis, see: `results/RANDOMIZATION_QUALITY_REPORT.txt`*
*For statistical test results, see: `results/randomization_quality_summary.json`*
*For visualization, see: `figures/randomization_quality_analysis.png`*
