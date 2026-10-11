# Predicting Credit Card Default with XGBoost

## 1. Problem & dataset

Banks lose money when clients stop paying. This project predicts whether a credit card client will default on next month's payment. A credit-risk team could use that prediction to act early, for example by lowering the limit or contacting the client. The data is the UCI Default of Credit Card Clients dataset (#350, Yeh 2009). It covers 30,000 clients in Taiwan with 23 features: credit limit, demographics, and six months of repayment status, bills and payments (April–September 2005). 22.1% of clients default, so the classes are imbalanced. That imbalance shapes every later choice of metric and threshold.

## 2. Data preparation decisions

Profiling showed no missing values and plausible ranges: age 21–79, no negative limits or payments. Negative bills reflect overpayment, so they were kept. 35 pairs of rows were identical apart from ID. Thirty of them show no activity, and five have identical bills, consistent with different customers paying fixed recurring charges, so all rows were kept. ID was dropped because it carries no credit information. SEX was dropped to avoid discriminatory decisions. Undocumented EDUCATION codes (0, 5, 6; 1.1%) and the undocumented MARRIAGE code (0.2%) were merged into "other". Both features were treated as categories, not quantities. Leakage check: every feature describes April–September 2005 and the target is the following month, so no feature contains future information.

## 3. Model selection rationale

Exploratory analysis showed that repayment history is the key signal. Clients who paid on time defaulted in 13–17% of cases, versus 50–78% for clients who were behind. The default rate rose with each extra month of delay. With this in mind, three models were compared on the validation set at default settings:

| Model | Accuracy | Precision | Recall | F1 | AUC (val) |
|---|---|---|---|---|---|
| Majority baseline | 0.779 | 0.000 | 0.000 | 0.000 | 0.500 |
| Logistic regression | 0.809 | 0.697 | 0.243 | 0.361 | 0.714 |
| XGBoost | 0.818 | 0.669 | 0.354 | 0.463 | 0.774 |

The baseline reaches 77.9% accuracy by predicting "no default" for everyone, so it catches no defaulters. Both learned models beat it. XGBoost ranks clients better (AUC 0.774 vs. 0.714) and catches about 46% more defaulters (recall 35.4% vs. 24.3%), so it was chosen and tuned on two hyperparameters:

| Hyperparameter | Value | Train AUC | Val AUC | Gap |
|---|---|---|---|---|
| max_depth | 2 | 0.801 | **0.777** | 0.024 |
| max_depth | 4 | 0.851 | 0.774 | 0.077 |
| max_depth | 8 | 0.982 | 0.762 | 0.220 |
| max_depth | 12 | 1.000 | 0.755 | 0.245 |
| learning_rate | 0.01 | 0.774 | 0.760 | 0.014 |
| learning_rate | 0.1 | 0.801 | **0.777** | 0.024 |
| learning_rate | 0.3 | 0.822 | 0.775 | 0.047 |

*max_depth tested with learning_rate = 0.1; learning_rate tested with max_depth = 2.*

The best settings are max_depth = 2 and learning_rate = 0.1:

- **max_depth = 2** gives the highest validation AUC (0.777). Its train–validation gap is three times smaller than at depth 4 (0.024 vs. 0.077). Deeper trees memorise the data: at depth 12, train AUC reaches 1.000 while validation AUC falls to 0.755.
- **learning_rate = 0.1** also gives the best validation AUC. At 0.01 the model underfits, with both AUCs lower. At 0.3 it starts to overfit: train AUC rises to 0.822 and validation AUC drops to 0.775.

Tuning therefore raised validation AUC slightly and chose a simpler model that generalises better at no cost in performance. Its shallow trees are also easier to justify to clients and regulators.

## 4. Evaluation

Because of the imbalance, accuracy is misleading; the baseline already scores 77.9%. The model is judged on three metrics instead:

- **Recall:** the share of defaulters caught.
- **Precision:** the share of flagged clients who actually default.
- **AUC:** how often a defaulter is scored as riskier than a non-defaulter. It does not depend on the threshold, so it drove model selection.

The test set was used once, after all choices were fixed. Both rows use the 0.25 threshold chosen in Section 5:

| | Precision | Recall | F1 | AUC |
|---|---|---|---|---|
| Validation | 0.511 | 0.567 | 0.537 | 0.777 |
| Test | 0.510 | 0.593 | 0.549 | 0.779 |

Test results match validation closely, so tuning did not overfit the validation set. In practice, the model catches about 59% of defaulters, and roughly every second flagged client actually defaults. Test accuracy (0.784) is only slightly above the baseline (0.779), which confirms that accuracy is not the right measure here: the baseline catches no defaulters at all.

## 5. Trade-offs & limitations

At the default 0.5 threshold, the tuned model's recall is 35.4%, so it misses 64.6% of defaulters. For a bank, missing a defaulter costs more than flagging a good client. The threshold was therefore lowered to 0.25, the value that maximises F1 on validation:

| Validation (995 defaulters) | Caught | Missed | False alarms |
|---|---|---|---|
| Threshold 0.5 | 352 | 643 | 179 |
| Threshold 0.25 | 564 | 431 | 540 |

Recall rises from 35.4% to 56.7%, while precision falls from 66.3% to 51.1%. The model now flags 24.5% of clients, close to the 22.1% who actually default. A threshold of 0.2 would raise recall to 65.8%, but at about three false alarms per extra defaulter caught. That is why 0.25 was chosen, although the right threshold ultimately depends on the bank's real cost of a missed default versus a false alarm, and on regulatory requirements.

Limitations:

- F1 stands in for the real cost ratio, which is unknown.
- AGE and MARRIAGE remain in the model and may act as partial proxies for protected attributes.
- The data covers one country and one period (2005).

With more time, engineered features such as the share of each bill repaid could improve the ranking itself (AUC). Threshold tuning, by contrast, only moves along the existing precision/recall trade-off.

## Appendix (not counted)

- **Endpoint:** `phase1-credit-default-plamena-v4` (SageMaker, us-east-1, ml.m5.large), deleted after the invocation test.
- **Invocation:** ![Endpoint invocation](images/endpoint_invocation.png)  
  The endpoint returned a default probability of 0.103 for test client 0, identical to the local model (not flagged at 0.25; actual outcome: no default).
- **Deployment notes:** the pipeline was rerun once with xgboost 2.1.4 to match the SageMaker container. Results changed only in the third decimal, and no decision was changed after seeing the test results.
- **AI assistance:** [your disclosure per the syllabus policy]
