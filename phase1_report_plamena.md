# Phase 1 Report — [Your name], [dataset]
*Target ~500 words total. Replace every bracket. Due: Session 5 (Oct 12).*

## 1. Problem & dataset (~80 words)
[What are you predicting, for whom, and why would a business care? Dataset, size, class balance.]

The data is the Default of Credit Card Clients dataset (UCI #350, Yeh 2009): 30,000 clients from Taiwan with 23 features

“22.1% of clients default, so the classes are imbalanced.”


## 2. Data preparation decisions (~100 words)
[≥2 concrete decisions: what you found (evidence), what you did, why. Leakage check goes here.]
    -profile shows no missing values, therefore no rows were removed:
    -age is reasonable - 21-79, no Credit limit under 0, no Negative payments: 0, PAY_0 have expected values, Negative bills (overpayment)
    -35 pairs of rows show up as duplicates apart from ID, 30 of them have no activity and 5 have the same amount billed with 4 of them repaid in full
    possibly due to fixed recurring charges from the same bank and 1 never repaid. Meaning they are most likely different customers, so all rows are kept.
    -dropped ID as it has no relationship with the credit risk and can confuse the model
    -EDUCATION 0, 5,6 merged(1.1%), as well as Marriage (0.2%)  into "others" category as they are not documented  
    -Education and Marriage values converted to labels as they are categories not quantities
    -sex is dropped as it is viewed as discriminatory 
    -all features describe past events - April - September 2005 and the target is the following month,  so there is no leakage


## 3. Model selection rationale (~100 words)
[What you tried, what you chose, why — reference a baseline comparison.]
-Exploratory analysis shows repayment status is strongly related to default:
    -the higher the months delayed for repayment, the higher the % of default next month
    -clients who paid on time defaulted in 13-17% of the cases, while those who left behind defaulted
    in 50-78% meaning payment history is very strong predictor


| Model | Accuracy | Precision | Recall | F1 | AUC (val) |
|---|---|---|---|---|---|
| Majority baseline | 0.779 | 0.000 | 0.000 | 0.000 | 0.500 |
| Logistic regression | 0.809 | 0.697 | 0.243 | 0.361 | 0.714 |
| XGBoost | 0.818 | 0.669 | 0.354 | 0.463 | 0.774 |

The majority baseline reaches 77.9%, however it does not catch any defaulters, which is the actual added value. As 22.1% of the 
clients actually default in the dataset, this model achieves such accuracy without learning anything useful.

Both logistic regression with 80.9% accuracy and XGBoost with 81.8% score better. XGBoost catches about 46% more 
defaulters -35.4% compared to 24.3%. Therefore, XGBoost was chosen to be tuned. 

XGBoost Tuning:
| Hyperparameter | Value | Train AUC | Val AUC | Gap |
|---|---|---|---|---|
| max_depth | 2 | 0.801 | **0.777** | 0.024 |
| max_depth | 4 | 0.851 | 0.774 | 0.077 |
| max_depth | 8 | 0.982 | 0.762 | 0.220 |
| max_depth | 12 | 1.000 | 0.755 | 0.245 |
| learning_rate | 0.01 | 0.774 | 0.760 | 0.014 |
| learning_rate | **0.1** | 0.801 | **0.777 ** | 0.024 |
| learning_rate | 0.3 | 0.822| 0.775 | 0.047 |

*max_depth tested with learning_rate = 0.1; learning_rate tested with max_depth = 2.*

Chosen hyperparameters are max_depth = 2 and learning_rate = 0.1.
Max_depth of 2 has the highest validation AUC with a train-validation gap 3 times smaller (0.024 VS 0.077),meaning it generalizes better.However, deeper trees - depth 12 memorizes data better - train AUC 1.00 but with  validation AUC dropped to 0.755.
Learning_rate of 0.1 outperforms in terms of AUC, as 0.01 underfits  - both train AUC and validation AUC are lower, while 0.3 starts overfitting slightly - train AUC rises to 0.822 and val AUC drop a little to 0.775.

Overall, tuning did increase slightly the validation AUC while reducing the training validation gap and it led to choosing a simpler model which overfits less at no cost at performance.
Additionally, simpler trees are especially useful in backing up credit decisions to clients or regulators. 

++ADD TABLE 

## 4. Evaluation (~120 words)
[Metrics appropriate to the problem and what they mean HERE — e.g., what does your recall
imply operationally? Include val vs. test numbers and your tuning evidence (≥2 hyperparameters).]

As there is imbalance in the dataset, the accuracy is not informative. Therefore, we use recall - what share of the
defaulters the model catches combined with precision - how many flagged clients actually default and AUC - identifying 
a chosen defaulter as riskier than non-defaulter. Combined, these metrics give a more in-depth evaluation.


## 5. Trade-offs & limitations (~100 words)
[Precision/recall choice, cost of errors, what you'd do with more time/data. One honest weakness.]
Given the default 0.5 threshold, XGBoost has a recall of 35.4%, meaning it misses 64.6% of the defaulters. Missing 
a defaulter costs more than flagging a non-defaulter for a bank credit risk prediction. Therefore, we lower the threshold from 0.5 to 0.25 - highest score in F1. 
By doing this, we raise the recall from 35.4% to 56.7%, while lowering precision from 66.3% to 51.1%. This results in flagging 24.5% of clients, 
while the data has 22.1% actual defaulters. If we drop the threshold even further to 0.2 this would result in 65.8% recall but at the expense of three false alarms per 
extra defaulter. That is the reason the chosen threshold is 0.25, however depending on the bank major objectives and regulatory requirements. 

++ADD CONFUSION MATRIX AND TABLES
++what you'd do with more time/data

Results: Test accuracy at 0.25 is 0.784 which is just slightly higher than the majority baseline - 0.779. The results
show that accuracy is not a meaningful metric here, as the new model catches 59% of the defaulters compared with the baseline which catches none. 

## Appendix (not counted)
Endpoint name + invocation screenshot/cell, AI-assistance disclosure (per syllabus policy).
