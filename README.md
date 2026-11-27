# Next-Day Rain Prediction for Mumbai: Classical ML vs a Neural Network

Compares Logistic Regression, Decision Tree, Random Forest, SVM and an MLP on a binary
task: will it rain tomorrow in Mumbai, given today's weather observations?
[Project report](https://drive.google.com/file/d/1DkIMeXenqMF87nxPRC7wWX6qzVpodgqG/view?usp=sharing)

## Task and data

- Hourly weather observations are aggregated to daily features: 9 am and 3 pm values for
  each measured variable, plus differences between them.
- `RainToday` is derived from the day's recorded weather phrases, and the target
  `RainTomorrow` is `RainToday` shifted by one day.
- The input file `cleaned_data.csv` is **not included in this repository**, so the notebook
  cannot be re-run from a fresh clone. Its original source is not documented here.
- Features are standardized with a scaler fit on the training split only.

## Evaluation setup

Days are sorted by date and split 70 / 15 / 15 into train, validation and test. The split is
done **separately within each class** (rainy and dry days each split by date) so that all
three sets keep the same rain rate. This keeps class balance but means the sets are not
strictly forward in time: a rainy day and a dry day from the same period can land in
different sets. Treat the numbers as optimistic compared with true forward-in-time testing.

Models were compared on the validation set (1,241 days); the best by validation AUC was then
scored once on the test set (1,460 days).

## Results

Validation set, rain class (the positive class, 175 of 1,241 days):

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.907 | 0.78 | 0.48 | 0.59 | 0.904 |
| Decision Tree | 0.883 | 0.59 | 0.58 | 0.58 | 0.873 |
| **Random Forest** | **0.930** | **0.89** | 0.58 | **0.70** | **0.928** |
| SVM | 0.890 | 0.62 | 0.57 | 0.60 | 0.893 |
| Neural network (MLP) | 0.904 | 0.67 | **0.64** | 0.65 | 0.909 |

Held-out **test** set for the selected model (Random Forest, 206 rain days of 1,460):

| Accuracy | Precision (rain) | Recall (rain) | F1 (rain) | ROC-AUC |
| ---: | ---: | ---: | ---: | ---: |
| 0.884 | 0.63 | 0.41 | 0.50 | 0.915 |

### How to read this

- About 14% of days are rainy, so accuracy alone is flattering: always predicting "no rain"
  would already score about 86%. Look at rain-class recall and F1.
- Random Forest ranks days best (highest AUC), but it misses most rainy days at the default
  threshold (test recall 0.41). The MLP has the best validation recall (0.64).
- Validation metrics are higher than test metrics, as expected after picking the best model on
  validation. The test numbers are the ones to quote.
- One split and one seed: there are no confidence intervals, and differences of a few points
  between models are not established.

## Run

```bash
git clone https://github.com/sachin2967/Mumbai-Rainfall-Prediction-MLvsDL.git
cd Mumbai-Rainfall-Prediction-MLvsDL
python -m venv .venv && source .venv/bin/activate   # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
# add cleaned_data.csv next to the notebook (not included), then:
jupyter notebook final_stat_ml_v2_0.ipynb
```

## Ideas for improvement

1. Use a strictly forward-in-time split and report a rolling-origin evaluation.
2. Tune the decision threshold for rain-class recall or F-score instead of using the default.
3. Add sequence models (LSTM / CNN-LSTM) and compare against a "persistence" baseline.
4. Extend to rainfall-intensity classes.

## References

1. [Predicting rainfall using ML/DL, Scientific Reports](https://www.nature.com/articles/s41598-024-77687-x)
2. [Hybrid CNN-LSTM models for weather prediction](https://iwaponline.com/ws/article/22/5/4902/88212)

## License

MIT
