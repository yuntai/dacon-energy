`ls -v models/*.json|xargs -I{} python eval.py --model {} 2>/dev/null |tee res/res1-20.csv`
`ls -v models/*.json|xargs -I{} python forecast.py --model {} 2>/dev/null |tee res/fres1-20.csv`

# modeling
- hours is numerical or categorical?
- val set to same size as test set
- non-recursive combo
- transformer/LSTM
- lag other variables?
- interploation (separate prediction model?)
- pacf - consider confidence interval?
- review KAGGLE solutions
- outlier removal
- train with full data
- feature selection

# eng
- use user in Docker
