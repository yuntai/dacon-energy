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
- log normality test

# Feat eng
https://www.analyticsvidhya.com/blog/2019/12/6-powerful-feature-engineering-techniques-time-series/

# data cleaning
- num=27 contains zero kwh

# eng
- use user in Docker

# links (SMAPE)
- https://stats.stackexchange.com/questions/425390/how-do-i-decide-when-to-use-mape-smape-and-mase-for-time-series-analysis-on-sto
- https://towardsdatascience.com/choosing-the-correct-error-metric-mape-vs-smape-5328dec53fac
- https://stats.stackexchange.com/questions/213897/best-way-to-optimize-mape
- https://stats.stackexchange.com/questions/417588/how-to-optimize-mape-in-regression-algorithms
- https://stats.stackexchange.com/questions/299712/what-are-the-shortcomings-of-the-mean-absolute-percentage-error-mape
- https://stats.stackexchange.com/questions/145490/minimizing-symmetric-mean-absolute-percentage-error-smape
- https://stats.stackexchange.com/questions/18844/when-and-why-should-you-take-the-log-of-a-distribution-of-numbers
