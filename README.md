`ls -v models/*.json|xargs -I{} python eval.py --model {} 2>/dev/null |tee res/res1-20.csv`
`ls -v models/*.json|xargs -I{} python forecast.py --model {} 2>/dev/null |tee res/fres1-20.csv`
