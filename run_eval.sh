set -uo pipefail
s=$(ls -v models/*.json|xargs -I{} basename {}|cut -d'_' -f2|head -1)
e=$(ls -v models/*.json|xargs -I{} basename {}|cut -d'_' -f2|tail -1)
ls -v models/*.json|xargs -I{} python eval.py --model {} 2>/dev/null |tee res/res$s-$e.csv`
ls -v models/*.json|xargs -I{} python forecast.py --model {} 2>/dev/null |tee res/fres$s-$e.csv`
