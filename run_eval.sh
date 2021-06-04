set -uo pipefail
s=$(ls -v models/*.json|xargs -I{} basename {}|cut -d'_' -f2|head -1)
e=$(ls -v models/*.json|xargs -I{} basename {}|cut -d'_' -f2|tail -1)
echo $s~$e
for m in $(ls -v models/*.json); do 
	echo evaluation $m ...
	python eval.py -m $m 2>/dev/null |tee -a res/res$s-$e.csv
done
echo
for m in $(ls -v models/*.json); do 
	echo forecasting $m ...
	python forecast.py -m $m 2>/dev/null |tee -a res/fres$s-$e.csv
done
