set -euo pipefail
s=$(ls -v models/*.json|xargs -I{} basename {}|cut -d'_' -f2|head -1)
e=$(ls -v models/*.json|xargs -I{} basename {}|cut -d'_' -f2|tail -1)
echo $s~$e
rm -rf ./subs
mkdir subs
for m in $(ls -v models/*.json); do 
    echo processing\($m\)
	python forecast_sub.py -m $m
done
