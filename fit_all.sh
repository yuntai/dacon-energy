SEED=42
for i in $(seq 1 50); do
    python fit_xgb.py --num $i --seed $SEED
done
