SEED=42
for i in $(seq 1 20); do
    python single_bldg_xgb.py --num $i --seed $SEED
done
