SEED=42
for i in $(seq 1 60); do
    python single_bldg_xgb.py --num $i --seed $SEED
done
