BASEMEAN=$1
DEVICE=$2
N_CLASSES=$3
FORMULA=$4
WEIGHT=$5
OUTPUT_DIR=$FORMULA"_unseen_"$WEIGHT

python ./main.py \
    --mode Train \
    --method CSI \
    --folds 5 \
    --save_gspread False \
    --epochs 300 \
    --batch_size 128 \
    --dataset DEAP \
    --optimizer Adam \
    --learning_rate 1e-4 \
    --weight_decay 1e-4 \
    --beta1 0.9 \
    --beta2 0.999 \
    --target arousal \
    --basemean ${BASEMEAN} \
    --alpha 1 \
    --feature DE \
    --file_length 1 \
    --n_classes ${N_CLASSES} \
    --device ${DEVICE} \
    --formula ${FORMULA} \
    --weight ${WEIGHT} \
    --output_dir ${OUTPUT_DIR} \
    --test_weights "" 