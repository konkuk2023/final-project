BASEMEAN=$1
DEVICE=$2
N_CLASSES=$3
FORMULA=$4
WEIGHT=$5
OUTPUT_DIR=$FORMULA"_"$WEIGHT

python ./main.py \
    --mode Train \
    --method CSV \
    --save_gspread True \
    --epochs 500 \
    --batch_size 256 \
    --dataset DEAP \
    --optimizer Adam \
    --learning_rate 1e-5 \
    --weight_decay 1e-4 \
    --beta1 0.9 \
    --beta2 0.999 \
    --target arousal \
    --basemean ${BASEMEAN} \
    --alpha 1 \
    --feature DE \
    --file_length 10 \
    --n_classes ${N_CLASSES} \
    --device ${DEVICE} \
    --formula ${FORMULA} \
    --weight ${WEIGHT} \
    --output_dir ${OUTPUT_DIR} \
    --test_weights "" 