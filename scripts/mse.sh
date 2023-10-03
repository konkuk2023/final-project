BASEMEAN=$1
DEVICE=$2
LENGTH=$3
TARGET=$4
BATCH=$5
LR=$6
OUTPUT_DIR="MSE_"$LENGTH

python ./main.py \
    --mode Train \
    --method CSI \
    --save_gspread True \
    --epochs 500 \
    --batch_size ${BATCH} \
    --dataset DEAP \
    --optimizer Adam \
    --learning_rate ${LR} \
    --weight_decay 1e-4 \
    --beta1 0.9 \
    --beta2 0.999 \
    --target ${TARGET} \
    --basemean ${BASEMEAN} \
    --alpha 1 \
    --feature DE \
    --file_length ${LENGTH} \
    --n_classes 1 \
    --device ${DEVICE} \
    --formula None \
    --weight None \
    --output_dir ${OUTPUT_DIR} \
    --test_weights "" 