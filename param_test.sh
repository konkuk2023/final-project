TARGET=$1
LEN=$2
BATCH=$3
LR=$4
OFFSET=$5
DEVICE=$6
OUTPUT_DIR=$BATCH"_"$LR

python ./main.py \
    --test_params True \
    --method CSV \
    --gspread_offset ${OFFSET} \
    --mode Train \
    --epochs 200 \
    --batch_size ${BATCH} \
    --dataset DEAP \
    --optimizer Adam \
    --learning_rate ${LR} \
    --weight_decay 1e-4 \
    --beta1 0.9 \
    --beta2 0.999 \
    --target ${TARGET} \
    --basemean True \
    --alpha 1 \
    --feature DE \
    --file_length ${LEN} \
    --n_classes 9 \
    --device ${DEVICE} \
    --formula Bernoulli \
    --weight None \
    --output_dir ${OUTPUT_DIR} \
    --test_weights "" 