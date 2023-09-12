python ./main.py \
    --mode Train \
    --epochs 200 \
    --batch_size 2048 \
    --dataset DEAP \
    --optimizer Adam \
    --learning_rate 1e-5 \
    --weight_decay 1e-4 \
    --beta1 0.9 \
    --beta2 0.999 \
    --target valence \
    --basemean True \
    --alpha 1 \
    --feature DE \
    --file_length 1 \
    --n_classes 9 \
    --device cuda:1 \
    --formula None \
    --weight None \
    --output_dir TEST \
    --test_weights "" 