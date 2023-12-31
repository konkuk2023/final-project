python ./main.py \
    --mode Test \
    --method CSI \
    --folds 5 \
    --test_params False \
    --batch_size 1 \
    --dataset DEAP \
    --beta1 0.9 \
    --beta2 0.999 \
    --target valence \
    --basemean True \
    --alpha 1 \
    --feature DE \
    --file_length 6 \
    --n_classes 9 \
    --device cuda:1 \
    --formula Bernoulli \
    --weight Square \
    --output_dir Bernoulli_Square \
    --test_weights "30 51 59 59 38"