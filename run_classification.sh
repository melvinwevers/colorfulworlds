#!/bin/bash

train_classifier() {
    model_path=$1
    methods=("all" "ac" "pc")

    for method in "${methods[@]}"; do
        echo "training $method"
        if [ "$method" == "all" ]; then
            python ~/work/projects/couleur_locale/src/train_classifier.py --model_path "$model_path"
        else
            python ~/work/projects/couleur_locale/src/train_classifier.py --model_path "$model_path" --method "$method"
        fi
    done
}

model_paths=(
     './models/buckets_8_4.pkl'
     './models/buckets_8_6.pkl'
     './models/buckets_8_8.pkl'
     './models/buckets_16_4.pkl'
     './models/buckets_16_6.pkl'
     './models/buckets_16_8.pkl'
)

for model_path in "${model_paths[@]}"; do
    echo "$model_path"
    train_classifier "$model_path"
done
