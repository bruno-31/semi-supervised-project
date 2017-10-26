#!/usr/bin/env bash

touch random_search00.txt

logpath='./log_chigan'

array1=(1 2 3 4 5)
array2=(a b c d e)
array3=(v w x y z)


for gpu in {1..3} # gpu to use
do
    export CUDA_VISIBLE_DEVICES=$gpu
    cpt=0
    name=array$gpu
    declare -n array=$name
    while [ $cpt -le 2 ]
    do
        echo cuda = $CUDA_VISIBLE_DEVICES  task:${array[$cpt]}   cpt =$cpt
#        python3 train_chi_gan.py
        ((cpt++))
        sleep 3
    done &
done
wait
exit