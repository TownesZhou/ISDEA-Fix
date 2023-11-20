#
set -e

#
rm -rf data
# python subgraph.py \
#     OpenEA_dataset_v2.0/D_W_15K_V1 \
#     --num-nodes 5000 --cap 100
# exit
for dataset in D_W D_Y EN_DE EN_FR; do
    #
    for n in 15; do
        #
        for v in 1 2; do
            #
            python subgraph.py \
                OpenEA_dataset_v2.0/${dataset}_${n}K_V${v} \
                --num-nodes 5000 --cap 100
        done
    done
done
