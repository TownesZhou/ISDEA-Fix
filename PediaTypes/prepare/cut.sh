#
set -e

#
declare -A rename
rename["D2W"]="DB WD"
rename["D2Y"]="DB YG"
rename["EN2DE"]="EN DE"
rename["EN2FR"]="EN FR"

#
for dataset in D2W D2Y EN2DE EN2FR; do
    #
    for n in 15; do
        #
        for v in 1 2; do
            #
            for suffix in ${rename[${dataset}]}; do
                #
                chmod 0444 "data/${dataset}${n}KV${v}-${suffix}/full.txt"
                for section in message predict observe train valid test; do
                    #
                    rm -f "data/${dataset}${n}KV${v}-${suffix}/${section}.txt"
                done
                python cut.py "data/${dataset}${n}KV${v}-${suffix}"
            done
        done
    done
done
