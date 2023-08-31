#
set -e

#
declare -A rename
rename["D2W"]="DB>WD WD>DB"
rename["D2Y"]="DB>YG YG>DB"
rename["EN2DE"]="EN>DE DE>EN"
rename["EN2FR"]="EN>FR FR>EN"

#
rm -rf task
for dataset in D2W D2Y EN2DE EN2FR; do
    #
    for n in 15; do
        #
        for v in 1 2; do
            #
            for dx in ${rename[${dataset}]}; do
                #
                src=${dx%%>*}
                dst=${dx##*>}
                name=${src}2${dst}-${n}K-V${v}

                #
                echo "${name}: ${src} (trans) ==> ${dst} (ind)"
                mkdir -p task/${name}-trans
                cp data/${dataset}${n}KV${v}-${src}/train.txt task/${name}-trans/train.txt
                cp data/${dataset}${n}KV${v}-${src}/valid.txt task/${name}-trans/valid.txt
                mkdir -p task/${name}-ind
                cp data/${dataset}${n}KV${v}-${dst}/train.txt task/${name}-ind/observe.txt
                cp data/${dataset}${n}KV${v}-${dst}/train.txt task/${name}-ind/train.txt
                cp data/${dataset}${n}KV${v}-${dst}/valid.txt task/${name}-ind/valid.txt
                cp data/${dataset}${n}KV${v}-${dst}/test.txt task/${name}-ind/test.txt
            done
        done
    done
done
