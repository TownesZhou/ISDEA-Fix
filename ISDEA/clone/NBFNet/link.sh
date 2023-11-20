#
for i in 1 2; do
    #
    top=../../../../../data
    mkdir -p datasets/FD${i}Trans/raw
    ln -sb ${top}/FD${i}-trans/entities.dict datasets/FD${i}Trans/raw/
    ln -sb ${top}/FD${i}-trans/relations.dict datasets/FD${i}Trans/raw/
    ln -sb ${top}/FD${i}-trans/observe.txt datasets/FD${i}Trans/raw/
    ln -sb ${top}/FD${i}-trans/train.txt datasets/FD${i}Trans/raw/
    ln -sb ${top}/FD${i}-trans/valid.txt datasets/FD${i}Trans/raw/
    mkdir -p datasets/FD${i}Ind/raw
    ln -sb ${top}/FD${i}-ind/entities.dict datasets/FD${i}Ind/raw/
    ln -sb ${top}/FD${i}-ind/relations.dict datasets/FD${i}Ind/raw/
    ln -sb ${top}/FD${i}-ind/observe.txt datasets/FD${i}Ind/raw/
    ln -sb ${top}/FD${i}-ind/test.txt datasets/FD${i}Ind/raw/
done

#
for prefix in WN18RR NELL995 FB237; do
    for suffix in 1 2 3 4; do
        #
        top=../../../../../data
        data=${prefix}${suffix}
        mkdir -p datasets/${data}Trans/raw
        ln -sb ${top}/${data}-trans/entities.dict datasets/${data}Trans/raw/
        ln -sb ${top}/${data}-trans/relations.dict datasets/${data}Trans/raw/
        ln -sb ${top}/${data}-trans/train.txt datasets/${data}Trans/raw/
        ln -sb ${top}/${data}-trans/valid.txt datasets/${data}Trans/raw/
        mkdir -p datasets/${data}Ind/raw
        ln -sb ${top}/${data}-ind/entities.dict datasets/${data}Ind/raw/
        ln -sb ${top}/${data}-ind/relations.dict datasets/${data}Ind/raw/
        ln -sb ${top}/${data}-ind/observe.txt datasets/${data}Ind/raw/
        ln -sb ${top}/${data}-ind/test.txt datasets/${data}Ind/raw/
    done
done

for prefix in WN18RR NELL995 FB237; do
    for suffix in 1 2 3 4; do
        #
        top=../../../../../data
        data=${prefix}${suffix}
        mkdir -p datasets/${data}PermInd/raw
        ln -sb ${top}/${data}-ind-perm/entities.dict datasets/${data}PermInd/raw/
        ln -sb ${top}/${data}-ind-perm/relations.dict datasets/${data}PermInd/raw/
        ln -sb ${top}/${data}-ind-perm/observe.txt datasets/${data}PermInd/raw/
        ln -sb ${top}/${data}-ind-perm/test.txt datasets/${data}PermInd/raw/
    done
done


#
for prefix in DB2WD WD2DB DB2YG YG2DB FR2EN EN2FR EN2DE DE2EN; do
    for suffix in -15K-V1 -15K-V2; do
        #
        top=../../../../../data
        data=${prefix}${suffix}
        mkdir -p datasets/${data}Trans/raw
        ln -sb ${top}/${data}-trans/entities.dict datasets/${data}Trans/raw/
        ln -sb ${top}/${data}-trans/relations.dict datasets/${data}Trans/raw/
        ln -sb ${top}/${data}-trans/train.txt datasets/${data}Trans/raw/
        ln -sb ${top}/${data}-trans/valid.txt datasets/${data}Trans/raw/
        mkdir -p datasets/${data}Ind/raw
        ln -sb ${top}/${data}-ind/entities.dict datasets/${data}Ind/raw/
        ln -sb ${top}/${data}-ind/relations.dict datasets/${data}Ind/raw/
        ln -sb ${top}/${data}-ind/observe.txt datasets/${data}Ind/raw/
        ln -sb ${top}/${data}-ind/test.txt datasets/${data}Ind/raw/
    done
done

for prefix in DB2WD WD2DB DB2YG YG2DB FR2EN EN2FR EN2DE DE2EN; do
    for suffix in -15K-V1 -15K-V2; do
        #
        top=../../../../../data
        data=${prefix}${suffix}
        mkdir -p datasets/${data}OrigInd/raw
        ln -sb ${top}/${data}-ind-orig/entities.dict datasets/${data}OrigInd/raw/
        ln -sb ${top}/${data}-ind-orig/relations.dict datasets/${data}OrigInd/raw/
        ln -sb ${top}/${data}-ind-orig/observe.txt datasets/${data}OrigInd/raw/
        ln -sb ${top}/${data}-ind-orig/test.txt datasets/${data}OrigInd/raw/
    done
done

for prefix in FB2371 FB2372 NELL9952 NELL9953 NELL9954; do
    for suffix in Dis; do
        #
        top=../../../../../data
        data=${prefix}${suffix}
        mkdir -p datasets/${data}Trans/raw
        ln -sb ${top}/${data}-trans/entities.dict datasets/${data}Trans/raw/
        ln -sb ${top}/${data}-trans/relations.dict datasets/${data}Trans/raw/
        ln -sb ${top}/${data}-trans/train.txt datasets/${data}Trans/raw/
        ln -sb ${top}/${data}-trans/valid.txt datasets/${data}Trans/raw/
        mkdir -p datasets/${data}Ind/raw
        ln -sb ${top}/${data}-ind/entities.dict datasets/${data}Ind/raw/
        ln -sb ${top}/${data}-ind/relations.dict datasets/${data}Ind/raw/
        ln -sb ${top}/${data}-ind/observe.txt datasets/${data}Ind/raw/
        ln -sb ${top}/${data}-ind/test.txt datasets/${data}Ind/raw/
    done
done