#
import sys
import os
import more_itertools as xitertools
import numpy as onp


#
root = sys.argv[1]
props = {"train": 8, "valid": 1, "test": 1}
num_relations = 10000


#
dataset = os.path.basename(root)
print(dataset)


# Group triplets by relation.
groupby_rel = {}
with open(os.path.join(root, "full.txt"), "r") as file:
    #
    for line in file:
        #
        (sub, rel, obj) = line.split()
        if rel in groupby_rel:
            #
            groupby_rel[rel].append((sub, obj))
        else:
            #
            groupby_rel[rel] = [(sub, obj)]


# Keep only top relations.
triplets_full = []
relations = list(
    rel
    for rel, _ in sorted(
        ((rel, len(group)) for rel, group in groupby_rel.items()),
        key=lambda rel_x_cnt: rel_x_cnt[1],
        reverse=True,
    )
)[:num_relations]
for rel in list(groupby_rel.keys()):
    #
    if rel not in relations:
        #
        del groupby_rel[rel]
    else:
        #
        for sub, obj in groupby_rel[rel]:
            #
            triplets_full.append((sub, rel, obj))


# Count entity and relation frequency.
counts_ent = {}
counts_rel = {}
for sub, rel, obj in triplets_full:
    #
    for ent in (sub, obj):
        #
        if ent in counts_ent:
            #
            counts_ent[ent] += 1
        else:
            #
            counts_ent[ent] = 1
    if rel in counts_rel:
        #
        counts_rel[rel] += 1
    else:
        #
        counts_rel[rel] = 1


#
triplets_section = {"train": [], "valid": [], "test": []}
rng = onp.random.RandomState(42)
while sum(len(group) for group in groupby_rel.values()) > 0:
    #
    for rid in rng.permutation(len(relations)).tolist():
        # Treat as sampled once regardless of success or not.
        rel = relations[rid]
        group = groupby_rel[rel]
        if len(group) == 0:
            #
            continue

        #
        buffer = list(
            sorted(
                group,
                key=lambda pair: counts_ent[pair[0]] * counts_ent[pair[1]],
                reverse=True,
            ),
        )
        remains = []
        for title, prop in props.items():
            #
            for sub, obj in buffer[:prop]:
                #
                counts_ent[sub] -= 1
                counts_rel[rel] -= 1
                counts_ent[obj] -= 1
                if counts_ent[sub] == 0 or counts_rel[rel] == 0 or counts_ent[obj] == 0:
                    #
                    remains.append((sub, rel, obj))
                else:
                    #
                    triplets_section[title].append((sub, rel, obj))
            buffer = buffer[prop:]
        triplets_section["train"].extend(remains)
        groupby_rel[rel] = buffer
print("- #Train: {:d}".format(len(triplets_section["train"])))
print("- #Valid: {:d}".format(len(triplets_section["valid"])))
print("- #Test: {:d}".format(len(triplets_section["test"])))


#
rels_section = {
    title: set(rel for _, rel, _ in triplets)
    for (title, triplets) in triplets_section.items()
}
ents_section = {
    title: set(xitertools.flatten((sub, obj) for sub, _, obj in triplets))
    for (title, triplets) in triplets_section.items()
}
assert sum(len(triplets) for triplets in triplets_section.values()) == len(
    triplets_full,
)
assert rels_section["valid"].issubset(rels_section["train"])
assert rels_section["test"].issubset(rels_section["train"])
assert ents_section["valid"].issubset(ents_section["train"])
assert ents_section["test"].issubset(ents_section["train"])

#
for title, triplets in triplets_section.items():
    #
    with open(os.path.join(root, "{:s}.txt".format(title)), "w") as file:
        #
        for sub, rel, obj in triplets:
            #
            file.write("{:s} {:s} {:s}\n".format(sub, rel, obj))
