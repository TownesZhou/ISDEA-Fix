#
import os
import argparse
import numpy as onp
import re
import networkx as nx


#
RENAME_ABBR = {
    "D_W": ("D2W", ("DB", "WD")),
    "D_Y": ("D2Y", ("DB", "YG")),
    "EN_DE": ("EN2DE", ("EN", "DE")),
    "EN_FR": ("EN2FR", ("EN", "FR")),
}
RENAME_FULL = {}
for abbr, (title, rename) in RENAME_ABBR.items():
    #
    for n in (15, 100):
        #
        for v in (1, 2):
            #
            RENAME_FULL["{:s}_{:d}K_V{:d}".format(abbr, n, v)] = (
                "{:s}{:d}KV{:d}".format(title, n, v),
                rename,
            )

#
parser = argparse.ArgumentParser()
parser.add_argument("root", type=str)
parser.add_argument("--num-nodes", type=int, required=False)
parser.add_argument("--num-edges", type=int, required=False)
parser.add_argument("--cap", type=int, required=False)
args = parser.parse_args()
root = args.root
num_nodes_exp_opt = args.num_nodes
num_edges_exp_opt = args.num_edges
cap_opt = args.cap
dataset = os.path.basename(root)


def load_triplet_raw(path):
    #
    ent2int = {}
    rel2int = {}
    buffer = []
    with open(path, "r") as file:
        #
        for line in file:
            #
            (sub, rel, obj) = re.split(r"\s+", line.strip())
            for ent in (sub, obj):
                #
                if ent not in ent2int:
                    #
                    ent2int[ent] = len(ent2int)
            if rel not in rel2int:
                #
                rel2int[rel] = len(rel2int)
            buffer.append((ent2int[sub], rel2int[rel], ent2int[obj]))
    array = onp.array(buffer)
    return array, ent2int, rel2int


def count_degrees(array, n):
    #
    srcs = array[:, 0]
    dsts = array[:, -1]
    degs = onp.zeros((n,), dtype=onp.int64)
    onp.add.at(degs, srcs, 1)
    onp.add.at(degs, dsts, 1)
    return degs


def subgraph_centered_sized(
    array,
    center,
    num_nodes_exp,
    num_edges_exp,
    degs,
    cap,
    seed,
):
    #
    srcs = array[:, 0]
    dsts = array[:, -1]
    rng = onp.random.RandomState(seed)
    n = len(degs)

    #
    mask_node = onp.zeros((n,), dtype=onp.bool_)
    mask_edge = onp.zeros((len(array),), dtype=onp.bool_)
    extending = [center]
    while True:
        #
        buffer = []
        for node in extending:
            #
            bounderies = onp.unique(
                onp.array(srcs[dsts == node].tolist() + dsts[srcs == node].tolist()),
            )
            bounderies = bounderies[onp.logical_not(mask_node[bounderies])]
            if len(bounderies) > cap:
                #
                weights = degs[bounderies]
                choice = rng.choice(
                    len(bounderies),
                    cap,
                    replace=False,
                    p=weights / weights.sum(),
                )
                bounderies = bounderies[choice]
            buffer.extend(bounderies.tolist())
        buffer = onp.unique(onp.array(buffer)).tolist()

        #
        assert not onp.any(mask_node[buffer]).item()

        #
        mask_node_buf = mask_node.copy()
        mask_edge_buf = mask_edge.copy()
        mask_node_buf[buffer] = True
        mask_edge_buf[onp.logical_and(mask_node_buf[srcs], mask_node_buf[dsts])] = True

        #
        if len(buffer) == 0 or (
            onp.sum(mask_node_buf).item() > num_nodes_exp
            and onp.sum(mask_edge_buf).item() > num_edges_exp
        ):
            #
            break
        else:
            #
            onp.copyto(mask_node, mask_node_buf)
            onp.copyto(mask_edge, mask_edge_buf)
            extending = buffer

    #
    (nodes,) = onp.nonzero(mask_node)
    order = onp.argsort(degs[nodes])[::-1][:num_nodes_exp]
    mask_node.fill(False)
    mask_node[nodes[order]] = True
    mask_edge = onp.logical_and(mask_node[srcs], mask_node[dsts])

    #
    (edges,) = onp.nonzero(mask_edge)
    order = onp.argsort(onp.sqrt(degs[srcs[edges]] * degs[dsts[edges]]))[::-1][
        :num_edges_exp,
    ]
    mask_edge.fill(False)
    mask_node.fill(False)
    mask_edge[edges[order]] = True
    mask_node[srcs[mask_edge]] = True
    mask_node[dsts[mask_edge]] = True

    #
    return array[mask_edge]


def save_triplet_raw(path, array, ent2id, rel2id):
    #
    id2ent = ["" for _ in range(len(ent2id))]
    id2rel = ["" for _ in range(len(rel2id))]
    for ent, eid in ent2id.items():
        #
        id2ent[eid] = ent
    for rel, rid in rel2id.items():
        #
        id2rel[rid] = rel
    with open(path, "w") as file:
        #
        for triplet in array:
            #
            (sid, rid, oid) = triplet.tolist()
            file.write("{:s} {:s} {:s}\n".format(id2ent[sid], id2rel[rid], id2ent[oid]))


def process(tid):
    #
    filename = "rel_triples_{:d}".format(tid)
    (array, ent2id, rel2id) = load_triplet_raw(os.path.join(root, filename))
    degs = count_degrees(array, len(ent2id))
    assert degs.sum().item() == len(array) * 2

    #
    center = onp.argsort(degs)[-1].item()
    subarray = subgraph_centered_sized(
        array,
        center,
        len(ent2id) if num_nodes_exp_opt is None else num_nodes_exp_opt,
        len(array) if num_edges_exp_opt is None else num_edges_exp_opt,
        degs.astype(onp.float64),
        len(ent2id) if cap_opt is None else cap_opt,
        42,
    )
    num_nodes_sub = len(
        onp.unique(onp.reshape(subarray[:, [0, -1]], (len(subarray) * 2,))),
    )
    num_edges_sub = len(subarray)

    #
    (title, pair) = RENAME_FULL[dataset]
    directory = os.path.join("data", "-".join((title, pair[tid - 1])))
    os.makedirs(directory)
    save_triplet_raw(
        os.path.join(directory, "full.txt"),
        subarray,
        ent2id,
        rel2id,
    )

    #
    subgraph = nx.Graph()
    subgraph.add_edges_from(subarray[:, [0, -1]].tolist())
    assert num_nodes_sub == subgraph.number_of_nodes()
    assert num_edges_sub >= subgraph.number_of_edges()

    #
    print("-".join((title, pair[tid - 1])))
    print("- #Nodes: {:d}".format(num_nodes_sub))
    print("- #Edges: {:d}".format(num_edges_sub))
    print("- #Relations: {:d}".format(len(onp.unique(subarray[:, 1]))))


#
process(1)
process(2)
