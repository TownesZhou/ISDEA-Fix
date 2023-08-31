"""
    Generate Wikidata-Topics dataset. For each topic of relations, create 3 separate datasets:
    - transductive. Default is 10000 entities. It consists of the following files:
        - entities.dict   # The list of entities.
        - relations.dict  # The list of relations.
        - train.txt       # The list of training triplets.
        - valid.txt       # The list of validation triplets.
    - inductive. Same number of entities as transductive, and sampled i.i.d. from the same distribution. It consists of
        the following files:
        - entities.dict   # The list of entities.
        - relations.dict  # The list of relations.
        - observe.txt     # The list of observed triplets (i.e. input triplets the model can see).
        - test.txt        # The list of test triplets (i.e. output triplets the model should predict).
    - inductive-small. Default is 1000 entities, and sampled i.i.d. from the same distribution. It consists of the
        following files:
        - entities.dict   # The list of entities.
        - relations.dict  # The list of relations.
        - observe.txt     # The list of observed triplets (i.e. input triplets the model can see).
        - test.txt        # The list of test triplets (i.e. output triplets the model should predict).

    Required file:
    - All raw triplets from Wikidata5m. Specify the path in the argument "--raw". File must be in the format of
        "head[WHITESPACE]relation[WHITESPACE]tail\n".
    - Relation aliases of the topic. Specify the path in the argument "--topic-relations". File must be in the format of
        "relation[WHITESPACE]descriptions\n".
"""
from typing import List, Set
import argparse
import pandas as pd
import numpy as np
import networkx as nx
import mmap
import os
from tqdm import tqdm
from littleballoffur import ForestFireSampler as Sampler

# Dataframe progress_apply
tqdm.pandas()


def get_num_lines(file_path: str) -> int:
    """
    Utility method to get number of lines in a file.
    """
    fp = open(file_path, "r+")
    buf = mmap.mmap(fp.fileno(), 0)
    lines = 0
    while buf.readline():
        lines += 1
    return lines


def load_triplets(path_triplets: str) -> pd.DataFrame:
    """
    Load triplets from file.
    """
    heads, rels, tails = [], [], []
    # Read from file line by line
    with open(path_triplets, 'r') as file:
        for line in tqdm(file, total=get_num_lines(path_triplets)):
            # Split line by whitespace
            line_parts = line.split()
            # Check there should only be three parts
            assert len(line_parts) == 3
            heads.append(line_parts[0])
            rels.append(line_parts[1])
            tails.append(line_parts[2])
    # Create dataframe
    trip_df = pd.DataFrame(
        data={
            'head': heads,
            'rel': rels,
            'tail': tails
        },
        columns=['head', 'rel', 'tail']
    )
    return trip_df


def load_relations(path_relation_alias:str) -> List[str]:
    """
    Load relation aliases from file.

    Args:
        path_relation_alias: Path to relation aliases file.

    Returns:
        A list of relation aliases.
    """
    rel_ids = []
    # Read from file line by line
    with open(path_relation_alias, 'r') as file:
        for line in tqdm(file, total=get_num_lines(path_relation_alias)):
            # Split line by whitespace
            line_parts = [part.strip() for part in line.split()]
            # First part is the id. We don't need the description in the later part.
            rel_ids.append(line_parts[0])
    return rel_ids


def create_graph(entity_ids: np.ndarray, triplets: pd.DataFrame) -> nx.MultiDiGraph:
    """
    Create a multi-directed graph from the given entities and triplets
    Args:
        entity_ids: Array of entity names.
        triplets: triplet dataframe.

    Returns:
        A multi-directed graph representing the knowledge graph
    """
    topic_graph = nx.MultiDiGraph()
    # Add nodes
    # Node are ent_id, which has string type
    # for ent_id, i in tqdm(topic_ent2id.items(), total=len(topic_ent2id)):
    #     topic_graph.add_node(i, ent_id=ent_id)
    for ent_id in tqdm(entity_ids):
        topic_graph.add_node(ent_id)
    # Iterate over triplet dataframe and add individual edges
    for row in tqdm(triplets.itertuples(index=False), total=triplets.shape[0]):
        topic_graph.add_edge(
            row.head,
            row.tail,
            rel_id=row.rel
        )
    return topic_graph


def create_split(sampled_graph: nx.MultiDiGraph, split_ratio: float, seed: float)\
        -> (Set[str], Set[str], pd.DataFrame, pd.DataFrame):
    """
    Create the observe and label split from the sampled graphs. Ensures that none of the triplets in the label split is
    unseen in the observe split.

    Args:
        sampled_graph: The sampled graph to create the split from.
        split_ratio: The ratio of the observe split to the label split.
        seed: The random seed to use for the split.

    Returns:
        A tuple of (entity_set, relation_set, observe_df, label_df).
    """
    # Create the dataframe for the sampled graph
    trip_df_data = list(sampled_graph.edges(data='rel_id'))
    trip_sampled_df = pd.DataFrame(
        data=trip_df_data,
        columns=['head', 'tail', 'rel']
    )
    # Rearange columns
    trip_sampled_df = trip_sampled_df[['head', 'rel', 'tail']]

    # Create the unique entities and relations set
    # Entity set obtained directly from the graph
    # Relation set obtained from the dataframe
    entity_sampled_set = set(sampled_graph.nodes)
    relation_sampled_set = trip_sampled_df['rel'].unique()
    print("# of total entities:", len(entity_sampled_set))
    print("# of unique relations:", len(relation_sampled_set))
    # Print size of full sampled graph
    print("# of total triplets:", len(sampled_graph.edges))
    # Print number of label triplets to sample
    n_label_triplets = int(len(sampled_graph.edges) * (1 - split_ratio))
    print("# of valid/test label triplets to be sampled:", n_label_triplets)

    # Iteratively sample
    n_max_trials = 10
    n_cur = 0
    n_no_improve = 0
    epoch_i = 0
    trip_cur_df = None
    trip_remain_df = trip_sampled_df.copy(deep=True)
    while n_cur < n_label_triplets:
        n_remain = n_label_triplets - n_cur
        print(f"Start epoch {epoch_i + 1}. Sampling remaining {n_remain} triplets")
        epoch_i += 1

        # Sample from remain
        new_sampled_df = trip_remain_df.sample(n=n_remain, random_state=seed)
        # Take current remain part
        remain_ids = list(set(trip_remain_df.index) - set(new_sampled_df.index))
        trip_new_remain_df = trip_remain_df.loc[remain_ids]

        # Check which sampled entities are not in the new remain df
        new_sampled_entities = set(new_sampled_df['head']).union(set(new_sampled_df['tail']))
        new_remain_entities = set(trip_new_remain_df['head']).union(set(trip_new_remain_df['tail']))
        entities_diff = new_sampled_entities - new_remain_entities

        print(f"\tIn total of {len(entities_diff)} sampled entities found to be isolated")

        # Remove these isolated entities from the new samples
        fixed_sampled_df = new_sampled_df.loc[
            (~new_sampled_df['head'].isin(entities_diff)) &
            (~new_sampled_df['tail'].isin(entities_diff))
            ]
        n_cur += fixed_sampled_df.shape[0]
        print(f"\tIn total of {n_cur} valid samples so far")

        # Concat with existing samples
        if trip_cur_df is None:
            trip_cur_df = fixed_sampled_df
        else:
            trip_cur_df = pd.concat([trip_cur_df, fixed_sampled_df], axis=0)

        # Take the new remaining part
        remain_ids = list(set(trip_remain_df.index) - set(trip_cur_df.index))
        trip_remain_df = trip_remain_df.loc[remain_ids]

        # See if no improvements
        if fixed_sampled_df.shape[0] == 0:
            print(f"\t\tNo Improvements!")
            n_no_improve += 1

        # assert n_no_improve < n_max_trials, "No improvement patience has been reached."
        if n_no_improve >= n_max_trials:
            print("### No improvement patience has been reached. ###")
            break
    #
    trip_sampled_label_df = trip_cur_df
    trip_sampled_observe_df = trip_remain_df

    # Print number of observe and label triplets
    print("# of sampled label triplets:", trip_sampled_label_df.shape[0])
    print("# of sampled observe triplets:", trip_sampled_observe_df.shape[0])

    # Sanity Check: See how many entities in label that is not seen in observer, and what is the ratio
    sampled_observe_entities = set(trip_sampled_observe_df['head']).union(set(trip_sampled_observe_df['tail']))
    sampled_label_entities = set(trip_sampled_label_df['head']).union(set(trip_sampled_label_df['tail']))
    sampled_entities_diff = sampled_label_entities - sampled_observe_entities
    print("# unexpected label entities:", len(sampled_entities_diff))
    print("Ratio of # unexpected label entities / total # label entities:",
          len(sampled_entities_diff) / len(sampled_label_entities))

    # Return unique entity set, unique relation set, observe triplets, and label triplets
    return entity_sampled_set, relation_sampled_set, trip_sampled_observe_df, trip_sampled_label_df


def save_entities(filename: str, entities: Set[str], sep: str):
    """
    Save entities to file.

    Args:
        filename: Path to file.
        entities: Set of entities.
        sep: Separator between entity integer id and entity name.
    """
    with open(filename, 'w') as file:
        for i, entity in enumerate(tqdm(entities)):
            line = sep.join([str(i), entity])
            line += "\n"
            file.write(line)


def save_relations(filename: str, relations: Set[str], sep: str):
    """
    Save relations to file.

    Args:
        filename: Path to file.
        relations: Set of relations.
        sep: Separator between relation integer id and relation name.
    """
    with open(filename, 'w') as file:
        for i, relation in enumerate(tqdm(relations)):
            line = sep.join([str(i), relation])
            line += "\n"
            file.write(line)


def main():
    # Parse arguments.
    parser = argparse.ArgumentParser()

    # Define and add arguments.
    parser.add_argument("--data-name", type=str, required=True, help="Name of the dataset.")
    parser.add_argument("--raw", type=str, required=True, help="Path to raw triplets.")
    parser.add_argument("--topic-relations", type=str, required=True, help="Path to topic relations aliases.")
    parser.add_argument("--out-dir", type=str, required=True, help="Path to output directory.")
    parser.add_argument("--n-entities", type=int, default=10000,
                        help="Number of entities to sample for transductive and inductive dataset.")
    parser.add_argument("--n-entities-small", type=int, default=1000,
                        help="Number of entities to sample for inductive-small dataset.")
    parser.add_argument("--split-ratio", type=float, default=0.9,
                        help="Ratio of triplets to split into train/observe and valid/test.")
    parser.add_argument("--burning-prob", type=float, default=0.8,
                        help="Burning probability for forest fire sampling.")
    parser.add_argument("--n-processes", type=int, default=4, help="Number of processes to use.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    # Load all raw triplets.
    print("Loading all raw triplets...")
    # trip_df = load_triplets(args.raw, args.n_processes)
    trip_df = load_triplets(args.raw)

    # Load topic relations.
    print("Loading topic relations...")
    rel_topic_ids = load_relations(args.topic_relations)

    # Check that all topic relations are in the raw triplets.
    print("Checking if topic relations are referenced in the raw triplets...")
    rel_topic_checked = []
    all_rels = trip_df['rel'].unique()
    for rel in tqdm(rel_topic_ids):
        if rel in all_rels:
            rel_topic_checked.append(rel)
        else:
            print(f"WARNING: Relation ID {rel} not found in the dataset!")

    # Extract triplets corresponding to topic relations.
    trip_topic_df = trip_df.loc[trip_df['rel'].isin(rel_topic_checked)]
    # Create union of head and tail entity set
    head_tail_entities = pd.concat([trip_topic_df['head'], trip_topic_df['tail']])
    entity_ids = head_tail_entities.unique()

    # Create topic graph.
    print("Creating topic graph...")
    topic_graph = create_graph(entity_ids, trip_topic_df)

    # Create the undirected graph and take the largest undirected component
    print("Creating undirected graph and taking the largest component...")
    topic_graph_undirected = topic_graph.to_undirected()
    topic_graph_undirected_largest_nodes = max(nx.connected_components(topic_graph_undirected), key=len)

    # Get the induced undirected and directed subgraph corresponding to the largest component,
    #   and compute connectivity
    print("Getting induced subgraphs and computing connectivity...")
    topic_graph_undirected_largest = topic_graph_undirected.subgraph(topic_graph_undirected_largest_nodes)
    # topic_graph_directed_largest = topic_graph.subgraph(topic_graph_undirected_largest_nodes)
    # print("\tIs the undirected largest component connected?",
    #       nx.is_connected(topic_graph_undirected_largest))
    # print("\tIs the directed largest component weakly connected?",
    #       nx.is_weakly_connected(topic_graph_directed_largest))
    # print("\tIs the directed largest component strongly connected?",
    #       nx.is_strongly_connected(topic_graph_directed_largest))

    # Downsample the transductive, inductive, and inductive-small graphs
    # Reindex the graph and create the target graph to sample from
    print("Reindexing the target graph to sample from...")
    node_mapping = {
        ent_id: i for i, ent_id in
        enumerate(tqdm(topic_graph_undirected_largest.nodes))
    }
    target_graph = nx.relabel_nodes(
        topic_graph_undirected_largest,
        node_mapping,
        copy=True
    )

    # Sample using Forest Fire algorithm. Sample one graph for each dataset
    print("Sampling using Forest Fire algorithm...")
    sampler_trans = Sampler(number_of_nodes=args.n_entities, p=args.burning_prob, seed=args.seed)
    sampler_ind = Sampler(number_of_nodes=args.n_entities, p=args.burning_prob, seed=args.seed + 1)
    sampler_small = Sampler(number_of_nodes=args.n_entities_small, p=args.burning_prob, seed=args.seed + 2)
    sampled_undirected_graphs = {
        "trans": sampler_trans.sample(target_graph),
        "ind": sampler_ind.sample(target_graph),
        "ind-small": sampler_small.sample(target_graph)
    }
    # Get the induced, directed subgraphs of the sampled undirected graphs
    node_mapping_inverse = list(topic_graph_undirected_largest.nodes)
    sampled_directed_graphs = {}
    for split, undirected in sampled_undirected_graphs.items():
        nodes_int = list(undirected.nodes)
        nodes_ids = [node_mapping_inverse[i] for i in nodes_int]
        sampled_directed_graphs[split] = topic_graph.subgraph(nodes_ids)

    # Create the observe and label triplets split for each of the transductive, inductive, and inductive-small graphs
    print("Creating observe and label triplets split...")
    sampled_splits = {}
    for split, sampled_graph in sampled_directed_graphs.items():
        print(f"\n### Creating split for the {split} dataset ###\n")
        e_set, _, observe, label = create_split(sampled_graph, args.split_ratio, args.seed)
        sampled_splits[split] = {
            "entities": e_set,
            # For consistency, we use pre-defined relations, even though they may not be all present in the sampled graph
            "relations": set(rel_topic_ids),
            "observe": observe,
            "label": label
        }

    # Save the sampled splits
    print("Saving the sampled splits...")
    # Separator
    sep = "    "

    # Create the directory if it does not exist
    dir_trans_topic = os.path.join(args.out_dir, f"{args.data_name}-trans")
    dir_ind_topic = os.path.join(args.out_dir, f"{args.data_name}-ind")
    dir_ind_small_topic = os.path.join(args.out_dir, f"{args.data_name}_small-ind")
    os.makedirs(dir_trans_topic, exist_ok=True)
    os.makedirs(dir_ind_topic, exist_ok=True)
    os.makedirs(dir_ind_small_topic, exist_ok=True)

    # Save entities.dict
    save_entities(os.path.join(dir_trans_topic, "entities.dict"),
                  sampled_splits['trans']['entities'], sep=sep)
    save_entities(os.path.join(dir_ind_topic, "entities.dict"),
                  sampled_splits['ind']['entities'], sep=sep)
    save_entities(os.path.join(dir_ind_small_topic, "entities.dict"),
                  sampled_splits['ind-small']['entities'], sep=sep)

    # Save relations.dict
    save_relations(os.path.join(dir_trans_topic, "relations.dict"),
                   sampled_splits['trans']['relations'], sep=sep)
    save_relations(os.path.join(dir_ind_topic, "relations.dict"),
                   sampled_splits['ind']['relations'], sep=sep)
    save_relations(os.path.join(dir_ind_small_topic, "relations.dict"),
                   sampled_splits['ind-small']['relations'], sep=sep)

    # Save train.txt and observe.txt
    def save_observe(filename, observe_df):
        observe_df.to_csv(filename, sep=" ", header=False, index=False)
    save_observe(os.path.join(dir_trans_topic, "train.txt"),
                 sampled_splits['trans']['observe'])
    save_observe(os.path.join(dir_ind_topic, "observe.txt"),
                 sampled_splits['ind']['observe'])
    save_observe(os.path.join(dir_ind_small_topic, "observe.txt"),
                 sampled_splits['ind-small']['observe'])

    # Save valid.txt and test.txt
    def save_label(filename, label_df):
        label_df.to_csv(filename, sep=" ", header=False, index=False)

    save_label(os.path.join(dir_trans_topic, "valid.txt"),
               sampled_splits['trans']['label'])
    save_label(os.path.join(dir_ind_topic, "test.txt"),
               sampled_splits['ind']['label'])
    save_label(os.path.join(dir_ind_small_topic, "test.txt"),
               sampled_splits['ind-small']['label'])

    print("Done!")


if __name__ == "__main__":
    main()
