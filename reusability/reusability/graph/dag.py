def recursive_count(node):
    """Recursively count the usage of all source nodes

    (Illustration purposes only)"""
    node.count += 1

    for source in get_sources(node):
        recursive_count(source)


def horizontal_unroll_count(leaf_nodes):
    """Recursively count the total number of contexts for all nodes

    (Illustration purposes only)"""

    for node in leaf_nodes:
        recursive_count(node)


def get_sources(node):
    return [e.source for e in node.input_edges]
