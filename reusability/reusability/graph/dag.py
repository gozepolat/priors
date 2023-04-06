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


class Edge:
    def __init__(self, source=None, target=None):
        self.source = source
        self.target = target


def get_targets(node):
    return [e.target for e in node.output_edges]


def get_sources(node):
    return [e.source for e in node.input_edges]


def get_target_names(node):
    return {e.target.name for e in node.output_edges}


def get_source_names(node):
    return {e.source.name for e in node.input_edges}


def get_target_edge(node, name):
    for e in node.output_edges:
        if e.target.name == name:
            return e
    return None


def get_source_edge(node, name):
    for e in node.input_edges:
        if e.source.name == name:
            return e
    return None


class Node:
    def __init__(self, name=''):
        self.output_edges = []
        self.input_edges = []
        self.name = name
        self.depth = -1
        self.count = 0
        self.shared = None
        self.masked = False

    def __str__(self):
        return self.name

    def __repr__(self):
        return f"Node: {self.name}"

    def remove_target(self, target, is_source_removed=False):
        num_targets = len(self.output_edges)
        for i in range(num_targets):
            current_target = self.output_edges[i].source
            if current_target.name == target.name:
                if not is_source_removed:
                    current_target.remove_source(self, is_target_removed=True)
                self.output_edges = self.output_edges[:i] + self.output_edges[i + 1:]
                return True

        return False

    def remove_source(self, source, is_target_removed=False):
        num_sources = len(self.input_edges)
        for i in range(num_sources):
            current_source = self.input_edges[i].source
            if current_source.name == source.name:
                if not is_target_removed:
                    current_source.remove_target(self, is_source_removed=True)
                # TODO switch to sets instead, will order ever matter? Not likely.
                self.input_edges = self.input_edges[:i] + self.input_edges[i + 1:]
                return True

        return False

    def add_output_target(self, target):
        """Add an output node, create an edge pointing towards the node

        Make sure that there is no duplicated edge towards the target"""
        if target.name in get_target_names(self):
            edge = get_target_edge(self, target.name)
            edge.target = target

            if self.name not in get_source_names(target):
                edge.target.add_input_edge(edge)
            else:
                source_edge = get_source_edge(target, self.name)
                source_edge.source = self

        edge = Edge(self, target)
        self.add_output_edge(edge)

    def add_output_edge(self, edge):
        assert edge.source == self
        if edge.target.name not in get_target_names(self):
            self.output_edges.append(edge)
        edge.target.add_input_edge(edge)

    def add_input_edge(self, edge):
        assert edge.target == self
        if edge.source.name not in get_source_names(self):
            self.input_edges.append(edge)
        else:
            source_edge = get_source_edge(self, edge.source.name)
            source_edge.source = edge.source


class DAG(Node):
    def __init__(self, name='root'):
        super().__init__(name)
        self.weight_set = set()
        self.attributes = {}

    def __repr__(self):
        return f"DAG: {self.name}"

    def add_weight(self, node):
        self.weight_set.add(node)
