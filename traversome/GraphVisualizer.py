import toyplot


class GraphVisualizer(object):
    def __init__(self, graph, alignment=None):
        self.graph = graph
        self.alignment = alignment

    def plot(self, width=500, height=500, vlshow=True):
        plot_graph = toyplot.graph(
            [],
            [],
            width=width,
            height=height,
            tmarker=">",
            vsize=5,
            vstyle={"stroke": "black", "stroke-width": 2, "fill": "black"},
            vlshow=vlshow,
            estyle={"stroke": "black", "stroke-width": 2},
            layout=toyplot.layout.FruchtermanReingold(edges=toyplot.layout.CurvedEdges())
        )
