# %%
from pathlib import Path

import graphviz
from lpr_sintef_bifrost.models import EMPSModelBuilder
from lpr_sintef_bifrost.models._collection import BiFrostCollection
from lpr_sintef_bifrost.models.common import (
    Busbar,
    Bypass,
    HydraulicCoupling,
    InflowSeries,
    Plant,
    Pump,
    Reservoir,
    Spill,
)
from lpr_sintef_bifrost.models.connection import ObjectType

def to_graphviz(config, ignore_busbars: bool = False):  # pragma: no cover
    """
    Export the BifrostRunConfig to a graphviz dot file.

    :param ignore_busbars: Whether to ignore busbars in the graph.

    :return:
    """
    watercourses = []
    for res in config.reservoirs:
        watercourses.append(res.metadata.watercourse)

    dot = graphviz.Digraph(comment="BifrostRunConfig")
    for item in config.__dict__.values():
        if isinstance(item, BiFrostCollection):
            for i in item.items:
                connection_name = f"{i.connection_type.value}{i.name}"
                prettified_name = i.name.capitalize().replace("_", " ")

                if isinstance(i, Busbar):
                    if ignore_busbars:
                        continue
                    dot.node(connection_name, prettified_name, shape="polygon")
                elif isinstance(i, Reservoir):
                    dot.node(
                        connection_name,
                        prettified_name,
                        shape="invtrapezium",
                        fillcolor="blue",
                        style="filled",
                        fontcolor="white",
                    )
                elif isinstance(i, Plant):
                    dot.node(connection_name, prettified_name, shape="box", color="yellow", style="filled")
                elif isinstance(i, Spill):
                    dot.node(connection_name, "spill", shape="plain", fontsize="10", fontcolor="red")
                elif isinstance(i, Bypass):
                    dot.node(connection_name, "bypass", shape="plain", fontsize="10", fontcolor="blue")
                elif isinstance(i, HydraulicCoupling):
                    dot.node(connection_name, prettified_name, shape="diamond", color="green", style="filled")
                elif isinstance(i, Pump):
                    dot.node(connection_name, prettified_name, shape="house", color="lightgreen", style="filled")
                elif isinstance(i, InflowSeries):
                    pass
                else:
                    dot.node(connection_name, prettified_name)

    for connection in config.get_connections():
        if ignore_busbars and (connection.from_type == ObjectType.BUSBAR or connection.to_type == ObjectType.BUSBAR):
            continue

        color = "black"
        style = "solid"
        if connection.from_type == ObjectType.SPILL or connection.to_type == ObjectType.SPILL:
            color = "red"
            style = "dashed"
        elif connection.from_type == ObjectType.BYPASS or connection.to_type == ObjectType.BYPASS:
            color = "blue"
            style = "dashed"

        dot.edge(
            f"{connection.from_type.value}{connection.from_name}",
            f"{connection.to_type.value}{connection.to_name}",
            color=color,
            style=style,
        )
    return dot
    
    

dataset_files = [i for i in (Path.cwd() / "ltm_output").glob("*/*") if i.is_dir()]
config = EMPSModelBuilder.from_json(filepath=dataset_files[1] / "results/model.json")
ignore_busbars = False
dot = to_graphviz(config=config, ignore_busbars=ignore_busbars)

dot.render(f"images/hydro_system_ignore_bussbar_{ignore_busbars}", format="pdf", cleanup=True)
    
    # return dot
