import pandas as pd
from lpr_sintef_bifrost.ltm import LTM


def get_market_steps(selected_path, selected_nodes: list[str]) -> pd.DataFrame:
    pyltm_session = LTM.session_from_folder(selected_path / "run_folder/emps")
    pyltm_model = pyltm_session.model

    market_step_data = []
    for busbar_name in selected_nodes:
        for market_step in pyltm_model.market_steps():
            if busbar_name == market_step.busbar_name:
                market_step_data.append(
                    {
                        "capacity": market_step.capacity.scenarios,
                        "name": market_step.name,
                        "price": getattr(market_step.price, "scenarios", market_step.price),
                        "busbar": busbar_name,
                    }
                )

    return pd.DataFrame(market_step_data)
