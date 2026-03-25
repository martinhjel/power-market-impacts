uprate_values = {
    "nore_i": {
        "eff": 1700 / 1500,
        "uprate": 2,
        "reservoirs": ["nore_1"],
        "elspot_area": "NO1",
    },  # Source: https://constructionreviewonline.com/news/statkraft-submits-plan-to-upgrade-nore-power-plant-in-norway-for-4-billion-nok/?utm_source=chatgpt.com
    "nore_ii": {
        "eff": 1700 / 1500,
        "uprate": 2,
        "reservoirs": ["nore_2"],
        "elspot_area": "NO1",
    },  # Source: https://constructionreviewonline.com/news/statkraft-submits-plan-to-upgrade-nore-power-plant-in-norway-for-4-billion-nok/?utm_source=chatgpt.com
    "mauranger": {
        "eff": (1150 + 75) / 1150,
        "uprate": 880 / 250,  # Source: https://no.wikipedia.org/wiki/Mauranger_kraftverk
        "reservoirs": ["mauranger"],
        "elspot_area": "NO5",
    },
    "aura": {
        "eff": 1.05,  # Assumed, 290 MW installed
        "uprate": 810 / 310,  # Source: https://energiwatch.no/nyheter/fornybar/article18204332.ece
        "reservoirs": ["aura"],
        "elspot_area": "NO3",
    },
    "osbu": {
        "eff": 1.05,  # Assumed, 20 MW installed
        "uprate": 810 / 310,  # Source: https://energiwatch.no/nyheter/fornybar/article18204332.ece
        "reservoirs": ["osbu"],
        "elspot_area": "NO3",
    },
    "alta": {
        "eff": ((100 + 150) / 2 + 694.7) / 694.7,  # 150 MW installed
        "uprate": (120 + 150)
        / 150,  # Source: https://www.nve.no/media/17040/alta-kraftverk-a3-prosjektbeskrivelse-med-utredningsprogram.pdf
        "reservoirs": ["alta"],
        "elspot_area": "NO4",
    },
    "svean": {
        "eff": (10 + 129.8) / 129.8,
        "uprate": 36
        / 27,  # Source: https://www.statkraft.no/om-statkraft/hvor-vi-har-virksomhet/norge/svean-vannkraftverk/, https://www.statkraft.no/presserom/nyheter-og-pressemeldinger/2025/statkraft-signerer-avtaler-for-kraftverk-verdt-12-milliarder-kroner/
        "reservoirs": ["svean"],
        "elspot_area": "NO3",
    },  # RSK development. Assumed flat 2x of the entire system with +5% increased efficiency
    "suldal_i": {
        "eff": 1.05,
        "uprate": 2,
        "reservoirs": ["suldal_1"],
        "elspot_area": "NO2",
    },
    "suldal_ii": {
        "eff": 1.05,
        "uprate": 2,
        "reservoirs": ["suldal_2"],
        "elspot_area": "NO2",
    },
    "roeldal": {
        "eff": 1.05,
        "uprate": 2,
        "reservoirs": ["roeldal"],
        "elspot_area": "NO2",
    },
    "kvanndal": {
        "eff": 1.05,
        "uprate": 2,
        "reservoirs": ["kvanndal"],
        "elspot_area": "NO2",
    },
    "novle": {
        "eff": 1.05,
        "uprate": 2,
        "reservoirs": ["novle"],
        "elspot_area": "NO2",
    },
    "svandalsflona": {
        "eff": 1.05,
        "uprate": 2,
        "reservoirs": ["svandalsflon"],
        "elspot_area": "NO2",
    },
}