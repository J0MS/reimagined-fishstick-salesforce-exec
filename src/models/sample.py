"""Samples for RCT-PC API."""

"""Power calculator input sample."""
power_calulator_sample = {
    "EXPERIMENTAL_FACTORS": {
        'promo_type': [['smdc', 'bau', 'no promo'], [35, 15, 25], 2],
        'execution_type': [['display', 'fridge', 'no ex'], [60, 45, 125], 3],
        'visits': [['frequent', 'no', 'seldom'], [7, 3, 4], 3]
    },
    "PRIMARY_OUTCOME_TYPE": "CONTINUOUS",
    "HISTORICAL_STD": 0.5,
    "HISTORICAL_PROPORTION": 0.3,
    "NOMINAL_PVALUE": 0.1,
    "SCOPE_SIZE": 9645,
    "MIN_EFFECT_SIZE": 0.02,
    "EFFECTIVE_PVALUE": 0.05,
    "POWER": 0.2577644642998115
}

"""Random Control Trial minimal input sample."""
rct_sample = {
    "EXP_ID": "123618",
    "EXP_NAME": "Experiment1",
    "EXP_UNIT": "store_id",
    "EXP_MARKET": "Mexico",
    "PRIMARY_OUTCOME_NAME": "net_revenue",
    "SCOPE_SIZE": 542,
    "BLOCKING_FACTORS": [
        "DRV",
        "POC_SEGMENTATION"
    ],
    "EXPERIMENTAL_FACTORS": {'execution_type': [['display', 'fridge'], [200, 100], 1],
                             'promo_type': [['smdc', 'bau', 'no promo'], [50, 100, 150], 2]},
    "CONTROL_GROUP": {
        "promo_type": "bau"
    }
}

"""Random Control Trial standard input sample."""
experiment_data_sample = {
            "EXP_NAME": "TestExperiment",
            "EXP_STATUS": "SUBMITTED",
            "EXP_MESSAGE": "CREATED",
            "EXP_MARKET": "Mexico",
            "EXP_TYPE": "Test_Experiment",
            "EXP_UNIT": "POC",
            "EXPERIMENT_START_DATE": "2021-04-19",
            "EXPERIMENT_END_DATE": "2021-05-19",
            "BASELINE_START_DATE": "2021-04-19",
            "BASELINE_END_DATE": "2021-05-19",
            "ANALYSIS_START_DATE": "2021-04-19",
            "ANALYSIS_END_DATE": "2021-05-19",
            "BLOCKING_FACTORS": [
                {
                    "id": "block_factor_1",
                    "selected": 1,
                    "name": "DRV",
                    "column": "sales_management"
                },
                {
                    "id": "block_factor_2",
                    "selected": 1,
                    "name": "REGION",
                    "column": "region"
                },
                {
                    "id": "block_factor_3",
                    "selected": 1,
                    "name": "Size Segmentation",
                    "column": "poc_segmentation"
                },
                {
                    "id": "block_factor_4",
                    "selected": 0,
                    "name": "",
                    "column": ""
                },
                {
                    "id": "block_factor_5",
                    "selected": 0,
                    "name": "",
                    "column": ""
                }
            ],
            "EXPERIMENTAL_FACTORS": {
              "promo_type": [
                [
                  "smdc",
                  "bau",
                  "nopromo"
                ],
                [
                  128989,
                  100000,
                  100000
                ],
                3,
                [
                  "smdcDescription",
                  "bauDescription",
                  "nopromoDescription"
                ],
                [
                ]
              ],
            },
            "SCOPE_SIZE": 328989,
            "CREATED_BY_NAME": "Test Automation",
            "CREATED_BY_EMAIL": "test@ab-inbev.com"
        }

