import numpy as np
import pandas as pd
import os
import logging
from math import exp
import re

##############################
# Setup Logging Configuration
##############################
logger = logging.getLogger(os.path.basename(__file__))
if not len(logger.handlers):
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s|%(name)s === %(message)s ===', datefmt='%Y-%m-%d %I:%M:%S')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    file_handler = logging.FileHandler('log.txt')
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
###############################


def convert_tenor_to_years(tenor):
    # convert tenor to year fraction

    freq = tenor[re.search('\D', tenor).start():]
    unit = float(tenor[:re.search('\D', tenor).start()])

    if freq == 'm':
        years = unit / 12
    elif freq == 'y':
        years = unit

    return years

def build_bucket_correlation(pos_delta, params, margin):
    risk_class = pos_delta.RiskClass.unique()[0]

    g = 0

    if risk_class == 'IR':
        all_curr = pos_delta.Group.unique()
        g = np.ones((len(all_curr), len(all_curr)))
        g.fill(params.IR_Gamma)
        np.fill_diagonal(g, 1)
    elif risk_class == 'CreditQ':
        g = params.CreditQ_Corr
    elif risk_class == 'CreditNonQ':
        g = params.CreditNonQ_Corr
    elif risk_class == 'Equity':
        g = params.Equity_Corr
    elif risk_class == 'Commodity':
        g = params.Commodity_Corr

    if margin == 'Curvature':
        g = pow(g, 2)

    g = np.mat(g)
    np.fill_diagonal(g, 0)

    return g

def build_non_residual_S(pos_gp, params):
    risk_class = pos_gp.RiskClass.unique()[0]
    # pos_gp has all 0 index, so has to reset for loop works
    pos_gp.reset_index(inplace=True)

    if risk_class == 'IR':
        S = pos_gp.S
    elif risk_class in ['CreditQ', 'CreditNonQ', 'Equity', 'Commodity']:
        if risk_class == 'CreditQ':
            S = np.zeros(len(params.CreditQ_Bucket) - 1)

            for i in range(len(params.CreditQ_Bucket) - 1):
                for j in range(len(pos_gp.Group)):
                    if pos_gp.Group[j] == params.CreditQ_Bucket[i]:
                        S[i] = pos_gp.S[j]
                        break

        elif risk_class == 'CreditNonQ':
            S = np.zeros(len(params.CreditNonQ_Bucket) - 1)

            for i in range(len(params.CreditNonQ_Bucket) - 1):
                for j in range(len(pos_gp.Group)):
                    if pos_gp.Group[j] == params.CreditNonQ_Bucket[i]:
                        S[i] = pos_gp.S[j]
                        break

        elif risk_class == 'Equity':
            S = np.zeros(len(params.Equity_Bucket) - 1)

            for i in range(len(params.Equity_Bucket) - 1):
                for j in range(len(pos_gp.Group)):
                    if pos_gp.Group[j] == params.Equity_Bucket[i]:
                        S[i] = pos_gp.S[j]
                        break

        elif risk_class == 'Commodity':
            S = np.zeros(len(params.Commodity_Bucket))

            for i in range(len(params.Commodity_Bucket)):
                for j in range(len(pos_gp.Group)):
                    if pos_gp.Group[j] == params.Commodity_Bucket[i]:
                        S[i] = pos_gp.S[j]
                        break

    elif risk_class == 'FX':
        S = 0

    return S