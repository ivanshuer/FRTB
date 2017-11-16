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

    if freq == 'M':
        years = unit / 12
    elif freq == 'Y':
        years = unit

    return years

def change_FX_ticker_order(gp):
    curr1 = gp['Bucket'][0:3]
    curr2 = gp['Bucket'][3:6]

    curr_pair = [curr1, curr2]
    curr_pair.sort()
    curr_pair = "".join(curr_pair)

    gp['Bucket'] = curr_pair

    return gp

def build_bucket_correlation(pos_delta, params, margin):
    risk_class = pos_delta.RiskClass.unique()[0]

    g = 0

    if risk_class == 'IR':
        all_curr = pos_delta.Group.unique()
        g = np.ones((len(all_curr), len(all_curr)))
        g.fill(params.IR_Gamma)
        np.fill_diagonal(g, 1)
    elif risk_class == 'FX':
        g = np.ones((pos_delta.Group.nunique(), pos_delta.Group.nunique()))
        g.fill(params.FX_Gamma)
        np.fill_diagonal(g, 1)
    elif risk_class == 'CSR':
        g_sector = params.CSR_Sector_Corr

        CSR_buckets = params.CSR_IG + params.CSR_HY
        g_rating = np.ones((len(CSR_buckets),len(CSR_buckets)))
        for i in range(len(CSR_buckets)):
            for j in range(len(CSR_buckets)):
                if (CSR_buckets[i] in params.CSR_IG and CSR_buckets[j] in params.CSR_HY) or \
                        (CSR_buckets[i] in params.CSR_HY and CSR_buckets[j] in params.CSR_IG):
                    g_rating[i, j] = params.CSR_Rho_Rating

        g = g_rating * g_sector
        g.columns = CSR_buckets
        g.index = CSR_buckets
        buckets = pos_delta.Group.unique()
        g = g.ix[buckets, buckets]

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