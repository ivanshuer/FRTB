import numpy as np
import pandas as pd
import os
import logging
import math
import re
import margin_lib as mlib
from vega_margin import VegaMargin

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

class CurvatureMargin(object):

    def __init__(self):
        self.__margin = 'Curvature'

    def margin_type(self):
        return self.__margin

    def net_sensitivities(self, pos, params):
        risk_class = pos.RiskClass.unique()[0]

        if risk_class == 'IR':
            factor_group = ['CombinationID', 'RiskType', 'Bucket', 'Qualifier', 'RiskClass']
        elif risk_class == 'CSR':
            factor_group = ['CombinationID', 'RiskType', 'Bucket', 'issuer_id', 'Qualifier', 'RiskClass']
        elif risk_class == 'CreditNonQ':
            factor_group = ['CombinationID', 'ProductClass', 'RiskType', 'Qualifier', 'Bucket', 'Label1', 'RiskClass']
        elif risk_class in ['Equity', 'Commodity']:
            factor_group = ['CombinationID', 'ProductClass', 'RiskType', 'Qualifier', 'Bucket', 'RiskClass']
        elif risk_class == 'FX':
            factor_group = ['CombinationID', 'ProductClass', 'RiskType', 'Qualifier', 'RiskClass']

        pos_gp = pos.groupby(factor_group)
        pos_delta = pos_gp.agg({'Stat_Value': np.sum, 'Raw_PV_Base': np.sum, 'Shifted_PV_Base': np.sum})
        pos_delta.reset_index(inplace=True)

        return pos_delta

    def build_risk_factors(self, pos_gp, params):
        risk_class = pos_gp.RiskClass.unique()[0]

        if risk_class == 'IR':
            RW = params.IR_Weights
            RW = RW.weight.max()
            factor = 'Bucket'

            # bucket = pos_gp.Bucket.unique()[0]
            # if bucket in params.IR_Adjust_Curr:
            #     RW = RW / math.sqrt(2)
        elif risk_class == 'CSR':
            RW = params.CSR_Weights
            RW = RW.weight.max()
            factor = 'issuer_id'

        factors = pos_gp[factor].sort_values().unique().tolist()
        s = np.zeros(len(factors))

        idx = 0
        for i in factors:
            pos_gp_factor = pos_gp[pos_gp[factor] == i].copy()
            CVR_up = pos_gp_factor[pos_gp_factor.Qualifier == 'UP'].copy()
            CVR_down = pos_gp_factor[pos_gp_factor.Qualifier == 'DOWN'].copy()

            CVR_up['CVR'] = CVR_up['Shifted_PV_Base'] - CVR_up['Raw_PV_Base'] - RW * CVR_up['Stat_Value']
            CVR_down['CVR'] = CVR_down['Shifted_PV_Base'] - CVR_down['Raw_PV_Base'] + RW * CVR_down['Stat_Value']

            CVR = -min(CVR_up['CVR'].values[0], CVR_down['CVR'].values[0])
            s[idx] = CVR
            idx = idx + 1

        return s

    def build_in_bucket_correlation(self, pos_gp, params):
        risk_class = pos_gp.RiskClass.unique()[0]

        if risk_class == 'IR':
            Corr = np.zeros((pos_gp.Bucket.nunique(), pos_gp.Bucket.nunique()))
        elif risk_class == 'CSR':
            num_issuers = pos_gp.issuer_id.nunique()

            rho_issuers = np.zeros((num_issuers, num_issuers))
            rho_issuers.fill(params.CSR_Rho_Name)
            np.fill_diagonal(rho_issuers, 0)

            Corr = rho_issuers

        Corr = pow(Corr, 2)

        return Corr

    def margin_risk_group(self, gp, params):

        risk_class = gp.RiskClass.unique()[0]

        logger.info('Calculate {0} Curvature Margin for {1}'.format(risk_class, gp.Bucket.unique()))

        s = self.build_risk_factors(gp, params)
        WS = s

        Corr = self.build_in_bucket_correlation(gp, params)

        for i in range(len(s)):
            for j in range(len(s)):
                if s[i] < 0 and s[j] < 0:
                    Corr[i, j] = 0

        K = np.mat(WS) * np.mat(Corr) * np.mat(np.reshape(WS, (len(WS), 1)))
        max_s = [max(x, 0) for x in s]
        K = K + np.dot(max_s, max_s)
        K = max(0, K.item(0))
        K = math.sqrt(K)

        if risk_class == 'CSR' and gp.Bucket.unique()[0] in params.CSR_Others:
            K = abs(WS).sum()

        ret = gp[['CombinationID', 'RiskType', 'RiskClass']].copy()
        ret.drop_duplicates(inplace=True)
        ret['K'] = K
        ret['S'] = WS.sum()
        ret['S_alt'] = max(min(WS.sum(), K), -K)
        ret['Group'] = gp['Bucket'].unique()[0]

        return ret