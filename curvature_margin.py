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
        elif risk_class == 'CreditQ':
            pos.ix[pos.Label2.isnull(), 'Label2'] = 'Non_Sec'
            factor_group = ['CombinationID', 'ProductClass', 'RiskType', 'Qualifier', 'Bucket', 'Label1', 'Label2', 'RiskClass']
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

            CVR_up = pos_gp[pos_gp.Qualifier == 'UP'].copy()
            CVR_down = pos_gp[pos_gp.Qualifier == 'DOWN'].copy()

            CVR_up['CVR'] = CVR_up['Shifted_PV_Base'] - CVR_up['Raw_PV_Base'] - RW * CVR_up['Stat_Value']
            CVR_down['CVR'] = CVR_down['Shifted_PV_Base'] - CVR_down['Raw_PV_Base'] + RW * CVR_down['Stat_Value']

            CVR = -min(CVR_up['CVR'].values[0], CVR_down['CVR'].values[0])
            s = np.ones(pos_gp.Bucket.nunique()) * CVR

        return s

    def build_in_bucket_correlation(self, pos_gp, params):
        risk_class = pos_gp.RiskClass.unique()[0]

        if risk_class == 'IR':
            Corr = np.zeros((pos_gp.Bucket.nunique(), pos_gp.Bucket.nunique()))

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

        ret = gp[['CombinationID', 'RiskType', 'RiskClass']].copy()
        ret.drop_duplicates(inplace=True)
        ret['K'] = K
        ret['S'] = WS.sum()
        ret['S_alt'] = max(min(WS.sum(), K), -K)

        if risk_class == 'IR':
            ret['Group'] = gp['Bucket'].unique()[0]
        elif risk_class == 'FX':
            ret['Group'] = gp['RiskType'].unique()[0]
        else:
            ret['Group'] = gp['Bucket'].unique()[0]

        return ret