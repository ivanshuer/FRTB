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
        self.__vega_loader = VegaMargin()

    def margin_type(self):
        return self.__margin

    def calc_scaling(self, gp):
        label = gp.Label1.unique()[0]
        [space, num, freq] = re.split('(\d+)', label)
        num = float(num)

        if freq == 'w':
            num_days = num * 7
        elif freq == 'm':
            num_days = num/12 * 365
        elif freq == 'y':
            num_days = num * 365
        else:
            logger.info('wrong label 1 {0}'.format(label))
            gp['SF'] = 1.0
            return gp

        scale = 0.5 * min(1, 14 / num_days)
        gp['SF'] = scale

        return gp

    def input_scaling(self, pos):

        pos = pos.groupby(['Label1']).apply(self.calc_scaling)
        pos['AmountUSD'] = pos['AmountUSD'] * pos['SF']

        return pos

    def net_sensitivities(self, pos, params):
        return self.__vega_loader.net_sensitivities(pos, params)

    def build_risk_factors(self, pos_gp, params):
        return self.__vega_loader.build_risk_factors(pos_gp, params)

    def calculate_CR_Threshold(self, gp, params):
        return self.__vega_loader.calculate_CR_Threshold(gp, params)

    def calculate_risk_group(self, gp, params):
        return self.__vega_loader.calculate_risk_group(gp, params)

    def calculate_CR_Threshold(self, gp, params):
        return self.__vega_loader.calculate_CR_Threshold(gp, params)

    def margin_risk_group(self, gp, params):

        risk_class = gp.RiskClass.unique()[0]

        if risk_class in ['IR', 'FX']:
            logger.info('Calculate {0} Curvature Margin for {1}'.format(risk_class, gp.Qualifier.unique()))
        else:
            logger.info('Calculate {0} Curvature Margin for {1}'.format(risk_class, gp.Bucket.unique()))

        gp = gp.apply(self.calculate_risk_group, axis=1, params=params)
        gp = self.calculate_CR_Threshold(gp, params)

        s = self.build_risk_factors(gp, params)
        CR = mlib.build_concentration_risk(gp, params, self.margin_type())

        WS = s

        Corr = mlib.build_in_bucket_correlation(gp, params, self.margin_type(), CR)

        K = np.mat(WS) * np.mat(Corr) * np.mat(np.reshape(WS, (len(WS), 1)))
        K = math.sqrt(K.item(0))

        ret = gp[['CombinationID', 'ProductClass', 'RiskType', 'RiskClass']].copy()
        ret.drop_duplicates(inplace=True)
        ret['K'] = K
        ret['S'] = max(min(WS.sum(), K), -K)

        ret['CVR_sum'] = WS.sum()
        ret['CVR_abs_sum'] = abs(WS).sum()

        if risk_class == 'IR':
            ret['Group'] = gp['Qualifier'].unique()[0]
        elif risk_class == 'FX':
            ret['Group'] = gp['RiskType'].unique()[0]
        else:
            ret['Group'] = gp['Bucket'].unique()[0]

        return ret