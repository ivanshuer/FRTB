import numpy as np
import pandas as pd
import os
import logging
import math
import margin_lib as mlib
from scipy.stats import norm

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

class VegaMargin(object):

    def __init__(self):
        self.__margin = 'Vega'

    def margin_type(self):
        return self.__margin

    def change_FX_ticker_order(self, gp):
        curr1 = gp['Qualifier'][0:3]
        curr2 = gp['Qualifier'][3:6]

        curr_pair = set([curr1, curr2])
        curr_pair = "".join(curr_pair)

        gp['Qualifier'] = curr_pair

        return gp

    def net_sensitivities(self, pos, params):
        risk_class = pos.RiskClass.unique()[0]

        if risk_class == 'Equity':
            pos = pd.merge(pos, params.Equity_Weights, left_on=['Bucket'], right_on=['bucket'], how='inner')
            pos['AmountUSD'] = pos['AmountUSD'] * pos['weight'] * math.sqrt(365.0 / 14) / norm.ppf(0.99)
            pos.drop(['bucket', 'weight'], axis=1, inplace=True)
        elif risk_class == 'Commodity':
            pos = pd.merge(pos, params.Commodity_Weights, left_on=['Bucket'], right_on=['bucket'], how='inner')
            pos['AmountUSD'] = pos['AmountUSD'] * pos['weight'] * math.sqrt(365.0 / 14) / norm.ppf(0.99)
            pos.drop(['bucket', 'weight'], axis=1, inplace=True)
        elif risk_class == 'FX':
            pos['AmountUSD'] = pos['AmountUSD'] * params.FX_Weights * math.sqrt(365.0 / 14) / norm.ppf(0.99)

        if risk_class == 'IR':
            factor_group = ['CombinationID','ProductClass', 'RiskType', 'Qualifier', 'Label1', 'RiskClass']
        elif risk_class == 'FX':
            pos = pos.apply(self.change_FX_ticker_order, axis=1)
            factor_group = ['CombinationID', 'ProductClass', 'RiskType', 'Qualifier', 'Label1', 'RiskClass']
        elif risk_class in ['CreditQ', 'CreditNonQ']:
            factor_group = ['CombinationID', 'ProductClass', 'RiskType', 'Qualifier', 'Bucket', 'Label1', 'RiskClass']
        elif risk_class in ['Equity', 'Commodity']:
            factor_group = ['CombinationID', 'ProductClass', 'RiskType', 'Qualifier', 'Bucket', 'RiskClass']

        pos_gp = pos.groupby(factor_group)
        pos_vega = pos_gp.agg({'AmountUSD': np.sum})
        pos_vega.reset_index(inplace=True)

        return pos_vega

    def find_factor_idx(self, tenor_factor, tenors):
        idx = 0

        for tenor in tenors:
            if tenor_factor == tenor:
                return idx
            else:
                idx = idx + 1

        return -1

    def build_risk_factors(self, pos_gp, params):

        risk_class = pos_gp.RiskClass.unique()[0]

        if risk_class == 'IR':
            s = np.zeros(len(params.IR_Tenor))

            for i, row in pos_gp.iterrows():
                idx = self.find_factor_idx(row['Label1'], params.IR_Tenor)
                if idx >= 0:
                    s[idx] = row['AmountUSD']
        elif risk_class in ['CreditQ', 'CreditNonQ']:
            if risk_class == 'CreditQ':
                tenors = params.CreditQ_Tenor
            if risk_class == 'CreditNonQ':
                tenors = params.CreditNonQ_Tenor

            s = np.zeros(pos_gp.Qualifier.nunique() * len(tenors))

            for j in range(pos_gp.Qualifier.nunique()):
                pos_gp_qualifier = pos_gp[pos_gp.Qualifier == pos_gp.sort_values(['Qualifier']).Qualifier.unique()[j]].copy()

                for i, row in pos_gp_qualifier.iterrows():
                    idx = self.find_factor_idx(row['Label1'], tenors)
                    if idx >= 0:
                        s[idx + j * len(tenors)] = row['AmountUSD']
        else:
            s = np.zeros(pos_gp.Qualifier.nunique())

            for i, row in pos_gp.iterrows():
                s[i] = row['AmountUSD']

        return s

    def build_risk_weights(self, pos_gp, params):
        risk_class = pos_gp.RiskClass.unique()[0]

        if risk_class == 'IR':
            VRW = params.IR_VRW
        elif risk_class == 'CreditQ':
            VRW = params.CreditQ_VRW
        elif risk_class == 'CreditNonQ':
            VRW = params.CreditNonQ_VRW
        elif risk_class == 'Equity':
            VRW = params.Equity_VRW
        elif risk_class == 'Commodity':
            VRW = params.Commodity_VRW
        elif risk_class == 'FX':
            VRW = params.FX_VRW

        return VRW

    def calculate_risk_group(self, gp, params):
        if gp['RiskClass'] == 'IR':
            if gp['Qualifier'] in params.IR_Low_Vol_Curr:
                risk_group = 'Low volatility'
            elif gp['Qualifier'] in params.IR_Reg_Vol_Less_Well_Traded_Curr:
                risk_group = 'Regular volatility, less well-traded'
            elif gp['Qualifier'] in params.IR_Reg_Vol_Well_Traded_Curr:
                risk_group = 'Regular volatility, well-traded'
            else:
                risk_group = 'High volatility'

        elif gp['RiskClass'] == 'CreditQ':
            risk_group = 'Qualifying'

        elif gp['RiskClass'] == 'CreditNonQ':
            risk_group = 'Non Qualifying'

        elif gp['RiskClass'] == 'Equity':
            if gp['Bucket'] in params.Equity_CR_Emerging_Large_Cap:
                risk_group = 'Emerging Markets - Large Cap'
            elif gp['Bucket'] in params.Equity_CR_Developed_Large_Cap:
                risk_group = 'Developed Markets - Large Cap'
            elif gp['Bucket'] in params.Equity_CR_Emerging_Small_Cap:
                risk_group = 'Emerging Markets - Small Cap'
            elif gp['Bucket'] in params.Equity_CR_Developed_Small_Cap:
                risk_group = 'Developed Markets - Small Cap'
            elif gp['Bucket'] in params.Equity_CR_Index_Funds_ETF:
                risk_group = 'Indexeds, Funds, ETFs'
            elif gp['Bucket'] in params.Equity_CR_Not_Classified:
                risk_group = 'Not classified'

        elif gp['RiskClass'] == 'Commodity':
            if gp['Bucket'] in params.Commodity_CR_Coal:
                risk_group = 'Coal'
            elif gp['Bucket'] in params.Commodity_CR_Crude_Oil:
                risk_group = 'Crude Oil'
            elif gp['Bucket'] in params.Commodity_CR_Light_End:
                risk_group = 'Light ends'
            elif gp['Bucket'] in params.Commodity_CR_Middle_Distilates:
                risk_group = 'Middle Distilates'
            elif gp['Bucket'] in params.Commodity_CR_Heavy_Distilates:
                risk_group = 'Heavy Distilates'
            elif gp['Bucket'] in params.Commodity_CR_NA_Natual_Gas:
                risk_group = 'NA Natural gas'
            elif gp['Bucket'] in params.Commodity_CR_EU_Natual_Gas:
                risk_group = 'EU Natual gas'
            elif gp['Bucket'] in params.Commodity_CR_NA_Power:
                risk_group = 'NA Power, On-Peak'
            elif gp['Bucket'] in params.Commodity_CR_EU_Power:
                risk_group = 'EU Power, On-Peak'
            elif gp['Bucket'] in params.Commodity_CR_Freight:
                risk_group = 'Freight, Dry or Wet'
            elif gp['Bucket'] in params.Commodity_CR_Base_Metals:
                risk_group = 'Base metals'
            elif gp['Bucket'] in params.Commodity_CR_Precious_Metals:
                risk_group = 'Precious Metals'
            elif gp['Bucket'] in params.Commodity_CR_Grains:
                risk_group = 'Grains'
            elif gp['Bucket'] in params.Commodity_CR_Softs:
                risk_group = 'Softs'
            elif gp['Bucket'] in params.Commodity_CR_Livestock:
                risk_group = 'Livestock'
            elif gp['Bucket'] in params.Commodity_CR_Others:
                risk_group = 'Other / Diversified Commodity Indices'

        elif gp['RiskClass'] == 'FX':
            curr1 = gp['Qualifier'][0:3]
            curr2 = gp['Qualifier'][3:6]

            if curr1 in params.FX_Significantly_Material and curr2 in params.FX_Significantly_Material:
                risk_group = 'C1_C1'
            elif (curr1 in params.FX_Significantly_Material and curr2 in params.FX_Frequently_Traded) or \
                    (curr1 in params.FX_Frequently_Traded and curr2 in params.FX_Significantly_Material):
                risk_group = 'C1_C2'
            elif curr1 in params.FX_Significantly_Material or curr2 in params.FX_Significantly_Material:
                risk_group = 'C1_C3'
            else:
                risk_group = 'Others'

        gp['Risk_Group'] = risk_group

        return gp

    def calculate_CR_Threshold(self, gp, params):

        risk_group = gp['RiskClass'].unique()[0]

        if risk_group == 'IR':
            thrd = params.IR_CR_Thrd[params.IR_CR_Thrd.Type == 'Vega'].copy()

        elif risk_group == 'CreditQ':
            thrd = params.CreditQ_CR_Thrd[params.CreditQ_CR_Thrd.Type == 'Vega'].copy()

        elif risk_group == 'CreditNonQ':
            thrd = params.CreditNonQ_CR_Thrd[params.CreditNonQ_CR_Thrd.Type == 'Vega'].copy()

        elif risk_group == 'Equity':
            thrd = params.Equity_CR_Thrd[params.Equity_CR_Thrd.Type == 'Vega'].copy()

        elif risk_group == 'Commodity':
            thrd = params.Commodity_CR_Thrd[params.Commodity_CR_Thrd.Type == 'Vega'].copy()

        elif risk_group == 'FX':
            thrd = params.FX_CR_Thrd[params.FX_CR_Thrd.Type == 'Vega'].copy()

        gp = pd.merge(gp, thrd[['Risk_Group', 'CR_THR']], how='left')

        return gp

    def margin_risk_group(self, gp, params):

        risk_class = gp.RiskClass.unique()[0]

        if risk_class in ['IR', 'FX']:
            logger.info('Calculate {0} Vega Margin for {1}'.format(risk_class, gp.Qualifier.unique()))
        else:
            logger.info('Calculate {0} Vega Margin for {1}'.format(risk_class, gp.Bucket.unique()))

        gp = gp.apply(self.calculate_risk_group, axis=1, params=params)
        gp = self.calculate_CR_Threshold(gp, params)

        s = self.build_risk_factors(gp, params)
        RW = self.build_risk_weights(gp, params)
        CR = mlib.build_concentration_risk(gp, params, self.margin_type())

        WS = RW * s * CR

        Corr = mlib.build_in_bucket_correlation(gp, params, self.margin_type(), CR)

        K = np.mat(WS) * np.mat(Corr) * np.mat(np.reshape(WS, (len(WS), 1)))
        K = math.sqrt(K.item(0))

        ret = gp[['CombinationID', 'ProductClass', 'RiskType', 'RiskClass']].copy()
        ret.drop_duplicates(inplace=True)
        ret['K'] = K
        ret['S'] = max(min(WS.sum(), K), -K)

        if risk_class == 'IR':
            ret['CR'] = CR
        else:
            ret['CR'] = CR[0]

        if risk_class == 'IR':
            ret['Group'] = gp['Qualifier'].unique()[0]
        elif risk_class == 'FX':
            ret['Group'] = gp['RiskType'].unique()[0]
        else:
            ret['Group'] = gp['Bucket'].unique()[0]

        return ret