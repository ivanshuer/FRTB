import numpy as np
import pandas as pd
import os
import logging
import math
import margin_lib as mlib

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

class DeltaMargin(object):

    def __init__(self):
        self.__margin = 'Delta'

    def margin_type(self):
        return self.__margin

    def net_sensitivities(self, pos, params):
        risk_class = pos.RiskClass.unique()[0]

        if risk_class == 'IR':
            factor_group = ['CombinationID', 'ProductClass', 'RiskType', 'Qualifier', 'Bucket', 'Label1', 'Label2', 'RiskClass']
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
        pos_delta = pos_gp.agg({'AmountUSD': np.sum})
        pos_delta.reset_index(inplace=True)

        # if there exists inflation, need to aggregate amount by each currency
        pos_inflation = pos[pos.RiskType == 'Risk_Inflation'].copy()
        if len(pos_inflation) > 0:
            pos_inflation = pos_inflation.groupby(['CombinationID', 'ProductClass', 'RiskType', 'Qualifier', 'RiskClass']).agg({'AmountUSD': np.sum})
            pos_inflation.reset_index(inplace=True)

            pos_delta = pd.concat([pos_delta, pos_inflation])

        return pos_delta

    def find_factor_idx(self, tenor_factor, curve_factor, tenors, curves, risk_class):
        idx = 0

        if risk_class == 'IR':
            for tenor in tenors:
                for curve in curves:
                    if tenor_factor == tenor and curve_factor == curve:
                        return idx
                    else:
                        idx = idx + 1

        elif risk_class in ['CreditQ', 'CreditNonQ']:
            for tenor in tenors:
                if tenor_factor == tenor:
                    return idx
                else:
                    idx = idx + 1

        return -1

    def build_risk_factors(self, pos_gp, params):

        risk_class = pos_gp.RiskClass.unique()[0]

        if risk_class == 'IR':
            pos_inflation = pos_gp[pos_gp.RiskType == 'Risk_Inflation'].copy()

            gp_curr = pos_gp.Qualifier.unique()[0]

            curve = params.IR_Sub_Curve
            if gp_curr == 'USD':
                curve = params.IR_USD_Sub_Curve

            s = np.zeros(len(params.IR_Tenor) * len(curve))
            if len(pos_inflation) > 0:
                s = np.zeros(len(params.IR_Tenor) * len(curve) + 1)

            for i, row in pos_gp.iterrows():
                idx = self.find_factor_idx(row['Label1'], row['Label2'], params.IR_Tenor, curve, risk_class)
                if idx >= 0:
                    s[idx] = row['AmountUSD']

            if len(pos_inflation) > 0:
                s[len(s) - 1] = pos_inflation.AmountUSD

        elif risk_class == 'CreditQ':
            tenors = params.CreditQ_Tenor

            # 2 for securitization
            s = np.zeros(pos_gp.Qualifier.nunique() * params.CreditQ_num_sec_type * len(tenors))

            pos_gp.sort_values(['Qualifier', 'Label2'], inplace=True, ascending=True)

            for j in range(pos_gp.Qualifier.nunique()):
                pos_gp_qualifier = pos_gp[pos_gp.Qualifier == pos_gp.Qualifier.unique()[j]].copy()

                pos_gp_qualifier_non_sec = pos_gp_qualifier[pos_gp_qualifier.Label2 == 'Non_Sec'].copy()

                for i, row in pos_gp_qualifier_non_sec.iterrows():
                    idx = self.find_factor_idx(row['Label1'], [], tenors, [], risk_class)
                    if idx >= 0:
                        s[idx + j * len(tenors) * params.CreditQ_num_sec_type] = row['AmountUSD']

                pos_gp_qualifier_sec = pos_gp_qualifier[pos_gp_qualifier.Label2 == 'Sec'].copy()

                for i, row in pos_gp_qualifier_sec.iterrows():
                    idx = self.find_factor_idx(row['Label1'], [], tenors, [], risk_class)
                    if idx >= 0:
                        s[idx + j * len(tenors) * params.CreditQ_num_sec_type + len(tenors)] = row['AmountUSD']

        elif risk_class == 'CreditNonQ':
            tenors = params.CreditNonQ_Tenor

            s = np.zeros(pos_gp.Qualifier.nunique() * len(tenors))

            for j in range(pos_gp.Qualifier.nunique()):
                pos_gp_qualifier = pos_gp[
                    pos_gp.Qualifier == pos_gp.sort_values(['Qualifier']).Qualifier.unique()[j]].copy()

                for i, row in pos_gp_qualifier.iterrows():
                    idx = self.find_factor_idx(row['Label1'], [], tenors, [], risk_class)
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
            bucket = pd.DataFrame(pos_gp.Bucket.unique(), columns=['curr_type'])
            RW = pd.merge(bucket, params.IR_Weights, left_on=['curr_type'], right_on=['curr'], how='inner')
            RW = RW.drop(['curr_type', 'curr'], axis=1)
            RW = RW.as_matrix()

            gp_curr = pos_gp.Qualifier.unique()[0]

            curve = params.IR_Sub_Curve
            if gp_curr == 'USD':
                curve = params.IR_USD_Sub_Curve

            RW = np.repeat(RW, len(curve))

            pos_inflation = pos_gp[pos_gp.RiskType == 'Risk_Inflation'].copy()
            if len(pos_inflation) > 0:
                RW = np.append(RW, params.IR_Inflation_Weights)
        else:
            if risk_class == 'CreditQ':
                weights = params.CreditQ_Weights
                num_factors = pos_gp.Qualifier.nunique() * len(params.CreditQ_Tenor) * params.CreditQ_num_sec_type
            elif risk_class == 'CreditNonQ':
                weights = params.CreditNonQ_Weights
                num_factors = pos_gp.Qualifier.nunique() * len(params.CreditNonQ_Tenor)
            elif risk_class == 'Equity':
                weights = params.Equity_Weights
                num_factors = pos_gp.Qualifier.nunique()
            elif risk_class == 'Commodity':
                weights = params.Commodity_Weights
                num_factors = pos_gp.Qualifier.nunique()
            elif risk_class == 'FX':
                weights = params.FX_Weights
                num_factors = pos_gp.Qualifier.nunique()

            if risk_class != 'FX':
                bucket = pd.DataFrame(pos_gp.Bucket.unique(), columns=['bucket'])
                RW = pd.merge(bucket, weights, left_on=['bucket'], right_on=['bucket'], how='inner')
                RW = np.array(RW.weight.values[0])
            else:
                RW = np.array([weights])

            RW = np.repeat(RW, num_factors)

        return RW

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
            if gp['Bucket'] in params.CreditQ_CR_Sov_incl_Central_Banks:
                risk_group = 'Sovereigns including central banks'
            elif gp['Bucket'] in params.CreditQ_CR_Corp_Entities:
                risk_group = 'Corporate entities'
            elif gp['Bucket'] in params.CreditQ_CR_Not_Classified:
                risk_group = 'Not classified'

        elif gp['RiskClass'] == 'CreditNonQ':
            if gp['Bucket'] in params.CreditNonQ_CR_IG:
                risk_group = 'IG (RMBS and CMBS)'
            elif gp['Bucket'] in params.CreditNonQ_CR_HY_Non_Rated:
                risk_group = 'HY/Non-rated (RMBS and CMBS)'
            elif gp['Bucket'] in params.CreditNonQ_CR_Not_Classified:
                risk_group = 'Not classified'

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
            if gp['Qualifier'] in params.FX_Significantly_Material:
                risk_group = 'C1'
            elif gp['Qualifier'] in params.FX_Frequently_Traded:
                risk_group ='C2'
            else:
                risk_group = 'C3'

        gp['Risk_Group'] = risk_group

        return gp

    def calculate_CR_Threshold(self, gp, params):

        risk_group = gp['RiskClass'].unique()[0]

        if risk_group == 'IR':
            thrd = params.IR_CR_Thrd[params.IR_CR_Thrd.Type == 'Delta'].copy()

        elif risk_group == 'CreditQ':
            thrd = params.CreditQ_CR_Thrd[params.CreditQ_CR_Thrd.Type == 'Delta'].copy()

        elif risk_group == 'CreditNonQ':
            thrd = params.CreditNonQ_CR_Thrd[params.CreditNonQ_CR_Thrd.Type == 'Delta'].copy()

        elif risk_group == 'Equity':
            thrd = params.Equity_CR_Thrd[params.Equity_CR_Thrd.Type == 'Delta'].copy()

        elif risk_group == 'Commodity':
            thrd = params.Commodity_CR_Thrd[params.Commodity_CR_Thrd.Type == 'Delta'].copy()

        elif risk_group == 'FX':
            thrd = params.FX_CR_Thrd[params.FX_CR_Thrd.Type == 'Delta'].copy()

        gp = pd.merge(gp, thrd[['Risk_Group', 'CR_THR']], how='left')

        return gp

    def margin_risk_group(self, gp, params):

        risk_class = gp.RiskClass.unique()[0]

        if risk_class in ['IR', 'FX']:
            logger.info('Calculate {0} Delta Margin for {1}'.format(risk_class, gp.Qualifier.unique()))
        else:
            logger.info('Calculate {0} Delta Margin for {1}'.format(risk_class, gp.Bucket.unique()))

        gp = gp.apply(self.calculate_risk_group, axis=1, params=params)
        gp = self.calculate_CR_Threshold(gp, params)

        s = self.build_risk_factors(gp, params)
        RW = self.build_risk_weights(gp, params)
        CR = mlib.build_concentration_risk(gp, params, self.margin_type())

        WS = RW * s * CR

        Corr = mlib.build_in_bucket_correlation(gp, params, self.__margin, CR)

        K = np.mat(WS) * np.mat(Corr) * np.mat(np.reshape(WS, (len(WS), 1)))
        K = math.sqrt(K.item(0))

        if gp.RiskType.nunique() > 1:
            risk_type = '_'.join(gp.RiskType.unique())
        else:
            risk_type = gp.RiskType.unique()[0]

        ret = gp[['CombinationID', 'ProductClass', 'RiskClass']].copy()
        ret.drop_duplicates(inplace=True)
        ret['RiskType'] = risk_type
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


