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
            factor_group = ['CombinationID', 'RiskType', 'Bucket', 'Label1', 'Label2', 'RiskClass']
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
        pos_delta = pos_gp.agg({'Stat_Value': np.sum})
        pos_delta.reset_index(inplace=True)

        # if there exists inflation, need to aggregate amount by each currency
        pos_inflation = pos[pos.RiskType == 'Risk_Inflation'].copy()
        if len(pos_inflation) > 0:
            pos_inflation = pos_inflation.groupby(['CombinationID', 'RiskType', 'Bucket', 'RiskClass']).agg({'AmountUSD': np.sum})
            pos_inflation.reset_index(inplace=True)

            pos_delta = pd.concat([pos_delta, pos_inflation])

        return pos_delta

    def find_factor_idx(self, tenor_factor, curve_factor, tenors, curves, risk_class):
        idx = 0

        if risk_class == 'IR':
            for curve in curves:
                for tenor in tenors:
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

            # Sort curves alphabetically
            curves = pos_gp.Label1.sort_values().unique().tolist()

            s = np.zeros(len(params.IR_Tenor) * len(curves))
            if len(pos_inflation) > 0:
                s = np.zeros(len(params.IR_Tenor) * len(curves) + 1)

            for i, row in pos_gp.iterrows():
                idx = self.find_factor_idx(row['Label2'], row['Label1'], params.IR_Tenor, curves, risk_class)
                if idx >= 0:
                    s[idx] = row['Stat_Value']

            if len(pos_inflation) > 0:
                s[len(s) - 1] = pos_inflation.Stat_Value

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
            RW = params.IR_Weights
            RW = RW.weight.tolist()
            curves = pos_gp.Label1.unique()

            RW = RW * len(curves)

            pos_inflation = pos_gp[pos_gp.RiskType == 'Risk_Inflation'].copy()
            if len(pos_inflation) > 0:
                RW = np.append(RW, params.IR_Inflation_Weights)

            RW = np.array(RW)
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

    def margin_risk_group(self, gp, params):

        risk_class = gp.RiskClass.unique()[0]

        logger.info('Calculate {0} Delta Margin for {1}'.format(risk_class, gp.Bucket.unique()))

        s = self.build_risk_factors(gp, params)
        RW = self.build_risk_weights(gp, params)

        WS = RW * s

        Corr = mlib.build_in_bucket_correlation(gp, params, self.__margin)

        K = np.mat(WS) * np.mat(Corr) * np.mat(np.reshape(WS, (len(WS), 1)))
        K = max(K, 0)
        K = math.sqrt(K.item(0))

        if gp.RiskType.nunique() > 1:
            risk_type = '_'.join(gp.RiskType.unique())
        else:
            risk_type = gp.RiskType.unique()[0]

        ret = gp[['CombinationID', 'RiskClass']].copy()
        ret.drop_duplicates(inplace=True)
        ret['RiskType'] = risk_type
        ret['K'] = K
        ret['S'] = max(min(WS.sum(), K), -K)

        if risk_class == 'IR':
            ret['Group'] = gp['Bucket'].unique()[0]
        elif risk_class == 'FX':
            ret['Group'] = gp['RiskType'].unique()[0]
        else:
            ret['Group'] = gp['Bucket'].unique()[0]

        return ret


