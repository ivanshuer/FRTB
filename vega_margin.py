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

    def net_sensitivities(self, pos, params):
        risk_class = pos.RiskClass.unique()[0]

        pos['Stat_Value'] = pos['Stat_Value'] * pos['ImpliedVol']

        if risk_class == 'Equity':
            pos = pd.merge(pos, params.Equity_Weights, left_on=['Bucket'], right_on=['bucket'], how='inner')
            pos['AmountUSD'] = pos['AmountUSD'] * pos['weight'] * math.sqrt(365.0 / 14) / norm.ppf(0.99)
            pos.drop(['bucket', 'weight'], axis=1, inplace=True)
        elif risk_class == 'Commodity':
            pos = pd.merge(pos, params.Commodity_Weights, left_on=['Bucket'], right_on=['bucket'], how='inner')
            pos['AmountUSD'] = pos['AmountUSD'] * pos['weight'] * math.sqrt(365.0 / 14) / norm.ppf(0.99)
            pos.drop(['bucket', 'weight'], axis=1, inplace=True)

        if risk_class == 'IR':
            factor_group = ['CombinationID', 'RiskType', 'Bucket', 'Label2', 'Label3', 'RiskClass']
        elif risk_class == 'FX':
            factor_group = ['CombinationID', 'RiskType', 'Bucket', 'Label3', 'RiskClass']
        elif risk_class in ['CreditQ', 'CreditNonQ']:
            factor_group = ['CombinationID', 'ProductClass', 'RiskType', 'Qualifier', 'Bucket', 'Label1', 'RiskClass']
        elif risk_class in ['Equity', 'Commodity']:
            factor_group = ['CombinationID', 'ProductClass', 'RiskType', 'Qualifier', 'Bucket', 'RiskClass']

        pos_gp = pos.groupby(factor_group)
        pos_vega = pos_gp.agg({'Stat_Value': np.sum})
        pos_vega.reset_index(inplace=True)

        return pos_vega

    def find_factor_idx(self, maturity_factor, residual_maturity_factor, maturities, residual_maturities):
        idx = 0

        for maturity in maturities:
            if len(residual_maturities) > 0:
                for residual_maturity in residual_maturities:
                    if maturity_factor == maturity and residual_maturity_factor == residual_maturity:
                        return idx
                    else:
                        idx = idx + 1
            else:
                if maturity_factor == maturity:
                    return idx
                else:
                    idx = idx + 1

        return -1

    def build_risk_factors(self, pos_gp, params):

        risk_class = pos_gp.RiskClass.unique()[0]

        if risk_class == 'IR':
            s = np.zeros(len(params.IR_Vega_Maturity) * len(params.IR_Vega_Residual_Maturity))

            for i, row in pos_gp.iterrows():
                idx = self.find_factor_idx(row['Label3'], row['Label2'], params.IR_Vega_Maturity, params.IR_Vega_Residual_Maturity)
                if idx >= 0:
                    s[idx] = row['Stat_Value']
        elif risk_class == 'FX':
            s = np.zeros(len(params.FX_Vega_Maturity))

            for i, row in pos_gp.iterrows():
                idx = self.find_factor_idx(row['Label3'], [], params.FX_Vega_Maturity, [])
                if idx >= 0:
                    s[idx] = row['Stat_Value']

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
            LH = params.IR_LH
        elif risk_class == 'CreditQ':
            VRW = params.CreditQ_VRW
        elif risk_class == 'CreditNonQ':
            VRW = params.CreditNonQ_VRW
        elif risk_class == 'Equity':
            VRW = params.Equity_VRW
        elif risk_class == 'Commodity':
            VRW = params.Commodity_VRW
        elif risk_class == 'FX':
            LH = params.FX_LH

        VRW = min(params.Vega_VRW * math.sqrt(LH) / math.sqrt(10), 1)

        return VRW

    def build_in_bucket_correlation(self, pos_gp, params):
        risk_class = pos_gp.RiskClass.unique()[0]

        if risk_class == 'IR':
            num_maturities = len(params.IR_Vega_Maturity)
            num_residual_maturities = len(params.IR_Vega_Residual_Maturity)

            maturity_years = [mlib.convert_tenor_to_years(tenor) for tenor in params.IR_Vega_Maturity]
            residual_maturity_years = [mlib.convert_tenor_to_years(tenor) for tenor in params.IR_Vega_Residual_Maturity]

            rho = np.zeros((num_maturities, num_maturities))
            for i in range(num_maturities):
                for j in range(num_maturities):
                    rho[i, j] = math.exp(-params.Vega_Alpha * abs(maturity_years[i] - maturity_years[j]) / min(maturity_years[i], maturity_years[j]))

            fai = np.zeros((num_residual_maturities, num_residual_maturities))
            for i in range(num_residual_maturities):
                for j in range(num_residual_maturities):
                    fai[i, j] = math.exp(-params.Vega_Alpha * abs(residual_maturity_years[i] - residual_maturity_years[j]) / min(residual_maturity_years[i], residual_maturity_years[j]))

            Corr = np.kron(rho, fai)
        elif risk_class == 'FX':
            num_maturities = len(params.FX_Vega_Maturity)

            maturity_years = [mlib.convert_tenor_to_years(tenor) for tenor in params.FX_Vega_Maturity]

            rho = np.zeros((num_maturities, num_maturities))
            for i in range(num_maturities):
                for j in range(num_maturities):
                    rho[i, j] = math.exp(-params.Vega_Alpha * abs(maturity_years[i] - maturity_years[j]) / min(maturity_years[i], maturity_years[j]))

            fai = np.ones((pos_gp.Bucket.nunique(), pos_gp.Bucket.nunique()))

            Corr = np.kron(fai, rho)

        for i in range(len(Corr)):
            for j in range(len(Corr)):
                Corr[i, j] = min(Corr[i, j], 1)

        return Corr

    def margin_risk_group(self, gp, params):

        risk_class = gp.RiskClass.unique()[0]

        logger.info('Calculate {0} Vega Margin for {1}'.format(risk_class, gp.Bucket.unique()))

        s = self.build_risk_factors(gp, params)
        RW = self.build_risk_weights(gp, params)

        WS = RW * s

        Corr = self.build_in_bucket_correlation(gp, params)

        K = np.mat(WS) * np.mat(Corr) * np.mat(np.reshape(WS, (len(WS), 1)))
        K = max(K.item(0), 0)
        K = math.sqrt(K)

        ret = gp[['CombinationID', 'RiskType', 'RiskClass']].copy()
        ret.drop_duplicates(inplace=True)
        ret['K'] = K
        ret['S'] = WS.sum()
        ret['S_alt'] = max(min(WS.sum(), K), -K)
        ret['Group'] = gp['Bucket'].unique()[0]

        return ret