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
        elif risk_class == 'CSR':
            factor_group = ['CombinationID', 'RiskType', 'Bucket', 'issuer_id', 'Label1', 'Label2', 'RiskClass']
        elif risk_class == 'CreditNonQ':
            factor_group = ['CombinationID', 'ProductClass', 'RiskType', 'Qualifier', 'Bucket', 'Label1', 'RiskClass']
        elif risk_class in ['Equity', 'Commodity']:
            factor_group = ['CombinationID', 'ProductClass', 'RiskType', 'Qualifier', 'Bucket', 'RiskClass']
        elif risk_class == 'FX':
            factor_group = ['CombinationID', 'RiskType', 'Bucket', 'RiskClass']

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

        if risk_class in ['IR', 'CSR']:
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

        elif risk_class == 'FX':
            s = np.ones(pos_gp.Bucket.nunique()) * pos_gp.Stat_Value.values[0]

        elif risk_class == 'CSR':
            tenors = params.CSR_Tenor
            issuers = pos_gp.issuer_id.sort_values().unique().tolist()

            s = np.zeros(pos_gp.issuer_id.nunique() * len(tenors))

            for i, row in pos_gp.iterrows():
                idx = self.find_factor_idx(row['Label2'], row['issuer_id'], tenors, issuers, risk_class)
                if idx >= 0:
                    s[idx] = row['Stat_Value']

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
        elif risk_class == 'FX':
            RW = params.FX_Weights
        else:
            if risk_class == 'CSR':
                weights = params.CSR_Weights
                num_factors = pos_gp.issuer_id.nunique() * len(params.CSR_Tenor)
            elif risk_class == 'CreditNonQ':
                weights = params.CreditNonQ_Weights
                num_factors = pos_gp.Qualifier.nunique() * len(params.CreditNonQ_Tenor)
            elif risk_class == 'Equity':
                weights = params.Equity_Weights
                num_factors = pos_gp.Qualifier.nunique()
            elif risk_class == 'Commodity':
                weights = params.Commodity_Weights
                num_factors = pos_gp.Qualifier.nunique()

            bucket = pd.DataFrame(pos_gp.Bucket.unique(), columns=['bucket'])
            RW = pd.merge(bucket, weights, left_on=['bucket'], right_on=['bucket'], how='inner')
            RW = np.array(RW.weight.values[0])


            RW = np.repeat(RW, num_factors)

        return RW

    def build_in_bucket_correlation(self, pos_gp, params):
        risk_class = pos_gp.RiskClass.unique()[0]

        if risk_class == 'IR':
            num_tenors = len(params.IR_Tenor)
            tenor_years = [mlib.convert_tenor_to_years(tenor) for tenor in params.IR_Tenor]

            # Same curve, diff vertex
            rho = np.zeros((num_tenors, num_tenors))
            for i in range(num_tenors):
                for j in range(num_tenors):
                    rho[i, j] = max(math.exp(-params.IR_Theta * abs(tenor_years[i] - tenor_years[j]) / min(tenor_years[i], tenor_years[j])), 0.4)

            curves = pos_gp.Label1.unique()
            fai = np.zeros((len(curves), len(curves)))
            fai.fill(params.IR_Fai)
            np.fill_diagonal(fai, 1)

            Corr = np.kron(fai, rho)

            pos_inflation = pos_gp[pos_gp.RiskType == 'Risk_Inflation'].copy()
            if len(pos_inflation) > 0:
                inflation_rho = np.ones(len(curves)*len(params.IR_Tenor)) * params.IR_Inflation_Rho
                inflation_rho_column = np.reshape(inflation_rho, (len(inflation_rho), 1))
                Corr = np.append(Corr, inflation_rho_column, axis=1)

                inflation_rho = np.append(inflation_rho, 1)
                inflation_rho = np.reshape(inflation_rho, (1, len(inflation_rho)))
                Corr = np.append(Corr, inflation_rho, axis=0)
        elif risk_class == 'FX':
            Corr = np.ones((pos_gp.Bucket.nunique(), pos_gp.Bucket.nunique()))
        elif risk_class == 'CSR':
            num_tenors = len(params.CSR_Tenor)
            num_issuers = pos_gp.issuer_id.nunique()

            rho_issuers = np.zeros((num_issuers, num_issuers))
            rho_issuers.fill(params.CSR_Rho_Name)
            np.fill_diagonal(rho_issuers, 1)

            rho_tenors = np.zeros((num_tenors, num_tenors))
            rho_tenors.fill(params.CSR_Rho_Tenor)
            np.fill_diagonal(rho_tenors, 1)

            Corr = np.kron(rho_issuers, rho_tenors)

        else:

            if risk_class in ['CreditQ', 'CreditNonQ']:
                if risk_class == 'CreditQ':
                    tenors = params.CreditQ_Tenor
                    same_is_rho = params.CreditQ_Rho_Agg_Same_IS
                    diff_is_rho = params.CreditQ_Rho_Agg_Diff_IS
                    if bucket == 'Residual':
                        same_is_rho = params.CreditQ_Rho_Res_Same_IS
                        diff_is_rho = params.CreditQ_Rho_Res_Diff_IS
                else:
                    tenors = params.CreditNonQ_Tenor
                    same_is_rho = params.CreditNonQ_Rho_Agg_Same_IS
                    diff_is_rho = params.CreditNonQ_Rho_Agg_Diff_IS
                    if bucket == 'Residual':
                        same_is_rho = params.CreditNonQ_Rho_Res_Same_IS
                        diff_is_rho = params.CreditNonQ_Rho_Res_Diff_IS

                rho = np.ones((num_qualifiers, num_qualifiers)) * diff_is_rho
                np.fill_diagonal(rho, same_is_rho)

                if risk_class == 'CreditQ' and margin == 'Delta':
                    one_mat = np.ones((len(tenors) * params.CreditQ_num_sec_type, len(tenors) * params.CreditQ_num_sec_type))
                else:
                    one_mat = np.ones((len(tenors), len(tenors)))
                rho = np.kron(rho, one_mat)

            elif risk_class in ['Equity', 'Commodity']:
                bucket_df = pd.DataFrame(pos_gp.Bucket.unique(), columns=['bucket'])

                if risk_class == 'Equity':
                    bucket_params = params.Equity_Rho
                elif risk_class == 'Commodity':
                    bucket_params = params.Commodity_Rho

                rho = pd.merge(bucket_df, bucket_params, left_on=['bucket'], right_on=['bucket'], how='inner')
                rho = rho['corr'][0]

            if margin == 'Curvature':
                rho = rho * rho
                F.fill(1)

            Corr = rho * F
            np.fill_diagonal(Corr, 1)

        return Corr

    def margin_risk_group(self, gp, params):

        risk_class = gp.RiskClass.unique()[0]

        logger.info('Calculate {0} Delta Margin for {1}'.format(risk_class, gp.Bucket.unique()))

        s = self.build_risk_factors(gp, params)
        RW = self.build_risk_weights(gp, params)

        WS = RW * s

        Corr = self.build_in_bucket_correlation(gp, params)

        K = np.mat(WS) * np.mat(Corr) * np.mat(np.reshape(WS, (len(WS), 1)))
        K = max(K.item(0), 0)
        K = math.sqrt(K)

        if risk_class == 'CSR' and gp.Bucket.unique()[0] in params.CSR_Others:
            K = abs(WS).sum()

        if gp.RiskType.nunique() > 1:
            risk_type = '_'.join(gp.RiskType.unique())
        else:
            risk_type = gp.RiskType.unique()[0]

        ret = gp[['CombinationID', 'RiskClass']].copy()
        ret.drop_duplicates(inplace=True)
        ret['RiskType'] = risk_type
        ret['K'] = K
        ret['S'] = WS.sum()
        ret['S_alt'] = max(min(WS.sum(), K), -K)
        ret['Group'] = gp['Bucket'].unique()[0]

        return ret


