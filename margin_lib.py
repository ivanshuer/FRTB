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


def build_in_bucket_correlation(pos_gp, params, margin):
    risk_class = pos_gp.RiskClass.unique()[0]
    if risk_class not in ['IR', 'FX']:
        bucket = pos_gp.Bucket.unique()[0]

    if margin == 'Vega':
        if risk_class == 'IR':
            num_maturities = len(params.IR_Vega_Maturity)
            num_residual_maturities = len(params.IR_Vega_Residual_Maturity)

            maturity_years = [convert_tenor_to_years(tenor) for tenor in params.IR_Vega_Maturity]
            residual_maturity_years = [convert_tenor_to_years(tenor) for tenor in params.IR_Vega_Residual_Maturity]

            rho = np.zeros((num_maturities, num_maturities))
            for i in range(num_maturities):
                for j in range(num_maturities):
                    rho[i, j] = exp(-params.IR_Alpha * abs(maturity_years[i] - maturity_years[j]) / min(maturity_years[i], maturity_years[j]))

            fai = np.zeros((num_residual_maturities, num_residual_maturities))
            for i in range(num_residual_maturities):
                for j in range(num_residual_maturities):
                    fai[i, j] = exp(-params.IR_Alpha * abs(residual_maturity_years[i] - residual_maturity_years[j]) / min(residual_maturity_years[i], residual_maturity_years[j]))

            Corr = np.kron(rho, fai)

            for i in range(len(Corr)):
                for j in range(len(Corr)):
                    Corr[i, j] = min(Corr[i, j], 1)
    else:
        if risk_class == 'IR':
            num_tenors = len(params.IR_Tenor)
            tenor_years = [convert_tenor_to_years(tenor) for tenor in params.IR_Tenor]

            # Same curve, diff vertex
            rho = np.zeros((num_tenors, num_tenors))
            for i in range(num_tenors):
                for j in range(num_tenors):
                    rho[i, j] = max(exp(-params.IR_Theta * abs(tenor_years[i] - tenor_years[j]) / min(tenor_years[i], tenor_years[j])), 0.4)

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
        else:
            num_qualifiers = pos_gp.Qualifier.nunique()

            F = np.zeros((len(CR), len(CR)))

            for i in range(len(CR)):
                for j in range(len(CR)):
                    CRi = CR[i]
                    CRj = CR[j]

                    F[i][j] = min(CRi, CRj) / max(CRi, CRj)

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

            elif risk_class == 'FX':
                rho = params.FX_Rho

            if margin == 'Curvature':
                rho = rho * rho
                F.fill(1)

            Corr = rho * F
            np.fill_diagonal(Corr, 1)

    return Corr


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