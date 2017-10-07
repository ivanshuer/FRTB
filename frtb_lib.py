import numpy as np
import pandas as pd
import os
import logging
import math
import shutil
import re
import delta_margin
import vega_margin
import curvature_margin
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

def prep_output_directory(params):
    """Setup output directory by product and risk class"""

    for prod in params.Product:
        output_path = '{0}\{1}'.format(os.getcwd(), prod)
        if os.path.exists(output_path):
            shutil.rmtree(output_path)

        os.mkdir(output_path)

    for prod in params.Product:
        for risk in params.RiskType:
            output_path = '{0}\{1}\{2}'.format(os.getcwd(), prod, risk)
            if not os.path.exists(output_path):
                os.mkdir(output_path)

    output_path = '{0}\simm_all_margin.csv'.format(os.getcwd())
    if os.path.exists(output_path):
        os.remove(output_path)

def risk_classification(trades_pos, params):
    """Risk class classification in terms of RiskType"""

    # Check product type
    trades_pos_no_product = trades_pos[~trades_pos.ProductClass.isin(params.Product)].copy()
    if len(trades_pos_no_product) > 0:
        logger.info('{0} trades missed Product Class'.format(len(trades_pos_no_product)))
        trades_pos_no_product['reason'] = 'Product'

    trades_pos = trades_pos[trades_pos.ProductClass.isin(params.Product)].copy()

    # Determine risk class
    trades_pos['RiskClass'] = np.NaN
    trades_pos.ix[trades_pos.RiskType.isin(params.IR), 'RiskClass'] = 'IR'
    trades_pos.ix[trades_pos.RiskType.isin(params.CreditQ), 'RiskClass'] = 'CreditQ'
    trades_pos.ix[trades_pos.RiskType.isin(params.CreditNonQ), 'RiskClass'] = 'CreditNonQ'
    trades_pos.ix[trades_pos.RiskType.isin(params.Equity), 'RiskClass'] = 'Equity'
    trades_pos.ix[trades_pos.RiskType.isin(params.FX), 'RiskClass'] = 'FX'
    trades_pos.ix[trades_pos.RiskType.isin(params.Commodity), 'RiskClass'] = 'Commodity'

    trades_pos_no_risk_class = trades_pos[~trades_pos.RiskClass.isin(params.RiskType)].copy()
    if len(trades_pos_no_risk_class) > 0:
        logger.info('{0} trades can not be classified for risk class'.format(len(trades_pos_no_risk_class)))
        trades_pos_no_risk_class['reason'] = 'RiskType'

    trades_pos = trades_pos[trades_pos.RiskClass.isin(params.RiskType)].copy()

    # Check qualifier
    trades_pos_no_qualifier = trades_pos[trades_pos.Qualifier.isnull()].copy()
    if len(trades_pos_no_qualifier) > 0:
        logger.info('{0} trades missed Qualifiers'.format(len(trades_pos_no_qualifier)))
        trades_pos_no_risk_class['reason'] = 'Qualifiers'

    trades_pos = trades_pos[trades_pos.Qualifier.notnull()].copy()
    trades_pos['reason'] = 'Good'

    trades_pos = pd.concat([trades_pos, trades_pos_no_product, trades_pos_no_risk_class, trades_pos_no_qualifier])

    return trades_pos

def prep_data_IRCurve(pos, params):
    """Check data quality for IR Curve factor"""

    # Check Bucket
    pos_no_bucket = pos[~pos.Bucket.isin(params.IR_Bucket)].copy()
    if len(pos_no_bucket) > 0:
        logger.info('{0} IR Curve trades have wrong Bucket'.format(len(pos_no_bucket)))
        pos_no_bucket['reason'] = 'Bucket'

    pos = pos[pos.Bucket.isin(params.IR_Bucket)].copy()

    # Check Label1
    pos_no_label1 = pos[~pos.Label1.isin(params.IR_Tenor)].copy()
    if len(pos_no_label1) > 0:
        logger.info('{0} IR Curve trades have wrong Label 1'.format(len(pos_no_label1)))
        pos_no_label1['reason'] = 'Label1'

    pos = pos[pos.Label1.isin(params.IR_Tenor)].copy()

    # Check Label2
    pos_no_label2 = pos[~(((pos.Qualifier == 'USD') & pos.Label2.isin(params.IR_USD_Sub_Curve)) |
                          ((pos.Qualifier != 'USD') & pos.Label2.isin(params.IR_Sub_Curve)))].copy()

    if len(pos_no_label2) > 0:
        logger.info('{0} IR Curve trades have wrong Label 2'.format(len(pos_no_label2)))
        pos_no_label2['reason'] = 'Label2'

    pos = pos[((pos.Qualifier == 'USD') & pos.Label2.isin(params.IR_USD_Sub_Curve)) |
              ((pos.Qualifier != 'USD') & pos.Label2.isin(params.IR_Sub_Curve))].copy()

    pos = pd.concat([pos, pos_no_bucket, pos_no_label1, pos_no_label2])

    return pos

def prep_data_IRVol(pos, params):
    """Check data quality for IR Vol factor"""

    # Check Label1
    pos_no_label1 = pos[~pos.Label1.isin(params.IR_Tenor)].copy()
    if len(pos_no_label1) > 0:
        logger.info('{0} IR Vol trades have wrong Label 1'.format(len(pos_no_label1)))
        pos_no_label1['reason'] = 'Label1'

    pos = pos[pos.Label1.isin(params.IR_Tenor)].copy()

    pos = pd.concat([pos, pos_no_label1])

    return pos

def prep_data_IR(pos, params):
    """Check data quality for IR factor"""

    # Check IR curve
    pos_IRCurve = pos[pos.RiskType == 'Risk_IRCurve'].copy()
    pos_IRCurve = prep_data_IRCurve(pos_IRCurve, params)

    # Check IR vol and curvature
    pos_IRVol = pos[pos.RiskType.isin(['Risk_IRVol', 'Risk_IRCV'])].copy()
    pos_IRVol = prep_data_IRVol(pos_IRVol, params)

    # Check IR inflation
    pos_Inflation = pos[pos.RiskType == 'Risk_Inflation'].copy()

    pos = pd.concat([pos_IRCurve, pos_IRVol, pos_Inflation])

    return pos

def prep_data_CreditQ(pos, params):
    """Check data quality for CreditQ factor"""

    # Check Bucket
    pos_no_bucket = pos[~pos.Bucket.isin(params.CreditQ_Bucket)].copy()
    if len(pos_no_bucket) > 0:
        logger.info('{0} CreditQ trades have wrong Bucket'.format(len(pos_no_bucket)))
        pos_no_bucket['reason'] = 'Bucket'

    pos = pos[pos.Bucket.isin(params.CreditQ_Bucket)].copy()

    # Check Label1
    pos_no_label1 = pos[~pos.Label1.isin(params.CreditQ_Tenor)].copy()
    if len(pos_no_label1) > 0:
        logger.info('{0} CreditQ trades have wrong Label 1'.format(len(pos_no_label1)))
        pos_no_label1['reason'] = 'Label1'

    pos = pos[pos.Label1.isin(params.CreditQ_Tenor)].copy()

    pos = pd.concat([pos, pos_no_bucket, pos_no_label1])

    return pos

def prep_data_CreditNonQ(pos, params):
    """Check data quality for CreditNonQ factor"""

    # Check Bucket
    pos_no_bucket = pos[~pos.Bucket.isin(params.CreditNonQ_Bucket)].copy()
    if len(pos_no_bucket) > 0:
        logger.info('{0} CreditNonQ trades have wrong Bucket'.format(len(pos_no_bucket)))
        pos_no_bucket['reason'] = 'Bucket'

    pos = pos[pos.Bucket.isin(params.CreditNonQ_Bucket)].copy()

    # Check Label1
    pos_no_label1 = pos[~pos.Label1.isin(params.CreditNonQ_Tenor)].copy()
    if len(pos_no_label1) > 0:
        logger.info('{0} CreditNonQ trades have wrong Label 1'.format(len(pos_no_label1)))
        pos_no_label1['reason'] = 'Label1'

    pos = pos[pos.Label1.isin(params.CreditNonQ_Tenor)].copy()

    pos = pd.concat([pos, pos_no_bucket, pos_no_label1])

    return pos

def prep_data_Equity(pos, params):
    """Check data quality for Equity factor"""

    # Check Bucket
    pos_no_bucket = pos[~pos.Bucket.isin(params.Equity_Bucket)].copy()
    if len(pos_no_bucket) > 0:
        logger.info('{0} Equity trades have wrong Bucket'.format(len(pos_no_bucket)))
        pos_no_bucket['reason'] = 'Bucket'

    pos = pos[pos.Bucket.isin(params.Equity_Bucket)].copy()

    pos = pd.concat([pos, pos_no_bucket])

    return pos

def prep_data_Commodity(pos, params):
    """Check data quality for Commodity factor"""

    # Check Bucket
    pos_no_bucket = pos[~pos.Bucket.isin(params.Commodity_Bucket)].copy()
    if len(pos_no_bucket) > 0:
        logger.info('{0} Commodity trades have wrong Bucket'.format(len(pos_no_bucket)))
        pos_no_bucket['reason'] = 'Bucket'

    pos = pos[pos.Bucket.isin(params.Commodity_Bucket)].copy()

    pos = pd.concat([pos, pos_no_bucket])

    return pos

def prep_data(pos, params):
    """Check data quality for all risk factors"""

    pos_IR = pos[pos.RiskClass == 'IR'].copy()
    pos_IR = prep_data_IR(pos_IR, params)

    pos_CreditQ = pos[pos.RiskClass == 'CreditQ'].copy()
    pos_CreditQ = prep_data_CreditQ(pos_CreditQ, params)

    pos_CreditNonQ = pos[pos.RiskClass == 'CreditNonQ'].copy()
    pos_CreditNonQ = prep_data_CreditNonQ(pos_CreditNonQ, params)

    pos_Equity = pos[pos.RiskClass == 'Equity'].copy()
    pos_Equity = prep_data_Equity(pos_Equity, params)

    pos_Commodity = pos[pos.RiskClass == 'Commodity'].copy()
    pos_Commodity = prep_data_Commodity(pos_Commodity, params)

    pos_FX = pos[pos.RiskClass == 'FX'].copy()

    return pd.concat([pos_IR, pos_CreditQ, pos_CreditNonQ, pos_Equity, pos_Commodity, pos_FX])

def calc_delta_margin(pos, params):
    pos_delta = pos[pos.RiskType.isin(params.Delta_Factor)].copy()

    pos_delta_margin_gp = []
    pos_delta_margin = []

    if len(pos_delta) > 0:
        delta_margin_loader = delta_margin.DeltaMargin()
        pos_delta_margin = margin_risk_factor(pos_delta, params, delta_margin_loader)

    if len(pos_delta_margin) > 0:
        pos_delta_margin_gp = pos_delta_margin.groupby(['CombinationID', 'ProductClass', 'RiskClass'])
        pos_delta_margin_gp = pos_delta_margin_gp.agg({'Margin': np.sum})
        pos_delta_margin_gp.reset_index(inplace=True)
        pos_delta_margin_gp['MarginType'] = 'Delta'

    return pos_delta_margin_gp

def calc_vega_margin(pos, params):
    pos_vega = pos[pos.RiskType.isin(params.Vega_Factor)].copy()

    pos_vega_margin_gp = []
    pos_vega_margin = []

    if len(pos_vega) > 0:
        vega_margin_loader = vega_margin.VegaMargin()
        pos_vega_margin = margin_risk_factor(pos_vega, params, vega_margin_loader)

    if len(pos_vega_margin) > 0:
        pos_vega_margin_gp = pos_vega_margin.groupby(['CombinationID', 'ProductClass', 'RiskClass'])
        pos_vega_margin_gp = pos_vega_margin_gp.agg({'Margin': np.sum})
        pos_vega_margin_gp.reset_index(inplace=True)
        pos_vega_margin_gp['MarginType'] = 'Vega'

    return pos_vega_margin_gp

def calc_curvature_margin(pos, params):
    pos_curvature = pos[pos.RiskType.isin(params.Curvature_Factor)].copy()

    pos_curvature_margin_gp = []
    pos_curvature_margin = []

    if len(pos_curvature) > 0:
        curvature_margin_loader = curvature_margin.CurvatureMargin()
        pos_curvature_margin = margin_risk_factor(pos_curvature, params, curvature_margin_loader)

    if len(pos_curvature_margin) > 0:
        pos_curvature_margin_gp = pos_curvature_margin.groupby(['CombinationID', 'ProductClass', 'RiskClass'])
        pos_curvature_margin_gp = pos_curvature_margin_gp.agg({'Margin': np.sum})
        pos_curvature_margin_gp.reset_index(inplace=True)
        pos_curvature_margin_gp['MarginType'] = 'Curvature'

    return pos_curvature_margin_gp

def margin_risk_factor(pos, params, margin_loader):
    """Calculate Delta Margin for IR Class"""

    if margin_loader.margin_type() == 'Curvature':
        pos = margin_loader.input_scaling(pos)

    pos_delta = margin_loader.net_sensitivities(pos, params)

    product_class = pos_delta.ProductClass.unique()[0]
    risk_class = pos_delta.RiskClass.unique()[0]
    risk_type = pos_delta.RiskType.unique()[0]

    if risk_class == 'IR':
        group = 'Qualifier'
    elif risk_class == 'FX':
        group = 'RiskType'
    else:
        group = 'Bucket'

    pos_delta_gp_all = []
    for gp in pos_delta[group].sort_values().unique():
        pos_delta_gp = pos_delta[pos_delta[group] == gp].copy()
        pos_delta_gp = margin_loader.margin_risk_group(pos_delta_gp, params)
        pos_delta_gp_all.append(pos_delta_gp)

    pos_delta_gp_all = pd.concat(pos_delta_gp_all)

    pos_delta = pos_delta_gp_all.copy()

    intermediate_path = '{0}\{1}\{2}'.format(os.getcwd(), product_class, risk_class)
    file_name = '{0}\{1}_margin_group.csv'.format(intermediate_path, risk_type)

    if margin_loader.margin_type() == 'Curvature':
        pos_delta_output = pos_delta.ix[:, pos_delta.columns != 'CVR_sum']
    else:
        pos_delta_output = pos_delta

    if not os.path.isfile(file_name):
        pos_delta_output.to_csv(file_name, index=False)
    else:  # else it exists so append without writing the header
        pos_delta_output.to_csv(file_name, mode='a', header=False, index=False)

    g = mlib.build_bucket_correlation(pos_delta, params, margin_loader.margin_type())

    pos_delta_non_residual = pos_delta[pos_delta.Group != 'Residual'].copy()
    pos_delta_residual = pos_delta[pos_delta.Group == 'Residual'].copy()

    delta_margin = 0
    if len(pos_delta_non_residual) > 0:
        S = mlib.build_non_residual_S(pos_delta_non_residual, params)

        if risk_class != 'FX':
            SS = np.mat(S) * np.mat(g) * np.mat(np.reshape(S, (len(S), 1)))
            SS = SS.item(0)
        else:
            SS = 0

        delta_margin = math.sqrt(np.dot(pos_delta_non_residual.K, pos_delta_non_residual.K) + SS)

        if margin_loader.margin_type() == 'Curvature':
            theta = min(pos_delta_non_residual.CVR_sum.sum() / pos_delta_non_residual.CVR_abs_sum.sum(), 0)
            lambda_const = (pow(norm.ppf(0.995), 2) - 1) * (1 + theta) - theta

            delta_margin = max(lambda_const * delta_margin + pos_delta_non_residual.CVR_sum.sum(), 0)

    if len(pos_delta_residual) > 0:
        K = pos_delta_residual.K.values[0]

        if margin_loader.margin_type() == 'Curvature':
            CVR_sum = pos_delta_residual.CVR_sum.values[0]
            CVR_abs_sum = pos_delta_residual.CVR_abs_sum.values[0]

            theta = min(CVR_sum / CVR_abs_sum, 0)
            lambda_const = (pow(norm.ppf(0.995), 2) - 1) * (1 + theta) - theta

            delta_margin = delta_margin + max(CVR_sum + lambda_const * K, 0)
        else:
            delta_margin = delta_margin + K

    if margin_loader.margin_type() == 'Curvature' and risk_class == 'IR':
        delta_margin = delta_margin * params.IR_Curvature_Margin_Scale

    ret_mm = pos_delta[['CombinationID','ProductClass', 'RiskClass']].copy()
    ret_mm.drop_duplicates(inplace=True)
    ret_mm['Margin'] = delta_margin

    return ret_mm

def calculate_in_product_margin(pos_gp, params):

    risk_class_corr = params.Risk_Class_Corr

    pos_product_margin = []
    for product in pos_gp.ProductClass.unique():
        logger.info('Calculate In-Product margin for {0}'.format(product))

        pos_product = pos_gp[pos_gp.ProductClass == product].copy()

        risk_margin = np.zeros(len(params.RiskType))

        for i in range(len(params.RiskType)):
            for j in range(len(pos_product.RiskClass)):
                if pos_product.RiskClass.values[j] == params.RiskType[i]:
                    risk_margin[i] = pos_product.Margin.values[j]
                    break

        product_margin = np.mat(risk_margin) * np.mat(risk_class_corr) * np.mat(np.reshape(risk_margin, (len(risk_margin), 1)))
        product_margin = math.sqrt(product_margin.item(0))

        pos_product = pos_product[['CombinationID', 'ProductClass']].copy()
        pos_product.drop_duplicates(inplace=True)
        pos_product['Margin'] = product_margin

        pos_product_margin.append(pos_product)

    if len(pos_product_margin) > 0:
        pos_product_margin = pd.concat(pos_product_margin)

    return pos_product_margin

def calculate_simm(pos, params):

    product_margin = []

    for product in pos.ProductClass.unique():
        for risk in pos[pos.ProductClass == product].RiskClass.unique():
            logger.info('Calcualte SIMM for {0} and {1}'.format(product, risk))
            pos_product = pos[(pos.ProductClass == product) & (pos.RiskClass == risk)].copy()

            pos_gp_delta_margin = calc_delta_margin(pos_product, params)
            if len(pos_gp_delta_margin) > 0:
                product_margin.append(pos_gp_delta_margin)

            pos_gp_vega_margin = calc_vega_margin(pos_product, params)
            if len(pos_gp_vega_margin) > 0:
                product_margin.append(pos_gp_vega_margin)

            pos_gp_curvature_margin = calc_curvature_margin(pos_product, params)
            if len(pos_gp_curvature_margin) > 0:
                product_margin.append(pos_gp_curvature_margin)

    product_margin = pd.concat(product_margin)

    if not os.path.isfile('simm_all_margin.csv'):
        product_margin.to_csv('simm_all_margin.csv', index=False)
    else:  # else it exists so append without writing the header
        product_margin.to_csv('simm_all_margin.csv', mode='a', header=False, index=False)

    product_margin_gp = product_margin.groupby(['CombinationID', 'ProductClass', 'RiskClass'])
    product_margin_gp = product_margin_gp.agg({'Margin': np.sum})
    product_margin_gp.reset_index(inplace=True)

    product_margin_all = calculate_in_product_margin(product_margin_gp, params)

    simm = product_margin_all[['CombinationID']].drop_duplicates()
    simm['SIMM_Benchmark'] = product_margin_all.Margin.sum()

    return simm

def generate_trade_pos(input_file, params):

    excl_file = pd.ExcelFile(input_file)

    trades_pos = excl_file.parse('simm_input', converters={'Bucket': str, 'Label1': str, 'Label2': str, 'Amount': np.float64, 'AmountUSD': np.float64})
    trades_pos.dropna(how='all', inplace=True)

    # Calculate risk classification
    trades_pos = risk_classification(trades_pos, params)
    trades_pos_no_classification = trades_pos[trades_pos.reason != 'Good'].copy()
    trades_pos = trades_pos[trades_pos.reason == 'Good'].copy()

    # Check input data quality
    trades_pos_all = prep_data(trades_pos, params)
    trades_pos_all = pd.concat([trades_pos_all, trades_pos_no_classification])
    trades_pos_all.to_csv('all_trades_pos.csv', index=False)

    # Prepare input data
    trades_simm = trades_pos_all[trades_pos_all.reason == 'Good'].copy()
    trades_simm = trades_simm[['SensitivityID', 'ProductClass', 'RiskType', 'Qualifier', 'Bucket', 'Label1', 'Label2', 'AmountUSD', 'RiskClass']].copy()
    trades_simm.AmountUSD.fillna(0, inplace=True)

    return trades_simm

def find_sentivitiy_id(gp, trades_simm):

    case_ids = gp['SensitivityID']

    if not re.search('All', case_ids) == None:
        case_ids = case_ids[re.search('All', case_ids).end():]
        case_ids = [id.strip() for id in case_ids.split(',')]

        if case_ids[0] == '':
            case_ids = trades_simm.SensitivityID.values
        else:
            case_all = []
            for case in case_ids:
                case_id = trades_simm[trades_simm.SensitivityID.str.match(case + '_')]
                case_all.append(case_id)

            case_all = pd.concat(case_all)
            case_ids = case_all.SensitivityID.values
    else:
        case_ids = [id.strip() for id in case_ids.split(',')]

    case_name = gp['CombinationID']
    CombinationID = np.repeat(case_name, len(case_ids))
    case_df = pd.DataFrame({'CombinationID': CombinationID, 'SensitivityID': case_ids})

    return case_df

def generate_run_cases(input_file, trades_simm):

    excl_file = pd.ExcelFile(input_file)

    run_cases = excl_file.parse('run_cases', converters={'Include': str})

    run_case_include = run_cases[run_cases.Include == 'x'].copy()
    if len(run_case_include) > 0:
        run_case_all = run_case_include.copy()
    else:
        run_case_all = run_cases.copy()

    run_case_all = run_case_all[run_case_all.SensitivityID.notnull()].copy()

    valid_sensitivities = []

    if len(run_case_all) > 0:
        run_cases_expand = []
        for index, row in run_case_all.iterrows():
            run_case = find_sentivitiy_id(row, trades_simm)
            run_cases_expand.append(run_case)

        run_cases_expand = pd.concat(run_cases_expand)
        run_cases_expand = pd.merge(run_cases_expand, trades_simm, how='left')

        invalid_sensitivities = run_cases_expand[run_cases_expand.ProductClass.isnull()].copy()
        valid_sensitivities = run_cases_expand[run_cases_expand.ProductClass.notnull()].copy()

        if len(invalid_sensitivities) > 0:
            for index, row in invalid_sensitivities.iterrows():
                logger.info('{0} has no sensitivity {1}.'.format(row['CombinationID'], row['SensitivityID']))

    return valid_sensitivities














