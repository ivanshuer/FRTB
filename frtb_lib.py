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
    """Setup output directory by risk class"""

    for prod in params.RiskClass:
        output_path = '{0}\{1}'.format(os.getcwd(), prod)
        if os.path.exists(output_path):
            shutil.rmtree(output_path)

        os.mkdir(output_path)

    output_path = '{0}\sensitivity_all_margin.csv'.format(os.getcwd())
    if os.path.exists(output_path):
        os.remove(output_path)


def risk_classification(trades_pos, params):
    """Risk class classification in terms of RiskType"""

    # Determine risk class
    trades_pos['RiskClass'] = np.NaN
    trades_pos.ix[trades_pos.RiskType.isin(params.IR), 'RiskClass'] = 'IR'
    trades_pos.ix[trades_pos.RiskType.isin(params.CSR), 'RiskClass'] = 'CSR'
    trades_pos.ix[trades_pos.RiskType.isin(params.CSRNonCTP), 'RiskClass'] = 'CSRNonCTP'
    trades_pos.ix[trades_pos.RiskType.isin(params.CSRCTP), 'RiskClass'] = 'CSRCTP'
    trades_pos.ix[trades_pos.RiskType.isin(params.Equity), 'RiskClass'] = 'Equity'
    trades_pos.ix[trades_pos.RiskType.isin(params.FX), 'RiskClass'] = 'FX'
    trades_pos.ix[trades_pos.RiskType.isin(params.Commodity), 'RiskClass'] = 'Commodity'

    trades_pos_no_risk_class = trades_pos[~trades_pos.RiskClass.isin(params.RiskClass)].copy()
    if len(trades_pos_no_risk_class) > 0:
        logger.info('{0} trades can not be classified for risk class'.format(len(trades_pos_no_risk_class)))
        trades_pos_no_risk_class['reason'] = 'RiskType'

    trades_pos = trades_pos[trades_pos.RiskClass.isin(params.RiskClass)].copy()
    trades_pos['reason'] = 'Good'

    trades_pos = pd.concat([trades_pos, trades_pos_no_risk_class])

    return trades_pos

def calc_delta_margin(pos, params):
    pos_delta = pos[pos.RiskType.isin(params.Delta_Factor)].copy()

    pos_delta_margin_gp = []
    pos_delta_margin = []

    if len(pos_delta) > 0:
        delta_margin_loader = delta_margin.DeltaMargin()
        pos_delta_margin = margin_risk_factor(pos_delta, params, delta_margin_loader)

    if len(pos_delta_margin) > 0:
        pos_delta_margin_gp = pos_delta_margin.groupby(['CombinationID', 'RiskClass'])
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
        pos_vega_margin_gp = pos_vega_margin.groupby(['CombinationID', 'RiskClass'])
        pos_vega_margin_gp = pos_vega_margin_gp.agg({'Margin': np.sum})
        pos_vega_margin_gp.reset_index(inplace=True)
        pos_vega_margin_gp['MarginType'] = 'Vega'

    return pos_vega_margin_gp

def calc_curvature_margin(pos, params):
    pos_curvature = pos[pos.RiskType.isin(params.Curvature_Factor)].copy()

    pos_curvature_margin_gp = []
    pos_curvature_margin = []

    if len(pos_curvature) > 0:
        # Aggregate Delta Sensitivities for Curvature
        pos_delta = pos[(pos.RiskType.isin(params.Delta_Factor)) & (~pos.RiskType.isin(['Risk_Inflation']))].copy()
        pos_delta = pos_delta.groupby(['RiskClass', 'Bucket', 'security_id']).agg({'Stat_Value': np.sum})
        pos_delta.reset_index(inplace=True)
        pos_delta.rename(columns={'Stat_Value': 'Delta'}, inplace=True)

        pos_curvature = pd.merge(pos_curvature, pos_delta, how='left')
        pos_curvature['Stat_Value'] = pos_curvature['Delta']
        pos_curvature.drop(['Delta'], axis=1, inplace=True)

        curvature_margin_loader = curvature_margin.CurvatureMargin()
        pos_curvature_margin = margin_risk_factor(pos_curvature, params, curvature_margin_loader)

    if len(pos_curvature_margin) > 0:
        pos_curvature_margin_gp = pos_curvature_margin.groupby(['CombinationID', 'RiskClass'])
        pos_curvature_margin_gp = pos_curvature_margin_gp.agg({'Margin': np.sum})
        pos_curvature_margin_gp.reset_index(inplace=True)
        pos_curvature_margin_gp['MarginType'] = 'Curvature'

    return pos_curvature_margin_gp

def margin_risk_factor(pos, params, margin_loader):
    """Calculate Sensitivity Risk Charge for IR Class"""

    risk_class = pos.RiskClass.unique()[0]
    risk_type = pos.RiskType.unique()[0]

    pos_delta = margin_loader.net_sensitivities(pos, params)

    group = 'Bucket'

    groups = pos_delta[group].unique().tolist()
    groups.sort()
    if risk_class in ['CSR']:
        groups = [int(d) for d in groups]
        groups.sort()
        groups = [str(d) for d in groups]

    pos_delta_gp_all = []
    for gp in groups:
        pos_delta_gp = pos_delta[pos_delta[group] == gp].copy()
        pos_delta_gp = margin_loader.margin_risk_group(pos_delta_gp, params)
        pos_delta_gp_all.append(pos_delta_gp)

    pos_delta_gp_all = pd.concat(pos_delta_gp_all)

    pos_delta = pos_delta_gp_all.copy()

    intermediate_path = '{0}\{1}'.format(os.getcwd(), risk_class)
    file_name = '{0}\{1}_margin_group.csv'.format(intermediate_path, risk_type)

    if not os.path.isfile(file_name):
        pos_delta.to_csv(file_name, index=False)
    else:  # else it exists so append without writing the header
        pos_delta.to_csv(file_name, mode='a', header=False, index=False)

    ret_mm = pos_delta[['CombinationID', 'RiskClass']].copy()
    ret_mm.drop_duplicates(inplace=True)

    if risk_class in ['IR', 'FX']:
        pos_delta_other = []
    else:
        if risk_class == 'CSR':
            other_buckets = params.CSR_Others

        pos_delta_other = pos_delta[pos_delta.Group.isin(other_buckets)].copy()
        pos_delta = pos_delta[~pos_delta.Group.isin(other_buckets)].copy()

    delta_margin = 0

    if len(pos_delta) > 0:

        g = mlib.build_bucket_correlation(pos_delta, params, margin_loader.margin_type())

        pos_delta.reset_index(inplace=True)
        if margin_loader.margin_type() == 'Curvature':
            for i in range(len(pos_delta)):
                for j in range(len(pos_delta)):
                    if pos_delta.S[i] < 0 and pos_delta.S[j] < 0:
                        g[i, j] = 0

        S = pos_delta.S
        S_alt = pos_delta.S_alt

        SS = np.mat(S) * np.mat(g) * np.mat(S.values.reshape((len(S), 1)))
        SS = SS.item(0)

        SS_alt = np.mat(S_alt) * np.mat(g) * np.mat(S_alt.values.reshape((len(S_alt), 1)))
        SS_alt = SS_alt.item(0)

        if np.dot(pos_delta.K, pos_delta.K) + SS >= 0:
            risk_charge_squared = np.dot(pos_delta.K, pos_delta.K) + SS
        else:
            risk_charge_squared = np.dot(pos_delta.K, pos_delta.K) + SS_alt

        delta_margin = math.sqrt(risk_charge_squared)

    if len(pos_delta_other) > 0:
        delta_margin += pos_delta_other.K.sum()

    ret_mm['Margin'] = delta_margin

    return ret_mm

def calculate_sensitivity_risk(pos, params):

    product_margin = []

    for risk in pos.RiskClass.unique():
        logger.info('Calcualte Sensitivity Risk for {0}'.format(risk))
        pos_risk = pos[pos.RiskClass == risk].copy()

        if risk == 'FX':
            pos_risk = pos_risk.apply(mlib.change_FX_ticker_order, axis=1)

        pos_gp_delta_margin = calc_delta_margin(pos_risk, params)
        if len(pos_gp_delta_margin) > 0:
            product_margin.append(pos_gp_delta_margin)

        pos_gp_vega_margin = calc_vega_margin(pos_risk, params)
        if len(pos_gp_vega_margin) > 0:
            product_margin.append(pos_gp_vega_margin)

        pos_gp_curvature_margin = calc_curvature_margin(pos_risk, params)
        if len(pos_gp_curvature_margin) > 0:
            product_margin.append(pos_gp_curvature_margin)

    product_margin = pd.concat(product_margin)

    if not os.path.isfile('sensitivity_all_margin.csv'):
        product_margin.to_csv('sensitivity_all_margin.csv', index=False)
    else:  # else it exists so append without writing the header
        product_margin.to_csv('sensitivity_all_margin.csv', mode='a', header=False, index=False)

    risk_charges = product_margin[['CombinationID']].drop_duplicates()
    risk_charges['Risk_Charge'] = product_margin.Margin.sum()

    return risk_charges

def generate_trade_pos(input_file, params):

    excl_file = pd.ExcelFile(input_file)

    trades_pos = excl_file.parse('inputs', converters={'security_id': str, 'issuer_id': str, 'Qualifier': str,
                                                       'Bucket': str, 'Label1': str, 'Label2': str, 'Label3': str,
                                                       'Stat_Value': np.float64, 'ImpliedVol': np.float64,
                                                       'Shifted_PV_Base': np.float64, 'Raw_PV_Base': np.float64})
    trades_pos.dropna(how='all', inplace=True)

    # Calculate risk classification
    trades_pos = risk_classification(trades_pos, params)
    trades_pos_no_classification = trades_pos[trades_pos.reason != 'Good'].copy()
    trades_pos = trades_pos[trades_pos.reason == 'Good'].copy()

    # Check input data quality
    trades_pos_all = pd.concat([trades_pos, trades_pos_no_classification])
    trades_pos_all.to_csv('all_trades_pos.csv', index=False)

    # Prepare input data
    trades_all = trades_pos_all[trades_pos_all.reason == 'Good'].copy()
    trades_all.drop(['reason'], axis=1, inplace=True)
    trades_all.Stat_Value.fillna(0, inplace=True)

    return trades_all

def find_sentivitiy_id(gp, trades_pos):

    case_ids = gp['SensitivityID']

    if not re.search('All', case_ids) == None:
        case_ids = case_ids[re.search('All', case_ids).end():]
        case_ids = [id.strip() for id in case_ids.split(',')]

        if case_ids[0] == '':
            case_ids = trades_pos.SensitivityID.values
        else:
            case_all = []
            for case in case_ids:
                case_id = trades_pos[trades_pos.SensitivityID.str.match(case + '_')]
                case_all.append(case_id)

            case_all = pd.concat(case_all)
            case_ids = case_all.SensitivityID.values
    else:
        case_ids = [id.strip() for id in case_ids.split(',')]

    case_name = gp['CombinationID']
    CombinationID = np.repeat(case_name, len(case_ids))
    case_df = pd.DataFrame({'CombinationID': CombinationID, 'SensitivityID': case_ids})

    return case_df

def generate_run_cases(input_file, trades_pos):

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
            run_case = find_sentivitiy_id(row, trades_pos)
            run_cases_expand.append(run_case)

        run_cases_expand = pd.concat(run_cases_expand)
        run_cases_expand = pd.merge(run_cases_expand, trades_pos, how='left')

        invalid_sensitivities = run_cases_expand[run_cases_expand.RiskClass.isnull()].copy()
        valid_sensitivities = run_cases_expand[run_cases_expand.RiskClass.notnull()].copy()

        if len(invalid_sensitivities) > 0:
            for index, row in invalid_sensitivities.iterrows():
                logger.info('{0} has no sensitivity {1}.'.format(row['CombinationID'], row['SensitivityID']))

    return valid_sensitivities














