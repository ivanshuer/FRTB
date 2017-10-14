import pandas as pd

configs = pd.ExcelFile('frtb_config.xlsx')

RiskClass = ['IR', 'CSR', 'CSRNonCTP', 'CSRCTP', 'Equity', 'Commodity', 'FX']

IR = ['Risk_IRDelta', 'Risk_IRVega', 'Risk_Inflation', 'Risk_IRCurvature']
CSR = ['Risk_CSRDelta', 'Risk_CSRVega', 'RiskCSRCurvature']
CSRNonCTP = ['Risk_CSRNonCTPDelta', 'Risk_CSRNonCTPVega', 'RiskCSRNonCTPCurvature']
CSRCTP = ['Risk_CSRCTPDelta', 'Risk_CSRCTPVega', 'RiskCSRCTPCurvature']
Equity = ['Risk_EquityDelta', 'Risk_EquityVega', 'RiskEquityCurvature']
FX = ['Risk_FXDelta', 'Risk_FXVega', 'RiskFXCurvature']
Commodity = ['Risk_CommodityDelta', 'Risk_CommodityVega', 'RiskCommodityCurvature']

Delta_Factor = ['Risk_IRDelta', 'Risk_Inflation', 'Risk_CSRDelta', 'Risk_CSRNonCTPDelta', 'Risk_CSRCTPDelta', 'Risk_EquityDelta', 'Risk_FXDelta', 'Risk_CommodityDelta']
Vega_Factor = ['Risk_IRVega', 'Risk_CSRVega', 'Risk_CSRNonCTPVega', 'Risk_CSRCTPVega', 'Risk_EquityVega', 'RiskFXCurvature', 'Risk_CommodityVega']
Curvature_Factor = ['Risk_IRCurvature', 'RiskCSRCurvature', 'RiskCSRNonCTPCurvature', 'RiskCSRCTPCurvature', 'RiskEquityCurvature', 'RiskFXCurvature', 'RiskCommodityCurvature']

IR_Tenor = ['3m', '6m', '1y', '2y', '3y', '5y', '10y', '15y', '20y', '30y']
IR_Vega_Maturity = ['6m', '1y', '3y', '5y', '10y']
IR_Vega_Residual_Maturity = ['6m', '1y', '3y', '5y', '10y']
IR_Weights = configs.parse('IR_weights', converters={'tenor': str})
IR_Theta = 0.03
IR_Fai = 0.999
IR_Gamma = 0.5
IR_Inflation_Weights = 0.0225
IR_Inflation_Rho = 0.4
IR_VRW = 0.55
IR_LH = 60
IR_Alpha = 0.01

CreditQ_Bucket = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', 'Residual']
CreditQ_Tenor = ['1y', '2y', '3y', '5y', '10y']
CreditQ_CR_Sov_incl_Central_Banks = ['1', '7']
CreditQ_CR_Corp_Entities = ['2', '3', '4', '5', '6', '8', '9', '10', '11', '12']
CreditQ_CR_Not_Classified = ['Residual']
CreditQ_Weights = configs.parse('CreditQ_weights', converters={'bucket': str})
CreditQ_Rho_Agg_Same_IS = 0.98
CreditQ_Rho_Agg_Diff_IS = 0.55
CreditQ_Rho_Res_Same_IS = 0.5
CreditQ_Rho_Res_Diff_IS = 0.5
CreditQ_Corr = configs.parse('CreditQ_correlation')
CreditQ_CR_Thrd = configs.parse('CreditQ_CR_THR')
CreditQ_VRW = 0.35
CreditQ_num_sec_type = 2

CreditNonQ_Bucket = ['1', '2', 'Residual']
CreditNonQ_Tenor = ['1y', '2y', '3y', '5y', '10y']
CreditNonQ_CR_IG = ['1']
CreditNonQ_CR_HY_Non_Rated = ['2']
CreditNonQ_CR_Not_Classified = ['Residual']
CreditNonQ_Weights = configs.parse('CreditNonQ_weights', converters={'bucket': str})
CreditNonQ_Rho_Agg_Same_IS = 0.6
CreditNonQ_Rho_Agg_Diff_IS = 0.21
CreditNonQ_Rho_Res_Same_IS = 0.5
CreditNonQ_Rho_Res_Diff_IS = 0.5
CreditNonQ_Corr = configs.parse('CreditNonQ_correlation')
CreditNonQ_CR_Thrd = configs.parse('CreditNonQ_CR_THR')
CreditNonQ_VRW = 0.35

Equity_Tenor = ['2w', '1m', '3m', '6m', '1y', '2y', '3y', '5y', '10y', '15y', '20y', '30y']
Equity_Bucket = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', 'Residual']
Equity_CR_Emerging_Large_Cap = ['1', '2', '3', '4']
Equity_CR_Developed_Large_Cap = ['5', '6', '7', '8']
Equity_CR_Emerging_Small_Cap = ['9']
Equity_CR_Developed_Small_Cap = ['10']
Equity_CR_Index_Funds_ETF = ['11']
Equity_CR_Not_Classified = ['Residual']
Equity_Weights = configs.parse('Equity_weights', converters={'bucket': str})
Equity_Rho = configs.parse('Equity_in_bucket_correlation', converters={'bucket': str})
Equity_Corr = configs.parse('Equity_correlation')
Equity_CR_Thrd = configs.parse('Equity_CR_THR')
Equity_VRW = 0.21

Commodity_Tenor = ['2w', '1m', '3m', '6m', '1y', '2y', '3y', '5y', '10y', '15y', '20y', '30y']
Commodity_Bucket = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16']
Commodity_CR_Coal = ['1']
Commodity_CR_Crude_Oil = ['2']
Commodity_CR_Light_End = ['3']
Commodity_CR_Middle_Distilates = ['4']
Commodity_CR_Heavy_Distilates = ['5']
Commodity_CR_NA_Natual_Gas = ['6']
Commodity_CR_EU_Natual_Gas = ['7']
Commodity_CR_NA_Power = ['8']
Commodity_CR_EU_Power = ['9']
Commodity_CR_Freight = ['10']
Commodity_CR_Base_Metals = ['11']
Commodity_CR_Precious_Metals = ['12']
Commodity_CR_Grains = ['13']
Commodity_CR_Softs = ['14']
Commodity_CR_Livestock = ['15']
Commodity_CR_Others = ['16']
Commodity_Weights = configs.parse('Commodity_weights', converters={'bucket': str})
Commodity_Rho = configs.parse('Commodity_in_bucket_correlation', converters={'bucket': str})
Commodity_Corr = configs.parse('Commodity_correlation')
Commodity_CR_Thrd = configs.parse('Commodity_CR_THR')
Commodity_VRW = 0.36

FX_Tenor = ['2w', '1m', '3m', '6m', '1y', '2y', '3y', '5y', '10y', '15y', '20y', '30y']
FX_Weights = 7.9
FX_Rho = 0.5
FX_VRW = 0.21
FX_Significantly_Material = ['USD', 'EUR', 'JPY', 'GBP', 'AUD', 'CHF', 'CAD']
FX_Frequently_Traded = ['BRL', 'CNY', 'HKD', 'INR', 'KRW', 'MXN', 'NOK', 'NZD', 'RUB', 'SEK', 'SGD', 'TRY', 'ZAR']
FX_CR_Thrd = configs.parse('FX_CR_THR')

