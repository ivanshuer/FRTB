import pandas as pd

configs = pd.ExcelFile('frtb_config.xlsx')

Product = ['RatesFX', 'Credit', 'Equity', 'Commodity']

RiskType = ['IR', 'CreditQ', 'CreditNonQ', 'Equity', 'Commodity', 'FX']

IR = ['Risk_IRCurve', 'Risk_IRVol', 'Risk_Inflation']
CreditQ = ['Risk_CreditQ', 'Risk_CreditVol']
CreditNonQ = ['Risk_CreditNonQ', 'Risk_CreditVolNonQ']
Equity = ['Risk_Equity', 'Risk_EquityVol']
FX = ['Risk_FX', 'Risk_FXVol']
Commodity = ['Risk_Commodity', 'Risk_CommodityVol']

Delta_Factor = ['Risk_IRCurve', 'Risk_Inflation', 'Risk_CreditQ', 'Risk_CreditNonQ', 'Risk_Equity', 'Risk_FX', 'Risk_Commodity']
Vega_Factor = ['Risk_IRVol', 'Risk_CreditVol', 'Risk_CreditVolNonQ', 'Risk_EquityVol', 'Risk_FXVol', 'Risk_CommodityVol']
Curvature_Factor = Vega_Factor
#Curvature_Factor = ['Risk_IRCV', 'Risk_CreditCV', 'Risk_EquityCV', 'Risk_FXCV', 'Risk_CommodityCV']

Risk_Class_Corr = configs.parse('Risk_class_correlation')

IR_Bucket = ['1', '2', '3']
IR_Tenor = ['2w', '1m', '3m', '6m', '1y', '2y', '3y', '5y', '10y', '15y', '20y', '30y']
IR_Sub_Curve = ['OIS', 'Libor1m', 'Libor3m', 'Libor6m', 'Libor12m']
IR_USD_Sub_Curve = IR_Sub_Curve + ['Prime']
IR_Reg_Vol_Well_Traded_Curr = ['USD', 'EUR', 'GBP']
IR_Reg_Vol_Less_Well_Traded_Curr = ['CHF', 'AUD', 'NZD', 'CAD', 'SEK', 'NOK', 'DKK', 'HKD', 'KRW', 'SGD', 'TWD']
IR_Low_Vol_Curr = ['JPY']
IR_CR_Thrd = configs.parse('IR_CR_THR')
IR_Weights = configs.parse('IR_weights', converters={'curr': str})
IR_Corr = configs.parse('IR_correlation')
IR_Fai = 0.982
IR_Gamma = 0.27
IR_Inflation_Weights = 32
IR_Inflation_Rho = 0.33
IR_VRW = 0.21
IR_Curvature_Margin_Scale = 2.3

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

