import pandas as pd

configs = pd.ExcelFile('frtb_config.xlsx')

RiskClass = ['IR', 'CSR', 'CSRNonCTP', 'CSRCTP', 'Equity', 'Commodity', 'FX']

IR = ['Risk_IRDelta', 'Risk_IRVega', 'Risk_Inflation', 'Risk_IRCurvature']
CSR = ['Risk_CSRDelta', 'Risk_CSRVega', 'Risk_CSRCurvature']
CSRNonCTP = ['Risk_CSRNonCTPDelta', 'Risk_CSRNonCTPVega', 'Risk_CSRNonCTPCurvature']
CSRCTP = ['Risk_CSRCTPDelta', 'Risk_CSRCTPVega', 'Risk_CSRCTPCurvature']
Equity = ['Risk_EquityDelta', 'Risk_EquityVega', 'Risk_EquityCurvature']
FX = ['Risk_FXDelta', 'Risk_FXVega', 'Risk_FXCurvature']
Commodity = ['Risk_CommodityDelta', 'Risk_CommodityVega', 'Risk_CommodityCurvature']

Delta_Factor = ['Risk_IRDelta', 'Risk_Inflation', 'Risk_CSRDelta', 'Risk_CSRNonCTPDelta', 'Risk_CSRCTPDelta', 'Risk_EquityDelta', 'Risk_FXDelta', 'Risk_CommodityDelta']
Vega_Factor = ['Risk_IRVega', 'Risk_CSRVega', 'Risk_CSRNonCTPVega', 'Risk_CSRCTPVega', 'Risk_EquityVega', 'Risk_FXVega', 'Risk_CommodityVega']
Curvature_Factor = ['Risk_IRCurvature', 'Risk_CSRCurvature', 'Risk_CSRNonCTPCurvature', 'Risk_CSRCTPCurvature', 'Risk_EquityCurvature', 'Risk_FXCurvature', 'Risk_CommodityCurvature']

Vega_VRW = 0.55
Vega_Alpha = 0.01

IR_Tenor = ['3M', '6M', '1Y', '2Y', '3Y', '5Y', '10Y', '15Y', '20Y', '30Y']
IR_Vega_Maturity = ['6M', '1Y', '3Y', '5Y', '10Y']
IR_Vega_Residual_Maturity = ['6M', '1Y', '3Y', '5Y', '10Y']
IR_Adjust_Curr = ['EUR', 'USD', 'GBP', 'AUD', 'JPY', 'SEK', 'CAD']
IR_Weights = configs.parse('IR_weights', converters={'tenor': str})
IR_Theta = 0.03
IR_Fai = 0.999
IR_Gamma = 0.5
IR_Inflation_Weights = 0.0225
IR_Inflation_Rho = 0.4
IR_LH = 60

CSR_Tenor = ['6M', '1Y', '3Y', '5Y', '10Y']
CSR_IG = ['1', '2', '3', '4', '5', '6', '7', '8']
CSR_HY = ['9', '10', '11', '12', '13', '14', '15']
CSR_Others = ['16']
CSR_Weights = configs.parse('CSR_weights', converters={'bucket': str})
CSR_Rho_Name = 0.35
CSR_Rho_Tenor = 0.65
CSR_Rho_Rating = 0.5
CSR_Sector_Corr = configs.parse('CSR_correlation')
CSR_VRW = 0.35

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

FX_Vega_Maturity = ['6M', '1Y', '3Y', '5Y', '10Y']
FX_Adjust_Curr = ['USDEUR', 'USDJPY', 'USDGBP', 'USDAUD', 'USDCAD', 'USDCHF', 'USDMXN', 'USDCNY', 'USDNZD', 'USDRUB',
                  'USDHKD', 'USDSGD', 'USDTRY', 'USDKRW', 'USDSEK', 'USDZAR', 'USDINR', 'USDNOK', 'USDBRL', 'EURJPY',
                  'EURGBP', 'EURCHF', 'JPYAUD']
FX_Weights = 0.3
FX_Gamma = 0.6
FX_LH = 40

