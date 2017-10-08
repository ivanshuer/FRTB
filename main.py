import pandas as pd
import logging
import os
import params
import frtb_lib
import argparse

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

    file_handler = logging.FileHandler('log.txt', mode='w')
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
###############################


def main():
    # Setup input argument
    parser = argparse.ArgumentParser(description='FRBT Calculation.')
    parser.add_argument('-f', dest='input_file', type=str, required=True, help='FRBT input csv file')
    args = parser.parse_args(['-f' 'frtb_config.xlsx'])
    # args = parser.parse_args()

    # Create output directory for product and risk class
    frtb_lib.prep_output_directory(params)

    # Read input file with specified data type
    input_file = args.input_file

    trades_pos = frtb_lib.generate_trade_pos(input_file, params)
    run_cases = frtb_lib.generate_run_cases(input_file, trades_pos)

    # Calculate FRTB risk charges and dump output
    results_all = []
    if len(run_cases) > 0:
        for case in run_cases.CombinationID.unique():
            logger.info('Run test {0}'.format(case))
            run_case = run_cases[run_cases.CombinationID == case].copy()
            result = frtb_lib.calculate_sensitivity_risk(run_case, params)
            results_all.append(result)

        results_all = pd.concat(results_all)
        results_all.to_csv('results_output.csv', index=False)

        for index, row in results_all.iterrows():
            # logger.info('{0}: Total Risk is {1:,}'.format(row['CombinationID'], int(round(row['SIMM_Benchmark']))))
            logger.info('{0}: Total Risk is {1:,}'.format(row['CombinationID'], row['Risk_Charge']))

    else:
        logger.info('No trade has Risk')

    return

if __name__ == '__main__':
    main()