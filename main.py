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
    parser = argparse.ArgumentParser(description='SIMM Calculation.')
    parser.add_argument('-f', dest='input_file', type=str, required=True, help='simm input csv file')
    args = parser.parse_args(['-f' 'frtb_config.xlsx'])
    #args = parser.parse_args()

    # Create output directory for product and risk class
    frtb_lib.prep_output_directory(params)

    # Read input file with specified data type
    #input_file = 'simm_input_1.csv'
    input_file = args.input_file

    trades_simm = frtb_lib.generate_trade_pos(input_file, params)
    run_cases = frtb_lib.generate_run_cases(input_file, trades_simm)

    # Calculate SIMM and dump output
    simm_all = []
    if len(run_cases) > 0:
        for case in run_cases.CombinationID.unique():
            logger.info('Run test {0}'.format(case))
            run_case = run_cases[run_cases.CombinationID == case].copy()
            simm = frtb_lib.calculate_simm(run_case, params)
            simm_all.append(simm)

        simm_all = pd.concat(simm_all)
        simm_all.to_csv('simm_output.csv', index=False)

        for index, row in simm_all.iterrows():
            logger.info('{0}: Total SIMM is {1:,}'.format(row['CombinationID'], int(round(row['SIMM_Benchmark']))))

    else:
        logger.info('No trade has SIMM')

    return

if __name__ == '__main__':
    main()