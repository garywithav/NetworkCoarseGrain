import argparse
import numpy as np
import os
from pathlib import Path
from modules.LDS import *

def main():
    parser = argparse.ArgumentParser(description="Run the HMM-EM model.")
    parser.add_argument('filename', type=str, help='Input filename')
    parser.add_argument('--noisemode', type=int, default=2, help='Noise mode')
    parser.add_argument('--ridge', type=float, default=0.0, help='Ridge penalty value')
    parser.add_argument('--penalty_mode', type=str, default='L2', help='Penalty mode (default: L2)')

    args = parser.parse_args()
    
    dirnameIO = ''
    input_path = f'HMM_EM/input/{dirnameIO}'
    output_path = f'HMM_EM/output/{dirnameIO}'

    print('filename:', args.filename)
    print('noisemode:',args.noisemode)
    print('ridge:',args.ridge)
    
    # Prepare the ridge matrix based on the ridge value
    ridge_mat = np.zeros((1, 1, 1))
    if args.ridge > 0.0:
        try:
            ridge_mat = np.load(f'{input_path}ridgemat_{args.filename}.npy')
        except FileNotFoundError:
            print("File not found. Using default zero matrix.")

    # Load the necessary datasets
    res_counts = np.load(f'{input_path}counts_{args.filename}.npy')
    counts_deme = np.load(f'{input_path}totcounts_{args.filename}.npy')
    Qtest = 'n'
    test_file_path = f'{input_path}testcounts_{args.filename}.npy'
    if os.path.isfile(test_file_path):
        res_test_counts = np.load(test_file_path)
        Qtest = 'y'

    # Process the counts data and perform the HMM-EM
    res_A, res_Ne, res_A_LS, res_Csn, res_LH = [], [], [], [], []
    for counts in res_counts:
        ND, Nmut, tmax = counts.shape
        counts += 1
        
        lnLH_record, A_EM, Ne_EM, A_LS, Csn = Kalman_EM(
            counts, counts_deme, em_step_max=50, terminate_th=0.0001,
            noisemode=args.noisemode, ridge=args.ridge, ridge_mat=ridge_mat,
            penalty_mode=args.penalty_mode
        )
        
        res_A.append(A_EM.copy())
        res_Ne.append(Ne_EM.copy())
        res_A_LS.append(A_LS.copy())
        res_Csn.append(Csn.copy())
        res_LH.append(lnLH_record.copy())


    Path(output_path).mkdir(parents=True, exist_ok=True)
    
    # Save results
    np.save(f'{output_path}A_ridge{args.ridge}_{args.filename}.npy', np.array(res_A))
    np.save(f'{output_path}Ne_ridge{args.ridge}_{args.filename}.npy',  np.array(res_Ne))
    np.save(f'{output_path}A_LS_{args.filename}.npy', np.array(res_A_LS))
    np.save(f'{output_path}Csn_ridge{args.ridge}_{args.filename}.npy', np.array(res_Csn))
    np.savez(f'{output_path}LH_ridge{args.ridge}_{args.filename}.npz',  *res_LH)
    
    if Qtest == 'y':
        res_test_LH = []
        for i, A in enumerate(res_A):
            test_LH = calc_LH_fixed_parameters(
                A=A, Ne=res_Ne[i], Csn=res_Csn[i],
                counts=res_test_counts[i] + 1, counts_deme=counts_deme,
                infer_samplenoise=True, noisemode=args.noisemode
            )
            res_test_LH.append(test_LH)
        np.save(f'{output_path}Test_LH_AEM_ridge{args.ridge}_{args.filename}.npy', np.array(res_test_LH))


if __name__ == "__main__":
    main()
