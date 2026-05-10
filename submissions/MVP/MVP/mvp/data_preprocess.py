import argparse
from mvp.utils.preprocessing import generate_cons_spec_formulas, generate_cons_spec
import os
import pickle
import pandas as pd
from rdkit.Chem import AllChem
from rdkit import Chem
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--spec_type", choices=('formSpec', 'binnedSpec'), required=True)
parser.add_argument("--dataset_pth", required=True, help="path to spectra data")
parser.add_argument("--candidates_pth", required=True, help="path to candidates data")
parser.add_argument("--output_dir", required=True, help="path to output directory")
parser.add_argument("--subformula_dir_pth",  default='', help="path to subformula directory if using formSpec")


def check_args():

    # create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # check files
    if args.spec_type == 'formSpec':
        assert(os.path.isdir(args.subformula_dir_pth))
    
    assert(os.path.exists(args.dataset_pth))
    assert(os.path.exists(args.candidates_pth))

def construct_smiles_to_fp(smiles_list, r=5, fp_size=1024):
    fpgen = AllChem.GetMorganGenerator(radius=r,fpSize=fp_size)
    smiles_to_fp = {}
    failed_ct = 0

    for s in tqdm(smiles_list, total=len(smiles_list)):
        try:
            mol = Chem.MolFromSmiles(s)
            fp = fpgen.GetFingerprint(mol)
            smiles_to_fp[s] = fp
        except:
            failed_ct+=1
    print(f'Failed to generate fingerprints for {failed_ct} smiles')

    # save smiles_to_fp
    with open(os.path.join(args.output_dir, f'morganfp_r{r}_{fp_size}.pickle'), 'wb') as f:
        pickle.dump(smiles_to_fp, f)

def construct_consensus_spectra():
    if args.spec_type == 'formSpec':
        df = generate_cons_spec_formulas(args.dataset_pth, args.subformula_dir_pth, args.output_dir)
    elif args.spec_type == 'binnedSpec':
        df = generate_cons_spec(args.dataset_pth, args.output_dir)

    # save consensus spectra df
    with open(os.path.join(args.output_dir, f'consensus_{args.spec_type}.pkl'), 'wb') as f:
        pickle.dump(df, f)

def main(data):

    # generate fingerpints
    print("Processing fingerprints...")
    unique_smiles = data['smiles'].unique().tolist()
    construct_smiles_to_fp(unique_smiles)

    # generate consensus spectra
    print("Processring consensus spectra...")
    construct_consensus_spectra()


if __name__ == '__main__':
    args = parser.parse_args([] if "__file__" not in globals() else None)

    check_args()

    # load data
    data = pd.read_csv(args.dataset_pth, sep='\t')

    main(data)