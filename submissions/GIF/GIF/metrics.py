import selfies
import sklearn.metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics.pairwise import cosine_similarity
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.DataStructs import TanimotoSimilarity
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
import re
from collections import Counter
from pymatgen.core import Composition
import json
import numpy as np
import matplotlib.pyplot as plt


def calculate_mz(mass, adduct):
    charge = adduct[-1]
    adduct_formula = adduct.split('+')[1].split(']')[0]
    adduct_mass = molar_mass(adduct_formula)
    assert adduct_mass !=False
    if charge == '+':
        mz = mass + adduct_mass
    else:
        assert charge == '-'
        mz = mass - adduct_mass
    return mz

def molar_mass(formula):
    try:
        comp = Composition(formula)
    except ValueError:
        return False
    try:
        mass = comp.weight
    except AttributeError:
        return False
    return mass

def is_subformula(sub_formula, full_formula):
    try:
        sub_comp = Composition(sub_formula)
    except ValueError: #isn't valid formula
        return False
    full_comp = Composition(full_formula)

    for el, sub_amt in sub_comp.get_el_amt_dict().items():
        full_amt = full_comp.get_el_amt_dict().get(el, 0)
        if sub_amt > full_amt:
            return False
    return True

def selfies_to_smiles(selfies_str):
    try:
        smiles = selfies.decoder(selfies_str)
    except:
        try:
            smiles = selfies.decoder(selfies_str+']')
        except:
            smiles = ''
    return smiles

def is_valid_selfies(selfies_str):
    # try:
    # Convert SELFIES to SMILES
    # try:
    #     smiles = selfies.decoder(selfies_str)
    # except:
    #     try:
    #         smiles = selfies.decoder(selfies_str+']')
    #     except:
    #         smiles = ''
    smiles = selfies_to_smiles(selfies_str)

    if smiles == '':
        return False
    else:
        # Convert SMILES to RDKit molecule
        mol = Chem.MolFromSmiles(smiles)

        # If mol is None, it's an invalid molecule
        return mol is not None
    # except Exception as e:
    #     print(f"Error during validation: {e}")
    #     return False

def tanimoto_similarity_selfies(selfies1, selfies2):
    # smiles = selfies.decoder(selfies1)
    smiles = selfies_to_smiles(selfies1)
    mol = Chem.MolFromSmiles(smiles)
    fp1 = GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)

    # smiles = selfies.decoder(selfies2)
    smiles = selfies_to_smiles(selfies2)
    mol = Chem.MolFromSmiles(smiles)
    fp2 = GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)

    return TanimotoSimilarity(fp1, fp2)

def is_structurally_equivalent(selfies1, selfies2):
    # smiles = selfies.decoder(selfies1)
    smiles = selfies_to_smiles(selfies1)
    mol = Chem.MolFromSmiles(smiles)
    ik1 = Chem.inchi.MolToInchiKey(mol).split('-')[0]

    # smiles = selfies.decoder(selfies2)
    smiles = selfies_to_smiles(selfies2)
    mol = Chem.MolFromSmiles(smiles)
    ik2 = Chem.inchi.MolToInchiKey(mol).split('-')[0]
    
    return ik1 == ik2

def is_substructure(frag_selfies, mol_smi):
    # smiles = selfies.decoder(frag_selfies)
    smiles = selfies_to_smiles(frag_selfies)
    frag_mol = Chem.MolFromSmiles(smiles)
    if frag_mol == None:
        return False
    else:
        target_mol = Chem.MolFromSmiles(mol_smi)

        return target_mol.HasSubstructMatch(frag_mol)

def is_substructure_smi(frag_smi, mol_smi):
    # smiles = selfies.decoder(frag_selfies)
    # smiles = selfies_to_smiles(frag_selfies)
    frag_mol = Chem.MolFromSmiles(frag_smi)
    if frag_mol == None:
        return False
    else:
        target_mol = Chem.MolFromSmiles(mol_smi)

        return target_mol.HasSubstructMatch(frag_mol)

def masked_selfies_evaluation(response, data):
    generated_selfies = response['completed_selfies']
    true_selfies = data['selfies']
    validity = is_valid_selfies(generated_selfies)
    accuracy = generated_selfies == true_selfies

    if validity:
        sim = tanimoto_similarity_selfies(generated_selfies, true_selfies)
    else:
        sim = 0

    #is everything that wasn't masked the same?
    gen_unmasked_accuracy = True
    selfies_masked_tokens = data['selfies_masked'].split('][')
    selfies_true_tokens = true_selfies.split('][')
    selfies_gen_tokens = generated_selfies.split('][')
    for i in range(0,len(selfies_masked_tokens)):
        if selfies_masked_tokens[i] != 'MASK':
            assert selfies_masked_tokens[i] == selfies_true_tokens[i]
            if selfies_true_tokens[i] != selfies_gen_tokens[i]:
                gen_unmasked_accuracy = False
                break
    #how many masked tokens are correct or wrong?
    masked_token_accuracy = []
    for i in range(0,len(selfies_masked_tokens)):
        if selfies_masked_tokens[i] == 'MASK':
            masked_token_accuracy.append(selfies_true_tokens[i] == selfies_gen_tokens[i])

    return validity, accuracy, sim, gen_unmasked_accuracy, masked_token_accuracy

def masked_fragment_evaluation(response, data):
    gen_frags = []
    for frag in response['completed_fragments']:
        gen_frags.append(frag['frag']) #TODO should be frag
    true_frags = data['sorted_frags']
    sequence_level_accuracy = gen_frags == true_frags
    # accuracy_irrespective_order = sorted(true_frags) == sorted(gen_frags)
    overall_frag_accuracy = len(set(true_frags).intersection(set(gen_frags)))

    #is everything that wasn't masked the same?
    gen_unmasked_accuracy = True
    for i in range(len(true_frags)):
        if data['frags_masked'][i]['frag'] != '[MASK]':
            assert data['frags_masked'][i]['frag'] == true_frags[i]
            if true_frags[i] not in gen_frags:
                gen_unmasked_accuracy = False
                break
            # if len(gen_frags)>i:
            #     if gen_frags[i] != true_frags[i]:
            #         gen_unmasked_accuracy = False
            #         break
            # else:
            #     gen_unmasked_accuracy = False
            #     break
    #are masked tokens correct?
    masked_frag_accuracy = []
    #tanimoto between masked frags
    masked_frag_sim = []
    masked_frag_structurally_equivalent = []
    for i in range(len(true_frags)):
        if data['frags_masked'][i]['frag'] == '[MASK]':
            masked_frag_accuracy.append(gen_frags[i] == true_frags[i])
            masked_frag_structurally_equivalent.append(is_structurally_equivalent(gen_frags[i], true_frags[i]))
            masked_frag_sim.append(tanimoto_similarity_selfies(gen_frags[i], true_frags[i]))


    #chemical validity of frags
    #true substructure
    validity = []
    true_substructure = []
    for frag in gen_frags:
        validity.append(is_valid_selfies(frag))
        true_substructure.append(is_substructure(frag, data['smiles']))
        
    #TODO do we want to do per query numbers or overall masked tokens numbers

    return sequence_level_accuracy, overall_frag_accuracy, gen_unmasked_accuracy, masked_frag_accuracy, masked_frag_structurally_equivalent, masked_frag_sim, validity, true_substructure

def fragment_list_prediction_evaluation(response, data):
    #add these "coverage terms" and sort fragments
    gen_frags = []
    if 'fragments' in response.keys():
        for frag in response['fragments']:
            gen_frags.append(frag['frag']) #TODO should be frag
    true_frags = data['sorted_frags']
    sequence_level_accuracy = gen_frags == true_frags
    # accuracy_irrespective_order = sorted(true_frags) == sorted(gen_frags)
    # how many SELFIES are the same?
    if len(set(true_frags)) == 0:
        overall_frag_accuracy = 0
    else:
        overall_frag_accuracy = len(set(true_frags).intersection(set(gen_frags)))/len(set(true_frags))
    #convert to smiles
    smi_gen_frags = []
    for frag in gen_frags:
        # TODO confirm this is consistent with everything
        # temp_smi = selfies_to_smiles(frag)
        smi_gen_frags.append(selfies_to_smiles(frag))
        # if temp_smi == '':
        #     #check if SMILES
        #     try:
        #         mol = Chem.MolFromSmiles(frag)
        #         # If mol is not None, it means RDKit successfully parsed the SMILES.
        #         if mol is not None:
        #             smi_gen_frags.append(frag)
        #     except Exception:
        #         smi_gen_frags.append('')
    smi_true_frags = []
    for frag in true_frags:
        smi_true_frags.append(selfies_to_smiles(frag))
    # how many inchikey block 1 are the same?
    ik_gen_frags=[]
    formula_gen_frags = []
    mass_gen_frags = []
    for i in range(len(smi_gen_frags)):
        m = Chem.MolFromSmiles(smi_gen_frags[i])
        if m != None:
            ik = Chem.inchi.MolToInchiKey(m).split('-')[0]
            ik_gen_frags.append(ik)

            inchi = Chem.inchi.MolToInchi(m)
            if '/' in inchi:
                formula = inchi.split('/')[1]
            else:
                formula = ''
            mass = Descriptors.ExactMolWt(m)

            formula_gen_frags.append(formula)
            mass_gen_frags.append(round(mass,2))
    ik_true_frags = []
    formula_true_frags = []
    mass_true_frags = []
    for i in range(len(smi_true_frags)):
        m = Chem.MolFromSmiles(smi_true_frags[i])
        ik = Chem.inchi.MolToInchiKey(m).split('-')[0]
        ik_true_frags.append(ik)

        inchi = Chem.inchi.MolToInchi(m)
        formula = inchi.split('/')[1]
        mass = Descriptors.ExactMolWt(m)

        formula_true_frags.append(formula)
        mass_true_frags.append(round(mass,2))
    # how many formula are the same?
    if len(set(formula_true_frags)) == 0:
        formula_accuracy = 0
    else:
        formula_accuracy = len(set(formula_true_frags).intersection(set(formula_gen_frags)))/len(set(formula_true_frags))
    # how many mass are the same rounded to the nearest integer?
    if len(set(mass_true_frags)) == 0:
        mass_accuracy = 0
    else:
        mass_accuracy = len(set(mass_true_frags).intersection(set(mass_gen_frags)))/len(set(mass_true_frags))

    # chemical validity of frags
    # true substructure
    validity = []
    true_substructure = []
    for frag in gen_frags:
        validity.append(is_valid_selfies(frag))
        true_substructure.append(is_substructure(frag, data['smiles']))


    return sequence_level_accuracy, overall_frag_accuracy, formula_accuracy, mass_accuracy, validity, true_substructure

def smi_fragment_list_prediction_evaluation(response, data):
    #add these "coverage terms" and sort fragments
    gen_frags = []
    if 'fragments' in response.keys():
        for frag in response['fragments']:
            gen_frags.append(frag['frag']) #TODO should be frag
    true_frags = data['sorted_frags']
    sequence_level_accuracy = gen_frags == true_frags
    # accuracy_irrespective_order = sorted(true_frags) == sorted(gen_frags)
    # how many SELFIES are the same?
    if len(set(true_frags)) == 0:
        overall_frag_accuracy = 0
    else:
        overall_frag_accuracy = len(set(true_frags).intersection(set(gen_frags)))/len(set(true_frags))
    #convert to smiles
    smi_gen_frags = []
    validity_smi = []
    for frag in gen_frags:
        #TODO confirm this is consistent with everything
        temp_smi = selfies_to_smiles(frag)
        # smi_gen_frags.append(selfies_to_smiles(frag))
        if temp_smi == '':
            #check if SMILES
            try:
                mol = Chem.MolFromSmiles(frag)
                # If mol is not None, it means RDKit successfully parsed the SMILES.
                if mol is not None:
                    smi_gen_frags.append(frag)
                    validity_smi.append(True)
                else:
                    smi_gen_frags.append('')
                    validity_smi.append(False)
            except Exception:
                smi_gen_frags.append('')
                validity_smi.append(False)
        else:
            smi_gen_frags.append(temp_smi)
            validity_smi.append(True)
    smi_true_frags = []
    for frag in true_frags:
        smi_true_frags.append(selfies_to_smiles(frag))
    # how many inchikey block 1 are the same?
    ik_gen_frags=[]
    formula_gen_frags = []
    mass_gen_frags = []
    for i in range(len(smi_gen_frags)):
        m = Chem.MolFromSmiles(smi_gen_frags[i])
        if m != None:
            ik = Chem.inchi.MolToInchiKey(m).split('-')[0]
            ik_gen_frags.append(ik)

            inchi = Chem.inchi.MolToInchi(m)
            if '/' in inchi:
                formula = inchi.split('/')[1]
            else:
                formula = ''
            mass = Descriptors.ExactMolWt(m)

            formula_gen_frags.append(formula)
            mass_gen_frags.append(round(mass,2))
    ik_true_frags = []
    formula_true_frags = []
    mass_true_frags = []
    for i in range(len(smi_true_frags)):
        m = Chem.MolFromSmiles(smi_true_frags[i])
        ik = Chem.inchi.MolToInchiKey(m).split('-')[0]
        ik_true_frags.append(ik)

        inchi = Chem.inchi.MolToInchi(m)
        formula = inchi.split('/')[1]
        mass = Descriptors.ExactMolWt(m)

        formula_true_frags.append(formula)
        mass_true_frags.append(round(mass,2))
    # how many formula are the same?
    if len(set(formula_true_frags)) == 0:
        formula_accuracy = 0
    else:
        formula_accuracy = len(set(formula_true_frags).intersection(set(formula_gen_frags)))/len(set(formula_true_frags))
    # how many mass are the same rounded to the nearest integer?
    if len(set(mass_true_frags)) == 0:
        mass_accuracy = 0
    else:
        mass_accuracy = len(set(mass_true_frags).intersection(set(mass_gen_frags)))/len(set(mass_true_frags))


    # validity_smi = len(smi_gen_frags)/len(gen_frags)
    true_substructure_smi = []
    for frag in smi_gen_frags:
        true_substructure_smi.append(is_substructure_smi(frag, data['smiles']))

    # TODO do we want to do per query numbers or overall masked tokens numbers

    return sequence_level_accuracy, overall_frag_accuracy, formula_accuracy, mass_accuracy, validity_smi, true_substructure_smi


def cumulative_fragment_list_prediction_evaluation(task, data, save_path):
    n = re.search(r'\d+$', task)
    # if the string ends in digits m will be a Match object, or None otherwise.
    # if n == None:
    #     n=1
    # else:
    n = int(n.group())
    gen_frags = []
    for i in range(0,n+1):
        if i == 0:
            with open(save_path + 'raw/json/' + 'FragmentListPrediction' + '/' + data['identifier'] + '_' + 'FragmentListPrediction' + '_model_response.json',
                      'r') as file:
                response = json.load(file)
        # elif i == 1:
        #     with open(save_path + 'raw/json/' + 'IterativeFragmentListPrediction' + '/' + data['identifier'] + '_' + 'IterativeFragmentListPrediction' + '_model_response.json',
        #               'r') as file:
        #         response = json.load(file)
        else:
            with open(save_path + 'raw/json/' + 'IterativeFragmentListPrediction' + str(i) + '/' + data['identifier'] + '_' + 'IterativeFragmentListPrediction' + str(i) + '_model_response.json',
                      'r') as file:
                response = json.load(file)
        if 'fragments' in response.keys():
            for frag in response['fragments']:
                gen_frags.append(frag['frag'])
    gen_frags = list(set(gen_frags))
    true_frags = data['sorted_frags']
    sequence_level_accuracy = gen_frags == true_frags
    # accuracy_irrespective_order = sorted(true_frags) == sorted(gen_frags)
    # how many SELFIES are the same?
    if len(set(true_frags)) == 0:
        overall_frag_accuracy = 0
    else:
        overall_frag_accuracy = len(set(true_frags).intersection(set(gen_frags))) / len(set(true_frags))
    # convert to smiles
    smi_gen_frags = []
    for frag in gen_frags:
        smi_gen_frags.append(selfies_to_smiles(frag))
    smi_true_frags = []
    for frag in true_frags:
        smi_true_frags.append(selfies_to_smiles(frag))
    # how many inchikey block 1 are the same?
    ik_gen_frags = []
    formula_gen_frags = []
    mass_gen_frags = []
    for i in range(len(smi_gen_frags)):
        m = Chem.MolFromSmiles(smi_gen_frags[i])
        if m != None:
            ik = Chem.inchi.MolToInchiKey(m).split('-')[0]
            ik_gen_frags.append(ik)

            inchi = Chem.inchi.MolToInchi(m)
            if '/' in inchi:
                formula = inchi.split('/')[1]
            else:
                formula = ''
            mass = Descriptors.ExactMolWt(m)

            formula_gen_frags.append(formula)
            mass_gen_frags.append(round(mass,2))
    ik_true_frags = []
    formula_true_frags = []
    mass_true_frags = []
    for i in range(len(smi_true_frags)):
        m = Chem.MolFromSmiles(smi_true_frags[i])
        ik = Chem.inchi.MolToInchiKey(m).split('-')[0]
        ik_true_frags.append(ik)

        inchi = Chem.inchi.MolToInchi(m)
        formula = inchi.split('/')[1]
        mass = Descriptors.ExactMolWt(m)

        formula_true_frags.append(formula)
        mass_true_frags.append(round(mass,2))
    # how many formula are the same?
    if len(set(formula_true_frags)) == 0:
        formula_accuracy = 0
    else:
        formula_accuracy = len(set(formula_true_frags).intersection(set(formula_gen_frags))) / len(
            set(formula_true_frags))
    # how many mass are the same rounded to the nearest integer?
    if len(set(mass_true_frags)) == 0:
        mass_accuracy = 0
    else:
        mass_accuracy = len(set(mass_true_frags).intersection(set(mass_gen_frags))) / len(set(mass_true_frags))

    # #based on order find structural equivalence or tanimoto similarity
    # frags_structurally_equivalent = []
    # frags_sim = []
    # for i in range(len(gen_frags)):
    #     if len(true_frags)<=i:
    #         break
    #     frags_structurally_equivalent.append(is_structurally_equivalent(gen_frags[i], true_frags[i]))
    #     frags_sim.append(tanimoto_similarity_selfies(gen_frags[i], true_frags[i]))

    # chemical validity of frags
    # true substructure
    validity = []
    true_substructure = []
    for frag in gen_frags:
        validity.append(is_valid_selfies(frag))
        true_substructure.append(is_substructure(frag, data['smiles']))

    return sequence_level_accuracy, overall_frag_accuracy, formula_accuracy, mass_accuracy, validity, true_substructure

def masked_intensity_evaluation(response, data):
    gen_spec = response['completed_spectrum']
    true_spec = data['frags_mz_int']
    # gen_mzs = []
    # gen_frags = []
    # for frag in true_spec:
    #     gen_frags.append(frag['frag'])
    # gen_ints = []
    # masked_ints = []
    sequence_level_accuracy = gen_spec == true_spec

    # is everything that wasn't masked the same?
    gen_unmasked_accuracy = True
    for i in range(0, len(true_spec)):
        if true_spec[i]['frag'] != gen_spec[i]['frag']:
            gen_unmasked_accuracy = False
            break
        if true_spec[i]['mz'] != gen_spec[i]['mz']:
            gen_unmasked_accuracy = False
            break
        if data['frags_int_masked'][i]['int'] != '[MASK]':
            # assert data['frags_int_masked'][i]['frag'] == true_spec[i]['frag']
            if true_spec[i]['int'] != gen_spec[i]['int']:
                gen_unmasked_accuracy = False
                break

    #are masked intensities correct?
    masked_int_accuracy = []
    masked_true_int = []
    masked_gen_int = []
    for i in range(0, len(true_spec)): #TODO by length, or create a dictionary based on fragments
        if data['frags_int_masked'][i]['int'] == '[MASK]':
            masked_int_accuracy.append(gen_spec[i]['int'] == true_spec[i]['int'])
            masked_true_int.append(true_spec[i]['int'])
            masked_gen_int.append(gen_spec[i]['int'])
    masked_int_mse = mean_squared_error(masked_true_int, masked_gen_int)
    masked_int_mae = mean_absolute_error(masked_true_int, masked_gen_int)

    return sequence_level_accuracy, gen_unmasked_accuracy, masked_int_accuracy, masked_int_mse, masked_int_mae

def intensity_frag_prediction_evaluation(response, data):
    if 'intensities' in response.keys():
        gen_spec = response['intensities']
    else:
        gen_spec = []
    true_spec = data['frags_mz_int']
    # gen_mzs = []
    # gen_frags = []
    # for frag in true_spec:
    #     gen_frags.append(frag['frag'])
    # gen_ints = []
    # masked_ints = []
    sequence_level_accuracy = gen_spec == true_spec

    # is everything that wasn't masked the same?
    gen_input_accuracy = True
    for i in range(0, len(true_spec)):
        if len(gen_spec)<=i:
            break
        #TODO if regenerating fragment too...
        # if true_spec[i]['frag'] != gen_spec[i]['frag']:
        #     gen_input_accuracy = False
        #     break
        if true_spec[i]['mz'] != gen_spec[i]['mz']:
            gen_input_accuracy = False
            break
        # if data['frags_int_masked'][i]['int'] != '[MASK]':
        #     # assert data['frags_int_masked'][i]['frag'] == true_spec[i]['frag']
        #     if true_spec[i]['int'] != gen_spec[i]['int']:
        #         gen_unmasked_accuracy = False
        #         break


    # are  intensities correct?
    int_accuracy = []
    true_int = []
    gen_int = []
    for i in range(0, len(gen_spec)): #TODO by length, or create a dictionary based on fragments
        if len(true_spec) <= i:
            break
        if 'int' in gen_spec[i].keys():
            int_accuracy.append(gen_spec[i]['int'] == true_spec[i]['int'])
            true_int.append(true_spec[i]['int'])
            gen_int.append(gen_spec[i]['int'])
    if len(gen_int) > 0:
        int_mse = mean_squared_error(true_int, gen_int)
        int_mae = mean_absolute_error(true_int, gen_int)
    else:
        int_mse = 0
        int_mae = 0

    return sequence_level_accuracy, gen_input_accuracy, int_accuracy, int_mse, int_mae


def parse_formula(formula):
    # Regular expression to parse elements and their counts
    pattern = r'([A-Z][a-z]?)(\d*)'
    elements = re.findall(pattern, formula)

    # Convert to element:count dictionary
    counts = Counter()
    for element, count in elements:
        counts[element] += int(count) if count else 1
    return counts

def subformula_prediction_evaluation(response, data):
    # add these "coverage terms" and sort fragments
    gen_sfs = []
    if 'subformulae' in response.keys():
        for pt in response['subformulae']:
            assert list(pt.keys()) == ['subformula']
            if pt['subformula'][-1] == '+':
                gen_sfs.append(pt['subformula'][:-1])
            gen_sfs.append(pt['subformula'])
    # if 'fragments' in response.keys():
    #     for frag in response['fragments']:
    #         gen_frags.append(frag['frag'])  # TODO should be frag
    true_sfs = data['sorted_subforms']
    sequence_level_accuracy = gen_sfs == true_sfs
    # accuracy_irrespective_order = sorted(true_frags) == sorted(gen_frags)
    # how many SELFIES are the same?
    if len(set(true_sfs)) == 0:
        overall_sf_accuracy = 0
    else:
        overall_sf_accuracy = len(set(true_sfs).intersection(set(gen_sfs))) / len(set(true_sfs))
    # TODO use parse_formula to convert so order doesn't matter
    true_sf_dicts = []
    for sf in true_sfs:
        true_sf_dicts.append(parse_formula(sf))
    gen_sf_dicts = []
    for sf in gen_sfs:
        gen_sf_dicts.append(parse_formula(sf))
    #TODO are they equivalent?
    c1 = Counter(frozenset(d.items()) for d in true_sf_dicts)
    c2 = Counter(frozenset(d.items()) for d in gen_sf_dicts)
    unordered_sf_equivalence = c1 == c2
    unordered_sf_accuracy = len(c1 & c2)/ len(c1)

    #get mass accuracy (rounded)
    validity = []
    gen_masses = []
    for sf in gen_sfs:
        temp = molar_mass(sf)
        if temp == False:
            validity.append(False)
        else:
            validity.append(True)
            gen_masses.append(round(temp,2))
    true_masses = []
    for sf in true_sfs:
        true_masses.append(round(molar_mass(sf),2))
    if len(set(gen_masses)) == 0:
        mass_accuracy = 0
    else:
        mass_accuracy = len(set(true_masses).intersection(set(gen_masses))) / len(set(true_masses))


    #true subformula
    actual_subformula = []
    for sf in gen_sfs:
        actual_subformula.append(is_subformula(sf, data['formula']))

    return sequence_level_accuracy, overall_sf_accuracy, unordered_sf_accuracy, unordered_sf_equivalence, mass_accuracy, sum(actual_subformula)/len(actual_subformula), sum(validity)/len(validity)


def intensity_prediction_evaluation(response, data):
    if 'intensities' in response.keys():
        gen_spec = response['intensities']
    else:
        gen_spec = []
    true_spec = data['subforms_mz_int']
    # gen_mzs = []
    # gen_frags = []
    # for frag in true_spec:
    #     gen_frags.append(frag['frag'])
    # gen_ints = []
    # masked_ints = []
    sequence_level_accuracy = gen_spec == true_spec

    # is everything that wasn't masked the same?
    gen_input_accuracy = True
    for i in range(0, len(true_spec)):
        if len(gen_spec)<=i:
            break
        if true_spec[i]['subformula'] != gen_spec[i]['subformula']:
            gen_input_accuracy = False
            break
        if true_spec[i]['mz'] != gen_spec[i]['mz']:
            gen_input_accuracy = False
            break
        # if data['frags_int_masked'][i]['int'] != '[MASK]':
        #     # assert data['frags_int_masked'][i]['frag'] == true_spec[i]['frag']
        #     if true_spec[i]['int'] != gen_spec[i]['int']:
        #         gen_unmasked_accuracy = False
        #         break

    # are  intensities correct?
    int_accuracy = []
    true_int = []
    gen_int = []
    for i in range(0, len(gen_spec)): #TODO by length, or create a dictionary based on fragments
        if len(true_spec) <= i:
            break
        if 'int' in gen_spec[i].keys():
            int_accuracy.append(gen_spec[i]['int'] == true_spec[i]['int'])
            true_int.append(true_spec[i]['int'])
            gen_int.append(gen_spec[i]['int'])
    if len(gen_int) > 0:
        int_mse = mean_squared_error(true_int, gen_int)
        int_mae = mean_absolute_error(true_int, gen_int)
    else:
        int_mse = 0
        int_mae = 0

    return sequence_level_accuracy, gen_input_accuracy, int_accuracy, int_mse, int_mae

def bin_aggregate(data, bins):
    try:
        x, y = data[:, 0], data[:, 1]
    except:
        print('error')
    bin_indices = np.digitize(x, bins) - 1
    vector = np.zeros(len(bins) - 1)
    for i in range(len(vector)):
        y_vals = y[bin_indices == i]
        vector[i] = np.sum(y_vals) if len(y_vals) > 0 else 0
    return vector

def bin_sparse_vector(spec, bin_size=0.01):
    """Return sparse vector as a dict {bin_idx: summed intensity}."""
    # if spec.ndim == 1:
    #     print('check')
    bins = np.floor(spec[:, 0] / bin_size).astype(np.int32)
    intensities = spec[:, 1]
    binned = {}
    for b, i in zip(bins, intensities):
        binned[b] = binned.get(b, 0.0) + i
    return binned

def sparse_cosine_similarity(pred_bins, true_bins):
    """Cosine similarity over sparse binned dicts."""
    all_bins = set(pred_bins.keys()).union(true_bins.keys())
    pred_vec = np.array([pred_bins.get(b, 0.0) for b in all_bins])
    true_vec = np.array([true_bins.get(b, 0.0) for b in all_bins])
    pred_norm = np.linalg.norm(pred_vec)
    true_norm = np.linalg.norm(true_vec)
    if pred_norm == 0 or true_norm == 0:
        return 0.0
    pred_vec /= pred_norm
    true_vec /= true_norm
    return float(np.dot(pred_vec, true_vec))

def jensen_shannon_similarity(pred_bins, true_bins, eps=1e-12):
    """Compute JSS = 1 - sqrt(JS divergence) between two sparse spectra."""
    all_bins = set(pred_bins.keys()).union(true_bins.keys())
    p = np.array([pred_bins.get(b, 0.0) for b in all_bins]) + eps
    q = np.array([true_bins.get(b, 0.0) for b in all_bins]) + eps
    p /= p.sum()
    q /= q.sum()
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log(p / m))
    kl_qm = np.sum(q * np.log(q / m))
    js_div = 0.5 * (kl_pm + kl_qm)
    # js_dist = np.sqrt(js_div)
    # return 1.0 - js_dist  # similarity version as used in the paper
    jss = 1.0 - (js_div / np.log(2))  # Normalize using log(2)
    return float(jss)

def baseline_metrics(response, data):
    if 'intensities' in response:
        list_gen_spec = response['intensities']
    else:
        list_gen_spec = []

    np_gen_spec = []
    for peak in list_gen_spec:
        if 'mz' in peak and 'int' in peak:
            np_gen_spec.append([float(peak['mz']), float(peak['int']) / 10])

    if len(np_gen_spec) == 0:
        return 0.0, None #0.0 #TODO should I change?

    np_gen_spec = np.array(np_gen_spec)

    np_true_spec = np.array([
        [data['mzs'][i], data['intensities'][i]]
        for i in range(len(data['mzs']))
    ])

    # # Compute sparse cosine similarity
    # cos_sim = sparse_cosine_similarity(
    #     np_gen_spec,
    #     np_true_spec,
    #     bin_size=0.01,
    #     mz_max=1005.0
    # )

    if np_true_spec.ndim == 1:
        return 0.0, None #TODO should I change?

    # Bin both spectra
    pred_bins = bin_sparse_vector(np_gen_spec, bin_size=0.01)
    true_bins = bin_sparse_vector(np_true_spec, bin_size=0.01)

    # Compute metrics
    cos_sim = sparse_cosine_similarity(pred_bins, true_bins)
    js_sim = jensen_shannon_similarity(pred_bins, true_bins)

    return cos_sim, js_sim

# #my original function
# def baseline_metrics(response, data):
#     # cosine_similarity(A.reshape(1, -1), B.reshape(1, -1))
#     if 'intensities' in response.keys():
#         list_gen_spec = response['intensities']
#     else:
#         list_gen_spec = []
#     np_gen_spec = []
#     for peak in list_gen_spec: #TODO or do we also use the m/z from the query!
#         if 'mz' in peak.keys() and 'int' in peak.keys():
#             np_gen_spec.append([peak['mz'], peak['int']/10])
#         # else:
#         #     print('check') #no int was generated usually
#     if len(np_gen_spec) == 0:
#         return 0
#     np_gen_spec = np.array(np_gen_spec)
#
#     np_true_spec = []
#     for i in range(0,len(data['mzs'])):
#         np_true_spec.append([data['mzs'][i], data['intensities'][i]])
#     np_true_spec = np.array(np_true_spec)
#
#     bin_size = 0.01
#     max_mz = 1005
#     min_mz = 0
#     bins = np.arange(min_mz, max_mz + bin_size, bin_size)
#
#     gen_vec = bin_aggregate(np_gen_spec, bins)
#     true_vec = bin_aggregate(np_true_spec, bins)
#
#     cos_sim = cosine_similarity([gen_vec], [true_vec])[0][0]
#
#     return cos_sim
#     # return compute_cosine_similarity_with_tolerance(np_true_spec, np_gen_spec, tolerance=0.01)

def calculate_best_case_cos_sim_from_frags(response,data):
    np_true_spec = []
    dict_true_spec = {}
    for i in range(0, len(data['mzs'])):
        np_true_spec.append([round(data['mzs'][i],2), data['intensities'][i]])
        dict_true_spec[round(data['mzs'][i],2)] = data['intensities'][i]
    np_true_spec = np.array(np_true_spec)

    np_gen_spec = []
    if 'fragments' in response.keys():
        for frag in response['fragments']:
            m = Chem.MolFromSmiles(selfies_to_smiles(frag['frag']))
            mass = Descriptors.ExactMolWt(m)
            mz = calculate_mz(mass, data['adduct'])
            if round(mz, 2) in dict_true_spec.keys():
                np_gen_spec.append([round(mz,2), dict_true_spec[round(mz,2)]])
    if len(np_gen_spec) == 0:
        return 0

    np_gen_spec = np.array(np_gen_spec)



    # Bin both spectra
    pred_bins = bin_sparse_vector(np_gen_spec, bin_size=0.01)
    true_bins = bin_sparse_vector(np_true_spec, bin_size=0.01)

    # Compute metrics
    cos_sim = sparse_cosine_similarity(pred_bins, true_bins)

    return cos_sim

def calculate_regen_mz(frag_response, int_response, data):
    # #how often are the mz values at least the same?
    #recalculate
    gen_frags = []
    if 'fragments' in frag_response.keys():
        for frag in frag_response['fragments']:
            gen_frags.append(frag['frag'])  # TODO should be frag
    # true_frags = data['sorted_frags']
    true_substructure = {}
    for frag in gen_frags:
        if is_substructure(frag, data['smiles']) and frag not in true_substructure.keys():
            m = Chem.MolFromSmiles(selfies_to_smiles(frag))
            mass = Descriptors.ExactMolWt(m)
            mz = calculate_mz(mass, data['adduct'])
            true_substructure[frag] = mz
    gen_frags_mz = []
    for frag in true_substructure.keys():
        # frags_mz_str.append({"frag": frag, "mz": round(frag_mz[frag], 2)})
        gen_frags_mz.append(round(true_substructure[frag], 2))

    gen_int_mz = []
    if 'intensities' in int_response.keys():
        for frag in int_response['intensities']:
            gen_int_mz.append(frag['mz'])

    if len(set(gen_frags_mz)) > 0:
        regen_accuracy = len(set(gen_int_mz).intersection(set(gen_frags_mz))) / len(set(gen_frags_mz))
    else:
        regen_accuracy = 0

    return regen_accuracy


def plot_distribution(data, task, data_name, save_path):
    plt.hist(data, bins=20, edgecolor='black')  # 'bins' controls the number of bins
    plt.xlabel(data_name)
    plt.ylabel("Count")
    plt.savefig(save_path + "figures/" + data_name.replace(' ', '_') + "_" + task + ".png")
    plt.clf()
    return

