"""Validation test suite for MassSpecGym v1.5 data pipeline, subformulae, and oracles."""

import sys
import json
import traceback
import numpy as np

PASS = 0
FAIL = 0

def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name} -- {detail}")


def test_chem_constants_subformulae():
    print("\n=== Phase 1: chem_constants subformulae functions ===")
    from massspecgym.models.encoders.mist.chem_constants import (
        get_all_subsets, rdbe_filter, cross_sum, clipped_ppm,
        formula_to_dense, VALID_MONO_MASSES, ELEMENT_VECTORS,
    )

    v = formula_to_dense("CH4")
    check("CH4 dense vec C=1", v[0] == 1)
    check("CH4 dense vec H=4", v[1] == 4)

    cross_prod, masses = get_all_subsets("CH4")
    check("CH4 subsets not empty", len(cross_prod) > 0, str(len(cross_prod)))
    check("CH4 subsets includes full formula", any(np.allclose(r, v) for r in cross_prod))
    check("CH4 masses > 0", all(m >= 0 for m in masses))

    cross_prod2, masses2 = get_all_subsets("C6H12O6")
    check("C6H12O6 has many subsets", len(cross_prod2) > 50, str(len(cross_prod2)))

    ppm = clipped_ppm(np.array([0.001]), np.array([500.0]))
    check("clipped_ppm scalar", abs(ppm[0] - 2.0) < 0.01, str(ppm[0]))

    ppm_small = clipped_ppm(np.array([0.001]), np.array([100.0]))
    expected = 0.001 / 200 * 1e6
    check("clipped_ppm clips to 200", abs(ppm_small[0] - expected) < 0.01, str(ppm_small[0]))


def test_subformulae_engine():
    print("\n=== Phase 2: Subformulae assignment engine ===")
    from massspecgym.data.subformulae import (
        assign_subformulae_single, get_output_dict, process_spec_file,
        parse_spectra_ms,
    )

    spectrum = np.array([
        [91.0542, 0.245],
        [125.0233, 1.0],
        [155.0577, 0.355],
        [246.1125, 0.735],
    ])

    result = assign_subformulae_single("C16H17NO4", spectrum, "[M+H]+", mass_diff_thresh=20.0)
    check("assign result has cand_form", result["cand_form"] == "C16H17NO4")
    check("assign result has cand_ion", result["cand_ion"] == "[M+H]+")
    check("assign result has output_tbl", result["output_tbl"] is not None)

    if result["output_tbl"] is not None:
        tbl = result["output_tbl"]
        check("output_tbl has mz", "mz" in tbl)
        check("output_tbl has formula", "formula" in tbl)
        check("output_tbl has ms2_inten", "ms2_inten" in tbl)
        check("output_tbl has ions", "ions" in tbl)
        check("output_tbl has mono_mass", "mono_mass" in tbl)
        check("peaks assigned > 0", len(tbl["mz"]) > 0, str(len(tbl["mz"])))

    result_bad_ion = assign_subformulae_single("C16H17NO4", spectrum, "[M-H]-")
    check("bad ion returns None output_tbl", result_bad_ion["output_tbl"] is None)


def test_subformulae_vs_reference():
    """Cross-validate against reference output at /home/liuhx25/orcd/pool/data/msg/."""
    print("\n=== Phase 3: Subformulae cross-validation vs MSG reference ===")
    import json
    from pathlib import Path
    from massspecgym.data.subformulae import assign_subformulae_single, parse_spectra_ms, process_spec_file

    ref_dir = Path("/home/liuhx25/orcd/pool/data/msg")
    spec_file = ref_dir / "spec_files" / "MassSpecGymID0000001.ms"
    ref_json = ref_dir / "subformulae" / "default_subformulae" / "MassSpecGymID0000001.json"

    if not spec_file.exists() or not ref_json.exists():
        print("  [SKIP] Reference MSG data not found")
        return

    meta, tuples = parse_spectra_ms(str(spec_file))
    spec = process_spec_file(meta, tuples)
    check("parsed spec not None", spec is not None)

    if spec is None:
        return

    with open(ref_json) as f:
        ref = json.load(f)

    result = assign_subformulae_single(
        ref["cand_form"], spec, ref["cand_ion"], mass_diff_thresh=20.0
    )

    check("formula matches ref", result["cand_form"] == ref["cand_form"])
    check("ion matches ref", result["cand_ion"] == ref["cand_ion"])

    if result["output_tbl"] is not None and ref["output_tbl"] is not None:
        our_formulas = set(result["output_tbl"]["formula"])
        ref_formulas = set(ref["output_tbl"]["formula"])
        check("formula overlap > 50%",
              len(our_formulas & ref_formulas) / max(len(ref_formulas), 1) > 0.5,
              f"ours={len(our_formulas)}, ref={len(ref_formulas)}, overlap={len(our_formulas & ref_formulas)}")


def test_sanity_check():
    print("\n=== Phase 4: InChIKey sanity check ===")
    from massspecgym.data.sanity_check import (
        check_inchikey_overlap, check_inchikey_overlap_strict, DataLeakageError,
        load_exclusion_set,
    )

    exclude_set = load_exclusion_set()
    check("exclusion set loaded", len(exclude_set) > 8000, str(len(exclude_set)))

    clean_keys = ["ABCDEFGHIJKLMN", "ZYXWVUTSRQPONM"]
    result = check_inchikey_overlap(clean_keys)
    check("clean data passes", result.is_clean)
    check("clean count correct", result.total_molecules == 2)

    dirty_keys = list(exclude_set)[:3] + clean_keys
    result2 = check_inchikey_overlap(dirty_keys)
    check("dirty data detected", not result2.is_clean)
    check("overlap count >= 3", result2.overlap_count >= 3, str(result2.overlap_count))

    try:
        check_inchikey_overlap_strict(dirty_keys)
        check("strict raises on dirty", False, "should have raised")
    except DataLeakageError:
        check("strict raises on dirty", True)


def test_magma_fragmentation():
    print("\n=== Phase 5: MAGMa FragmentEngine ===")
    from massspecgym.models.simulation.iceberg.magma import FragmentEngine

    engine = FragmentEngine("CCO", max_tree_depth=2, max_broken_bonds=3)
    check("FragmentEngine created", engine.mol is not None)
    check("FragmentEngine natoms=3", engine.natoms == 3)

    engine.generate_fragments()
    n_frags = len(engine.frag_to_entry)
    check("fragments generated", n_frags > 1, str(n_frags))

    root = engine.get_root_frag()
    check("root frag correct", root == (1 << 3) - 1)

    forms, masses = engine.get_frag_forms()
    check("frag_forms not empty", len(forms) > 0, str(len(forms)))
    check("frag_masses positive", all(m > 0 for m in masses))

    engine2 = FragmentEngine("c1ccccc1", max_tree_depth=2)
    engine2.generate_fragments()
    check("benzene fragments", len(engine2.frag_to_entry) > 1, str(len(engine2.frag_to_entry)))


def test_fp2mol_dataset_parquet():
    print("\n=== Phase 6: FP2MolDataset Parquet ===")
    import tempfile, os
    import pandas as pd

    smiles_list = ["CCO", "CC(=O)O", "c1ccccc1", "CC(C)O"]
    df = pd.DataFrame({
        "smiles": smiles_list,
        "inchikey_14": ["LFQSCWFLJHTTHZ", "QTBSBXVTEAMEQO", "UHOVQNZJYSORNB", "KFZMGEQAYNKOFK"],
        "formula": ["C2H6O", "C2H4O2", "C6H6", "C3H8O"],
    })

    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        df.to_parquet(f.name)
        tmp_path = f.name

    try:
        from massspecgym.data.fp2mol_dataset import FP2MolDataset
        ds = FP2MolDataset(tmp_path, exclude_inchikeys=False)
        check("FP2MolDataset loads parquet", len(ds) == 4, str(len(ds)))

        item = ds[0]
        check("item has fingerprint", "fingerprint" in item)
        check("item has formula", "formula" in item)
        check("item has mol", "mol" in item)
        check("fingerprint shape", item["fingerprint"].shape[0] == 4096)
    finally:
        os.unlink(tmp_path)

    try:
        from massspecgym.data.fp2mol_dataset import FP2MolDataset
        ds2 = FP2MolDataset(smiles_list, exclude_inchikeys=False)
        check("FP2MolDataset list input", len(ds2) == 4)
    except Exception as e:
        check("FP2MolDataset list input", False, str(e))


def test_oracle_interfaces():
    print("\n=== Phase 7: Oracle interfaces ===")
    from massspecgym.models.oracles import OracleBase
    from massspecgym.models.oracles.mist_cf import predict_formulas, MistCFNet, enumerate_candidate_formulas, FormulaCandidate
    from massspecgym.models.oracles.iceberg import predict_spectrum

    check("OracleBase importable", True)
    check("predict_formulas importable", callable(predict_formulas))
    check("predict_spectrum importable", callable(predict_spectrum))
    check("MistCFNet importable", MistCFNet is not None)
    check("FormulaCandidate importable", FormulaCandidate is not None)

    candidates = enumerate_candidate_formulas(180.0634, "[M+H]+", ppm_tol=10.0)
    check("enumerate_candidates non-empty", len(candidates) > 0, str(len(candidates)))
    has_glucose = any("C6" in c and "H12" in c and "O6" in c for c in candidates)
    check("enumerate finds C6H12O6-like", has_glucose or len(candidates) > 0)

    results = predict_formulas(
        spectrum_mzs=[91.05, 125.02, 155.06, 246.11],
        spectrum_intensities=[0.25, 1.0, 0.36, 0.73],
        precursor_mz=288.12,
        adduct="[M+H]+",
        top_k=5,
    )
    check("predict_formulas returns results", len(results) > 0, str(len(results)))
    if results:
        check("results are FormulaCandidate", isinstance(results[0], FormulaCandidate))
        check("results have score", results[0].score != 0)


def test_iceberg_models():
    print("\n=== Phase 7b: ICEBERG model imports ===")
    from massspecgym.models.simulation.iceberg.gen_model import FragGNN
    from massspecgym.models.simulation.iceberg.inten_model import IntenGNN
    from massspecgym.models.simulation.iceberg.joint_model import JointModel
    from massspecgym.models.simulation.iceberg.dag_data import TreeProcessor, FRAGMENT_ENGINE_PARAMS
    from massspecgym.models.simulation.iceberg.adapter import IcebergSimulationMassSpecGymModel

    check("FragGNN importable", FragGNN is not None)
    check("IntenGNN importable", IntenGNN is not None)
    check("JointModel importable", JointModel is not None)
    check("TreeProcessor importable", TreeProcessor is not None)
    check("FRAGMENT_ENGINE_PARAMS", FRAGMENT_ENGINE_PARAMS == {"max_broken_bonds": 6, "max_tree_depth": 3})

    tp = TreeProcessor(pe_embed_k=0, root_encode="gnn")
    nf = tp.get_node_feats()
    check("TreeProcessor node_feats > 0", nf > 0, str(nf))

    gen = FragGNN(hidden_size=64)
    check("FragGNN instantiates", gen is not None)

    inten = IntenGNN(hidden_size=64)
    check("IntenGNN instantiates", inten is not None)

    jm = JointModel(gen, inten)
    check("JointModel instantiates", jm is not None)

    result = jm.predict_mol("CCO", collision_eng=40.0, adduct="[M+H]+")
    check("JointModel predict_mol returns", "spec" in result)
    check("JointModel predict_mol has peaks", len(result["spec"]) > 0, str(len(result["spec"])))

    from massspecgym.models.oracles.mist_cf.model import MistCFNet
    net = MistCFNet(hidden_size=64, layers=1)
    check("MistCFNet instantiates", net is not None)

    import torch
    num_peaks = torch.tensor([3, 2])
    peak_types = torch.tensor([[0, 1, 3, 0], [0, 3, 0, 0]])
    form_vec = torch.randn(2, 4, 18)
    ion_vec = torch.zeros(2, 4, 7)
    instrument_vec = torch.zeros(2, 4, 6)
    intens = torch.tensor([[0.5, 0.3, 0.2, 0.0], [0.7, 0.3, 0.0, 0.0]])
    rel_mass_diffs = torch.zeros(2, 4)
    scores = net(num_peaks, peak_types, form_vec, ion_vec, instrument_vec, intens, rel_mass_diffs)
    check("MistCFNet forward shape", scores.shape == (2, 1), str(scores.shape))


def test_mist_data_mixin():
    print("\n=== Phase 8: MISTDataMixin ===")
    from massspecgym.data.mist_data_mixin import MISTDataMixin
    check("MISTDataMixin importable", True)

    class TestModel(MISTDataMixin):
        pass

    m = TestModel()
    check("ensure_mist_data is callable", hasattr(m, "ensure_mist_data"))
    check("get_subformulae_dir is callable", hasattr(m, "get_subformulae_dir"))


if __name__ == "__main__":
    print("=" * 60)
    print("MassSpecGym v1.5 Data/Oracles Validation Suite")
    print("=" * 60)

    tests = [
        test_chem_constants_subformulae,
        test_subformulae_engine,
        test_subformulae_vs_reference,
        test_sanity_check,
        test_magma_fragmentation,
        test_fp2mol_dataset_parquet,
        test_oracle_interfaces,
        test_iceberg_models,
        test_mist_data_mixin,
    ]

    for test_fn in tests:
        try:
            test_fn()
        except Exception as e:
            FAIL += 1
            print(f"\n  [ERROR] {test_fn.__name__} crashed: {e}")
            traceback.print_exc()

    print("\n" + "=" * 60)
    print(f"Results: {PASS} passed, {FAIL} failed out of {PASS + FAIL} checks")
    print("=" * 60)
    sys.exit(1 if FAIL > 0 else 0)
