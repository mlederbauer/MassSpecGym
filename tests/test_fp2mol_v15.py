"""Comprehensive validation of MassSpecGym v1.5 FP2Mol implementations.

Tests all new modules against reference implementations for correctness.
Run on a GPU node: ssh node3509, conda activate massspecgym, python tests/test_fp2mol_v15.py
"""

import sys
import math
import traceback
import torch
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


def test_chem_constants():
    print("\n=== Phase 1a: Chemistry Constants ===")
    from massspecgym.models.encoders.mist.chem_constants import (
        VALID_ELEMENTS, NORM_VEC, formula_to_dense, vec_to_formula,
        max_instr_idx, ion_to_idx, element_to_ind,
    )
    check("VALID_ELEMENTS count", len(VALID_ELEMENTS) == 18)
    check("VALID_ELEMENTS starts with C,H", VALID_ELEMENTS[0] == "C" and VALID_ELEMENTS[1] == "H")
    check("NORM_VEC shape", len(NORM_VEC) == 18)
    check("NORM_VEC[0] (C max)", NORM_VEC[0] == 81)

    # formula_to_dense
    v = formula_to_dense("C6H12O6")
    check("formula_to_dense C6H12O6: C=6", v[element_to_ind["C"]] == 6)
    check("formula_to_dense C6H12O6: H=12", v[element_to_ind["H"]] == 12)
    check("formula_to_dense C6H12O6: O=6", v[element_to_ind["O"]] == 6)

    # vec_to_formula roundtrip
    f = vec_to_formula(v)
    check("vec_to_formula roundtrip", "C6" in f and "H12" in f and "O6" in f, f)

    # max_instr_idx
    check("max_instr_idx > 0", max_instr_idx > 0, str(max_instr_idx))

    # ion_to_idx
    check("ion_to_idx has [M+H]+", "[M+H]+" in ion_to_idx)


def test_form_embedders():
    print("\n=== Phase 1b: Form Embedders ===")
    from massspecgym.models.encoders.mist.form_embedders import (
        get_embedder, FloatFeaturizer, FourierFeaturizerPosCos, FourierFeaturizer,
        RBFFeaturizer, OneHotFeaturizer, LearnedFeaturizer,
    )
    from massspecgym.models.encoders.mist.chem_constants import NORM_VEC

    # FloatFeaturizer
    fe = get_embedder("float")
    check("get_embedder('float') type", isinstance(fe, FloatFeaturizer))
    check("FloatFeaturizer.num_dim", fe.num_dim == 1)
    check("FloatFeaturizer.full_dim", fe.full_dim == len(NORM_VEC))
    t = torch.tensor([[6.0, 12.0, 0.0, 6.0] + [0.0]*14])
    out = fe(t)
    check("FloatFeaturizer output shape", out.shape == (1, 18), str(out.shape))

    # FourierFeaturizerPosCos
    fc = get_embedder("pos-cos")
    check("get_embedder('pos-cos') type", isinstance(fc, FourierFeaturizerPosCos))
    check("FourierFeaturizerPosCos.num_funcs", fc.num_funcs == 9)
    out = fc(t)
    check("FourierFeaturizerPosCos output shape", out.shape == (1, 18 * 9), str(out.shape))

    # All embedders
    for name in ["fourier", "rbf", "one-hot", "learnt", "float", "fourier-sines", "abs-sines", "pos-cos"]:
        try:
            emb = get_embedder(name)
            out = emb(t)
            check(f"get_embedder('{name}') works", out.shape[0] == 1 and out.shape[1] > 0)
        except Exception as e:
            check(f"get_embedder('{name}') works", False, str(e))


def test_transformer_layer():
    print("\n=== Phase 1c: Transformer Layer ===")
    from massspecgym.models.encoders.mist.transformer_layer import TransformerEncoderLayer, MultiheadAttention

    # Basic TransformerEncoderLayer
    layer = TransformerEncoderLayer(d_model=64, nhead=4, dim_feedforward=128, dropout=0.0)
    src = torch.randn(10, 2, 64)  # (seq, batch, dim)
    out, pw = layer(src)
    check("TransformerEncoderLayer output shape", out.shape == (10, 2, 64), str(out.shape))
    check("TransformerEncoderLayer no pairwise", pw is None)

    # With key_padding_mask
    mask = torch.zeros(2, 10, dtype=torch.bool)
    mask[0, 7:] = True
    out2, _ = layer(src, src_key_padding_mask=mask)
    check("TransformerEncoderLayer with mask shape", out2.shape == (10, 2, 64))

    # Pairwise featurization
    layer_pw = TransformerEncoderLayer(d_model=64, nhead=4, pairwise_featurization=True, dropout=0.0)
    pw_feats = torch.randn(2, 10, 10, 64)
    out3, pw3 = layer_pw(src, pairwise_features=pw_feats)
    check("TransformerEncoderLayer pairwise output shape", out3.shape == (10, 2, 64))

    # Additive attention
    layer_add = TransformerEncoderLayer(d_model=64, nhead=4, additive_attn=True, dropout=0.0)
    out4, _ = layer_add(src)
    check("TransformerEncoderLayer additive attn shape", out4.shape == (10, 2, 64))


def test_modules():
    print("\n=== Phase 1d: MIST Modules ===")
    from massspecgym.models.encoders.mist.modules import FormulaTransformer, FPGrowingModule, MLPBlocks

    # MLPBlocks
    mlp = MLPBlocks(input_size=32, hidden_size=64, dropout=0.0, num_layers=3)
    out = mlp(torch.randn(2, 32))
    check("MLPBlocks output shape", out.shape == (2, 64))

    # FPGrowingModule
    fpg = FPGrowingModule(hidden_input_dim=64, final_target_dim=4096, num_splits=4, reduce_factor=2)
    out = fpg(torch.randn(2, 64))
    check("FPGrowingModule returns list", isinstance(out, list))
    check("FPGrowingModule num outputs", len(out) == 5, str(len(out)))  # num_splits + 1
    check("FPGrowingModule final dim", out[-1].shape == (2, 4096), str(out[-1].shape))

    # FormulaTransformer - basic
    ft = FormulaTransformer(
        hidden_size=64, peak_attn_layers=2, set_pooling="intensity",
        num_heads=4, output_size=2048, form_embedder="float",
    )
    batch = {
        "num_peaks": torch.tensor([3, 2]),
        "types": torch.tensor([[0, 1, 3, 0], [0, 3, 0, 0]]),
        "instruments": torch.tensor([0, 1]),
        "ion_vec": torch.tensor([[0, 0, 0, 0], [0, 0, 0, 0]]),
        "form_vec": torch.randn(2, 4, 18),
        "intens": torch.tensor([[0.5, 0.3, 0.2, 0.0], [0.7, 0.3, 0.0, 0.0]]),
    }
    out, aux = ft(batch, return_aux=True)
    check("FormulaTransformer output shape", out.shape == (2, 64), str(out.shape))
    check("FormulaTransformer has peak_tensor aux", "peak_tensor" in aux)


def test_encoder():
    print("\n=== Phase 1e: SpectraEncoder ===")
    from massspecgym.models.encoders.mist.encoder import SpectraEncoder, SpectraEncoderGrowing

    batch = {
        "num_peaks": torch.tensor([3, 2]),
        "types": torch.tensor([[0, 1, 3, 0], [0, 3, 0, 0]]),
        "instruments": torch.tensor([0, 1]),
        "ion_vec": torch.tensor([[0, 0, 0, 0], [0, 0, 0, 0]]),
        "form_vec": torch.randn(2, 4, 18),
        "intens": torch.tensor([[0.5, 0.3, 0.2, 0.0], [0.7, 0.3, 0.0, 0.0]]),
    }

    enc = SpectraEncoder(
        form_embedder="float", output_size=4096, hidden_size=64,
        peak_attn_layers=2, num_heads=4,
    )
    out, aux = enc(batch)
    check("SpectraEncoder output shape", out.shape == (2, 4096), str(out.shape))
    check("SpectraEncoder output range [0,1]", out.min() >= 0 and out.max() <= 1, f"[{out.min():.4f}, {out.max():.4f}]")
    check("SpectraEncoder has h0", "h0" in aux)
    check("SpectraEncoder has pred_frag_fps", "pred_frag_fps" in aux)

    enc_g = SpectraEncoderGrowing(
        form_embedder="float", output_size=4096, hidden_size=64,
        peak_attn_layers=2, num_heads=4, refine_layers=3,
    )
    out_g, aux_g = enc_g(batch)
    check("SpectraEncoderGrowing output shape", out_g.shape == (2, 4096), str(out_g.shape))
    check("SpectraEncoderGrowing has int_preds", "int_preds" in aux_g)
    check("SpectraEncoderGrowing int_preds count", len(aux_g["int_preds"]) == 3, str(len(aux_g["int_preds"])))


def test_formula_encoder():
    print("\n=== Phase 2: FormulaEncoder ===")
    from massspecgym.models.de_novo.fp2mol.formula_utils import FormulaEncoder

    enc = FormulaEncoder(normalize="none")
    check("FormulaEncoder vocab_size", enc.vocab_size == 30)

    v = enc.encode("C6H12O6")
    check("encode C6H12O6 shape", v.shape == (30,), str(v.shape))
    check("encode C6H12O6 C=6", v[0].item() == 6.0)
    check("encode C6H12O6 H=12", v[1].item() == 12.0)
    check("encode C6H12O6 O=6", v[3].item() == 6.0)
    check("encode C6H12O6 N=0", v[2].item() == 0.0)

    # Batch
    vb = enc.encode_batch(["C6H12O6", "CH4", "C2H5OH"])
    check("encode_batch shape", vb.shape == (3, 30), str(vb.shape))

    # Decode roundtrip
    d = enc.decode(v)
    check("decode roundtrip contains C6", "C6" in d, d)

    # Normalization
    v_sum = enc.encode("C6H12O6", normalize="sum")
    check("sum normalization sums to 1", abs(v_sum.sum().item() - 1.0) < 1e-5, str(v_sum.sum().item()))


def test_mdlm():
    print("\n=== Phase 3a: MDLM ===")
    from massspecgym.models.de_novo.fp2mol.frigid.mdlm import MDLM, LogLinearExpNoiseSchedule

    # Noise schedule
    ns = LogLinearExpNoiseSchedule(alpha_max=1.0, alpha_min=1e-3)
    t0 = torch.tensor([0.0])
    t1 = torch.tensor([1.0])
    check("alpha(0) ~ 1", abs(ns.alpha(t0).item() - 1.0) < 1e-5)
    check("alpha(1) ~ 1e-3", abs(ns.alpha(t1).item() - 1e-3) < 1e-5)

    # MDLM
    mdlm = MDLM(mask_token_id=4, vocab_size=100, sampling_eps=1e-3)

    # sample_time
    t = mdlm.sample_time(8, antithetic=True)
    check("sample_time antithetic shape", t.shape == (8,), str(t.shape))
    check("sample_time in range", t.min() >= 1e-3 and t.max() <= 1.0, f"[{t.min():.4f}, {t.max():.4f}]")

    # forward_process (avoid mask_token_id=4 in original data)
    x0 = torch.randint(5, 100, (4, 20))
    t_fp = torch.tensor([0.0, 0.5, 0.9, 1.0])
    xt = mdlm.forward_process(x0, t_fp)
    check("forward_process shape", xt.shape == x0.shape)
    check("forward_process t=0 no masks", (xt[0] == 4).sum().item() == 0)
    check("forward_process t=1 mostly masks", (xt[3] == 4).sum().item() >= 15, str((xt[3] == 4).sum().item()))

    # loss
    logits = torch.randn(4, 20, 100)
    mask = torch.ones(4, 20)
    t_loss = torch.tensor([0.3, 0.5, 0.7, 0.9])
    xt_loss = mdlm.forward_process(x0, t_loss)
    loss = mdlm.loss(logits, x0, xt_loss, t_loss, mask=mask, global_mean=True)
    check("loss is scalar", loss.dim() == 0)
    check("loss is finite", torch.isfinite(loss), str(loss.item()))

    # step_confidence
    x_masked = torch.full((2, 10), 4, dtype=torch.long)
    x_masked[:, 0] = 1  # BOS
    x_masked[:, -1] = 2  # EOS
    logits_sc = torch.randn(2, 10, 100)
    x_new = mdlm.step_confidence(logits_sc, x_masked, step_idx=0, num_steps=8, temperature=1.0, randomness=0.5)
    check("step_confidence shape", x_new.shape == (2, 10))
    masks_before = (x_masked == 4).sum().item()
    masks_after = (x_new == 4).sum().item()
    check("step_confidence reduces masks", masks_after < masks_before, f"{masks_before} -> {masks_after}")


def test_frigid_components():
    print("\n=== Phase 3b: FRIGID Components ===")
    from massspecgym.models.de_novo.fp2mol.frigid.components import (
        FormulaSequenceEncoder, FingerprintSequenceEncoder,
        CrossAttentionLayer, CrossAttentionFormulaConditioner,
        CrossAttentionFingerprintConditioner, SetSelfAttention,
    )
    from massspecgym.models.de_novo.fp2mol.formula_utils import FormulaEncoder

    # FormulaSequenceEncoder
    fse = FormulaSequenceEncoder(num_atom_types=30, embedding_dim=64)
    fe = FormulaEncoder()
    fv = fe.encode_batch(["C6H12O6", "CH4"])
    emb, mask = fse(fv)
    check("FormulaSequenceEncoder emb shape", emb.shape == (2, 30, 64), str(emb.shape))
    check("FormulaSequenceEncoder mask shape", mask.shape == (2, 30), str(mask.shape))
    check("FormulaSequenceEncoder mask nonzero", mask[0].sum() > 0 and mask[1].sum() > 0)

    # FingerprintSequenceEncoder
    fpse = FingerprintSequenceEncoder(
        num_bits=4096, embedding_dim=64, max_seq_len=64,
        num_self_attention_layers=2, num_attention_heads=4,
    )
    fp = torch.zeros(2, 4096)
    fp[0, [1, 50, 100, 200, 500]] = 1.0
    fp[1, [5, 10]] = 1.0
    emb_fp, mask_fp = fpse(fp)
    check("FingerprintSequenceEncoder emb shape[0]", emb_fp.shape[0] == 2)
    check("FingerprintSequenceEncoder mask[0] active=5", mask_fp[0].sum().item() == 5, str(mask_fp[0].sum().item()))
    check("FingerprintSequenceEncoder mask[1] active=2", mask_fp[1].sum().item() == 2, str(mask_fp[1].sum().item()))

    # CrossAttentionLayer
    cal = CrossAttentionLayer(hidden_size=64, num_attention_heads=4, dropout=0.0)
    hidden = torch.randn(2, 10, 64)
    cond = torch.randn(2, 5, 64)
    cond_mask = torch.ones(2, 5)
    out = cal(hidden, cond, cond_mask)
    check("CrossAttentionLayer output shape", out.shape == (2, 10, 64), str(out.shape))

    # SetSelfAttention
    ssa = SetSelfAttention(hidden_size=64, num_attention_heads=4, dropout=0.0)
    x = torch.randn(2, 8, 64)
    m = torch.ones(2, 8)
    out_ssa = ssa(x, m)
    check("SetSelfAttention output shape", out_ssa.shape == (2, 8, 64))


def test_diffms_diffusion():
    print("\n=== Phase 4a: DiffMS Diffusion Utils ===")
    from massspecgym.models.de_novo.fp2mol.diffms.diffusion_utils import (
        PredefinedNoiseScheduleDiscrete, MarginalUniformTransition,
        DiscreteUniformTransition, PlaceHolder,
        compute_batched_over0_posterior_distribution,
        sample_discrete_features, sample_discrete_feature_noise,
    )

    # Noise schedule
    ns = PredefinedNoiseScheduleDiscrete("cosine", 500)
    beta_0 = ns(t_int=torch.tensor([0]))
    beta_250 = ns(t_int=torch.tensor([250]))
    beta_499 = ns(t_int=torch.tensor([499]))
    check("cosine schedule beta increasing", beta_0.item() < beta_250.item() < beta_499.item(),
          f"{beta_0.item():.6f} < {beta_250.item():.6f} < {beta_499.item():.6f}")
    ab = ns.get_alpha_bar(t_int=torch.tensor([0]))
    check("alpha_bar(0) close to 1", ab.item() > 0.99, str(ab.item()))

    # Transition matrices
    x_marg = torch.tensor([0.5, 0.1, 0.05, 0.05, 0.05, 0.05, 0.1, 0.1])
    e_marg = torch.tensor([0.9, 0.04, 0.03, 0.02, 0.01])
    mt = MarginalUniformTransition(x_marg, e_marg, 0)
    Qt = mt.get_Qt(torch.tensor([0.1]), "cpu")
    check("MarginalTransition Qt.X shape", Qt.X.shape == (1, 8, 8), str(Qt.X.shape))
    check("MarginalTransition Qt.E shape", Qt.E.shape == (1, 5, 5), str(Qt.E.shape))
    check("Qt.X rows sum to 1", (Qt.X.sum(dim=-1) - 1.0).abs().max().item() < 1e-5)

    # sample_discrete_feature_noise
    node_mask = torch.ones(2, 5, dtype=torch.bool)
    limit = PlaceHolder(X=x_marg, E=e_marg, y=torch.zeros(0))
    noise = sample_discrete_feature_noise(limit, node_mask)
    check("sample_noise X shape", noise.X.shape == (2, 5, 8), str(noise.X.shape))
    check("sample_noise E shape", noise.E.shape == (2, 5, 5, 5), str(noise.E.shape))
    check("sample_noise E symmetric", (noise.E - noise.E.transpose(1, 2)).abs().max().item() < 1e-5)


def test_mdlm_loss_weight():
    """Verify MDLM loss weight matches bionemo formula: dsigma/expm1(sigma)."""
    print("\n=== Phase 3c: MDLM Loss Weight Verification ===")
    from massspecgym.models.de_novo.fp2mol.frigid.mdlm import MDLM, LogLinearExpNoiseSchedule

    ns = LogLinearExpNoiseSchedule(alpha_max=1.0, alpha_min=1e-3)

    # Check sigma(t)
    t = torch.tensor([0.0, 0.5, 1.0])
    sigma = ns.sigma(t)
    check("sigma(0) = 0", abs(sigma[0].item()) < 1e-5, str(sigma[0].item()))
    check("sigma(1) = -log(1e-3) ~ 6.9", abs(sigma[2].item() - 6.9078) < 0.01, str(sigma[2].item()))

    # Check d_sigma/dt
    dsig = ns.d_sigma_dt(t)
    expected = -math.log(1e-3)  # log(1000) ~ 6.9078
    check("d_sigma/dt is constant ~6.9", abs(dsig[0].item() - expected) < 0.01, str(dsig[0].item()))

    # Check loss_weight = dsigma / expm1(sigma)
    t_mid = torch.tensor([0.5])
    w = ns.loss_weight(t_mid)
    sig_mid = ns.sigma(t_mid)
    dsig_mid = ns.d_sigma_dt(t_mid)
    expected_w = dsig_mid / torch.expm1(sig_mid)
    check("loss_weight formula", torch.allclose(w, expected_w, atol=1e-6),
          f"got={w.item():.6f}, expected={expected_w.item():.6f}")

    # Verify weight is larger at low t (few masked) and smaller at high t (many masked)
    t_test = torch.linspace(0.01, 0.99, 10)
    weights = ns.loss_weight(t_test)
    check("loss_weight decreasing with t", all(weights[i] >= weights[i+1] for i in range(len(weights)-1)),
          str(weights.tolist()))


def test_diffms_graph_transformer():
    print("\n=== Phase 4b: DiffMS GraphTransformer ===")
    from massspecgym.models.de_novo.fp2mol.diffms.graph_transformer import GraphTransformer, XEyTransformerLayer

    gt = GraphTransformer(
        n_layers=2,
        input_dims={"X": 9, "E": 6, "y": 33},
        hidden_mlp_dims={"X": 32, "E": 16, "y": 32},
        hidden_dims={"dx": 32, "de": 16, "dy": 32, "n_head": 4, "dim_ffX": 32, "dim_ffE": 16},
        output_dims={"X": 8, "E": 5, "y": 0},
    )
    X = torch.randn(2, 5, 9)
    E = torch.randn(2, 5, 5, 6)
    y = torch.randn(2, 33)
    mask = torch.ones(2, 5)
    out = gt(X, E, y, mask)
    check("GraphTransformer X out shape", out.X.shape == (2, 5, 8), str(out.X.shape))
    check("GraphTransformer E out shape", out.E.shape == (2, 5, 5, 5), str(out.E.shape))


def test_diffms_mol_from_graphs():
    print("\n=== Phase 4c: DiffMS mol_from_graphs ===")
    from massspecgym.models.de_novo.fp2mol.diffms.model import mol_from_graphs
    from rdkit import Chem

    # Simple ethanol-like: C-C-O
    nodes = np.array([0, 0, 3])  # C=0, O=3 in ATOM_DECODER
    adj = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])  # single bonds
    mol = mol_from_graphs(nodes, adj)
    check("mol_from_graphs returns mol", mol is not None)
    if mol:
        smi = Chem.MolToSmiles(mol)
        check("mol_from_graphs SMILES valid", smi is not None and len(smi) > 0, smi)

    # Benzene-like: aromatic
    nodes_benz = np.array([0]*6)  # 6 carbons
    adj_benz = np.zeros((6, 6), dtype=int)
    for i in range(6):
        adj_benz[i, (i+1) % 6] = 4  # aromatic
        adj_benz[(i+1) % 6, i] = 4
    mol_benz = mol_from_graphs(nodes_benz, adj_benz)
    check("mol_from_graphs benzene", mol_benz is not None)


def test_molforge_search():
    print("\n=== Phase 5a: MolForge Search ===")
    from massspecgym.models.de_novo.fp2mol.molforge.decoder_search import greedy_search, beam_search, _subsequent_mask

    mask = _subsequent_mask(5, torch.device("cpu"))
    check("subsequent_mask shape", mask.shape == (5, 5), str(mask.shape))
    check("subsequent_mask causal", mask[0, 0].item() == False)
    check("subsequent_mask upper tri", mask[0, 1].item() == True)


def test_formula_utils_vs_reference():
    """Cross-validate our FormulaEncoder against the reference FRIGID implementation."""
    print("\n=== Phase 6: Cross-validation vs Reference ===")
    from massspecgym.models.de_novo.fp2mol.formula_utils import FormulaEncoder
    sys.path.insert(0, "/home/liuhx25/MassSpecGym/external/genms/src")
    try:
        from genmol.utils.formula_encoder import FormulaEncoder as RefFormulaEncoder
        ours = FormulaEncoder(normalize="none")
        ref = RefFormulaEncoder(normalize="none")

        check("vocab_size match", ours.vocab_size == ref.vocab_size, f"{ours.vocab_size} vs {ref.vocab_size}")
        check("ATOM_VOCAB match", ours.ATOM_VOCAB == ref.ATOM_VOCAB)

        test_formulas = ["C6H12O6", "CH4", "C2H5OH", "C9H10N2O2", "C20H30BrClN2O4S"]
        for f in test_formulas:
            v_ours = ours.encode(f)
            v_ref = ref.encode(f)
            match = torch.allclose(v_ours, v_ref)
            check(f"FormulaEncoder '{f}' matches ref", match,
                  f"diff={torch.abs(v_ours - v_ref).max().item()}" if not match else "")

        # Batch
        vb_ours = ours.encode_batch(test_formulas)
        vb_ref = ref.encode_batch(test_formulas)
        check("encode_batch matches ref", torch.allclose(vb_ours, vb_ref))
    except ImportError as e:
        print(f"  [SKIP] Reference FRIGID not importable: {e}")
    finally:
        if "/home/liuhx25/MassSpecGym/external/genms/src" in sys.path:
            sys.path.remove("/home/liuhx25/MassSpecGym/external/genms/src")


def test_mist_encoder_vs_reference():
    """Cross-validate our MIST modules against the reference FRIGID implementation."""
    print("\n=== Phase 7: MIST Encoder Cross-validation ===")
    from massspecgym.models.encoders.mist.modules import FormulaTransformer as OurFT, FPGrowingModule as OurFPG
    from massspecgym.models.encoders.mist.form_embedders import get_embedder as our_get_embedder
    from massspecgym.models.encoders.mist.chem_constants import NORM_VEC as OUR_NORM_VEC

    sys.path.insert(0, "/home/liuhx25/MassSpecGym/external/genms/src")
    try:
        from mist.models.modules import FormulaTransformer as RefFT, FPGrowingModule as RefFPG
        from mist.models.form_embedders import get_embedder as ref_get_embedder
        from mist.utils.chem_utils import NORM_VEC as REF_NORM_VEC

        # NORM_VEC
        check("NORM_VEC matches ref", np.array_equal(OUR_NORM_VEC, REF_NORM_VEC))

        # Float embedder output
        our_fe = our_get_embedder("float")
        ref_fe = ref_get_embedder("float")
        test_in = torch.tensor([[6.0, 12.0, 0.0, 6.0] + [0.0]*14])
        our_out = our_fe(test_in)
        ref_out = ref_fe(test_in)
        check("FloatFeaturizer output matches ref", torch.allclose(our_out, ref_out, atol=1e-6),
              f"max_diff={torch.abs(our_out - ref_out).max().item()}")

        # pos-cos embedder
        our_pc = our_get_embedder("pos-cos")
        ref_pc = ref_get_embedder("pos-cos")
        our_out_pc = our_pc(test_in)
        ref_out_pc = ref_pc(test_in)
        check("PosCos embedder output matches ref", torch.allclose(our_out_pc, ref_out_pc, atol=1e-6))

        # FormulaTransformer constructor compatibility (same kwargs)
        our_ft = OurFT(hidden_size=64, peak_attn_layers=2, num_heads=4, form_embedder="float")
        ref_ft = RefFT(hidden_size=64, peak_attn_layers=2, num_heads=4, form_embedder="float")
        our_params = sum(p.numel() for p in our_ft.parameters())
        ref_params = sum(p.numel() for p in ref_ft.parameters())
        check("FormulaTransformer param count matches", our_params == ref_params,
              f"ours={our_params}, ref={ref_params}")

        # FPGrowingModule constructor compatibility
        our_fpg = OurFPG(hidden_input_dim=64, final_target_dim=4096, num_splits=4)
        ref_fpg = RefFPG(hidden_input_dim=64, final_target_dim=4096, num_splits=4)
        our_fp_params = sum(p.numel() for p in our_fpg.parameters())
        ref_fp_params = sum(p.numel() for p in ref_fpg.parameters())
        check("FPGrowingModule param count matches", our_fp_params == ref_fp_params,
              f"ours={our_fp_params}, ref={ref_fp_params}")

    except ImportError as e:
        print(f"  [SKIP] Reference MIST not importable: {e}")
    finally:
        if "/home/liuhx25/MassSpecGym/external/genms/src" in sys.path:
            sys.path.remove("/home/liuhx25/MassSpecGym/external/genms/src")


if __name__ == "__main__":
    print("=" * 60)
    print("MassSpecGym v1.5 FP2Mol Validation Suite")
    print("=" * 60)

    tests = [
        test_chem_constants,
        test_form_embedders,
        test_transformer_layer,
        test_modules,
        test_encoder,
        test_formula_encoder,
        test_mdlm,
        test_mdlm_loss_weight,
        test_frigid_components,
        test_diffms_diffusion,
        test_diffms_graph_transformer,
        test_diffms_mol_from_graphs,
        test_molforge_search,
        test_formula_utils_vs_reference,
        test_mist_encoder_vs_reference,
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
