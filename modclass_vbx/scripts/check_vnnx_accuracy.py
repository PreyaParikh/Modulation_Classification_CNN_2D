
"""
check_vnnx_accuracy.py -- accuracy of the compiled VNNX model using
VectorBlox's bit-accurate Python simulator (vbx.sim.Model). This is the
SAME simulator that models what the actual CoreVectorBlox hardware will
compute, so this is the real pre-deployment accuracy number -- not just
the TFLite INT8 number, which can differ slightly due to op-fusion /
rounding differences introduced during VNNX graph lowering.

MUST be run inside the VectorBlox SDK's Python venv:
    source $VBX_SDK/vbx_env/bin/activate
1
Usage:
    python3 check_vnnx_accuracy.py --vnnx modclass_vbx.vnnx --mat testData.mat
"""

import argparse

import numpy as np
from scipy.io import loadmat
import vbx.sim


def load_data(mat_path: str):
    d = loadmat(mat_path)
    X = d["XTestONNX"]                              # raw MATLAB shape (2, 1024, 1, N)
    y = d["testTargetsNum"].flatten().astype(int)    # 1-indexed labels, 1..11

    # Same transpose used throughout the TFLite/VNNX conversion:
    # -> (N, 1, 2, 1024) NCHW -> (N, 2, 1024, 1) NHWC
    X_nchw = np.transpose(X, (3, 2, 0, 1)).astype(np.float32)
    X_nhwc = np.transpose(X_nchw, (0, 2, 3, 1)).astype(np.float32)
    return X_nhwc, y


def load_vnnx(path: str):
    with open(path, "rb") as mf:
        model = vbx.sim.Model(mf.read())
    print("VNNX input shape:", model.input_shape[0], "dtype:", model.input_dtypes[0])
    print("VNNX input scale/zero:", model.input_scale_factor[0], model.input_zeropoint[0])
    print("VNNX output shape:", model.output_shape[0], "dtype:", model.output_dtypes[0])
    print("VNNX output scale/zero:", model.output_scale_factor[0], model.output_zeropoint[0])
    return model


def sanity_check_builtin_test_vector(model):

    test_in = model.test_input
    expected_out = model.test_output
    actual_out = model.run(test_in)
    all_match = True
    for o, (exp, act) in enumerate(zip(expected_out, actual_out)):
        match = np.array_equal(exp, act)
        all_match &= match
        print(f"Built-in test vector, output {o}: {'MATCH' if match else 'MISMATCH'}")
        if not match:
            print("  expected:", exp[:20], "...")
            print("  actual:  ", act[:20], "...")
    return all_match


def evaluate(model, X_nhwc, y):
    in_scale, in_zp = model.input_scale_factor[0], model.input_zeropoint[0]
    in_dtype = model.input_dtypes[0]
    out_scale, out_zp = model.output_scale_factor[0], model.output_zeropoint[0]

    preds = []
    for i in range(X_nhwc.shape[0]):
        x = X_nhwc[i]
        xq = np.round(x / in_scale + in_zp).astype(in_dtype)

        outputs = model.run([xq.flatten()])
        out = outputs[0]
        out_deq = out_scale * (out.astype(np.float32) - out_zp)
        preds.append(np.argmax(out_deq) + 1)  # +1: MATLAB labels are 1-indexed

    preds = np.array(preds)
    acc = (preds == y).mean()
    return acc, preds


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vnnx", default="modclass_vbx.vnnx", help="path to compiled .vnnx model")
    ap.add_argument("--mat", default="testData.mat", help="path to MATLAB test data .mat file")
    args = ap.parse_args()

    X_nhwc, y = load_data(args.mat)
    model = load_vnnx(args.vnnx)

    print("Built-in test vector sanity check")
    ok = sanity_check_builtin_test_vector(model)
    if not ok:
        print(
            "WARNING: built-in test vector mismatch -- investigate the "
            "vnnx_compile step before trusting the accuracy number below."
        )

    print("\n--- Full dataset accuracy (VNNX simulator) ---")
    acc, preds = evaluate(model, X_nhwc, y)
    print(f"VNNX accuracy: {acc:.4f}  ({int(acc * len(y))}/{len(y)} correct)")


if __name__ == "__main__":
    main()
