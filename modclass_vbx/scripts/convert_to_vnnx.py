"""
convert_to_vnnx.py -- ONNX -> onnxsim -> TFLite (FP32 & INT8) conversion

This file is normally driven by ../modclass_cnn.sh, one stage at a time,
but every stage is also runnable standalone for debugging:

    python3 convert_to_vnnx.py simplify        --in-onnx model.onnx --out-onnx model_sim.onnx
    python3 convert_to_vnnx.py to-saved-model   --in-onnx model_sim.onnx --out-dir tf_saved
    python3 convert_to_vnnx.py eval-float32     --saved-model tf_saved --mat testData.mat
    python3 convert_to_vnnx.py quantize-int8    --saved-model tf_saved --mat testData.mat --out model_int8.tflite
    python3 convert_to_vnnx.py eval-int8        --tflite model_int8.tflite --mat testData.mat

`simplify` (onnxsim, FIXED input shape):
    MATLAB's ONNX exporter emits SAME-padding as a dynamic runtime
    computation (Shape/Cast/Div/Ceil/Floor/Sub/Mul/Add/Split/Concat/Pad) in
    front of every Conv, instead of a static `pads` attribute, even though
    the input shape is fixed at export time. TFLite cannot lower that
    arithmetic chain, so it falls back to FlexConv2D, which is unsupported
    outside the full TensorFlow runtime and unsupported by vnnx_compile.
    It also breaks INT8 calibration, because the calibrator ends up
    "seeing" shape-arithmetic tensors instead of only real activations.

    Fix: run onnxsim with a FIXED input shape (batch = 1) so the whole
    dynamic-padding subgraph gets constant-folded into a static Conv
    `pads` attribute. This also fuses Conv+BatchNorm and MatMul+Add into
    Gemm, which is what takes the graph from ~125 nodes down to ~19
    standard ops with zero Flex/custom ops.

`to-saved-model` (onnx2tf):
    Converts the simplified ONNX graph to a TensorFlow SavedModel. A
    SavedModel (rather than a direct .tflite export) is required so that
    TFLiteConverter can attach a representative_dataset for INT8
    calibration in the next stage.

`eval-float32` / `quantize-int8` / `eval-int8`:
    INT8 quantization is done via `tf.lite.TFLiteConverter` directly, with
    a representative dataset drawn from real input samples

  MATLAB .mat  "XTestONNX"   raw shape (2, 1024, 1, N)
       -> transpose(3,2,0,1) -> ONNX/NCHW           (N, 1, 2, 1024)
       -> transpose(0,2,3,1) -> TFLite/VNNX NHWC    (N, 2, 1024, 1)

  MATLAB labels are 1-indexed (1..11), so every argmax prediction needs
  "+ 1" before comparing against the ground-truth label array.
"""

from __future__ import annotations 

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.io import loadmat

def load_data(mat_path: str):

    d = loadmat(mat_path)
    X = d["XTestONNX"]                                # (2, 1024, 1, N)
    y = d["testTargetsNum"].flatten().astype(int)  

    X_nchw = np.transpose(X, (3, 2, 0, 1)).astype(np.float32)   # (N,1,2,1024)
    X_nhwc = np.transpose(X_nchw, (0, 2, 3, 1)).astype(np.float32)  # (N,2,1024,1)
    return X_nhwc, y


def die(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


# Stage: simplify 
def cmd_simplify(args):
    import onnx
    from onnxsim import simplify

    if not Path(args.in_onnx).exists():
        die(f"input ONNX not found: {args.in_onnx}")

    m = onnx.load(args.in_onnx)
    shape = [1] + [int(x) for x in args.input_shape.split(",")]
    model_simp, ok = simplify(
        m, overwrite_input_shapes={args.input_name: shape}
    )
    if not ok:
        die("onnxsim simplification check failed")

    Path(args.out_onnx).parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model_simp, args.out_onnx)

    before, after = len(m.graph.node), len(model_simp.graph.node)
    print(f"[simplify] {before} nodes -> {after} nodes")
    remaining_flex_like = {"Shape", "Cast", "Ceil", "Floor", "Split"} & {
        n.op_type for n in model_simp.graph.node
    }
    if remaining_flex_like:
        print(
            "[simplify] WARNING: dynamic-padding-style ops still present "
            f"after simplification: {sorted(remaining_flex_like)}. "
            "Double-check --input-name / --input-shape match the real "
            "ONNX input tensor."
        )
    print(f"[simplify] wrote {args.out_onnx}")

# Stage: to-saved-model (onnx2tf)

def cmd_to_saved_model(args):
    import subprocess

    if not Path(args.in_onnx).exists():
        die(f"input ONNX not found: {args.in_onnx}")

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    r = subprocess.run(
        [
            "onnx2tf",
            "-i", args.in_onnx,
            "-o", args.out_dir,
            "-osd",               
        ],
        capture_output=True, text=True,
    )
    print(r.stdout[-2000:])
    if r.returncode != 0:
        print(r.stderr[-3000:], file=sys.stderr)
        die("onnx2tf failed")
    print(f"[to-saved-model] wrote SavedModel to {args.out_dir}")


# Stage: eval-float32

def _find_float32_tflite(saved_model_dir: str, explicit: str | None) -> str:
    if explicit:
        return explicit
    candidates = sorted(Path(saved_model_dir).glob("*float32.tflite"))
    if not candidates:
        die(
            f"no *float32.tflite found under {saved_model_dir}; "
            "pass tflite explicitly"
        )
    return str(candidates[0])


def _eval_tflite(model_path: str, X_nhwc, y, is_int8: bool):
    import tensorflow as tf

    interp = tf.lite.Interpreter(model_path=model_path)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]

    if is_int8:
        in_scale, in_zp = inp["quantization"]

    preds = []
    for i in range(X_nhwc.shape[0]):
        x = X_nhwc[i : i + 1]
        if is_int8:
            x = np.round(x / in_scale + in_zp).astype(np.int8)
        interp.set_tensor(inp["index"], x)
        interp.invoke()
        o = interp.get_tensor(out["index"])
        preds.append(np.argmax(o[0]) + 1) 
    preds = np.array(preds)
    return (preds == y).mean(), preds


def cmd_eval_float32(args):
    tflite_path = _find_float32_tflite(args.saved_model, args.tflite)
    X_nhwc, y = load_data(args.mat)
    acc, _ = _eval_tflite(tflite_path, X_nhwc, y, is_int8=False)
    print(f"[eval-float32] {tflite_path}")
    print(f"[eval-float32] accuracy: {acc:.4f}")


# Stage: quantize-int8

def cmd_quantize_int8(args):
    import tensorflow as tf

    X_nhwc, _ = load_data(args.mat)
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(X_nhwc.shape[0])
    calib = X_nhwc[idx[: args.n_calib]]

    def representative_dataset():
        for i in range(calib.shape[0]):
            yield [calib[i : i + 1]]

    converter = tf.lite.TFLiteConverter.from_saved_model(args.saved_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_int8 = converter.convert()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as f:
        f.write(tflite_int8)
    print(
        f"[quantize-int8] wrote {args.out} "
        f"({len(tflite_int8) / 1024:.1f} KB, {args.n_calib} calibration samples)"
    )


# Stage: eval-int8

def cmd_eval_int8(args):
    X_nhwc, y = load_data(args.mat)
    acc, _ = _eval_tflite(args.tflite, X_nhwc, y, is_int8=True)
    print(f"[eval-int8] {args.tflite}")
    print(f"[eval-int8] accuracy: {acc:.4f}")

def build_parser():
    p = argparse.ArgumentParser(description=__doc__.split("---")[0])
    sub = p.add_subparsers(dest="stage", required=True)

    s = sub.add_parser("simplify", help="onnxsim with a fixed input shape")
    s.add_argument("--in-onnx", required=True)
    s.add_argument("--out-onnx", required=True)
    s.add_argument("--input-name", default="InputLayer")
    s.add_argument(
        "--input-shape", default="1,2,1024",
        help="NCHW shape excluding batch dim, e.g. '1,2,1024'",
    )
    s.set_defaults(func=cmd_simplify)

    s = sub.add_parser("to-saved-model", help="onnx2tf: ONNX -> TF SavedModel")
    s.add_argument("--in-onnx", required=True)
    s.add_argument("--out-dir", required=True)
    s.set_defaults(func=cmd_to_saved_model)

    s = sub.add_parser("eval-float32", help="evaluate the float32 TFLite model")
    s.add_argument("--saved-model", required=True)
    s.add_argument("--mat", required=True)
    s.add_argument("--tflite", default=None, help="override auto-discovered float32 .tflite")
    s.set_defaults(func=cmd_eval_float32)

    s = sub.add_parser("quantize-int8", help="INT8 post-training quantization")
    s.add_argument("--saved-model", required=True)
    s.add_argument("--mat", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--n-calib", type=int, default=150)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=cmd_quantize_int8)

    s = sub.add_parser("eval-int8", help="evaluate the INT8 TFLite model")
    s.add_argument("--tflite", required=True)
    s.add_argument("--mat", required=True)
    s.set_defaults(func=cmd_eval_int8)

    return p


if __name__ == "__main__":
    parser = build_parser()
    ns = parser.parse_args()
    ns.func(ns)
