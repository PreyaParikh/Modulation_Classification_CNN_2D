# modclass_vbx — MATLAB → ONNX → TFLite INT8 → VNNX Deployment 

End-to-end code that takes a MATLAB-trained 11-class RadioML modulation
classifier (`modclass_vbx`) and deploys it to a Microchip PolarFire
FPGA via the CoreVectorBlox IP, using Microchip's VectorBlox SDK.

```
MATLAB (.mat model)
   │  
   ▼
modclass_vbx.onnx          raw ONNX export
   │  onnxsim (fixed input shape)
   ▼
modclass_vbx11_sim.onnx      static graph, 125 → 19 nodes, no Flex ops
   │  onnx2tf
   ▼
tf_saved/                    TensorFlow SavedModel
   │  TFLiteConverter (float32 export + sanity check)
   ▼
*_float32.tflite             accuracy check
   │  TFLiteConverter INT8 (representative dataset, 150 samples)
   ▼
modclass_vbx_int8.tflite   quantized model
   │  vnnx_compile (VectorBlox SDK)
   ▼
modclass_vbx.vnnx          deployable CoreVectorBlox model
   │  vbx.sim (bit-accurate simulator)
   ▼
final pre-deployment accuracy number
```

## Quick start

```bash
git clone <this repo>
cd Modulation_Classification_2D
export VBX_SDK=~/VectorBlox-SDK      # path to your built VectorBlox SDK
./modclass_cnn.sh
```

That single command runs all six stages. See `./modclass_cnn.sh --help`
for resuming from a specific stage (`--from`) or running only the
ONNX/TFLite half (`--only convert`) or only the VNNX half (`--only vnnx`).

## Repository layout

```
modclass_cnn.sh 
scripts/
  convert_to_vnnx.py             
  check_vnnx_accuracy.py         
build/                           all generated outputs (created on first run)
```

Inputs (`modclass_vbx.onnx`, `testData.mat`) are read directly from the
 `../ModClass2DCNN/` MATLAB project's output — run
`ModClass2DCNN.m` there first so both files exist before running this
pipeline. See the top-level repository README for the full two-stage
layout.

`modulation_classification_2D.sh` runs entirely inside Python environment: the
VectorBlox SDK's own `vbx_env` (`$VBX_SDK/setup_vars.sh`), since that environment already ships
`onnx`, `onnxsim`, `onnx2tf`, `tensorflow`, and the VectorBlox-specific
tools (`vnnx_compile`, `vbx.sim`) needed by every stage. This code does
not build `vbx_env`; follow the SDK's own setup instructions
first. If `onnx2tf` fails with `ModuleNotFoundError: No module named
'tf_keras'` on your SDK version, `pip install tf_keras` into `vbx_env` --
the script auto-detects and uses it if present (`TF_USE_LEGACY_KERAS=1`).


