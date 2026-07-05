# Modulation Classification — Training to FPGA Deployment

End-to-end pipeline for an 11-class RadioML modulation classifier: train a
2-D CNN in MATLAB, export it to ONNX, then convert and compile it to a
`.vnnx` model deployable on a Microchip PolarFire FPGA via the
CoreVectorBlox IP.

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
*_float32.tflite             baseline accuracy check
   │  TFLiteConverter INT8 (representative dataset, 150 samples)
   ▼
modclass_vbx11_int8.tflite   quantized model
   │  vnnx_compile (VectorBlox SDK)
   ▼
modclass_vbx11.vnnx          deployable CoreVectorBlox model
   │  vbx.sim (bit-accurate simulator)
   ▼
final pre-deployment accuracy number
```

## Repository layout

| Folder | Description |
|---|---|
| [`ModClass2DCNN/`](ModClass2DCNN/) | MATLAB project: generates synthetic training data, trains the 2-D CNN, evaluates it, and exports `modclass_vbx.onnx` + `testData.mat`. |
| [`modclass_vbx/`](modclass_vbx/) | Deployment pipeline: converts the ONNX export to a quantized `.vnnx` model for CoreVectorBlox, and verifies accuracy at every stage. |

Each folder has its own README with full details.

## Quick start

1. Train and export (MATLAB)
   ```
   cd ModClass2DCNN
   # In MATLAB: run ModClass2DCNN.mlx
   ```
   This produces `ModClass2DCNN/modclass_vbx.onnx` and
   `ModClass2DCNN/testData.mat`.

2. Convert and compile for hardware (VectorBlox SDK)
   ```bash
   cd modclass_vbx
   export VBX_SDK=~/VectorBlox-SDK      # path to your built VectorBlox SDK
   ./modulation_classification_2D.sh
   ```
   This pipeline reads `modclass_vbx.onnx` and `testData.mat` directly from
   `../ModClass2DCNN/` (no manual copying needed) and produces
   `modclass_vbx/build/modclass_vbx.vnnx`, ready for CoreVectorBlox.

## Requirements

- MATLAB R2023b+ with Deep Learning, Communications, and Signal Processing
  Toolboxes (see `ModClass2DCNN/README.md`)
- A built [VectorBlox SDK](https://github.com/Microchip-Vectorblox/VectorBlox-SDK)
  environment (see `modclass_vbx/README.md`)
