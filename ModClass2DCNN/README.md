# Modulation Classification with a 2-D CNN

Train a 2-D convolutional neural network in MATLAB to classify 11 digital
and analog modulation types from raw I/Q samples, then export the trained
network to ONNX for deployment with VectorBlox.

Adapted from MathWorks' [Modulation Classification with Deep
Learning](https://www.mathworks.com/help/deeplearning/ug/modulation-classification-with-deep-learning.html)
example, using a 2-D CNN input layout ([2 x SPF x 1], I/Q as two rows)
instead of a 1-D I/Q vector so the trained model can be exported to ONNX
and run outside MATLAB.

## Modulation types

`BPSK`, `QPSK`, `8PSK`, `16QAM`, `64QAM`, `PAM4`, `GFSK`, `CPFSK`,
`B-FM`, `DSB-AM`, `SSB-AM`

## Requirements

- MATLAB R2023b or later
- Deep Learning Toolbox
- Communications Toolbox
- Signal Processing Toolbox
- Parallel Computing Toolbox (optional, used automatically if available)

## Getting started

1. Clone this repository and open it in MATLAB (or add it to your MATLAB path).
2. Run `ModClass2DCNN.mlx`.

By default (`trainNow = true`) the script generates a fresh synthetic
dataset, trains the network from scratch, evaluates it on a held-out test
set, and exports the result to ONNX. Set `trainNow = false` at the top of
the script to instead load a previously trained network from
`trainedModClassVBX.mat` and skip straight to evaluation/export.

## How it works

1. Signal generation – For each modulation type, a source generates
   random symbols/audio, which is modulated and pulse-shaped
   (`helperModClassGetSource.m`, `helperModClassGetModulator.m`).
2. Channel impairment – Each signal passes through a channel with
   Rician multipath fading, clock offset, and AWGN
   (`helperModClassTestChannel.m`).
3. Framing – The received signal is segmented into fixed-length,
   unit-energy frames (`helperModClassFrameGenerator.m`).
4. Network – A 5-layer 2-D CNN classifies each frame
   (`helperModClassCNN_VBX.m`).
5. Export & verification – The trained network is exported to ONNX
   and re-imported into MATLAB to confirm the export preserves accuracy.

## File overview

| File | Description |
|---|---|
| `ModClass2DCNN.mlx` | Main script: generates data, trains the network, evaluates it, and exports it to ONNX. |
| `helperModClassCNN_VBX.m` | Defines the 2-D CNN architecture. |
| `helperModClassGetSource.m` | Produces the random symbol/audio source for each modulation type. |
| `helperModClassGetModulator.m` | Modulates and pulse-shapes symbols for each modulation type. |
| `helperModClassTestChannel.m` | `matlab.System` object modeling multipath fading, clock offset, and noise. |
| `helperModClassFrameGenerator.m` | Segments a signal into normalized, fixed-length frames. |
| `getPoolSafe.m` | Starts a parallel pool if Parallel Computing Toolbox is available, otherwise returns `[]`. |
| `audio_mix_441.wav` | Audio source used to generate `B-FM`, `DSB-AM`, and `SSB-AM` frames. |

## Notes

- This repository only includes the files that `ModClass2DCNN.mlx`
  actually uses. An earlier iteration of this project also generated data
  via MATLAB datastores. 
