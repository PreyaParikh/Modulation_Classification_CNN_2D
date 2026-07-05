# MATLAB ONNX export -> onnxsim -> TFLite (FP32 & INT8) -> VNNX (PolarFire)
#
# Target hardware : Microchip PolarFire FPGA, CoreVectorBlox soft IP
# Model           : 11-class RadioML modulation classifier
#
# Stages: 
#   1. simplify        onnxsim with a FIXED input shape -> folds MATLAB's
#                       dynamic SAME-padding subgraph into static Conv
#                       `pads` attributes (125 nodes -> ~19 nodes).
#   2. to-saved-model   onnx2tf: simplified ONNX -> TensorFlow SavedModel.
#   3. eval-float32     sanity-check accuracy of the un-quantized model.
#   4. quantize-int8    INT8 post-training quantization + accuracy check.
#   5. vnnx-compile     vnnx_compile: INT8 TFLite -> .vnnx for CoreVectorBlox.
#   6. vnnx-accuracy    bit-accurate VNNX simulator (vbx.sim) accuracy run,
#                       the real pre-deployment number.
#
# ENVIRONMENT
# run setup_vars.sh first to activate virtual environment

# USAGE
#   ./modclass_cnn.sh                     # run every stage, start to finish
#   ./modclass_cnn.sh --from vnnx-compile # resume from a specific stage
#   ./modclass_cnn.sh --only convert      # only stages 1-4 (skip VNNX)
#   ./modclass_cnn.sh --only vnnx         # only stages 5-6 (skip conversion)
#   ./modclass_cnn.sh -h                  # help

set -euo pipefail

# Paths 
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS_DIR="${PROJECT_ROOT}/scripts"
BUILD_DIR="${PROJECT_ROOT}/build"

MATLAB_DIR="${PROJECT_ROOT}/../ModClass2DCNN"       # MATLAB training project

INPUT_ONNX="${MATLAB_DIR}/modclass_vbx.onnx"        # raw MATLAB ONNX export
TEST_MAT="${MATLAB_DIR}/testData.mat"               # calibration + test data

# Model Names
MODEL_NAME="modclass_vbx"
INPUT_TENSOR_NAME="InputLayer"          # ONNX graph input name
INPUT_SHAPE_NCHW="1,2,1024"             # NCHW shape, excluding batch dim

# INT8 quantization
N_CALIB=150
CALIB_SEED=0

# VectorBlox SDK / VNNX compile

VBX_SDK="${VBX_SDK:-$HOME/VectorBlox-SDK}"
SIZE_CONF="${SIZE_CONF:-V1000}"         # V250 | V500 | V1000 -- MUST match
                                         # your CoreVectorBlox IP config on
                                         # the PolarFire bitstream.
COMPRESSION="${COMPRESSION:-ncomp}"     # ncomp | comp | ucomp -- ncomp is
                                         # the safe default

# Derived paths
SIM_ONNX="${BUILD_DIR}/${MODEL_NAME}_sim.onnx"
SAVED_MODEL_DIR="${BUILD_DIR}/tf_saved"
INT8_TFLITE="${BUILD_DIR}/${MODEL_NAME}_int8.tflite"
VNNX_OUT="${BUILD_DIR}/${MODEL_NAME}.vnnx"

log()  { printf '\n\033[1;34m\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[WARN]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[ERROR]\033[0m %s\n' "$*" >&2; exit 1; }

require_file() { [[ -f "$1" ]] || die "required file not found: $1"; }
require_cmd()  { command -v "$1" >/dev/null 2>&1 || die "required command not found on PATH: $1"; }


VBX_ENV_ACTIVATED=0
activate_vbx_env() {
    [[ "${VBX_ENV_ACTIVATED}" -eq 1 ]] && return 0
    [[ -d "${VBX_SDK}" ]] || die "VBX_SDK not found at ${VBX_SDK}. Set \$VBX_SDK."
    [[ -f "${VBX_SDK}/vbx_env/bin/activate" ]] || die \
        "No vbx_env found under ${VBX_SDK}. "
    [[ -f "${VBX_SDK}/setup_vars.sh" ]] && source "${VBX_SDK}/setup_vars.sh"

    source "${VBX_SDK}/vbx_env/bin/activate"
    VBX_ENV_ACTIVATED=1
    log "Activated vbx_env: $(command -v python3)"

    if python3 -c "import tf_keras" >/dev/null 2>&1; then
        export TF_USE_LEGACY_KERAS=1
    fi
}

# Stages


stage_simplify() {
    log "Stage 1/6: onnxsim"
    require_file "${INPUT_ONNX}"
    mkdir -p "${BUILD_DIR}"
    activate_vbx_env
    python3 "${SCRIPTS_DIR}/convert_to_vnnx.py" simplify \
        --in-onnx "${INPUT_ONNX}" \
        --out-onnx "${SIM_ONNX}" \
        --input-name "${INPUT_TENSOR_NAME}" \
        --input-shape "${INPUT_SHAPE_NCHW}"
}

stage_to_saved_model() {
    log "Stage 2/6: onnx2tf (simplified ONNX -> TensorFlow SavedModel)"
    require_file "${SIM_ONNX}"
    activate_vbx_env
    python3 "${SCRIPTS_DIR}/convert_to_vnnx.py" to-saved-model \
        --in-onnx "${SIM_ONNX}" \
        --out-dir "${SAVED_MODEL_DIR}"
}

stage_eval_float32() {
    log "Stage 3/6: evaluate float32 TFLite accuracy"
    require_file "${TEST_MAT}"
    activate_vbx_env
    python3 "${SCRIPTS_DIR}/convert_to_vnnx.py" eval-float32 \
        --saved-model "${SAVED_MODEL_DIR}" \
        --mat "${TEST_MAT}"
}

stage_quantize_int8() {
    log "Stage 4/6: INT8 quantization + accuracy check"
    require_file "${TEST_MAT}"
    activate_vbx_env
    python3 "${SCRIPTS_DIR}/convert_to_vnnx.py" quantize-int8 \
        --saved-model "${SAVED_MODEL_DIR}" \
        --mat "${TEST_MAT}" \
        --out "${INT8_TFLITE}" \
        --n-calib "${N_CALIB}" \
        --seed "${CALIB_SEED}"
    python3 "${SCRIPTS_DIR}/convert_to_vnnx.py" eval-int8 \
        --tflite "${INT8_TFLITE}" \
        --mat "${TEST_MAT}"
}

stage_vnnx_compile() {
    log "Stage 5/6: vnnx_compile (INT8 TFLite -> .vnnx)"
    require_file "${INT8_TFLITE}"
    activate_vbx_env
    require_cmd vnnx_compile
    log "Compiling with size=${SIZE_CONF}, compression=${COMPRESSION}"

    vnnx_compile \
        -t "${INT8_TFLITE}" \
        -s "${SIZE_CONF}" \
        -c "${COMPRESSION}" \
        -o "${VNNX_OUT}"
    log "Wrote ${VNNX_OUT}"
}

stage_vnnx_accuracy() {
    log "Stage 6/6: VNNX simulator accuracy"
    require_file "${VNNX_OUT}"
    require_file "${TEST_MAT}"
    activate_vbx_env
    python3 "${SCRIPTS_DIR}/check_vnnx_accuracy.py" \
        --vnnx "${VNNX_OUT}" \
        --mat "${TEST_MAT}"
}


STAGE_NAMES=(simplify to-saved-model eval-float32 quantize-int8 vnnx-compile vnnx-accuracy)
STAGE_FUNCS=(stage_simplify stage_to_saved_model stage_eval_float32 stage_quantize_int8 stage_vnnx_compile stage_vnnx_accuracy)

usage() {
    grep -E '^#( |$)' "${BASH_SOURCE[0]}" | sed -E 's/^# ?//' | sed -n '1,55p'
    exit 0
}

FROM_STAGE=""
ONLY_GROUP=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --from) FROM_STAGE="$2"; shift 2 ;;
        --only) ONLY_GROUP="$2"; shift 2 ;;
        -h|--help) usage ;;
        *) die "Unknown argument: $1 (see --help)" ;;
    esac
done

start_idx=0
end_idx=$((${#STAGE_NAMES[@]} - 1))

if [[ -n "${FROM_STAGE}" ]]; then
    for i in "${!STAGE_NAMES[@]}"; do
        [[ "${STAGE_NAMES[$i]}" == "${FROM_STAGE}" ]] && start_idx=$i
    done
fi

case "${ONLY_GROUP}" in
    convert) start_idx=0; end_idx=3 ;;
    vnnx)    start_idx=4; end_idx=5 ;;
    "" ) : ;;
    *) die "--only must be 'convert' or 'vnnx'" ;;
esac

log "Modulation Classification (stages ${start_idx}..${end_idx})"
log "PROJECT_ROOT = ${PROJECT_ROOT}"
log "BUILD_DIR    = ${BUILD_DIR}"
log "VBX_SDK      = ${VBX_SDK}"

for i in $(seq "${start_idx}" "${end_idx}"); do
    "${STAGE_FUNCS[$i]}"
done


