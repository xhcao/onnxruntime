import onnxruntime
import os
import numpy as np
import sys

from pathlib import Path

# set to the directory the ONNX Runtime repo is in
# `git checkout https://github.com/microsoft/onnxruntime.git` if needed.
ORT_ROOT_DIR = Path(r'D:\src\github\ort')
sys.path.append(str(ORT_ROOT_DIR / 'tools' / 'python'))
import ort_test_dir_utils as utils  # noqa

# See https://github.com/microsoft/onnxruntime/blob/main/tools/python/PythonTools.md#creating-a-test-directory-for-a-model
# for info on providing specific input or expected output
# Copy your model to the Resources\Raw directory and rename to model.onnx
RAW_RESOURCE_DIR = Path(r'Resources\Raw')
MODEL_PATH = Path(r'D:\src\github\ort\onnxruntime\test\testdata\mnist.onnx')
OUTPUT_PATH = RAW_RESOURCE_DIR
TEST_NAME = 'test_data'
# Generate input and optionally output data for your model.

# when using the default data generation any symbolic dimension values must be provided
# check the model inputs/outputs using Netron and provide a value for any symbolic dimension name
symbolic_vals = {'batch': 1}  # symbolic dim named 'batch' will have data created using value of 1

# we can also explicitly provide input/expected output.
# As the test model expects normalized float values in the range 0..1 we create the input explicitly as the default
# data generation uses a range of -10..10
inputs = {'Input3': np.random.rand(1, 1, 28, 28).astype(np.float32)}

utils.create_test_dir(str(MODEL_PATH),
                      str(OUTPUT_PATH),
                      TEST_NAME,
                      # Explicit input data. Any missing required inputs will have data generated for them.
                      name_input_map=inputs,
                      # Optional map for any symbolic values.
                      symbolic_dim_values_map=symbolic_vals,
                      # Expected output can be provided if you want to validate model output against this.
                      name_output_map=None)

# rename the copied model
copied_model = OUTPUT_PATH / TEST_NAME / MODEL_PATH.name
renamed_model = OUTPUT_PATH / TEST_NAME / "model.onnx"
os.rename(str(copied_model), str(renamed_model))
