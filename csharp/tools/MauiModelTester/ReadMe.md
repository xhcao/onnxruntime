Usage:

Update create_test_data.py to specify
  - path to the model you wish to use
  - symbolic dimension values to use
  - any specific input if the randomly generated input will not be good enough
  - expected output if saving the output from running the model locally is not good enough

This will copy the model to Resources\Raw\test_data\model.onnx and the test data files to Resources\Raw\test_data\test_data_set_0

The MAUI application will read the model and test data from there and should need no other changes to be able to execute the model.

NOTES:

OnnxMl.cs was copied from the Perf tool.
  - TODO: The BeforeBuild target used in \csharp\tools\Microsoft.ML.OnnxRuntime.PerfTool\Microsoft.ML.OnnxRuntime.PerfTool.csproj
    doesn't generate an OnnxMl.cs in the MAUI project. Not sure why. Copied OnnxMl.cs short term.
