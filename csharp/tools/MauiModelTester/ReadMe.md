Usage:

Update create_test_data.py to specify
  - path to the model you wish to use
  - symbolic dimension values to use if needed
  - any specific input if the randomly generated input will not be good enough
	- you can create specific input with /tools/python/onnx_test_data_utils.py 
  - expected output if saving the output from running the model locally is not good enough

This will copy the model to Resources\Raw\test_data\model.onnx and the test data files to 
Resources\Raw\test_data\test_data_set_0

The MAUI application will read the model and test data from there and should need no other changes to be able to 
execute the model.

NOTES:

OnnxMl.cs was copied from the Perf tool. Update by copying the latest version as needed. We don't codegen it here
as the setup that works in the main C# solution doesn't seem to work for a MAUI project. 
We also want this to be a standalone solution so the nuget config can point to the ORT nightly feed (we want to avoid 
doing that in the main C# solution as it would add confusion about where the ORT C# package is coming from).
