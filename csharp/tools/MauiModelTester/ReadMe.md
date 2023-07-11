Usage:

Update create_test_data.py to specify
  - path to the model you wish to use
  - symbolic dimension values to use if needed
  - any specific input if the randomly generated input will not be good enough
	- you can create specific input with /tools/python/onnx_test_data_utils.py
      - see the comments in create_test_data.py for more details
  - expected output if saving the output from running the model locally is not good enough

This will copy the model to Resources\Raw\test_data\model.onnx and the test data files to
Resources\Raw\test_data\test_data_set_0

The MAUI application will read the model and test data from there and should need no other changes to be able to
execute the model.

NOTE: The project uses builds from the nightly feed to keep things simple.

If it was part of the main ONNX Runtime C# solutions we'd have to
  - add the ORT nightly feed to the top level nuget.config
    - this potentially adds confusion about nuget packages come from in unit tests
  - keep updating the referenced night packages so they remain valid so the complete solution builds in the CI

If you have new code to test the easiest way is to run the nuget packaging pipeline against that branch. Download the
native and managed nuget packages from the CI artifacts and update the nuget.config to point to the directory they are
in. This can be used to test both native and C# code changes.

If you wish to test the latest C# code, you can create a local package and add the directory it's in to nuget.config.
With the current setup you'd first need to build ORT with the `--build_csharp` param as it creates the native and
managed packages at the same time.

TODO: Can we just use a project reference? No - the ORT C# project requires the values from Directory.build.props in
the root directory. Would need to add the csproj to the ORT sln, and replace the nuget reference for
Microsoft.ML.OnnxRuntime.Managed with a project reference.

---

The following commands _should_ install the necessary workloads to create the managed package including mobile targets,
build the managed library and create the native (local build only) and managed packages. The packages will be in the
ORT build output directory (e.g. build/Windows/Debug/Debug). The native package will only contain the runtime for the
current platform (e.g. Windows 64-bit if you're building on Windows) so can't be used for testing other platforms. Use
the native package from the nightly feed or a packaging CI.

```
dotnet workload install ios android macos
dotnet workload restore .\src\Microsoft.ML.OnnxRuntime\Microsoft.ML.OnnxRuntime.csproj -p:SelectedTargets=All
msbuild -t:restore .\src\Microsoft.ML.OnnxRuntime\Microsoft.ML.OnnxRuntime.csproj -p:SelectedTargets=All
msbuild -t:build .\src\Microsoft.ML.OnnxRuntime\Microsoft.ML.OnnxRuntime.csproj -p:SelectedTargets=All
msbuild .\OnnxRuntime.CSharp.proj -t:CreatePackage -p:OrtPackageId=Microsoft.ML.OnnxRuntime -p:Configuration=Debug -p:Platform="Any CPU"
```
