using MauiModelTester;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Diagnostics;

namespace MauiModelTester
{
    public enum ExecutionProviders
    {
        CPU,    // CPU execution provider is always available by default
        NNAPI,  // NNAPI is available on Android
        CoreML, // CoreML is available on iOS/macOS
        XNNPACK // XNNPACK is available on ARM/ARM64 platforms and benefits 32-bit float models
    }

    // An inference session runs an ONNX model
    internal class OrtInferenceSession
    {
        public OrtInferenceSession(ExecutionProviders provider = ExecutionProviders.CPU)
        {
            _sessionOptions = new SessionOptions();
            switch (_executionProvider)
            {
                case ExecutionProviders.CPU:
                    break;
                case ExecutionProviders.NNAPI:
                    _sessionOptions.AppendExecutionProvider_Nnapi();
                    break;
                case ExecutionProviders.CoreML:
                    _sessionOptions.AppendExecutionProvider_CoreML();
                    break;
                case ExecutionProviders.XNNPACK:
                    _sessionOptions.AppendExecutionProvider("XNNPACK");
                    break;
            }

            // enable pre/post processing custom operators from onnxruntime-extensions
            _sessionOptions.RegisterOrtExtensions();
        }

        // async task to create the inference session which is an expensive operation.
        public async Task Create()
        {
            // create the InferenceSession. this is an expensive operation so only do this when necessary.
            // the InferenceSession supports multiple calls to Run, including concurrent calls.
            var modelBytes = await Utils.LoadResource("test_data/model.onnx");

            _inferenceSession = new InferenceSession(modelBytes, _sessionOptions);

            // read test data and model metadata
            foreach (var entry in _inferenceSession.InputMetadata)
            {
            }
        }

        public byte[] Run(byte[] jpgOrPngBytes)
        {
            // wrap the image bytes in a tensor
            var tensor = new DenseTensor<byte>(new Memory<byte>(jpgOrPngBytes), new[] { jpgOrPngBytes.Length });

            // create model input. the input name 'image' is defined in the model
            var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("image", tensor) };

            // Run inference
            var stopwatch = new Stopwatch();
            stopwatch.Start();
            using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = _inferenceSession.Run(inputs);
            stopwatch.Stop();

            var output = results.First().AsEnumerable<byte>().ToArray();

            return output;
        }

        private SessionOptions _sessionOptions;
        private InferenceSession _inferenceSession;
        private ExecutionProviders _executionProvider = ExecutionProviders.CPU;
    }
}
