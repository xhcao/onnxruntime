using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.ML.OnnxRuntime;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Runtime.CompilerServices;
using System.Reflection;

namespace MauiModelTester
{
    internal class Utils
    {
        internal static async Task<byte[]> LoadResource(string name)
        {
            using Stream fileStream = await FileSystem.Current.OpenAppPackageFileAsync(name);
            using MemoryStream memoryStream = new MemoryStream();
            fileStream.CopyTo(memoryStream);
            return memoryStream.ToArray();
        }

        static async Task<Dictionary<string, OrtValue>> LoadTestData(bool load_input_data)
        {
            var data = new Dictionary<string, OrtValue>();

            int idx = 0;
            var prefix = load_input_data ? "input_" : "output_";

            do
            {
                var filename = "test_data/test_data_set_0/" + prefix + idx + ".pb";
                var exists = await FileSystem.Current.AppPackageFileExistsAsync(filename);

                if (!exists)
                {
                    // we expect sequentially named files for all inputs so as soon as one is missing we're done
                    break;
                }

                var tensorProtoData = await LoadResource(filename);

                // get name and tensor data and create OrtValue
                Onnx.TensorProto tensorProto = null;
                tensorProto = Onnx.TensorProto.Parser.ParseFrom(tensorProtoData);
                var ortValue = CreateOrtValueFromTensorProto(tensorProto);

                data[tensorProto.Name] = ortValue;

                idx++;
            }
            while (true);

            return data;
        }

        static OrtValue CreateOrtValueFromTensorProto(Onnx.TensorProto tensorProto)
        {
            Type tensorElementType = GetElementType((TensorElementType)tensorProto.DataType);

            // use reflection to call generic method
            var func = typeof(Utils)
                           .GetMethod(nameof(TensorProtoToOrtValue), BindingFlags.Static | BindingFlags.NonPublic)
                           .MakeGenericMethod(tensorElementType);

            return (OrtValue)func.Invoke(null, new[] { tensorProto });
        }

        static Type GetElementType(TensorElementType elemType)
        {
            switch (elemType)
            {
                case TensorElementType.Float:
                    return typeof(float);
                case TensorElementType.Double:
                    return typeof(double);
                case TensorElementType.Int16:
                    return typeof(short);
                case TensorElementType.UInt16:
                    return typeof(ushort);
                case TensorElementType.Int32:
                    return typeof(int);
                case TensorElementType.UInt32:
                    return typeof(uint);
                case TensorElementType.Int64:
                    return typeof(long);
                case TensorElementType.UInt64:
                    return typeof(ulong);
                case TensorElementType.UInt8:
                    return typeof(byte);
                case TensorElementType.Int8:
                    return typeof(sbyte);
                case TensorElementType.String:
                    return typeof(byte);
                case TensorElementType.Bool:
                    return typeof(bool);
                default:
                    throw new ArgumentException("Unexpected element type of " + elemType);
            }
        }

        static OrtValue TensorProtoToOrtValue<T>(Onnx.TensorProto tensorProto)
        {
            unsafe
            {
                var elementSize = sizeof(T);
                T[] data = new T[tensorProto.RawData.Length / elementSize];

                fixed(byte *bytes = tensorProto.RawData.Span) fixed(void *target = data)
                {
                    Buffer.MemoryCopy(bytes, target, tensorProto.RawData.Length, tensorProto.RawData.Length);
                }

                return OrtValue.CreateTensorValueFromMemory(data, tensorProto.Dims.ToArray());
            }
        }
    }
}
