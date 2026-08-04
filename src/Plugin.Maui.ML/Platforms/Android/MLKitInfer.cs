#if ANDROID
using Java.Nio;
using Java.Nio.Channels;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Reflection;
using Xamarin.TensorFlow.Lite;
using Application = Android.App.Application;
using Object = Java.Lang.Object;

namespace Plugin.Maui.ML.Platforms.Android;

public sealed class MLKitInfer : IMLInfer, IDisposable
{
    private readonly Lock _sync = new();
    private bool _disposed;
    private Interpreter? _interpreter;
    private byte[]? _modelBytes;
#pragma warning disable CS0414 // Field is assigned intentionally to keep the mapped model buffer alive for the interpreter lifetime.
    private ByteBuffer? _modelBuffer;
#pragma warning restore CS0414

    public void Dispose()
    {
        if (_disposed) return;
        UnloadModel();
        _disposed = true;
        GC.SuppressFinalize(this);
    }

    public MLBackend Backend => MLBackend.MLKit;
    public bool IsModelLoaded => _interpreter != null;

    public Task LoadModelAsync(string modelPath, CancellationToken cancellationToken = default)
    {
        if (string.IsNullOrWhiteSpace(modelPath))
            throw new ArgumentException("Model path cannot be null or empty", nameof(modelPath));

        return Task.Run(() =>
        {
            cancellationToken.ThrowIfCancellationRequested();

            try
            {
                if (!System.IO.File.Exists(modelPath))
                    throw new FileNotFoundException($"Model file not found: {modelPath}");

                using var inputStream = new Java.IO.FileInputStream(modelPath);

                var channel = inputStream.Channel
                              ?? throw new InvalidOperationException("Could not get FileChannel for model file.");

                var length = new FileInfo(modelPath).Length;

                var mappedBuffer = channel.Map(
                    Java.Nio.Channels.FileChannel.MapMode.ReadOnly!,
                    0,
                    length);

                Initialize(mappedBuffer);
            }
            catch (Exception ex) when (ex is not OperationCanceledException)
            {
                throw new InvalidOperationException(
                    $"Failed to memory-map TFLite model file '{modelPath}': {ex.Message}", ex);
            }
        }, cancellationToken);
    }

    public async Task LoadModelAsync(Stream modelStream, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(modelStream);
        await using var ms = new MemoryStream();
        await modelStream.CopyToAsync(ms, cancellationToken).ConfigureAwait(false);
        Initialize(ms.ToArray());
    }

    public Task LoadModelFromAssetAsync(string assetName, CancellationToken cancellationToken = default)
    {
        if (string.IsNullOrWhiteSpace(assetName))
            throw new ArgumentException("Asset name cannot be null or empty", nameof(assetName));

        return Task.Run(() =>
        {
            cancellationToken.ThrowIfCancellationRequested();

            try
            {
                var assets = Application.Context.Assets
                             ?? throw new InvalidOperationException("Android AssetManager is unavailable.");

                try
                {
                    using var afd = assets.OpenFd(assetName)
                                    ?? throw new FileNotFoundException(
                                        $"Asset '{assetName}' not found in Android assets.");

                    using var inputStream = new Java.IO.FileInputStream(afd.FileDescriptor);

                    var channel = inputStream.Channel
                                  ?? throw new InvalidOperationException("Could not get FileChannel for asset.");

                    var mappedBuffer = channel.Map(
                        FileChannel.MapMode.ReadOnly!,
                        afd.StartOffset,
                        afd.DeclaredLength);

                    Initialize(mappedBuffer);
                }
                catch (Exception ex) when (ex is not OperationCanceledException)
                {
                    cancellationToken.ThrowIfCancellationRequested();

                    using var assetStream = assets.Open(assetName)
                        ?? throw new FileNotFoundException($"Asset '{assetName}' not found in Android assets.", ex);
                    using var ms = new System.IO.MemoryStream();
                    assetStream.CopyTo(ms);
                    Initialize(ms.ToArray());
                }
            }
            catch (Exception ex) when (ex is not OperationCanceledException)
            {
                throw new InvalidOperationException(
                    $"Failed to load TFLite model asset '{assetName}': {ex.Message}", ex);
            }
        }, cancellationToken);
    }

    public Task<Dictionary<string, Tensor<float>>> RunInferenceAsync(Dictionary<string, Tensor<float>> inputs,
        CancellationToken cancellationToken = default)
    {
        if (!IsModelLoaded) throw new InvalidOperationException("No TFLite model loaded. Call LoadModelAsync first.");
        if (inputs == null || inputs.Count == 0)
            throw new ArgumentException("Inputs cannot be null or empty", nameof(inputs));

        return Task.Run(() =>
        {
            lock (_sync)
            {
                if (_interpreter == null) throw new InvalidOperationException("Interpreter disposed.");

                var first = inputs.First();
                var inputTensor = first.Value;
                var inputIndex = 0; // assume single input

                // Prepare input buffer
                var flat = inputTensor.ToArray();
                var inputShape = inputTensor.Dimensions.ToArray();
                try
                {
                    _interpreter.ResizeInput(inputIndex, inputShape);
                }
                catch
                {
                    /* ignore if immutable */
                }

                _interpreter.AllocateTensors();

                // Build multidimensional Java array (fallback approach) or supply direct buffer
                // Simpler: use float[] and rely on internal mapping if shape is 1-D. For N-D we flatten.
                var inputData = Object.FromArray(flat);

                // Prepare output container (float[])
                // Query output shape via tensor
                var outputTensor = _interpreter.GetOutputTensor(0);
                if (outputTensor != null)
                {
                    var oshape = outputTensor.Shape();
                    var outCount = 1;
                    if (oshape != null)
                    {
                        foreach (var d in oshape) outCount *= d;
                        var outputArray = new float[outCount];

                        // Run
                        _interpreter.Run(inputData, outputArray);

                        // Wrap into DenseTensor
                        var dense = new DenseTensor<float>(oshape.Select(i => i).ToArray());
                        var span = dense.Buffer.Span;
                        for (var i = 0; i < outputArray.Length; i++) span[i] = outputArray[i];

                        return new Dictionary<string, Tensor<float>> { ["output0"] = dense };
                    }
                }
            }

            throw new InvalidOperationException("Failed to run inference or retrieve output.");
        }, cancellationToken);
    }

    public Task<Dictionary<string, Tensor<float>>> RunInferenceLongInputsAsync(Dictionary<string, Tensor<long>> inputs,
    CancellationToken cancellationToken = default)
    {
        if (!IsModelLoaded) throw new InvalidOperationException("No TFLite model loaded. Call LoadModelAsync first.");
        if (inputs == null || inputs.Count == 0)
            throw new ArgumentException("Inputs cannot be null or empty", nameof(inputs));

        return Task.Run(() =>
        {
            lock (_sync)
            {
                if (_interpreter == null) throw new InvalidOperationException("Interpreter disposed.");

                var first = inputs.First();
                var inputTensor = first.Value;
                var inputIndex = 0; // assume single input

                var flat = inputTensor.ToArray();
                var inputShape = inputTensor.Dimensions.ToArray();

                try
                {
                    _interpreter.ResizeInput(inputIndex, inputShape);
                }
                catch
                {
                    /* ignore if immutable */
                }

                _interpreter.AllocateTensors();

                var nativeOrder = ByteOrder.NativeOrder() ?? ByteOrder.BigEndian;

                // INT64 input buffer
                var inputData = ByteBuffer.AllocateDirect(flat.Length * 8).Order(nativeOrder!);
                inputData.AsLongBuffer()!.Put(flat);
                inputData.Rewind();

                var outputTensor = _interpreter.GetOutputTensor(0);
                if (outputTensor == null)
                    throw new InvalidOperationException("Failed to retrieve output tensor.");

                var oshape = outputTensor.Shape();
                if (oshape == null)
                    throw new InvalidOperationException("Failed to retrieve output shape.");

                var outCount = 1;
                foreach (var d in oshape) outCount *= d;

                // FLOAT32 output buffer
                var outputData = ByteBuffer.AllocateDirect(outCount * 4).Order(nativeOrder!);

                _interpreter.Run(inputData, outputData);

                outputData.Rewind();

                var outputArray = new float[outCount];
                outputData.AsFloatBuffer()!.Get(outputArray);

                var dense = new DenseTensor<float>(oshape.Select(i => i).ToArray());
                var span = dense.Buffer.Span;
                for (var i = 0; i < outputArray.Length; i++) span[i] = outputArray[i];

                return new Dictionary<string, Tensor<float>> { ["output0"] = dense };
            }
        }, cancellationToken);
    }

    public Dictionary<string, MLNodeMetadata> GetInputMetadata()
    {
        if (!IsModelLoaded) throw new InvalidOperationException("No TFLite model loaded. Call LoadModelAsync first.");

        lock (_sync)
        {
            return _interpreter == null
                ? throw new InvalidOperationException("Interpreter disposed.")
                : GetTensorMetadata(_interpreter, isInput: true);
        }
    }

    public Dictionary<string, MLNodeMetadata> GetOutputMetadata()
    {
        if (!IsModelLoaded) throw new InvalidOperationException("No TFLite model loaded. Call LoadModelAsync first.");

        lock (_sync)
        {
            return _interpreter == null
                ? throw new InvalidOperationException("Interpreter disposed.")
                : GetTensorMetadata(_interpreter, isInput: false);
        }
    }

    public void UnloadModel()
    {
        lock (_sync)
        {
            _interpreter?.Close();
            _interpreter?.Dispose();
            _interpreter = null;
            _modelBytes = null;
            _modelBuffer = null;
        }
    }

    private void Initialize(byte[] bytes)
    {
        lock (_sync)
        {
            UnloadModel();
            _modelBytes = bytes;
            var nativeOrder = ByteOrder.NativeOrder() ?? ByteOrder.BigEndian;
            var byteBuffer = ByteBuffer.AllocateDirect(bytes.Length).Order(nativeOrder!);
            byteBuffer.Put(bytes);
            byteBuffer.Rewind();
            _interpreter = new Interpreter(byteBuffer,
                (Interpreter.Options?)new Interpreter.Options().SetNumThreads(Math.Max(1,
                    Environment.ProcessorCount - 1)));
            _interpreter.AllocateTensors();
        }
    }
    private void Initialize(ByteBuffer modelBuffer)
    {
        lock (_sync)
        {
            UnloadModel();

            _modelBytes = null;
            _modelBuffer = modelBuffer;

            _interpreter = new Interpreter(
                modelBuffer,
                (Interpreter.Options?)new Interpreter.Options().SetNumThreads(
                    Math.Max(1, Environment.ProcessorCount - 1)));

            _interpreter.AllocateTensors();
        }
    }

    private static Dictionary<string, MLNodeMetadata> GetTensorMetadata(Interpreter interpreter, bool isInput)
    {
        var count = GetTensorCount(interpreter, isInput);
        var metadata = new Dictionary<string, MLNodeMetadata>(count);
        var prefix = isInput ? "input" : "output";

        for (var i = 0; i < count; i++)
        {
            var tensor = GetTensor(interpreter, isInput, i);
            if (tensor == null) continue;
            var name = GetTensorName(tensor) ?? $"{prefix}{i}";
            metadata[name] = new MLNodeMetadata(GetElementType(tensor), GetTensorDimensions(tensor));
        }

        return metadata;
    }

    private static int GetTensorCount(Interpreter interpreter, bool isInput)
    {
        return isInput ? interpreter.InputTensorCount : interpreter.OutputTensorCount;
    }

    private static ITensor? GetTensor(Interpreter interpreter, bool isInput, int index)
    {
        return isInput ? interpreter.GetInputTensor(index) : interpreter.GetOutputTensor(index);
    }

    private static int[] GetTensorDimensions(ITensor tensor)
    {
        return tensor.Shape() ?? [];
    }

    private static string? GetTensorName(ITensor tensor)
    {
        ArgumentNullException.ThrowIfNull(tensor);

        return tensor.Name();
    }

    private static Type GetElementType(ITensor tensor)
    {
        var dataTypeValue = tensor.DataType();

        if (dataTypeValue == null)
        {
            return typeof(float);
        }

        return MapTensorElementType(dataTypeValue);
    }

    private static Type MapTensorElementType(DataType dataTypeValue)
    {
        return dataTypeValue.GetType();
    }
}
#endif