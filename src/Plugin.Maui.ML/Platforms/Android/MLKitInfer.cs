#if ANDROID
using Java.Nio;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
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

    public void Dispose()
    {
        if (_disposed) return;
        UnloadModel();
        _disposed = true;
        GC.SuppressFinalize(this);
    }

    public MLBackend Backend => MLBackend.MLKit;
    public bool IsModelLoaded => _interpreter != null;

    public async Task LoadModelAsync(string modelPath, CancellationToken cancellationToken = default)
    {
        if (string.IsNullOrWhiteSpace(modelPath))
            throw new ArgumentException("Model path cannot be null or empty", nameof(modelPath));
        if (!File.Exists(modelPath))
            throw new FileNotFoundException($"Model file not found: {modelPath}");
        var bytes = await File.ReadAllBytesAsync(modelPath, cancellationToken).ConfigureAwait(false);
        Initialize(bytes);
    }

    public async Task LoadModelAsync(Stream modelStream, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(modelStream);
        await using var ms = new MemoryStream();
        await modelStream.CopyToAsync(ms, cancellationToken).ConfigureAwait(false);
        Initialize(ms.ToArray());
    }

    public async Task LoadModelFromAssetAsync(string assetName, CancellationToken cancellationToken = default)
    {
        if (string.IsNullOrWhiteSpace(assetName))
            throw new ArgumentException("Asset name cannot be null or empty", nameof(assetName));
        try
        {
            await using var assetStream = Application.Context.Assets?.Open(assetName)
                                          ?? throw new FileNotFoundException(
                                              $"Asset '{assetName}' not found in Android assets.");
            await LoadModelAsync(assetStream, cancellationToken).ConfigureAwait(false);
        }
        catch (Exception ex)
        {
            throw new InvalidOperationException($"Failed to load TFLite model asset '{assetName}': {ex.Message}", ex);
        }
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
        if (inputs == null || inputs.Count == 0)
            throw new ArgumentException("Inputs cannot be null or empty", nameof(inputs));
        var floatInputs = new Dictionary<string, Tensor<float>>();
        foreach (var (k, v) in inputs)
        {
            var cast = new DenseTensor<float>(v.Dimensions.ToArray());
            var arr = v.ToArray();
            for (var i = 0; i < arr.Length; i++) cast.Buffer.Span[i] = arr[i];
            floatInputs[k] = cast;
        }

        return RunInferenceAsync(floatInputs, cancellationToken);
    }

    public Dictionary<string, MLNodeMetadata> GetInputMetadata()
    {
        throw new NotSupportedException("TFLite metadata not exposed in Phase 1 implementation.");
    }

    public Dictionary<string, MLNodeMetadata> GetOutputMetadata()
    {
        throw new NotSupportedException("TFLite metadata not exposed in Phase 1 implementation.");
    }

    public void UnloadModel()
    {
        lock (_sync)
        {
            _interpreter?.Close();
            _interpreter?.Dispose();
            _interpreter = null;
            _modelBytes = null;
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
}
#endif