using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace Plugin.Maui.ML;

/// <summary>
///     Implementation of IMLInfer using ONNX Runtime for ML inference.
/// </summary>
public class OnnxRuntimeInfer : IMLInfer, IDisposable
{
    private readonly Lock _lock = new();
    private bool _disposed;
    private InferenceSession? _session;
    private SessionOptions? _sessionOptions;

    /// <summary>
    ///     Initializes a new instance of the OnnxRuntimeInfer class.
    /// </summary>
    public OnnxRuntimeInfer()
    {
        _sessionOptions = new SessionOptions();
        ConfigurePlatformSpecificOptions();
    }

    /// <summary>
    ///     Gets the backend type - ONNX Runtime
    /// </summary>
    public MLBackend Backend => MLBackend.OnnxRuntime;

    /// <summary>
    ///     Releases resources used by the OnnxRuntimeInfer instance.
    /// </summary>
    public void Dispose()
    {
        Dispose(true);
        GC.SuppressFinalize(this);
    }

    /// <summary>
    ///     Gets whether a model is currently loaded.
    /// </summary>
    public bool IsModelLoaded => _session != null;

    /// <summary>
    ///     Loads an ONNX model from the specified file path asynchronously.
    /// </summary>
    /// <param name="modelPath">
    ///     Path to the ONNX model file.
    /// </param>
    /// <param name="cancellationToken">
    ///     Cancellation token to cancel the operation.
    /// </param>
    /// <returns>
    ///     A task that completes when the model is loaded.
    /// </returns>
    /// <exception cref="ArgumentException">
    ///     Thrown if the model path is null or empty.
    /// </exception>
    /// <exception cref="FileNotFoundException">
    ///     Thrown if the model file does not exist.
    /// </exception>
    public async Task LoadModelAsync(string modelPath, CancellationToken cancellationToken = default)
    {
        if (string.IsNullOrEmpty(modelPath))
            throw new ArgumentException("Model path cannot be null or empty", nameof(modelPath));
        if (!File.Exists(modelPath)) throw new FileNotFoundException($"Model file not found: {modelPath}");
        await Task.Run(() =>
        {
            lock (_lock)
            {
                UnloadModel();
                _session = new InferenceSession(modelPath, _sessionOptions);
            }
        }, cancellationToken);
    }

    /// <summary>
    ///     Loads an ONNX model from the provided stream asynchronously.
    /// </summary>
    /// <param name="modelStream">
    ///     Stream containing the ONNX model data.
    /// </param>
    /// <param name="cancellationToken">
    ///     Cancellation token to cancel the operation.
    /// </param>
    /// <returns>
    ///     A task that completes when the model is loaded.
    /// </returns>
    public async Task LoadModelAsync(Stream modelStream, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(modelStream);
        await Task.Run(() =>
        {
            lock (_lock)
            {
                UnloadModel();
                using var ms = new MemoryStream();
                modelStream.CopyTo(ms);
                _session = new InferenceSession(ms.ToArray(), _sessionOptions);
            }
        }, cancellationToken);
    }

    /// <summary>
    ///     Loads an ONNX model from MAUI assets asynchronously.
    /// </summary>
    /// <param name="assetName">
    ///     Name of the asset file (e.g., "model.onnx").
    /// </param>
    /// <param name="cancellationToken">
    ///     Cancellation token to cancel the operation.
    /// </param>
    /// <returns>
    ///     A task that completes when the model is loaded.
    /// </returns>
    /// <exception cref="ArgumentException">
    ///     Thrown if the asset name is null or empty.
    /// </exception>
    /// <exception cref="FileNotFoundException">
    ///     Thrown if the asset file does not exist.
    /// </exception>
    public async Task LoadModelFromAssetAsync(string assetName, CancellationToken cancellationToken = default)
    {
        if (string.IsNullOrEmpty(assetName))
            throw new ArgumentException("Asset name cannot be null or empty", nameof(assetName));
        var assetPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, assetName);
        if (!File.Exists(assetPath)) throw new FileNotFoundException($"Asset '{assetName}' not found");
        await LoadModelAsync(assetPath, cancellationToken);
    }

    /// <summary>
    ///     Runs inference using float inputs (all inputs must be float-compatible and
    /// </summary>
    /// <param name="inputs">
    ///     Input tensors for the model, provided as a dictionary mapping input names to Tensor&lt;float&gt; objects.
    /// </param>
    /// <param name="cancellationToken">
    ///     Cancellation token to cancel the operation.
    /// </param>
    /// <returns>
    ///     A dictionary of output tensors, mapping output names to Tensor&lt;float&gt; objects.
    /// </returns>
    /// <exception cref="ArgumentException">
    ///     Thrown if the inputs dictionary is null or empty.
    /// </exception>
    public Task<Dictionary<string, Tensor<float>>> RunInferenceAsync(Dictionary<string, Tensor<float>> inputs,
        CancellationToken cancellationToken = default)
    {
        if (inputs == null || inputs.Count == 0)
            throw new ArgumentException("Inputs cannot be null or empty", nameof(inputs));
        var named = inputs.Select(kvp => NamedOnnxValue.CreateFromTensor(kvp.Key, kvp.Value));
        return RunInternal(named, cancellationToken);
    }

    /// <summary>
    ///     Runs inference using Int64 inputs (e.g., token ids / attention masks).
    ///     Will cast to model-expected element types (Int64 / Int32 / Float).
    /// </summary>
    /// <param name="inputs">
    ///     Input tensors for the model, provided as a dictionary mapping input names to Tensor&lt;long&gt; objects.
    /// </param>
    /// <param name="cancellationToken">
    ///     Cancellation token to cancel the operation.
    /// </param>
    /// <returns>
    ///     A dictionary of output tensors, mapping output names to Tensor&lt;float&gt; objects.
    /// </returns>
    /// <exception cref="ArgumentException">
    ///     Thrown if the inputs dictionary is null or empty.
    /// </exception>
    public Task<Dictionary<string, Tensor<float>>> RunInferenceLongInputsAsync(Dictionary<string, Tensor<long>> inputs,
        CancellationToken cancellationToken = default)
    {
        if (inputs == null || inputs.Count == 0)
            throw new ArgumentException("Inputs cannot be null or empty", nameof(inputs));
        if (_session == null) throw new InvalidOperationException("No model is loaded. Call LoadModelAsync first.");

        // Build NamedOnnxValue list casting as needed based on input metadata
        var meta = _session.InputMetadata;
        var named = new List<NamedOnnxValue>();
        foreach (var (key, src) in inputs)
        {
            if (!meta.TryGetValue(key, out var nodeMeta))
                throw new ArgumentException($"Input '{key}' not found in model metadata.");
            var targetType = nodeMeta.ElementType;
            if (targetType == typeof(long))
            {
                named.Add(NamedOnnxValue.CreateFromTensor(key, src));
            }
            else if (targetType == typeof(float))
            {
                var cast = new DenseTensor<float>(src.Dimensions.ToArray());
                var span = cast.Buffer.Span;
                var arr = src.ToArray();
                for (var i = 0; i < arr.Length; i++) span[i] = arr[i];
                named.Add(NamedOnnxValue.CreateFromTensor(key, cast));
            }
            else if (targetType == typeof(int))
            {
                var cast = new DenseTensor<int>(src.Dimensions.ToArray());
                var span = cast.Buffer.Span;
                var arr = src.ToArray();
                for (var i = 0; i < arr.Length; i++) span[i] = checked((int)arr[i]);
                named.Add(NamedOnnxValue.CreateFromTensor(key, cast));
            }
            else
            {
                throw new InvalidOperationException(
                    $"Unsupported target input element type '{targetType}' for provided Int64 tensor.");
            }
        }

        return RunInternal(named, cancellationToken);
    }

    /// <summary>
    ///     Gets input metadata for the loaded model.
    /// </summary>
    /// <returns>
    ///     A dictionary mapping input names to their corresponding NodeMetadata.
    /// </returns>
    /// <exception cref="InvalidOperationException">
    ///     Thrown if no model is loaded.
    /// </exception>
    public Dictionary<string, NodeMetadata> GetInputMetadata()
    {
        if (_session == null) throw new InvalidOperationException("No model is loaded. Call LoadModelAsync first.");
        lock (_lock)
        {
            return _session.InputMetadata.ToDictionary(k => k.Key, v => v.Value);
        }
    }

    /// <summary>
    ///     Gets output metadata for the loaded model.
    /// </summary>
    /// <returns>
    ///     A dictionary mapping output names to their corresponding NodeMetadata.
    /// </returns>
    /// <exception cref="InvalidOperationException">
    ///     Thrown if no model is loaded.
    /// </exception>
    public Dictionary<string, NodeMetadata> GetOutputMetadata()
    {
        if (_session == null) throw new InvalidOperationException("No model is loaded. Call LoadModelAsync first.");
        lock (_lock)
        {
            return _session.OutputMetadata.ToDictionary(k => k.Key, v => v.Value);
        }
    }

    /// <summary>
    ///     Unloads the currently loaded model and releases resources.
    /// </summary>
    public void UnloadModel()
    {
        lock (_lock)
        {
            _session?.Dispose();
            _session = null;
        }
    }

    /// <summary>
    ///     Finalizer to ensure unmanaged resources are released.
    /// </summary>
    ~OnnxRuntimeInfer()
    {
        Dispose(false);
    }

    /// <summary>
    ///     Protected implementation of Dispose pattern.
    /// </summary>
    /// <param name="disposing">True if called from Dispose; false if called from finalizer.</param>
    protected virtual void Dispose(bool disposing)
    {
        if (_disposed)
            return;

        if (disposing)
        {
            // Dispose managed resources
            UnloadModel();
            _sessionOptions?.Dispose();
            _sessionOptions = null;
        }

        // Free unmanaged resources (if any) here

        _disposed = true;
    }

    private Task<Dictionary<string, Tensor<float>>> RunInternal(IEnumerable<NamedOnnxValue> inputs,
        CancellationToken ct)
    {
        if (_session == null) throw new InvalidOperationException("No model is loaded. Call LoadModelAsync first.");
        return Task.Run(() =>
        {
            lock (_lock)
            {
                using var results = _session!.Run(inputs.ToList().AsReadOnly());
                var outputs = new Dictionary<string, Tensor<float>>();
                foreach (var r in results)
                    switch (r.Value)
                    {
                        case Tensor<float> ft:
                            outputs[r.Name] = ft;
                            break;

                        case Tensor<long> lt:
                        {
                            var castL = new DenseTensor<float>(lt.Dimensions.ToArray());
                            var i = 0;
                            foreach (var v in lt.ToArray()) castL.Buffer.Span[i++] = v;

                            outputs[r.Name] = castL;
                            break;
                        }

                        case Tensor<int> it:
                        {
                            var castI = new DenseTensor<float>(it.Dimensions.ToArray());
                            var j = 0;
                            foreach (var v in it.ToArray()) castI.Buffer.Span[j++] = v;

                            outputs[r.Name] = castI;
                            break;
                        }

                        default:
                            throw new InvalidOperationException(
                                $"Unsupported output tensor type: {r.Value?.GetType()}");
                    }

                return outputs;
            }
        }, ct);
    }

    private void ConfigurePlatformSpecificOptions()
    {
        if (_sessionOptions == null) return;
#if ANDROID
        _sessionOptions.AppendExecutionProvider_Nnapi();
        _sessionOptions.AppendExecutionProvider_CPU();
#elif IOS || MACCATALYST
        _sessionOptions.AppendExecutionProvider_CoreML();
        _sessionOptions.AppendExecutionProvider_CPU();
#elif WINDOWS
        try { _sessionOptions.AppendExecutionProvider_DML(); }
        catch { }

        _sessionOptions.AppendExecutionProvider_CPU();
#else
        _sessionOptions.AppendExecutionProvider_CPU();
#endif
        _sessionOptions.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
    }
}