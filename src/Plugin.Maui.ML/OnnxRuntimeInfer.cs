using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace Plugin.Maui.ML;

public class OnnxRuntimeInfer : IMLInfer, IDisposable
{
    private readonly object _lock = new();
    private bool _disposed;
    private InferenceSession? _session;
    private SessionOptions? _sessionOptions;

    public OnnxRuntimeInfer()
    {
        _sessionOptions = new SessionOptions();
        ConfigurePlatformSpecificOptions();
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            UnloadModel();
            _sessionOptions?.Dispose();
            _sessionOptions = null;
            _disposed = true;
        }
    }

    public bool IsModelLoaded => _session != null;

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

    public async Task LoadModelAsync(Stream modelStream, CancellationToken cancellationToken = default)
    {
        if (modelStream == null) throw new ArgumentNullException(nameof(modelStream));
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

    public async Task LoadModelFromAssetAsync(string assetName, CancellationToken cancellationToken = default)
    {
        if (string.IsNullOrEmpty(assetName))
            throw new ArgumentException("Asset name cannot be null or empty", nameof(assetName));
        var assetPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, assetName);
        if (!File.Exists(assetPath)) throw new FileNotFoundException($"Asset '{assetName}' not found");
        await LoadModelAsync(assetPath, cancellationToken);
    }

    public Task<Dictionary<string, Tensor<float>>> RunInferenceAsync(Dictionary<string, Tensor<float>> inputs,
        CancellationToken cancellationToken = default)
    {
        if (inputs == null || inputs.Count == 0)
            throw new ArgumentException("Inputs cannot be null or empty", nameof(inputs));
        var named = inputs.Select(kvp => NamedOnnxValue.CreateFromTensor(kvp.Key, kvp.Value));
        return RunInternal(named, cancellationToken);
    }

    public Task<Dictionary<string, Tensor<float>>> RunInferenceLongInputsAsync(Dictionary<string, Tensor<long>> inputs,
        CancellationToken cancellationToken = default)
    {
        if (inputs == null || inputs.Count == 0)
            throw new ArgumentException("Inputs cannot be null or empty", nameof(inputs));
        var named = inputs.Select(kvp => NamedOnnxValue.CreateFromTensor(kvp.Key, kvp.Value));
        return RunInternal(named, cancellationToken);
    }

    public Dictionary<string, NodeMetadata> GetInputMetadata()
    {
        if (_session == null) throw new InvalidOperationException("No model is loaded. Call LoadModelAsync first.");
        lock (_lock)
        {
            return _session.InputMetadata.ToDictionary(k => k.Key, v => v.Value);
        }
    }

    public Dictionary<string, NodeMetadata> GetOutputMetadata()
    {
        if (_session == null) throw new InvalidOperationException("No model is loaded. Call LoadModelAsync first.");
        lock (_lock)
        {
            return _session.OutputMetadata.ToDictionary(k => k.Key, v => v.Value);
        }
    }

    public void UnloadModel()
    {
        lock (_lock)
        {
            _session?.Dispose();
            _session = null;
        }
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
                {
                    switch (r.Value)
                    {
                        case Tensor<float> ft:
                            outputs[r.Name] = ft;
                            break;

                        case Tensor<long> lt:
                        {
                            var castL = new DenseTensor<float>(lt.Dimensions.ToArray());
                            var i = 0;
                            foreach (var v in lt.ToArray())
                            {
                                castL.Buffer.Span[i++] = v;
                            }

                            outputs[r.Name] = castL;
                            break;
                        }

                        case Tensor<int> it:
                        {
                            var castI = new DenseTensor<float>(it.Dimensions.ToArray());
                            var j = 0;
                            foreach (var v in it.ToArray())
                            {
                                castI.Buffer.Span[j++] = v;
                            }

                            outputs[r.Name] = castI;
                            break;
                        }

                        default:
                            throw new InvalidOperationException(
                                $"Unsupported output tensor type: {r.Value?.GetType()}");
                    }
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
