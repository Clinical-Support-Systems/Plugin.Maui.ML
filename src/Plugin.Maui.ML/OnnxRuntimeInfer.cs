using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Collections.Concurrent;

namespace Plugin.Maui.ML;

/// <summary>
/// ONNX Runtime implementation of ML inference
/// </summary>
public class OnnxRuntimeInfer : IMLInfer, IDisposable
{
    private InferenceSession? _session;
    private SessionOptions? _sessionOptions;
    private readonly object _lock = new();
    private bool _disposed;

    /// <summary>
    /// Initializes a new instance of the OnnxRuntimeInfer class
    /// </summary>
    public OnnxRuntimeInfer()
    {
        _sessionOptions = new SessionOptions();
        ConfigurePlatformSpecificOptions();
    }

    /// <inheritdoc/>
    public bool IsModelLoaded => _session != null;

    /// <inheritdoc/>
    public async Task LoadModelAsync(string modelPath, CancellationToken cancellationToken = default)
    {
        if (string.IsNullOrEmpty(modelPath))
            throw new ArgumentException("Model path cannot be null or empty", nameof(modelPath));

        if (!File.Exists(modelPath))
            throw new FileNotFoundException($"Model file not found: {modelPath}");

        await Task.Run(() =>
        {
            lock (_lock)
            {
                UnloadModel();
                _session = new InferenceSession(modelPath, _sessionOptions);
            }
        }, cancellationToken);
    }

    /// <inheritdoc/>
    public async Task LoadModelAsync(Stream modelStream, CancellationToken cancellationToken = default)
    {
        if (modelStream == null)
            throw new ArgumentNullException(nameof(modelStream));

        await Task.Run(() =>
        {
            lock (_lock)
            {
                UnloadModel();
                
                // Read stream to byte array
                using var memoryStream = new MemoryStream();
                modelStream.CopyTo(memoryStream);
                var modelBytes = memoryStream.ToArray();
                
                _session = new InferenceSession(modelBytes, _sessionOptions);
            }
        }, cancellationToken);
    }

    /// <inheritdoc/>
    public async Task LoadModelFromAssetAsync(string assetName, CancellationToken cancellationToken = default)
    {
        if (string.IsNullOrEmpty(assetName))
            throw new ArgumentException("Asset name cannot be null or empty", nameof(assetName));

        try
        {
            // This would be implemented by platform-specific versions
            // For now, try to load from current directory
            var assetPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, assetName);
            if (File.Exists(assetPath))
            {
                await LoadModelAsync(assetPath, cancellationToken);
            }
            else
            {
                throw new FileNotFoundException($"Asset '{assetName}' not found");
            }
        }
        catch (Exception ex)
        {
            throw new InvalidOperationException($"Failed to load model from asset '{assetName}': {ex.Message}", ex);
        }
    }

    /// <inheritdoc/>
    public async Task<Dictionary<string, Tensor<float>>> RunInferenceAsync(
        Dictionary<string, Tensor<float>> inputs, 
        CancellationToken cancellationToken = default)
    {
        if (_session == null)
            throw new InvalidOperationException("No model is loaded. Call LoadModelAsync first.");

        if (inputs == null || inputs.Count == 0)
            throw new ArgumentException("Inputs cannot be null or empty", nameof(inputs));

        return await Task.Run(() =>
        {
            lock (_lock)
            {
                if (_session == null)
                    throw new InvalidOperationException("Model was unloaded during inference");

                // Convert inputs to NamedOnnxValue
                var onnxInputs = inputs.Select(kvp => 
                    NamedOnnxValue.CreateFromTensor(kvp.Key, kvp.Value)).ToList();

                // Run inference
                using var results = _session.Run(onnxInputs);
                
                // Convert results back to dictionary
                var outputs = new Dictionary<string, Tensor<float>>();
                foreach (var result in results)
                {
                    if (result.Value is Tensor<float> tensor)
                    {
                        outputs[result.Name] = tensor;
                    }
                    else
                    {
                        // Handle type conversion if needed
                        throw new InvalidOperationException($"Unsupported output tensor type for '{result.Name}': {result.Value?.GetType()}");
                    }
                }

                return outputs;
            }
        }, cancellationToken);
    }

    /// <inheritdoc/>
    public Dictionary<string, NodeMetadata> GetInputMetadata()
    {
        if (_session == null)
            throw new InvalidOperationException("No model is loaded. Call LoadModelAsync first.");

        lock (_lock)
        {
            return _session.InputMetadata.ToDictionary(
                kvp => kvp.Key,
                kvp => kvp.Value
            );
        }
    }

    /// <inheritdoc/>
    public Dictionary<string, NodeMetadata> GetOutputMetadata()
    {
        if (_session == null)
            throw new InvalidOperationException("No model is loaded. Call LoadModelAsync first.");

        lock (_lock)
        {
            return _session.OutputMetadata.ToDictionary(
                kvp => kvp.Key,
                kvp => kvp.Value
            );
        }
    }

    /// <inheritdoc/>
    public void UnloadModel()
    {
        lock (_lock)
        {
            _session?.Dispose();
            _session = null;
        }
    }

    private void ConfigurePlatformSpecificOptions()
    {
        if (_sessionOptions == null) return;

        // Configure execution providers based on platform
#if ANDROID
        // Android-specific optimizations
        _sessionOptions.AppendExecutionProvider_Nnapi();
        _sessionOptions.AppendExecutionProvider_CPU();
#elif IOS || MACCATALYST
        // iOS/macOS-specific optimizations
        _sessionOptions.AppendExecutionProvider_CoreML();
        _sessionOptions.AppendExecutionProvider_CPU();
#elif WINDOWS
        // Windows-specific optimizations
        try
        {
            // Try to use DirectML if available
            _sessionOptions.AppendExecutionProvider_DML();
        }
        catch
        {
            // Fall back to CPU if DirectML is not available
        }
        _sessionOptions.AppendExecutionProvider_CPU();
#else
        // Default CPU execution
        _sessionOptions.AppendExecutionProvider_CPU();
#endif

        // General optimizations
        _sessionOptions.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
    }

    /// <summary>
    /// Dispose of resources
    /// </summary>
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
}