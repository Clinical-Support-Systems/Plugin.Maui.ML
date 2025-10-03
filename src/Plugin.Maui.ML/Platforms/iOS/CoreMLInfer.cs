#if IOS || MACCATALYST
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace Plugin.Maui.ML.Platforms.iOS;

/// <summary>
///     iOS/macOS-specific ML inference using CoreML
///     Provides native Apple Neural Engine acceleration
///     Note: This is a placeholder for future CoreML implementation
/// </summary>
public class CoreMLInfer : IMLInfer, IDisposable
{
    private bool _disposed;

    /// <summary>
    ///     Gets the backend type - CoreML
    /// </summary>
    public MLBackend Backend => MLBackend.CoreML;

    /// <summary>
    ///     Gets whether a model is currently loaded
    /// </summary>
    public bool IsModelLoaded => false;

    /// <summary>
    ///     Initializes a new instance of the CoreMLInfer class
    /// </summary>
    public CoreMLInfer()
    {
        // CoreML implementation coming soon
    }

    public Task LoadModelAsync(string modelPath, CancellationToken cancellationToken = default)
    {
        throw new NotImplementedException("CoreML support is under development. Use OnnxRuntimeInfer with CoreML execution provider for now.");
    }

    public Task LoadModelAsync(Stream modelStream, CancellationToken cancellationToken = default)
    {
        throw new NotImplementedException("CoreML support is under development. Use OnnxRuntimeInfer with CoreML execution provider for now.");
    }

    public Task LoadModelFromAssetAsync(string assetName, CancellationToken cancellationToken = default)
    {
        throw new NotImplementedException("CoreML support is under development. Use OnnxRuntimeInfer with CoreML execution provider for now.");
    }

    public Task<Dictionary<string, Tensor<float>>> RunInferenceAsync(
        Dictionary<string, Tensor<float>> inputs,
        CancellationToken cancellationToken = default)
    {
        throw new NotImplementedException("CoreML support is under development. Use OnnxRuntimeInfer with CoreML execution provider for now.");
    }

    public Task<Dictionary<string, Tensor<float>>> RunInferenceLongInputsAsync(
        Dictionary<string, Tensor<long>> inputs,
        CancellationToken cancellationToken = default)
    {
        throw new NotImplementedException("CoreML support is under development. Use OnnxRuntimeInfer with CoreML execution provider for now.");
    }

    public Dictionary<string, NodeMetadata> GetInputMetadata()
    {
        throw new NotImplementedException("CoreML support is under development.");
    }

    public Dictionary<string, NodeMetadata> GetOutputMetadata()
    {
        throw new NotImplementedException("CoreML support is under development.");
    }

    public void UnloadModel()
    {
        // Nothing to unload
    }

    public void Dispose()
    {
        Dispose(true);
        GC.SuppressFinalize(this);
    }

    protected virtual void Dispose(bool disposing)
    {
        if (_disposed)
            return;

        _disposed = true;
    }
}
#endif
