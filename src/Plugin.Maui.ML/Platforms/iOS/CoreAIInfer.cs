using Microsoft.ML.OnnxRuntime.Tensors;

#if IOS
namespace Plugin.Maui.ML.Platforms.iOS;

/// <summary>
///     iOS Core AI inference placeholder.
///     The public .NET bindings for Apple's Core AI framework are not yet available in this repository,
///     so this backend exists as the opt-in surface for future native integration.
/// </summary>
public sealed class CoreAIInfer : IMLInfer
{
    private const string CoreAIUnavailableMessage =
        "Core AI support is not yet available in this .NET MAUI package. Add Apple Core AI bindings before using this backend.";

    public MLBackend Backend => MLBackend.CoreAI;

    public bool IsModelLoaded => false;

    public Task LoadModelAsync(string modelPath, CancellationToken cancellationToken = default)
    {
        return ThrowNotSupportedTask();
    }

    public Task LoadModelAsync(Stream modelStream, CancellationToken cancellationToken = default)
    {
        return ThrowNotSupportedTask();
    }

    public Task LoadModelFromAssetAsync(string assetName, CancellationToken cancellationToken = default)
    {
        return ThrowNotSupportedTask();
    }

    public Task<Dictionary<string, Tensor<float>>> RunInferenceAsync(
        Dictionary<string, Tensor<float>> inputs,
        CancellationToken cancellationToken = default)
    {
        return ThrowNotSupported<Dictionary<string, Tensor<float>>>();
    }

    public Task<Dictionary<string, Tensor<float>>> RunInferenceLongInputsAsync(
        Dictionary<string, Tensor<long>> inputs,
        CancellationToken cancellationToken = default)
    {
        return ThrowNotSupported<Dictionary<string, Tensor<float>>>();
    }

    public Dictionary<string, MLNodeMetadata> GetInputMetadata()
    {
        throw new PlatformNotSupportedException(CoreAIUnavailableMessage);
    }

    public Dictionary<string, MLNodeMetadata> GetOutputMetadata()
    {
        throw new PlatformNotSupportedException(CoreAIUnavailableMessage);
    }

    public void UnloadModel()
    {
    }

    private static Task ThrowNotSupportedTask()
    {
        throw new PlatformNotSupportedException(CoreAIUnavailableMessage);
    }

    private static Task<T> ThrowNotSupported<T>()
    {
        throw new PlatformNotSupportedException(CoreAIUnavailableMessage);
    }
}
#endif