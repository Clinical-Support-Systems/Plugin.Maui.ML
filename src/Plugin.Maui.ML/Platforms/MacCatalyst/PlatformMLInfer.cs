using System.Runtime.InteropServices;

namespace Plugin.Maui.ML.Platforms.MacCatalyst;

/// <summary>
///     macOS Catalyst-specific ML inference implementation
/// </summary>
public class PlatformMLInfer : OnnxRuntimeInfer
{
    /// <summary>
    ///     Initializes a new instance of the PlatformMLInfer class for macOS Catalyst
    /// </summary>
    public PlatformMLInfer()
    {
        // macOS Catalyst-specific initialization can be added here
    }

    /// <summary>
    ///     Load model from macOS app bundle resources
    /// </summary>
    /// <param name="resourceName">Name of the resource in the app bundle</param>
    /// <param name="resourceExtension">Extension of the resource file (default: "onnx")</param>
    /// <param name="cancellationToken">Cancellation token</param>
    /// <returns>Task that completes when the model is loaded</returns>
    public async Task LoadModelFromBundleAsync(string resourceName, string resourceExtension = "onnx",
        CancellationToken cancellationToken = default)
    {
        if (string.IsNullOrEmpty(resourceName))
            throw new ArgumentException("Resource name cannot be null or empty", nameof(resourceName));

        try
        {
            // This would use NSBundle.MainBundle.PathForResource(resourceName, resourceExtension)
            // For now, fallback to base implementation
            var assetName = $"{resourceName}.{resourceExtension}";
            await LoadModelFromAssetAsync(assetName, cancellationToken);
        }
        catch (Exception ex)
        {
            throw new InvalidOperationException(
                $"Failed to load model from app bundle resource '{resourceName}.{resourceExtension}': {ex.Message}",
                ex);
        }
    }

    /// <summary>
    ///     Get available execution providers for macOS Catalyst
    /// </summary>
    /// <returns>List of available execution provider names</returns>
    public static List<string> GetAvailableExecutionProviders()
    {
        var providers = new List<string>
        {
            "CPUExecutionProvider",
            // CoreML is available on macOS 10.13+
            "CoreMLExecutionProvider"
        };

        return providers;
    }

    /// <summary>
    ///     Check if Apple Neural Engine is available on Apple Silicon Macs
    /// </summary>
    /// <returns>True if likely running on Apple Silicon with Neural Engine</returns>
    public static bool IsNeuralEngineAvailable()
    {
        try
        {
            // This is a heuristic check for Apple Silicon
            // On Apple Silicon Macs, the Neural Engine is available
            var architecture = RuntimeInformation.ProcessArchitecture;
            return architecture == Architecture.Arm64;
        }
        catch
        {
            return false;
        }
    }
}
