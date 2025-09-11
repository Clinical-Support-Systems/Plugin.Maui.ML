using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace Plugin.Maui.ML.Platforms.iOS;

/// <summary>
/// iOS-specific ML inference implementation
/// </summary>
public class PlatformMLInfer : OnnxRuntimeInfer
{
    /// <summary>
    /// Initializes a new instance of the PlatformMLInfer class for iOS
    /// </summary>
    public PlatformMLInfer() : base()
    {
        // iOS-specific initialization can be added here
    }

    /// <summary>
    /// Load model from iOS bundle resources
    /// </summary>
    /// <param name="resourceName">Name of the resource in the iOS bundle</param>
    /// <param name="resourceExtension">Extension of the resource file (default: "onnx")</param>
    /// <param name="cancellationToken">Cancellation token</param>
    /// <returns>Task that completes when the model is loaded</returns>
    public async Task LoadModelFromBundleAsync(string resourceName, string resourceExtension = "onnx", CancellationToken cancellationToken = default)
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
            throw new InvalidOperationException($"Failed to load model from iOS bundle resource '{resourceName}.{resourceExtension}': {ex.Message}", ex);
        }
    }

    /// <summary>
    /// Get available execution providers for iOS
    /// </summary>
    /// <returns>List of available execution provider names</returns>
    public static List<string> GetAvailableExecutionProviders()
    {
        var providers = new List<string> { "CPUExecutionProvider" };
        
        // CoreML is available on iOS 11+
        providers.Add("CoreMLExecutionProvider");

        return providers;
    }

    /// <summary>
    /// Check if Neural Engine is available (iOS 12+ on A12+ chips)
    /// </summary>
    /// <returns>True if Neural Engine is likely available</returns>
    public static bool IsNeuralEngineAvailable()
    {
        // Neural Engine is available on A12+ chips (iPhone XS, iPad Pro 2018, etc.)
        // This is a best-effort check
        return Environment.OSVersion.Version.Major >= 12;
    }
}