namespace Plugin.Maui.ML.Platforms.Android;

/// <summary>
///     Android-specific ML inference implementation
///     By default uses ONNX Runtime with NNAPI execution provider for hardware acceleration
///     For TensorFlow Lite models, consider using ML Kit or TFLite directly
/// </summary>
public class PlatformMLInfer : OnnxRuntimeInfer
{
    /// <summary>
    ///     Initializes a new instance of the PlatformMLInfer class for Android
    ///     Uses ONNX Runtime with NNAPI execution provider by default
    /// </summary>
    public PlatformMLInfer() : base()
    {
        // Android-specific initialization
        // NNAPI execution provider is already configured in OnnxRuntimeInfer
    }

    /// <summary>
    ///     Load model from Android assets folder
    /// </summary>
    /// <param name="assetName">Name of the asset in the assets folder</param>
    /// <param name="cancellationToken">Cancellation token</param>
    /// <returns>Task that completes when the model is loaded</returns>
    public async Task LoadModelFromAndroidAssetsAsync(string assetName, CancellationToken cancellationToken = default)
    {
        if (string.IsNullOrEmpty(assetName))
            throw new ArgumentException("Asset name cannot be null or empty", nameof(assetName));

        try
        {
#if ANDROID
            // Try to use Android Assets API
            if (global::Android.App.Application.Context?.Assets != null)
            {
                using var assetStream = global::Android.App.Application.Context.Assets.Open(assetName);
                await LoadModelAsync(assetStream, cancellationToken);
                return;
            }
#endif
            // Fallback to base implementation
            await LoadModelFromAssetAsync(assetName, cancellationToken);
        }
        catch (Exception ex)
        {
            throw new InvalidOperationException($"Failed to load model from Android asset '{assetName}': {ex.Message}",
                ex);
        }
    }

    /// <summary>
    ///     Get available execution providers for Android
    /// </summary>
    /// <returns>List of available execution provider names</returns>
    public static List<string> GetAvailableExecutionProviders()
    {
        var providers = new List<string>
        {
            "CPUExecutionProvider",
            // NNAPI would be available on Android API 27+ (Android 8.1+)
            "NnapiExecutionProvider"
        };

        return providers;
    }

    /// <summary>
    ///     Check if NNAPI is available on this Android device
    /// </summary>
    /// <returns>True if NNAPI is available (Android 8.1+)</returns>
    public static bool IsNnapiAvailable()
    {
#if ANDROID
        return global::Android.OS.Build.VERSION.SdkInt >= global::Android.OS.BuildVersionCodes.O;
#else
        return false;
#endif
    }
}
