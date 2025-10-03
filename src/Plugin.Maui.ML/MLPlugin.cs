namespace Plugin.Maui.ML;

/// <summary>
///     Static entry point for ML inference functionality
/// </summary>
public static class MLPlugin
{
    static IMLInfer? defaultImplementation;

    /// <summary>
    ///     Gets the default implementation of <see cref="IMLInfer"/>
    ///     Uses platform-specific implementation when available, falls back to ONNX Runtime
    /// </summary>
    public static IMLInfer Default =>
        defaultImplementation ??= CreatePlatformDefault();

    /// <summary>
    ///     Sets the default implementation (useful for testing or custom implementations)
    /// </summary>
    internal static void SetDefault(IMLInfer? implementation) =>
        defaultImplementation = implementation;

    /// <summary>
    ///     Creates a new platform-specific implementation instance
    /// </summary>
    internal static IMLInfer CreatePlatformDefault()
    {
#if IOS
        return new Platforms.iOS.PlatformMLInfer();
#elif MACCATALYST
        return new Platforms.MacCatalyst.PlatformMLInfer();
#elif ANDROID
        return new Platforms.Android.PlatformMLInfer();
#elif WINDOWS
        return new Platforms.Windows.PlatformMLInfer();
#else
        return new OnnxRuntimeInfer();
#endif
    }
}
