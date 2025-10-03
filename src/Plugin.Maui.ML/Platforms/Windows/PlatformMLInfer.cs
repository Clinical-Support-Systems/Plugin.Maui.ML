using System.Runtime.InteropServices;

namespace Plugin.Maui.ML.Platforms.Windows;

/// <summary>
///     Windows-specific ML inference implementation
///     By default uses ONNX Runtime with DirectML execution provider for GPU acceleration
/// </summary>
public class PlatformMLInfer : OnnxRuntimeInfer
{
    /// <summary>
    ///     Initializes a new instance of the PlatformMLInfer class for Windows
    ///     Uses ONNX Runtime with DirectML execution provider by default
    /// </summary>
    public PlatformMLInfer()
    {
        // Windows-specific initialization
        // DirectML execution provider is already configured in OnnxRuntimeInfer
    }

    /// <summary>
    ///     Load model from Windows app package
    /// </summary>
    /// <param name="packagePath">Relative path within the app package</param>
    /// <param name="cancellationToken">Cancellation token</param>
    /// <returns>Task that completes when the model is loaded</returns>
    public async Task LoadModelFromPackageAsync(string packagePath, CancellationToken cancellationToken = default)
    {
        if (string.IsNullOrEmpty(packagePath))
            throw new ArgumentException("Package path cannot be null or empty", nameof(packagePath));

        try
        {
#if WINDOWS
            // Try to use Windows Package API
            var installedLocation = global::Windows.ApplicationModel.Package.Current.InstalledLocation;
            var file = await installedLocation.GetFileAsync(packagePath);
            if (file != null)
            {
                await LoadModelAsync(file.Path, cancellationToken);
                return;
            }
#endif
            // Fallback to base implementation
            await LoadModelFromAssetAsync(packagePath, cancellationToken);
        }
        catch (Exception ex)
        {
            throw new InvalidOperationException(
                $"Failed to load model from Windows app package '{packagePath}': {ex.Message}", ex);
        }
    }

    /// <summary>
    ///     Get available execution providers for Windows
    /// </summary>
    /// <returns>List of available execution provider names</returns>
    public static List<string> GetAvailableExecutionProviders()
    {
        var providers = new List<string>
        {
            "CPUExecutionProvider"
        };

        // DirectML is available on Windows 10 version 1903+ with compatible GPU
        var version = Environment.OSVersion.Version;
        if (version.Major >= 10)
        {
            providers.Add("DmlExecutionProvider");
        }

        return providers;
    }

    /// <summary>
    ///     Check if DirectX 12 is available
    /// </summary>
    /// <returns>True if DirectX 12 is likely available</returns>
    public static bool IsDirectX12Available()
    {
        try
        {
            // Basic check for Windows 10+
            var version = Environment.OSVersion.Version;
            return version.Major >= 10;
        }
        catch
        {
            return false;
        }
    }

    /// <summary>
    ///     Get system information for ML inference optimization
    /// </summary>
    /// <returns>Dictionary containing system information</returns>
    public static Dictionary<string, object> GetSystemInfo()
    {
        var info = new Dictionary<string, object>
        {
            ["ProcessorCount"] = Environment.ProcessorCount,
            ["OSVersion"] = Environment.OSVersion.VersionString,
            ["Is64BitProcess"] = Environment.Is64BitProcess,
            ["Is64BitOperatingSystem"] = Environment.Is64BitOperatingSystem,
            ["WorkingSet"] = Environment.WorkingSet,
            ["Architecture"] = RuntimeInformation.ProcessArchitecture.ToString(),
            ["DirectML"] = IsDirectX12Available()
        };

        return info;
    }
}
