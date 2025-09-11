using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace Plugin.Maui.ML.Platforms.Windows;

/// <summary>
/// Windows-specific ML inference implementation
/// </summary>
public class PlatformMLInfer : OnnxRuntimeInfer
{
    /// <summary>
    /// Initializes a new instance of the PlatformMLInfer class for Windows
    /// </summary>
    public PlatformMLInfer() : base()
    {
        // Windows-specific initialization can be added here
    }

    /// <summary>
    /// Load model from Windows app package
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
            // This would use Windows.ApplicationModel.Package.Current.InstalledLocation
            // For now, fallback to base implementation
            await LoadModelFromAssetAsync(packagePath, cancellationToken);
        }
        catch (Exception ex)
        {
            throw new InvalidOperationException($"Failed to load model from Windows app package '{packagePath}': {ex.Message}", ex);
        }
    }

    /// <summary>
    /// Get available execution providers for Windows
    /// </summary>
    /// <returns>List of available execution provider names</returns>
    public static List<string> GetAvailableExecutionProviders()
    {
        var providers = new List<string> { "CPUExecutionProvider" };
        
        // DirectML is available on Windows 10 version 1903+ with compatible GPU
        var version = Environment.OSVersion.Version;
        if (version.Major >= 10)
        {
            providers.Add("DmlExecutionProvider");
        }

        return providers;
    }

    /// <summary>
    /// Check if DirectX 12 is available
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
    /// Get system information for ML inference optimization
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
            ["Architecture"] = System.Runtime.InteropServices.RuntimeInformation.ProcessArchitecture.ToString()
        };

        return info;
    }
}