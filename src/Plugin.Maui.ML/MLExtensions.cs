using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.DependencyInjection.Extensions;

namespace Plugin.Maui.ML;

/// <summary>
///     Extension methods for registering ML inference services with dependency injection
/// </summary>
public static class MLExtensions
{
    /// <summary>
    ///     Adds ML inference services to the service collection
    /// </summary>
    /// <param name="services">The service collection</param>
    /// <returns>The service collection for chaining</returns>
    public static IServiceCollection AddMauiML(this IServiceCollection services)
    {
        return services.AddMauiML(null);
    }

    /// <summary>
    ///     Adds ML inference services to the service collection with configuration
    /// </summary>
    /// <param name="services">The service collection</param>
    /// <param name="configure">Optional configuration action for ML services</param>
    /// <returns>The service collection for chaining</returns>
    public static IServiceCollection AddMauiML(this IServiceCollection services, Action<MLConfiguration>? configure)
    {
        var config = new MLConfiguration();
        configure?.Invoke(config);

        // Register the ML inference service
        if (config.UseTransientService)
        {
            services.TryAddTransient<IMLInfer, OnnxRuntimeInfer>();
        }
        else
        {
            services.TryAddSingleton<IMLInfer, OnnxRuntimeInfer>();
        }

        // Register configuration
        services.TryAddSingleton(config);

        return services;
    }
}

/// <summary>
///     Configuration options for ML services
/// </summary>
public class MLConfiguration
{
    /// <summary>
    ///     Gets or sets whether to use transient service lifetime (default: false, uses singleton)
    /// </summary>
    public bool UseTransientService { get; set; }

    /// <summary>
    ///     Gets or sets the default model asset path
    /// </summary>
    public string? DefaultModelAssetPath { get; set; }

    /// <summary>
    ///     Gets or sets whether to enable performance logging
    /// </summary>
    public bool EnablePerformanceLogging { get; set; }

    /// <summary>
    ///     Gets or sets the maximum number of concurrent inference operations
    /// </summary>
    public int MaxConcurrentInferences { get; set; } = Environment.ProcessorCount;
}
