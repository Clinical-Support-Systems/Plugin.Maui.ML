using JetBrains.Annotations;
using Plugin.Maui.ML.Configuration;
using Xunit;

namespace Plugin.Maui.ML.Tests;

/// <summary>
///     Contains unit tests for extension methods that register machine learning services and NLP model configuration
///     providers in a dependency injection container.
/// </summary>
/// <remarks>
///     These tests verify the correct registration and configuration of ML-related services, including
///     singleton and transient lifetimes, configuration options, and support for multiple NLP model config providers. The
///     class is intended for validating the integration of MauiML and NLP model config provider extensions with the
///     dependency injection system.
/// </remarks>
public class MLExtensionsTests
{
    [Fact]
    public void AddMauiML_RegistersServices()
    {
        // Arrange
        var services = new ServiceCollection();

        // Act
        services.AddMauiML();

        // Assert
        var serviceProvider = services.BuildServiceProvider();
        var mlService = serviceProvider.GetService<IMLInfer>();
        var config = serviceProvider.GetService<MLExtensions.MLConfiguration>();

        Assert.NotNull(mlService);
        Assert.NotNull(config);
        Assert.IsType<OnnxRuntimeInfer>(mlService);
    }

    [Fact]
    public void AddMauiML_WithConfiguration_AppliesConfiguration()
    {
        // Arrange
        var services = new ServiceCollection();

        // Act
        services.AddMauiML(config =>
        {
            config.UseTransientService = true;
            config.EnablePerformanceLogging = true;
            config.MaxConcurrentInferences = 2;
            config.DefaultModelAssetPath = "test-model.onnx";
        });

        // Assert
        var serviceProvider = services.BuildServiceProvider();
        var config = serviceProvider.GetRequiredService<MLExtensions.MLConfiguration>();

        Assert.True(config.UseTransientService);
        Assert.True(config.EnablePerformanceLogging);
        Assert.Equal(2, config.MaxConcurrentInferences);
        Assert.Equal("test-model.onnx", config.DefaultModelAssetPath);
    }

    [Fact]
    public void AddMauiML_WithTransientConfiguration_RegistersTransientService()
    {
        // Arrange
        var services = new ServiceCollection();

        // Act
        services.AddMauiML(config =>
        {
            config.UseTransientService = true;
        });

        // Assert
        var serviceProvider = services.BuildServiceProvider();
        var mlService1 = serviceProvider.GetService<IMLInfer>();
        var mlService2 = serviceProvider.GetService<IMLInfer>();

        Assert.NotNull(mlService1);
        Assert.NotNull(mlService2);
        // For transient services, each instance should be different
        Assert.NotSame(mlService1, mlService2);
    }

    [Fact]
    public void AddMauiML_WithSingletonConfiguration_RegistersSingletonService()
    {
        // Arrange
        var services = new ServiceCollection();

        // Act
        services.AddMauiML(config =>
        {
            config.UseTransientService = false;
        });

        // Assert
        var serviceProvider = services.BuildServiceProvider();
        var mlService1 = serviceProvider.GetService<IMLInfer>();
        var mlService2 = serviceProvider.GetService<IMLInfer>();

        Assert.NotNull(mlService1);
        Assert.NotNull(mlService2);
        // For singleton services, instances should be the same
        Assert.Same(mlService1, mlService2);
    }

    // NEW COVERAGE BELOW

    [Fact]
    public void AddMauiMl_RegistersSingletonOnnxRuntimeInfer()
    {
        var services = new ServiceCollection();
        services.AddMauiMl();
        var sp = services.BuildServiceProvider();
        var a = sp.GetRequiredService<IMLInfer>();
        var b = sp.GetRequiredService<IMLInfer>();
        Assert.Same(a, b); // singleton
        Assert.IsType<OnnxRuntimeInfer>(a);
    }

    [Fact]
    public void AddNlpModelConfigProvider_RegistersSingletonAndReturnsServices()
    {
        var services = new ServiceCollection();
        var returned = services.AddNlpModelConfigProvider<TestConfigProviderA>();
        Assert.Same(services, returned); // fluent
        var sp = services.BuildServiceProvider();
        var provider1 = sp.GetRequiredService<INlpModelConfigProvider>();
        var provider2 = sp.GetRequiredService<INlpModelConfigProvider>();
        Assert.Same(provider1, provider2); // singleton
        Assert.IsType<TestConfigProviderA>(provider1);
    }

    [Fact]
    public void AddNlpModelConfigProvider_MultipleRegistrations_LastWinsForSingleButAllEnumerated()
    {
        var services = new ServiceCollection();
        services.AddNlpModelConfigProvider<TestConfigProviderA>();
        services.AddNlpModelConfigProvider<TestConfigProviderB>();
        var sp = services.BuildServiceProvider();
        var single = sp.GetRequiredService<INlpModelConfigProvider>();
        Assert.IsType<TestConfigProviderB>(single); // last wins
        var all = sp.GetServices<INlpModelConfigProvider>().ToList();
        Assert.Equal(2, all.Count);
        Assert.IsType<TestConfigProviderA>(all[0]);
        Assert.IsType<TestConfigProviderB>(all[1]);
        Assert.Same(all[1], single);
    }

    /// <summary>
    ///     Provides configuration data for NLP models of type "A".
    /// </summary>
    /// <remarks>
    ///     This class implements <see cref="INlpModelConfigProvider" /> to supply model configuration
    ///     information specific to model type "A". It is intended for use in testing scenarios where a fixed configuration
    ///     is required.
    /// </remarks>
    [UsedImplicitly]
    private sealed class TestConfigProviderA : INlpModelConfigProvider
    {
        public Task<NlpModelConfig?> GetConfigAsync(string modelKey, CancellationToken ct = default)
        {
            return Task.FromResult<NlpModelConfig?>(new NlpModelConfig
            {
                ModelType = "A"
            });
        }
    }

    /// <summary>
    ///     Provides configuration data for NLP models of type "B".
    /// </summary>
    [UsedImplicitly]
    private sealed class TestConfigProviderB : INlpModelConfigProvider
    {
        public Task<NlpModelConfig?> GetConfigAsync(string modelKey, CancellationToken ct = default)
        {
            return Task.FromResult<NlpModelConfig?>(new NlpModelConfig
            {
                ModelType = "B"
            });
        }
    }
}
