using Microsoft.Extensions.DependencyInjection;
using Plugin.Maui.ML;
using Xunit;

namespace Plugin.Maui.ML.Tests;

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
        var config = serviceProvider.GetService<MLConfiguration>();

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
        var config = serviceProvider.GetRequiredService<MLConfiguration>();

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
}