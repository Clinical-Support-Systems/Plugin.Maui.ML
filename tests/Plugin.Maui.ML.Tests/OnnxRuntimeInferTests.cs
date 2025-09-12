using Microsoft.ML.OnnxRuntime.Tensors;
using Xunit;

namespace Plugin.Maui.ML.Tests;

public class OnnxRuntimeInferTests
{
    [Fact]
    public void Constructor_InitializesCorrectly()
    {
        // Act
        using var infer = new OnnxRuntimeInfer();

        // Assert
        Assert.NotNull(infer);
        Assert.False(infer.IsModelLoaded);
    }

    [Fact]
    public async Task LoadModelAsync_WithNullPath_ThrowsArgumentException()
    {
        // Arrange
        using var infer = new OnnxRuntimeInfer();

        // Act & Assert
        await Assert.ThrowsAsync<ArgumentException>(() => infer.LoadModelAsync((string)null!));
    }

    [Fact]
    public async Task LoadModelAsync_WithEmptyPath_ThrowsArgumentException()
    {
        // Arrange
        using var infer = new OnnxRuntimeInfer();

        // Act & Assert
        await Assert.ThrowsAsync<ArgumentException>(() => infer.LoadModelAsync(string.Empty));
    }

    [Fact]
    public async Task LoadModelAsync_WithNonExistentFile_ThrowsFileNotFoundException()
    {
        // Arrange
        using var infer = new OnnxRuntimeInfer();
        var nonExistentPath = "non-existent-model.onnx";

        // Act & Assert
        await Assert.ThrowsAsync<FileNotFoundException>(() => infer.LoadModelAsync(nonExistentPath));
    }

    [Fact]
    public async Task LoadModelAsync_WithNullStream_ThrowsArgumentNullException()
    {
        // Arrange
        using var infer = new OnnxRuntimeInfer();

        // Act & Assert
        await Assert.ThrowsAsync<ArgumentNullException>(() => infer.LoadModelAsync((Stream)null!));
    }

    [Fact]
    public async Task LoadModelFromAssetAsync_WithNullAssetName_ThrowsArgumentException()
    {
        // Arrange
        using var infer = new OnnxRuntimeInfer();

        // Act & Assert
        await Assert.ThrowsAsync<ArgumentException>(() => infer.LoadModelFromAssetAsync(null!));
    }

    [Fact]
    public async Task LoadModelFromAssetAsync_WithEmptyAssetName_ThrowsArgumentException()
    {
        // Arrange
        using var infer = new OnnxRuntimeInfer();

        // Act & Assert
        await Assert.ThrowsAsync<ArgumentException>(() => infer.LoadModelFromAssetAsync(string.Empty));
    }

    [Fact]
    public async Task RunInferenceAsync_WithoutLoadedModel_ThrowsInvalidOperationException()
    {
        // Arrange
        using var infer = new OnnxRuntimeInfer();
        var inputs = new Dictionary<string, Tensor<float>>
        {
            ["input"] = new DenseTensor<float>(new[] { 1.0f }, new[] { 1 })
        };

        // Act & Assert
        await Assert.ThrowsAsync<InvalidOperationException>(() => infer.RunInferenceAsync(inputs));
    }

    [Fact]
    public async Task RunInferenceAsync_WithNullInputs_ThrowsArgumentException()
    {
        // Arrange
        using var infer = new OnnxRuntimeInfer();

        // Act & Assert
        await Assert.ThrowsAsync<ArgumentException>(() => infer.RunInferenceAsync(null!));
    }

    [Fact]
    public async Task RunInferenceAsync_WithEmptyInputs_ThrowsArgumentException()
    {
        // Arrange
        using var infer = new OnnxRuntimeInfer();
        var inputs = new Dictionary<string, Tensor<float>>();

        // Act & Assert
        await Assert.ThrowsAsync<ArgumentException>(() => infer.RunInferenceAsync(inputs));
    }

    [Fact]
    public void GetInputMetadata_WithoutLoadedModel_ThrowsInvalidOperationException()
    {
        // Arrange
        using var infer = new OnnxRuntimeInfer();

        // Act & Assert
        Assert.Throws<InvalidOperationException>(() => infer.GetInputMetadata());
    }

    [Fact]
    public void GetOutputMetadata_WithoutLoadedModel_ThrowsInvalidOperationException()
    {
        // Arrange
        using var infer = new OnnxRuntimeInfer();

        // Act & Assert
        Assert.Throws<InvalidOperationException>(() => infer.GetOutputMetadata());
    }

    [Fact]
    public void UnloadModel_WithoutLoadedModel_DoesNotThrow()
    {
        // Arrange
        using var infer = new OnnxRuntimeInfer();

        // Act & Assert (should not throw)
        infer.UnloadModel();
        Assert.False(infer.IsModelLoaded);
    }

    [Fact]
    public void Dispose_DisposesCorrectly()
    {
        // Arrange
        var infer = new OnnxRuntimeInfer();

        // Act
        infer.Dispose();

        // Assert - should not throw
        Assert.False(infer.IsModelLoaded);
    }

    [Fact]
    public void Dispose_MultipleCalls_DoesNotThrow()
    {
        // Arrange
        var infer = new OnnxRuntimeInfer();

        // Act & Assert - multiple dispose calls should not throw
        infer.Dispose();
        infer.Dispose();
        infer.Dispose();
    }
}
