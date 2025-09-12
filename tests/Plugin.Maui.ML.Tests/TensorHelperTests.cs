using Microsoft.ML.OnnxRuntime.Tensors;
using Plugin.Maui.ML.Utilities;
using Xunit;

namespace Plugin.Maui.ML.Tests;

public class TensorHelperTests
{
    [Fact]
    public void CreateTensor_FromFloatArray_CreatesTensorCorrectly()
    {
        // Arrange
        var data = new[] { 1.0f, 2.0f, 3.0f, 4.0f };
        var dimensions = new[] { 2, 2 };

        // Act
        var tensor = TensorHelper.CreateTensor(data, dimensions);

        // Assert
        Assert.NotNull(tensor);
        Assert.Equal(dimensions, tensor.Dimensions.ToArray());
        Assert.Equal(data.Length, tensor.Length);
    }

    [Fact]
    public void CreateTensor_From2DArray_CreatesTensorCorrectly()
    {
        // Arrange
        var data = new[,] { { 1.0f, 2.0f }, { 3.0f, 4.0f } };

        // Act
        var tensor = TensorHelper.CreateTensor(data);

        // Assert
        Assert.NotNull(tensor);
        Assert.Equal(new[] { 2, 2 }, tensor.Dimensions.ToArray());
        Assert.Equal(4, tensor.Length);
    }

    [Fact]
    public void ToArray_ConvertsTensorToArray()
    {
        // Arrange
        var data = new[] { 1.0f, 2.0f, 3.0f, 4.0f };
        var tensor = new DenseTensor<float>(data, new[] { 2, 2 });

        // Act
        var result = TensorHelper.ToArray(tensor);

        // Assert
        Assert.Equal(data, result);
    }

    [Fact]
    public void GetShapeString_ReturnsCorrectFormat()
    {
        // Arrange
        var data = new[] { 1.0f, 2.0f, 3.0f, 4.0f };
        var tensor = new DenseTensor<float>(data, new[] { 2, 2 });

        // Act
        var shapeString = TensorHelper.GetShapeString(tensor);

        // Assert
        Assert.Equal("[2, 2]", shapeString);
    }

    [Fact]
    public void Reshape_WithValidDimensions_ReshapesTensor()
    {
        // Arrange
        var data = new[] { 1.0f, 2.0f, 3.0f, 4.0f };
        var tensor = new DenseTensor<float>(data, new[] { 2, 2 });
        var newDimensions = new[] { 4, 1 };

        // Act
        var reshapedTensor = TensorHelper.Reshape(tensor, newDimensions);

        // Assert
        Assert.Equal(newDimensions, reshapedTensor.Dimensions.ToArray());
        Assert.Equal(data, TensorHelper.ToArray(reshapedTensor));
    }

    [Fact]
    public void Reshape_WithInvalidDimensions_ThrowsException()
    {
        // Arrange
        var data = new[] { 1.0f, 2.0f, 3.0f, 4.0f };
        var tensor = new DenseTensor<float>(data, new[] { 2, 2 });
        var newDimensions = new[] { 2, 3 }; // Total size mismatch

        // Act & Assert
        Assert.Throws<ArgumentException>(() => TensorHelper.Reshape(tensor, newDimensions));
    }

    [Fact]
    public void Normalize_NormalizesToZeroOneRange()
    {
        // Arrange
        var data = new[] { 0.0f, 10.0f, 20.0f, 30.0f };
        var tensor = new DenseTensor<float>(data, new[] { 4 });

        // Act
        var normalizedTensor = TensorHelper.Normalize(tensor);
        var normalizedData = TensorHelper.ToArray(normalizedTensor);

        // Assert
        Assert.Equal(0.0f, normalizedData[0], 0.001f);
        Assert.Equal(1.0f, normalizedData[3], 0.001f);
        Assert.True(normalizedData.All(x => x >= 0.0f && x <= 1.0f));
    }

    [Fact]
    public void Normalize_WithSameValues_ReturnsOriginalTensor()
    {
        // Arrange
        var data = new[] { 5.0f, 5.0f, 5.0f, 5.0f };
        var tensor = new DenseTensor<float>(data, new[] { 4 });

        // Act
        var normalizedTensor = TensorHelper.Normalize(tensor);

        // Assert
        Assert.Equal(tensor.Dimensions.ToArray(), normalizedTensor.Dimensions.ToArray());
        // When all values are the same, normalization should return the original values
        Assert.Equal(data, TensorHelper.ToArray(normalizedTensor));
    }

    [Fact]
    public void Softmax_AppliesSoftmaxCorrectly()
    {
        // Arrange
        var data = new[] { 1.0f, 2.0f, 3.0f };
        var tensor = new DenseTensor<float>(data, new[] { 3 });

        // Act
        var softmaxTensor = TensorHelper.Softmax(tensor);
        var softmaxData = TensorHelper.ToArray(softmaxTensor);

        // Assert
        // Softmax values should sum to 1
        var sum = softmaxData.Sum();
        Assert.Equal(1.0f, sum, 0.00001f);

        // All values should be positive
        Assert.True(softmaxData.All(x => x > 0.0f));

        // Higher input values should result in higher softmax values
        Assert.True(softmaxData[2] > softmaxData[1]);
        Assert.True(softmaxData[1] > softmaxData[0]);
    }
}
