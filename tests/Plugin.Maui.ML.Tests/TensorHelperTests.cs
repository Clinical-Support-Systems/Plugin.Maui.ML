using Microsoft.ML.OnnxRuntime.Tensors;
using Plugin.Maui.ML.Utilities;
using Xunit;

namespace Plugin.Maui.ML.Tests;

/// <summary>
///     Provides unit tests for the TensorHelper class, verifying tensor creation, manipulation, and mathematical
///     operations
///     such as normalization and softmax.
/// </summary>
/// <remarks>
///     These tests ensure that TensorHelper methods behave as expected for various input scenarios,
///     including multidimensional arrays, reshaping, and numerical stability. The tests cover both typical and edge cases
///     to help maintain reliability and correctness of tensor-related functionality.
/// </remarks>
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
        Assert.Equal([2, 2], tensor.Dimensions.ToArray());
        Assert.Equal(4, tensor.Length);
    }

    [Fact]
    public void ToArray_ConvertsTensorToArray()
    {
        // Arrange
        var data = new[] { 1.0f, 2.0f, 3.0f, 4.0f };
        var tensor = new DenseTensor<float>(data, [2, 2]);

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
        var tensor = new DenseTensor<float>(data, [2, 2]);

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
        var tensor = new DenseTensor<float>(data, [2, 2]);
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
        var tensor = new DenseTensor<float>(data, [2, 2]);
        var newDimensions = new[] { 2, 3 }; // Total size mismatch

        // Act & Assert
        Assert.Throws<ArgumentException>(() => TensorHelper.Reshape(tensor, newDimensions));
    }

    [Fact]
    public void Normalize_NormalizesToZeroOneRange()
    {
        // Arrange
        var data = new[] { 0.0f, 10.0f, 20.0f, 30.0f };
        var tensor = new DenseTensor<float>(data, [4]);

        // Act
        var normalizedTensor = TensorHelper.Normalize(tensor);
        var normalizedData = TensorHelper.ToArray(normalizedTensor);

        // Assert
        Assert.Equal(0.0f, normalizedData[0], 0.001f);
        Assert.Equal(1.0f, normalizedData[3], 0.001f);
        Assert.True(normalizedData.All(x => x is >= 0.0f and <= 1.0f));
    }

    [Fact]
    public void Normalize_WithSameValues_ReturnsOriginalTensor()
    {
        // Arrange
        var data = new[] { 5.0f, 5.0f, 5.0f, 5.0f };
        var tensor = new DenseTensor<float>(data, [4]);

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
        var tensor = new DenseTensor<float>(data, [3]);

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

    // Additional comprehensive tests

    [Fact]
    public void CreateTensor_From3DArray_CreatesTensorCorrectly()
    {
        // Arrange
        var data3D = new[,,] { { { 1f, 2f }, { 3f, 4f } }, { { 5f, 6f }, { 7f, 8f } } };

        // Act
        var tensor = TensorHelper.CreateTensor(data3D);
        var flat = TensorHelper.ToArray(tensor);

        // Assert
        Assert.Equal([2, 2, 2], tensor.Dimensions.ToArray());
        Assert.Equal(8, tensor.Length);
        Assert.Equal([1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f], flat);
    }

    [Fact]
    public void CreateTensor_From2DArray_VerifyFlattenOrder()
    {
        // Arrange (2 x 3)
        var data2D = new[,] { { 1f, 2f, 3f }, { 4f, 5f, 6f } };

        // Act
        var tensor = TensorHelper.CreateTensor(data2D);
        var flat = TensorHelper.ToArray(tensor);

        // Assert
        Assert.Equal([2, 3], tensor.Dimensions.ToArray());
        Assert.Equal([1f, 2f, 3f, 4f, 5f, 6f], flat);
    }

    [Fact]
    public void GetShapeString_For3DTensor()
    {
        // Arrange dims 2,2,1
        var data = new[,,] { { { 0f }, { 1f } }, { { 2f }, { 3f } } };
        var tensor = TensorHelper.CreateTensor(data);

        // Act
        var shape = TensorHelper.GetShapeString(tensor);

        // Assert
        Assert.Equal("[2, 2, 1]", shape);
    }

    [Fact]
    public void Softmax_NumericalStability_LargeValues()
    {
        // Arrange
        var data = new[] { 1000f, 1001f, 1002f };
        var tensor = new DenseTensor<float>(data, [data.Length]);

        // Act
        var softmaxTensor = TensorHelper.Softmax(tensor);
        var softmaxData = TensorHelper.ToArray(softmaxTensor);

        // Assert
        var sum = softmaxData.Sum();
        Assert.Equal(1f, sum, 1e-5);
        Assert.DoesNotContain(softmaxData, float.IsNaN);
        Assert.DoesNotContain(softmaxData, float.IsInfinity);
        Assert.True(softmaxData[2] > softmaxData[1]);
        Assert.True(softmaxData[1] > softmaxData[0]);
    }

    [Fact]
    public void Normalize_PreservesOrderingAndMapsEndpoints()
    {
        // Arrange (duplicate min value to ensure stable handling)
        var data = new[] { 5.5f, 7.0f, 6.1f, 9.9f, 5.5f };
        var tensor = new DenseTensor<float>(data, [data.Length]);

        // Act
        var normalized = TensorHelper.Normalize(tensor);
        var normalizedData = TensorHelper.ToArray(normalized);

        // Assert endpoints
        var min = data.Min();
        var max = data.Max();
        for (var i = 0; i < data.Length; i++)
        {
            if (NearlyEqual(data[i], min)) Assert.Equal(0f, normalizedData[i], 1e-5);
            if (NearlyEqual(data[i], max)) Assert.Equal(1f, normalizedData[i], 1e-5);
        }

        Assert.All(normalizedData, v => Assert.InRange(v, 0f, 1f));
        // Monotonic ordering for distinct values
        for (var i = 0; i < data.Length; i++)
        {
            for (var j = 0; j < data.Length; j++)
            {
                if (data[i] < data[j])
                {
                    Assert.True(normalizedData[i] <= normalizedData[j]);
                }
            }
        }
    }

    /// <summary>
    ///     Determines whether two single-precision floating-point values are nearly equal within specified relative and
    ///     absolute tolerances.
    /// </summary>
    /// <remarks>
    ///     This method is useful for comparing floating-point values where exact equality is unreliable
    ///     due to rounding errors.
    /// </remarks>
    /// <param name="a">The first value to compare.</param>
    /// <param name="b">The second value to compare.</param>
    /// <param name="rtol">
    ///     The relative tolerance. Specifies the maximum allowed difference between the values, relative to their
    ///     magnitude. Must be non-negative.
    /// </param>
    /// <param name="atol">
    ///     The absolute tolerance. Specifies the minimum threshold for considering the values nearly equal, regardless of
    ///     their magnitude. Must be non-negative.
    /// </param>
    /// <returns>true if the values are nearly equal within the specified tolerances; otherwise, false.</returns>
    private static bool NearlyEqual(float a, float b, float rtol = 1e-4f, float atol = 1e-6f)
    {
        return MathF.Abs(a - b) <= MathF.Max(atol, rtol * MathF.Max(MathF.Abs(a), MathF.Abs(b)));
    }

    [Fact]
    public void Reshape_SameDimensions_ReturnsEquivalentTensor()
    {
        // Arrange
        var data = new[] { 2f, 4f, 6f, 8f };
        var original = new DenseTensor<float>(data, [2, 2]);

        // Act
        var reshaped = TensorHelper.Reshape(original, [2, 2]);

        // Assert
        Assert.NotSame(original, reshaped); // New tensor instance
        Assert.Equal(original.Dimensions.ToArray(), reshaped.Dimensions.ToArray());
        Assert.Equal(TensorHelper.ToArray(original), TensorHelper.ToArray(reshaped));
    }
}
