using Microsoft.ML.OnnxRuntime.Tensors;

namespace Plugin.Maui.ML.Utilities;

/// <summary>
/// Helper utilities for working with tensors
/// </summary>
public static class TensorHelper
{
    /// <summary>
    /// Create a tensor from a float array
    /// </summary>
    /// <param name="data">The float data</param>
    /// <param name="dimensions">The tensor dimensions</param>
    /// <returns>A new tensor</returns>
    public static Tensor<float> CreateTensor(float[] data, int[] dimensions)
    {
        return new DenseTensor<float>(data, dimensions);
    }

    /// <summary>
    /// Create a tensor from a 2D float array
    /// </summary>
    /// <param name="data">The 2D float data</param>
    /// <returns>A new tensor</returns>
    public static Tensor<float> CreateTensor(float[,] data)
    {
        var dimensions = new[] { data.GetLength(0), data.GetLength(1) };
        var flatData = new float[data.Length];
        
        for (int i = 0; i < data.GetLength(0); i++)
        {
            for (int j = 0; j < data.GetLength(1); j++)
            {
                flatData[i * data.GetLength(1) + j] = data[i, j];
            }
        }
        
        return new DenseTensor<float>(flatData, dimensions);
    }

    /// <summary>
    /// Create a tensor from a 3D float array
    /// </summary>
    /// <param name="data">The 3D float data</param>
    /// <returns>A new tensor</returns>
    public static Tensor<float> CreateTensor(float[,,] data)
    {
        var dimensions = new[] { data.GetLength(0), data.GetLength(1), data.GetLength(2) };
        var flatData = new float[data.Length];
        
        for (int i = 0; i < data.GetLength(0); i++)
        {
            for (int j = 0; j < data.GetLength(1); j++)
            {
                for (int k = 0; k < data.GetLength(2); k++)
                {
                    flatData[i * data.GetLength(1) * data.GetLength(2) + j * data.GetLength(2) + k] = data[i, j, k];
                }
            }
        }
        
        return new DenseTensor<float>(flatData, dimensions);
    }

    /// <summary>
    /// Convert tensor to float array
    /// </summary>
    /// <param name="tensor">The tensor to convert</param>
    /// <returns>Float array representation</returns>
    public static float[] ToArray(Tensor<float> tensor)
    {
        if (tensor is DenseTensor<float> denseTensor)
        {
            return denseTensor.Buffer.ToArray();
        }
        
        var result = new float[tensor.Length];
        for (int i = 0; i < tensor.Length; i++)
        {
            result[i] = tensor.GetValue(i);
        }
        
        return result;
    }

    /// <summary>
    /// Get tensor shape as a string
    /// </summary>
    /// <param name="tensor">The tensor</param>
    /// <returns>Shape string representation</returns>
    public static string GetShapeString(Tensor<float> tensor)
    {
        return $"[{string.Join(", ", tensor.Dimensions.ToArray())}]";
    }

    /// <summary>
    /// Reshape a tensor to new dimensions
    /// </summary>
    /// <param name="tensor">The original tensor</param>
    /// <param name="newDimensions">The new dimensions</param>
    /// <returns>Reshaped tensor</returns>
    public static Tensor<float> Reshape(Tensor<float> tensor, int[] newDimensions)
    {
        var data = ToArray(tensor);
        
        // Verify total size matches
        var originalSize = 1;
        foreach (var dim in tensor.Dimensions)
        {
            originalSize *= dim;
        }
        
        var newSize = 1;
        foreach (var dim in newDimensions)
        {
            newSize *= dim;
        }
        
        if (originalSize != newSize)
        {
            throw new ArgumentException($"Cannot reshape tensor from size {originalSize} to size {newSize}");
        }
        
        return new DenseTensor<float>(data, newDimensions);
    }

    /// <summary>
    /// Normalize tensor values to 0-1 range
    /// </summary>
    /// <param name="tensor">The tensor to normalize</param>
    /// <returns>Normalized tensor</returns>
    public static Tensor<float> Normalize(Tensor<float> tensor)
    {
        var data = ToArray(tensor);
        var min = data.Min();
        var max = data.Max();
        var range = max - min;
        
        if (range == 0)
        {
            return tensor; // All values are the same
        }
        
        var normalizedData = data.Select(x => (x - min) / range).ToArray();
        return new DenseTensor<float>(normalizedData, tensor.Dimensions.ToArray());
    }

    /// <summary>
    /// Apply softmax function to tensor
    /// </summary>
    /// <param name="tensor">The tensor</param>
    /// <returns>Tensor with softmax applied</returns>
    public static Tensor<float> Softmax(Tensor<float> tensor)
    {
        var data = ToArray(tensor);
        var maxValue = data.Max();
        
        // Subtract max for numerical stability
        var expValues = data.Select(x => Math.Exp(x - maxValue)).ToArray();
        var sum = expValues.Sum();
        
        var softmaxData = expValues.Select(x => (float)(x / sum)).ToArray();
        return new DenseTensor<float>(softmaxData, tensor.Dimensions.ToArray());
    }
}