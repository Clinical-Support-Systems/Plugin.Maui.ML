using System.Reflection;
using Xunit;

namespace Plugin.Maui.ML.Tests;

/// <summary>
///     Provides a test attribute that conditionally skips a test based on the result of a specified static method.
/// </summary>
/// <remarks>
///     Use this attribute to skip a test when a runtime condition is not met. The condition is determined by
///     invoking a static method on the test class with the specified name; if the method returns <see langword="true" />,
///     the test is skipped. This is useful for scenarios where tests should only run under certain environmental or
///     configuration conditions.
/// </remarks>
public sealed class ConditionalFactAttribute : FactAttribute
{
    public ConditionalFactAttribute(string skipConditionMethodName)
    {
        var type = typeof(OnnxRuntimeInferTests);
        var method = type.GetMethod(skipConditionMethodName,
            BindingFlags.Static | BindingFlags.NonPublic | BindingFlags.Public);

        if (method != null && method.Invoke(null, null) is true)
        {
            Skip = "Condition not met";
        }
    }
}
