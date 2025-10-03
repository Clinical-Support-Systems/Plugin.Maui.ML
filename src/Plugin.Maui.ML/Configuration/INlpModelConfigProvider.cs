namespace Plugin.Maui.ML.Configuration;

/// <summary>
///     Abstraction for supplying NLP model configuration (e.g., from embedded assets, network, or disk)
/// </summary>
public interface INlpModelConfigProvider
{
    /// <summary>
    ///     Asynchronously retrieves the configuration for the specified NLP model.
    /// </summary>
    /// <param name="modelKey">
    ///     The unique key identifying the NLP model whose configuration is to be retrieved. Cannot be null
    ///     or empty.
    /// </param>
    /// <param name="ct">A cancellation token that can be used to cancel the asynchronous operation.</param>
    /// <returns>
    ///     A task that represents the asynchronous operation. The task result contains the <see cref="NlpModelConfig" /> for
    ///     the specified model, or <see langword="null" /> if the model is not found.
    /// </returns>
    Task<NlpModelConfig?> GetConfigAsync(string modelKey, CancellationToken ct = default);
}