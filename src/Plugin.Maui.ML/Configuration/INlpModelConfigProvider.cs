namespace Plugin.Maui.ML.Configuration;

/// <summary>
/// Abstraction for supplying NLP model configuration (e.g., from embedded assets, network, or disk)
/// </summary>
public interface INlpModelConfigProvider
{
    Task<NlpModelConfig?> GetConfigAsync(string modelKey, CancellationToken ct = default);
}
