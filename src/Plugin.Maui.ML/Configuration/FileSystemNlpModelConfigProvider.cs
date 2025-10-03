namespace Plugin.Maui.ML.Configuration;

/// <summary>
///     Loads configuration from an absolute or base-directory relative path.
/// </summary>
public sealed class FileSystemNlpModelConfigProvider : INlpModelConfigProvider
{
    private readonly string _directory;

    /// <summary>
    ///     Initializes a new instance of the FileSystemNlpModelConfigProvider class using the specified directory for model
    ///     configuration files.
    /// </summary>
    /// <param name="directory">
    ///     The path to the directory containing NLP model configuration files. If null, empty, or whitespace, the
    ///     application's base directory is used.
    /// </param>
    public FileSystemNlpModelConfigProvider(string? directory = null)
    {
        _directory = string.IsNullOrWhiteSpace(directory) ? AppDomain.CurrentDomain.BaseDirectory : directory;
    }

    /// <summary>
    ///     Asynchronously retrieves the configuration for the specified NLP model.
    /// </summary>
    /// <param name="modelKey">
    ///     The key identifying the NLP model whose configuration is to be loaded. Cannot be null, empty, or consist only of
    ///     whitespace.
    /// </param>
    /// <param name="ct">A cancellation token that can be used to cancel the asynchronous operation.</param>
    /// <returns>
    ///     A <see cref="NlpModelConfig" /> instance containing the model's configuration if found; otherwise,
    ///     <see
    ///         langword="null" />
    ///     .
    /// </returns>
    public async Task<NlpModelConfig?> GetConfigAsync(string modelKey, CancellationToken ct = default)
    {
        if (string.IsNullOrWhiteSpace(modelKey)) return null;
        var path = Path.Combine(_directory, modelKey + ".config.json");
        if (!File.Exists(path)) return null;
        await using var fs = File.OpenRead(path);
        return await NlpModelConfig.LoadAsync(fs, ct);
    }
}