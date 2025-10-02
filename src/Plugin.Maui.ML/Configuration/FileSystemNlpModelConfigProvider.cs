namespace Plugin.Maui.ML.Configuration;

/// <summary>
/// Loads configuration from an absolute or base-directory relative path.
/// </summary>
public sealed class FileSystemNlpModelConfigProvider : INlpModelConfigProvider
{
    private readonly string _directory;

    public FileSystemNlpModelConfigProvider(string? directory = null)
    {
        _directory = string.IsNullOrWhiteSpace(directory) ? AppDomain.CurrentDomain.BaseDirectory : directory!;
    }

    public async Task<NlpModelConfig?> GetConfigAsync(string modelKey, CancellationToken ct = default)
    {
        if (string.IsNullOrWhiteSpace(modelKey)) return null;
        var path = Path.Combine(_directory, modelKey + ".config.json");
        if (!File.Exists(path)) return null;
        await using var fs = File.OpenRead(path);
        return await NlpModelConfig.LoadAsync(fs, ct);
    }
}
