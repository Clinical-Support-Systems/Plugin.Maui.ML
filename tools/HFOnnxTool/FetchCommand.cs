using System.ComponentModel;
using JetBrains.Annotations;
using Spectre.Console;
using Spectre.Console.Cli;

namespace HFOnnxTool;

/// <summary>
///     Provides configuration settings for controlling the behavior of a fetch operation, including output directory, file
///     selection, and optional resource copying.
/// </summary>
/// <remarks>
///     Use this class to specify options when performing a fetch, such as where downloaded files are stored,
///     whether to automatically select the first ONNX file if multiple are present, and an optional destination for MAUI
///     raw resources. Inherit from HfBaseSettings to include base configuration options.
/// </remarks>
[UsedImplicitly]
public class FetchSettings : HfBaseSettings
{
    [CommandOption("--output <DIR>")]
    [Description("Destination directory (default ./downloaded)")]
    public string OutputDir { get; set; } = "./downloaded";

    [CommandOption("--pick-first")]
    [Description("Auto-pick first ONNX if multiple.")]
    public bool PickFirst { get; set; }

    [CommandOption("--raw-dir <DIR>")]
    [Description("Optional MAUI Resources/Raw copy destination.")]
    public string? MauiRawDir { get; set; }
}

/// <summary>
///     Represents an asynchronous command that downloads ONNX files from a specified repository using the provided fetch
///     settings.
/// </summary>
/// <remarks>
///     This command locates ONNX files in the target repository and downloads the selected file to the
///     specified output directory. If multiple ONNX files are present, the user may be prompted to select one unless the
///     settings specify to pick the first automatically. Optionally, the downloaded file can be copied to a MAUI Raw
///     directory if configured in the settings. The command returns a nonzero exit code if no ONNX files are found. This
///     type is typically used in command-line scenarios for model management workflows.
/// </remarks>
[UsedImplicitly]
public class FetchCommand : AsyncCommand<FetchSettings>
{
    public async override Task<int> ExecuteAsync(CommandContext context, FetchSettings s)
    {
        if (string.IsNullOrEmpty(s.Repo))
        {
            AnsiConsole.MarkupLine("[red]Error: --repo is required.[/]");
            return 1;
        }

        // Also error if it's not a proper url
        if (s.Repo.StartsWith("http", StringComparison.OrdinalIgnoreCase))
        {
            if (!Uri.TryCreate(s.Repo, UriKind.Absolute, out var uri) ||
                uri.Scheme != Uri.UriSchemeHttp && uri.Scheme != Uri.UriSchemeHttps ||
                uri.Segments.Length < 3)
            {
                AnsiConsole.MarkupLine("[red]Error: --repo must be a valid URL or org/name.[/]");
                return 1;
            }
        }

        AnsiConsole.MarkupLine($"[blue]Fetching ONNX from[/] {s.Repo} [blue]revision[/] {s.Revision}");

        var token = s.Token ?? Environment.GetEnvironmentVariable("HF_TOKEN");
        var repo = HfApi.NormalizeRepo(s.Repo);
        var tree = await HfApi.GetTreeAsync(repo, s.Revision, token);

        var onnx = tree.Where(t => t.Type == "file" && t.Path.EndsWith(".onnx", StringComparison.OrdinalIgnoreCase))
            .Select(t => t.Path).OrderBy(p => p).ToList();
        if (onnx.Count == 0)
        {
            AnsiConsole.MarkupLine("[red]No ONNX files present. Use 'convert' command to export.[/]");
            return 1;
        }

        var chosen = onnx.Count == 1 || s.PickFirst
            ? onnx[0]
            : HfApi.PickFrom(onnx, "Select ONNX file");

        Directory.CreateDirectory(s.OutputDir);
        var dest = Path.Combine(s.OutputDir, Path.GetFileName(chosen));
        AnsiConsole.MarkupLine($"[green]Downloading[/] {chosen} -> {dest}");
        await HfApi.DownloadFileAsync(repo, chosen, s.Revision, dest, token);

        if (!string.IsNullOrWhiteSpace(s.MauiRawDir))
        {
            Directory.CreateDirectory(s.MauiRawDir);
            var copyDest = Path.Combine(s.MauiRawDir, Path.GetFileName(chosen));
            File.Copy(dest, copyDest, true);
            AnsiConsole.MarkupLine($"[yellow]Copied to MAUI Raw:[/] {copyDest}");
        }

        AnsiConsole.MarkupLine("[green]Done.[/]");
        return 0;
    }
}
