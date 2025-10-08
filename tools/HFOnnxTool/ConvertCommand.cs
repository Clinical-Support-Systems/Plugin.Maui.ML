using System.ComponentModel;
using System.Diagnostics;
using JetBrains.Annotations;
using Spectre.Console;
using Spectre.Console.Cli;

namespace HFOnnxTool;

/// <summary>
///     Provides configuration settings for converting models to ONNX format using the command-line interface.
/// </summary>
/// <remarks>
///     This class encapsulates options for specifying the conversion task, output directory, opset version,
///     Python executable, and additional behaviors such as skipping existing exports or copying output for MAUI
///     applications. It is typically used to supply arguments to the conversion process when exporting models.
/// </remarks>
[UsedImplicitly]
public class ConvertSettings : HfBaseSettings
{
    [CommandOption("--task <TASK>")]
    [Description("Task (e.g. token-classification, text-classification, question-answering).")]
    public string Task { get; set; } = "token-classification";

    [CommandOption("--output <DIR>")]
    [Description("Output directory for ONNX export (default ./onnx-out)")]
    public string OutputDir { get; set; } = "./onnx-out";

    [CommandOption("--opset <N>")]
    [Description("Opset version (default 17)")]
    public int Opset { get; set; } = 17;

    [CommandOption("--python <PATH>")]
    [Description("Python executable (default 'python' on PATH).")]
    public string PythonExe { get; set; } = "python";

    [CommandOption("--skip-existing")]
    [Description("Skip if directory already has model.onnx")]
    public bool SkipExisting { get; set; }

    [CommandOption("--maui-raw <DIR>")]
    [Description("Optional copy of final *.onnx into MAUI Resources/Raw.")]
    public string? MauiRawDir { get; set; }

    [CommandOption("--no-precheck")]
    [Description("Skip python dependency import pre-check.")]
    public bool NoPrecheck { get; set; }
}

/// <summary>
///     Represents a command that exports a Hugging Face model to ONNX format using Python tooling and specified settings.
/// </summary>
/// <remarks>
///     This command validates the provided repository and required Python packages before attempting export.
///     It supports skipping the export if ONNX files already exist and can copy the main ONNX model to a specified
///     directory for MAUI Raw usage. The command outputs progress and error information to the console. It is typically
///     used in automation or CLI scenarios where model conversion is required.
/// </remarks>
[UsedImplicitly]
public class ConvertCommand : AsyncCommand<ConvertSettings>
{
    public async override Task<int> ExecuteAsync(CommandContext context, ConvertSettings s)
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

        var repo = HfApi.NormalizeRepo(s.Repo);
        if (s.SkipExisting &&
            Directory.Exists(s.OutputDir) &&
            Directory.GetFiles(s.OutputDir, "*.onnx").Length != 0)
        {
            AnsiConsole.MarkupLine("[yellow]ONNX already exists. Skipping.[/]");
            return 0;
        }

        if (!s.NoPrecheck)
        {
            var missing = await PreflightAsync(s.PythonExe, ["onnx", "onnxruntime", "transformers", "optimum"]);
            if (missing.Count > 0)
            {
                AnsiConsole.MarkupLine("[red]Missing python packages:[/] " + string.Join(", ", missing));
                AnsiConsole.MarkupLine(
                    "Install with: [grey]pip install \"optimum[exporters]\" onnx onnxruntime transformers[/]");
                return 3;
            }
        }

        Directory.CreateDirectory(s.OutputDir);

        var args =
            $" -m optimum.exporters.onnx --model {repo} --task {s.Task} --opset {s.Opset} {Quote(s.OutputDir)}";
        AnsiConsole.MarkupLine(
            $"[cyan]Executing:[/] {s.PythonExe} -m optimum.exporters.onnx --model {repo} --task {s.Task}");

        var psi = new ProcessStartInfo
        {
            FileName = s.PythonExe,
            Arguments = args,
            RedirectStandardError = true,
            RedirectStandardOutput = true,
            UseShellExecute = false
        };

        var proc = Process.Start(psi) ?? throw new InvalidOperationException("Failed to start python process.");
        proc.OutputDataReceived += (_, e) =>
        {
            if (e.Data != null) SafePlainOut(e.Data, false);
        };
        proc.ErrorDataReceived += (_, e) =>
        {
            if (e.Data != null) SafePlainOut(e.Data, true);
        };
        proc.BeginOutputReadLine();
        proc.BeginErrorReadLine();
        await proc.WaitForExitAsync();

        if (proc.ExitCode != 0)
        {
            AnsiConsole.MarkupLine(
                $"[red]Export failed (exit {proc.ExitCode}). Ensure required python packages are installed.[/]");
            return proc.ExitCode;
        }


        var onnxFiles = Directory.GetFiles(s.OutputDir, "*.onnx", SearchOption.AllDirectories);
        if (onnxFiles.Length == 0)
        {
            AnsiConsole.MarkupLine("[red]No ONNX files produced.[/]");
            return 2;
        }

        AnsiConsole.MarkupLine($"[green]Export complete. Found {onnxFiles.Length} ONNX file(s).[/]");
        foreach (var f in onnxFiles)
        {
            AnsiConsole.WriteLine(" - " + f);
        }

        if (string.IsNullOrWhiteSpace(s.MauiRawDir))
        {
            return 0;
        }

        Directory.CreateDirectory(s.MauiRawDir);
        var primary = onnxFiles.OrderByDescending(f => new FileInfo(f).Length).First();
        var dest = Path.Combine(s.MauiRawDir, Path.GetFileName(primary));
        File.Copy(primary, dest, true);
        AnsiConsole.MarkupLine($"[yellow]Copied main model to MAUI Raw:[/] {dest}");

        return 0;
    }

    private static string Quote(string path)
    {
        return path.Contains(' ') ? $"\"{path}\"" : path;
    }

    private static void SafePlainOut(string line, bool isErr)
    {
        // Avoid Spectre markup parsing by writing plain text and stripping unprintable chars
        var sanitized = line.Replace("\e", ""); // strip ESC if any
        if (isErr)
        {
            AnsiConsole.WriteLine("[red]" + sanitized + "[/]");
        }
        else
        {
            AnsiConsole.WriteLine(sanitized);
        }
    }

    private static async Task<List<string>> PreflightAsync(string pythonExe, IEnumerable<string> modules)
    {
        var missing = new List<string>();
        foreach (var m in modules)
        {
            var psi = new ProcessStartInfo
            {
                FileName = pythonExe,
                Arguments = $"-c \"import {m}\"",
                RedirectStandardError = true,
                RedirectStandardOutput = true,
                UseShellExecute = false
            };
            try
            {
                using var p = Process.Start(psi)!;
                await p.WaitForExitAsync();
                if (p.ExitCode != 0)
                    missing.Add(m);
            }
            catch
            {
                missing.Add(m);
            }
        }

        return missing;
    }
}
