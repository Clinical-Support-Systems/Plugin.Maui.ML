using JetBrains.Annotations;
using Spectre.Console;
using Spectre.Console.Cli;

namespace HFOnnxTool;

/// <summary>
///     Represents a command that inspects the contents of a specified Hugging Face repository revision and displays
///     information about its files.
/// </summary>
/// <remarks>
///     This command retrieves the file tree for the given repository and revision, presenting details such
///     as file type, path, and size in a tabular format. It also identifies and reports the number of ONNX files found.
///     The
///     command requires valid repository information and may use an authentication token if provided in the settings or
///     available as an environment variable.
/// </remarks>
[UsedImplicitly]
public class InspectCommand : AsyncCommand<HfBaseSettings>
{
    public async override Task<int> ExecuteAsync(CommandContext context, HfBaseSettings settings)
    {
        var token = settings.Token ?? Environment.GetEnvironmentVariable("HF_TOKEN");
        var repo = HfApi.NormalizeRepo(settings.Repo);
        AnsiConsole.MarkupLine($"[cyan]Inspecting[/] {repo}@{settings.Revision} ...");
        var tree = await HfApi.GetTreeAsync(repo, settings.Revision, token);
        var table = new Table().AddColumns("Type", "Path", "Size");
        foreach (var item in tree.OrderBy(i => i.Path))
        {
            table.AddRow(item.Type, item.Path, item.Size.ToString());
        }

        AnsiConsole.Write(table);
        var onnx = tree.Where(t => t.Type == "file" && t.Path.EndsWith(".onnx", StringComparison.OrdinalIgnoreCase))
            .ToList();
        AnsiConsole.MarkupLine($"[green]Found {onnx.Count} ONNX file(s).[/]");
        return 0;
    }
}
