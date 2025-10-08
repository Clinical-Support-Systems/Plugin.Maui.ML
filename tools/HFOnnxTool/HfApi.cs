using System.Net.Http.Headers;
using System.Net.Http.Json;
using JetBrains.Annotations;
using Spectre.Console;

namespace HFOnnxTool;

/// <summary>
///     Represents an item in a Hugging Face tree structure, including its path, type, and size.
/// </summary>
/// <param name="Path">
///     The relative path of the item within the tree. This typically indicates the item's location or
///     hierarchy.
/// </param>
/// <param name="Type">
///     The type of the item, such as 'file' or 'directory'. Determines how the item is interpreted within
///     the tree.
/// </param>
/// <param name="Size">The size of the item in bytes. For directories, this value may be zero or undefined.</param>
[UsedImplicitly]
public record HfTreeItem(string Path, string Type, long Size);

/// <summary>
///     Provides static methods for interacting with Hugging Face and GitHub repositories, including normalization of
///     repository identifiers, retrieval of repository file trees, file downloads, and interactive console selection
///     prompts.
/// </summary>
/// <remarks>
///     The methods in this class facilitate common operations when working with Hugging Face repositories,
///     such as accessing repository contents, downloading files, and handling user selection in console applications. All
///     methods are thread-safe and designed for asynchronous usage where applicable. Authentication tokens may be required
///     for accessing private repositories or performing authenticated API requests.
/// </remarks>
public static class HfApi
{
    private static readonly HttpClient Http = new()
    {
        DefaultRequestHeaders =
        {
            { "User-Agent", "HFOnnxTool/1.0" }
        }
    };

    /// <summary>
    ///     Extracts the "org/name" repository identifier from a GitHub repository URL or returns the input string if it is
    ///     already in "org/name" format.
    /// </summary>
    /// <remarks>
    ///     This method supports both full GitHub repository URLs and direct "org/name" identifiers.
    ///     Trailing slashes in URLs are ignored. Only the first two path segments after the domain are used to construct
    ///     the identifier.
    /// </remarks>
    /// <param name="repoOrUrl">
    ///     The GitHub repository URL or the repository identifier in "org/name" format. If a URL is provided, it must
    ///     contain at least the organization and repository segments.
    /// </param>
    /// <returns>A string in the format "org/name" representing the repository identifier.</returns>
    /// <exception cref="ArgumentException">
    ///     Thrown if the provided URL does not contain enough segments to extract the
    ///     organization and repository name.
    /// </exception>
    public static string NormalizeRepo(string? repoOrUrl)
    {
        if (string.IsNullOrWhiteSpace(repoOrUrl))
            throw new ArgumentException("Repository or URL cannot be null or empty.");

        if (!repoOrUrl.StartsWith("http", StringComparison.OrdinalIgnoreCase))
            return repoOrUrl;

        var uri = new Uri(repoOrUrl.TrimEnd('/'));
        var segs = uri.AbsolutePath.Trim('/').Split('/', StringSplitOptions.RemoveEmptyEntries);

        return segs.Length >= 2
            ? $"{segs[0]}/{segs[1]}"
            : throw new ArgumentException("Cannot extract org/name from URL.");
    }

    /// <summary>
    ///     Retrieves the file tree for a specified Hugging Face repository and revision asynchronously.
    /// </summary>
    /// <remarks>
    ///     This method sends a GET request to the Hugging Face API to obtain the repository tree. If
    ///     authentication is required, provide a valid token. The returned list may be empty if the repository or revision
    ///     does not contain any files or folders.
    /// </remarks>
    /// <param name="repo">The name of the Hugging Face repository to query. This should be in the format "owner/repo".</param>
    /// <param name="revision">The revision of the repository to retrieve the tree for. Defaults to "main" if not specified.</param>
    /// <param name="token">
    ///     An optional access token used for authentication. If provided, the request will include the token in the
    ///     authorization header.
    /// </param>
    /// <param name="recursive">
    ///     Specifies whether to retrieve the tree recursively. Set to <see langword="true" /> to include all nested files
    ///     and folders; otherwise, only the top-level items are returned.
    /// </param>
    /// <returns>
    ///     A task that represents the asynchronous operation. The task result contains a list of <see cref="HfTreeItem" />
    ///     objects representing the files and folders in the repository tree. Returns an empty list if no items are found.
    /// </returns>
    /// <exception cref="InvalidOperationException">Thrown if the Hugging Face API responds with an error status code.</exception>
    public static async Task<List<HfTreeItem>> GetTreeAsync(string repo, string revision = "main", string? token = null,
        bool recursive = true)
    {
        var url = $"https://huggingface.co/api/models/{repo}/tree/{revision}?recursive={(recursive ? 1 : 0)}";
        var req = new HttpRequestMessage(HttpMethod.Get, url);
        if (!string.IsNullOrWhiteSpace(token))
            req.Headers.Authorization = new AuthenticationHeaderValue("Bearer", token);
        using var resp = await Http.SendAsync(req);
        if (!resp.IsSuccessStatusCode)
            throw new InvalidOperationException($"HF API error: {(int)resp.StatusCode} {resp.ReasonPhrase}");
        var data = await resp.Content.ReadFromJsonAsync<List<HfTreeItem>>() ??
                   [];
        return data;
    }

    /// <summary>
    ///     Asynchronously downloads a file from a specified Hugging Face repository and saves it to the given destination
    ///     path.
    /// </summary>
    /// <remarks>
    ///     If the destination directory does not exist, it will be created automatically. The method
    ///     overwrites the destination file if it already exists.
    /// </remarks>
    /// <param name="repo">The name of the Hugging Face repository from which to download the file. Cannot be null or empty.</param>
    /// <param name="path">The relative path to the file within the repository to download. Cannot be null or empty.</param>
    /// <param name="revision">
    ///     The revision identifier (such as a branch name, tag, or commit SHA) to resolve the file from. Cannot be null or
    ///     empty.
    /// </param>
    /// <param name="destFile">The local file path where the downloaded file will be saved. Cannot be null or empty.</param>
    /// <param name="token">
    ///     An optional authentication token used for accessing private repositories. If null or empty, the request will be
    ///     unauthenticated.
    /// </param>
    /// <returns>
    ///     A task that represents the asynchronous download operation. The task completes when the file has been
    ///     successfully saved to the specified location.
    /// </returns>
    /// <exception cref="InvalidOperationException">
    ///     Thrown if the download request does not succeed, such as when the file does
    ///     not exist or access is denied.
    /// </exception>
    public static async Task DownloadFileAsync(string repo, string path, string revision, string destFile,
        string? token = null)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(destFile)!);
        var url = $"https://huggingface.co/{repo}/resolve/{revision}/{path}?download=1";
        var req = new HttpRequestMessage(HttpMethod.Get, url);
        if (!string.IsNullOrWhiteSpace(token))
            req.Headers.Authorization = new AuthenticationHeaderValue("Bearer", token);
        using var resp = await Http.SendAsync(req);
        if (!resp.IsSuccessStatusCode)
            throw new InvalidOperationException($"Download failed {url}: {(int)resp.StatusCode}");
        await using var fs = File.Create(destFile);
        await resp.Content.CopyToAsync(fs);
    }

    /// <summary>
    ///     Prompts the user to select an item from the specified list of choices using a console selection interface.
    /// </summary>
    /// <remarks>
    ///     If the list contains only one item, the prompt is skipped and the single item is returned
    ///     automatically. The selection interface displays up to 15 choices per page. This method requires a console
    ///     environment that supports interactive prompts.
    /// </remarks>
    /// <param name="items">The list of string items to present as selectable choices. Must contain at least one item.</param>
    /// <param name="title">The title displayed above the selection prompt to guide the user.</param>
    /// <returns>
    ///     The string value of the item selected by the user. If the list contains only one item, that item is returned
    ///     without prompting.
    /// </returns>
    public static string PickFrom(IReadOnlyList<string> items, string title)
    {
        if (items.Count == 1) return items[0];
        return AnsiConsole.Prompt(
            new SelectionPrompt<string>()
                .Title(title)
                .PageSize(15)
                .AddChoices(items));
    }
}
