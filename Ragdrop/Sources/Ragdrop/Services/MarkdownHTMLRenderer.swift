import Foundation

struct PreviewDocument: Sendable {
    let markdown: String
    let html: String
}

enum MarkdownPreviewError: LocalizedError {
    case unreadable
    case conversionFailed(String)

    var errorDescription: String? {
        switch self {
        case .unreadable: "The converter produced unreadable Markdown."
        case .conversionFailed(let details): "Markdown rendering failed. \(details)"
        }
    }
}

enum MarkdownHTMLRenderer {
    static func render(text: String) async throws -> PreviewDocument {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("ragdrop-review-\(UUID().uuidString).md")
        try await Task.detached { try text.write(to: url, atomically: true, encoding: .utf8) }.value
        defer { try? FileManager.default.removeItem(at: url) }
        return try await render(url)
    }

    static func render(_ markdownURL: URL) async throws -> PreviewDocument {
        let task = Task.detached(priority: .userInitiated) {
            guard let markdown = try? String(contentsOf: markdownURL, encoding: .utf8) else {
                throw MarkdownPreviewError.unreadable
            }

            if LibraryLocation.resolve() == .local, let connection = LocalEngine.current.connection {
                let result = try await ProcessRunner.run(executable: connection.python,
                    arguments: ["-c", "import pypandoc,sys; print(pypandoc.convert_file(sys.argv[1], 'html5', format='gfm+tex_math_dollars', extra_args=['--mathml', '--wrap=none']))", markdownURL.path],
                    label: "Markdown rendering", timeout: 30, environment: LocalEngine.cleanEnvironment())
                return PreviewDocument(markdown: markdown, html: wrapHTML(result.output))
            }
            let candidates = ["/opt/homebrew/bin/pandoc", "/usr/local/bin/pandoc"]
            guard let pandoc = candidates.first(where: FileManager.default.fileExists(atPath:)) else {
                return PreviewDocument(markdown: markdown, html: wrapHTML("<pre>\(escapeHTML(markdown))</pre>"))
            }

            let temporary = FileManager.default.temporaryDirectory
            let outputURL = temporary.appendingPathComponent("ragdrop-preview-\(UUID().uuidString).html")
            let errorURL = temporary.appendingPathComponent("ragdrop-preview-\(UUID().uuidString).err")
            FileManager.default.createFile(atPath: errorURL.path, contents: nil)
            defer {
                try? FileManager.default.removeItem(at: outputURL)
                try? FileManager.default.removeItem(at: errorURL)
            }

            let errorHandle = try FileHandle(forWritingTo: errorURL)
            defer { try? errorHandle.close() }

            let process = Process()
            process.executableURL = URL(fileURLWithPath: pandoc)
            process.arguments = [
                "--from=gfm+tex_math_dollars",
                "--to=html5",
                "--mathml",
                "--wrap=none",
                markdownURL.path,
                "--output", outputURL.path
            ]
            process.standardError = errorHandle
            try Task.checkCancellation()
            try process.run()
            defer { if process.isRunning { process.terminate() } }
            let deadline = Date().addingTimeInterval(30)
            while process.isRunning && Date() < deadline { try await Task.sleep(for: .milliseconds(50)) }
            if process.isRunning {
                process.terminate()
                throw MarkdownPreviewError.conversionFailed("Rendering exceeded 30 seconds. Use the Source tab.")
            }
            try? errorHandle.synchronize()

            guard process.terminationStatus == 0,
                  let fragment = try? String(contentsOf: outputURL, encoding: .utf8) else {
                let details = String(decoding: (try? Data(contentsOf: errorURL)) ?? Data(), as: UTF8.self)
                throw MarkdownPreviewError.conversionFailed(String(details.suffix(800)))
            }
            return PreviewDocument(markdown: markdown, html: wrapHTML(fragment))
        }
        return try await withTaskCancellationHandler { try await task.value } onCancel: { task.cancel() }
    }

    static func wrapHTML(_ body: String) -> String {
        """
        <!doctype html>
        <html><head><meta charset="utf-8">
        <meta name="color-scheme" content="light dark">
        <meta http-equiv="Content-Security-Policy" content="default-src 'none'; img-src file: data:; style-src 'unsafe-inline'">
        <style>
        :root { \(RagdropTheme.cssVariables(.light)) color-scheme: light; font: 14px -apple-system, BlinkMacSystemFont, sans-serif; }
        :root[data-theme="dark"] { \(RagdropTheme.cssVariables(.dark)) color-scheme: dark; }
        body { max-width: 880px; margin: 0 auto; padding: 34px 42px 70px; line-height: 1.62; color: var(--text); background: var(--canvas); }
        h1 { font-size: 2rem; line-height: 1.18; margin: .2em 0 .8em; letter-spacing: -.02em; }
        h2 { font-size: 1.45rem; margin: 1.8em 0 .55em; border-bottom: 1px solid var(--line); padding-bottom: .25em; }
        h3 { font-size: 1.16rem; margin: 1.5em 0 .45em; }
        p { margin: 0 0 1em; }
        a { color: var(--link); text-decoration: none; }
        blockquote { margin: 1.3em 0; padding: .15em 1.1em; border-left: 3px solid var(--accent); color: var(--secondary); }
        code { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: .9em; background: var(--panel); padding: .12em .3em; border-radius: 4px; }
        pre { white-space: pre-wrap; overflow-wrap: anywhere; padding: 18px; border-radius: 10px; background: var(--panel); }
        table { display: block; overflow-x: auto; width: max-content; max-width: 100%; border-collapse: collapse; margin: 1.4em 0; font-size: .9em; }
        th, td { border: 1px solid var(--line); padding: 7px 10px; vertical-align: top; }
        th { background: var(--raised); text-align: left; }
        img { max-width: 100%; height: auto; border-radius: 8px; }
        math { font-size: 1.04em; }
        sup { font-size: .72em; }
        </style></head><body>\(body)</body></html>
        """
    }

    private static func escapeHTML(_ value: String) -> String {
        value
            .replacingOccurrences(of: "&", with: "&amp;")
            .replacingOccurrences(of: "<", with: "&lt;")
            .replacingOccurrences(of: ">", with: "&gt;")
    }
}
