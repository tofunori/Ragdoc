import Foundation

/// Stage a reviewed extraction on the same volume; publish Markdown last.
enum LocalTransfer {
    static func save(_ artifact: ConversionArtifact, root: URL) throws {
        let name = artifact.remoteFilename
        guard name.range(of: "^[A-Za-z0-9_.-]+\\.md$", options: .regularExpression) != nil,
              URL(fileURLWithPath: name).lastPathComponent == name else { throw PipelineError.invalidRemoteFilename }
        let fm = FileManager.default
        let articles = root.appendingPathComponent("articles_markdown")
        let visuals = root.appendingPathComponent("ragdoc_artifacts")
        try fm.createDirectory(at: articles, withIntermediateDirectories: true)
        try fm.createDirectory(at: visuals, withIntermediateDirectories: true)
        let stem = String(name.dropLast(3))
        if let bundle = artifact.artifactBundleURL {
            let target = visuals.appendingPathComponent(stem)
            if !fm.fileExists(atPath: target.path) {
                let staging = visuals.appendingPathComponent(".staging-\(UUID().uuidString)")
                defer { try? fm.removeItem(at: staging) }
                try fm.copyItem(at: bundle, to: staging)
                try fm.moveItem(at: staging, to: target)
            }
        }
        if let metadata = artifact.metadataURL {
            try Data(contentsOf: metadata).write(to: articles.appendingPathComponent(stem + ".metadata.json"), options: .atomic)
        }
        try Data(contentsOf: artifact.markdownURL).write(to: articles.appendingPathComponent(name), options: .atomic)
    }
}
