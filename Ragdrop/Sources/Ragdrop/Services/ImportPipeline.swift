import CryptoKit
import Darwin
import Foundation

struct PipelineConfiguration: Sendable {
    let converterPath: String
    let nasHost: String
    let remoteRoot: String
    var location: LibraryLocation = .server
    var localConnection: LocalConnection? = nil

    enum ConverterKind: String, CaseIterable, Identifiable, Sendable {
        case mistral
        case mineru
        case custom

        var id: Self { self }
        var title: String {
            switch self {
            case .mistral: "Mistral OCR"
            case .mineru: "MinerU (fallback)"
            case .custom: "Custom"
            }
        }
        var commandLabel: String {
            switch self {
            case .mistral: "Mistral OCR"
            case .mineru: "MinerU"
            case .custom: "The PDF converter"
            }
        }
        var parser: String {
            switch self {
            case .mistral: "mistral-ocr"
            case .mineru: "mineru"
            case .custom: "custom"
            }
        }
        var parserVersion: String {
            switch self {
            case .mistral: "mistral-ocr-latest"
            case .mineru: "api-v4-pipeline"
            case .custom: "external"
            }
        }
        var maximumBytes: Int? {
            switch self {
            case .mistral: 512 * 1024 * 1024
            case .mineru: 200 * 1024 * 1024
            case .custom: nil
            }
        }
        var timeout: TimeInterval {
            self == .mistral ? 30 * 60 : 90 * 60
        }
    }

    static var defaultConverterPath: String {
        defaultMistralConverterPath
    }

    static var defaultMistralConverterPath: String {
        converterPath(named: "ragdrop_mistral_convert.py", skillCandidates: [])
    }

    static var defaultMinerUConverterPath: String {
        let home = FileManager.default.homeDirectoryForCurrentUser
        return converterPath(named: "ragdrop_mineru_convert.py", skillCandidates: [
            home.appendingPathComponent(".codex/skills/mineru-pdf/mineru_convert.py"),
            home.appendingPathComponent(".Codex/skills/mineru-pdf/mineru_convert.py"),
            home.appendingPathComponent(".agents/skills/mineru-pdf/mineru_convert.py")
        ])
    }

    private static func converterPath(named filename: String, skillCandidates: [URL]) -> String {
        let bundled = Bundle.main.resourceURL?.appendingPathComponent(filename)
        var candidates = [bundled].compactMap { $0 }
        #if DEBUG
        let project = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("scripts/\(filename)")
        candidates.append(project)
        #endif
        candidates += skillCandidates
        return candidates.first(where: { FileManager.default.fileExists(atPath: $0.path) })?.path
            ?? candidates[0].path
    }

    static func current() -> PipelineConfiguration {
        let defaults = UserDefaults.standard
        let location = LibraryLocation.resolve(defaults: defaults)
        if defaults.string(forKey: "libraryLocation") == nil { defaults.set(location.rawValue, forKey: "libraryLocation") }
        let savedConverter = defaults.string(forKey: "converterPath")?.nonEmpty
        let converter: String
        if defaults.integer(forKey: "converterMigrationVersion") < 2,
           savedConverter == nil || savedConverter.map(isLegacyConverterPath) == true {
            converter = defaultConverterPath
            defaults.set(converter, forKey: "converterPath")
            defaults.set(2, forKey: "converterMigrationVersion")
        } else {
            converter = savedConverter ?? defaultConverterPath
        }
        let host = defaults.string(forKey: "nasHost")?.nonEmpty ?? "ragdoc-server"
        let root = defaults.string(forKey: "remoteRoot")?.nonEmpty
            ?? "/srv/ragdoc"
        if location == .local {
            let engine = LocalEngine.current
            return PipelineConfiguration(converterPath: converter, nasHost: "", remoteRoot: engine.recordedConnection?.library ?? defaults.string(forKey: "localLibraryPath") ?? engine.defaultLibrary.path,
                                         location: .local, localConnection: engine.connection)
        }
        return PipelineConfiguration(converterPath: converter, nasHost: host, remoteRoot: root)
    }

    static func isLegacyConverterPath(_ path: String) -> Bool {
        path.hasSuffix("/skills/mineru-pdf/mineru_convert.py")
            || path.hasSuffix("/ragdrop_mineru_convert.py")
    }

    static func converterKind(for path: String) -> ConverterKind {
        let name = URL(fileURLWithPath: path).lastPathComponent.lowercased()
        if name.contains("mistral") { return .mistral }
        if name.contains("mineru") { return .mineru }
        return .custom
    }

    var converterKind: ConverterKind { Self.converterKind(for: converterPath) }

    func validate() throws {
        if location == .local {
            guard remoteRoot.hasPrefix("/"), !remoteRoot.contains("\0") else {
                throw PipelineError.invalidConfiguration("Choose an absolute local library folder.")
            }
            guard let localConnection, localConnection.library == remoteRoot else {
                throw PipelineError.invalidConfiguration("Prepare your local library in Settings first.")
            }
            return
        }
        guard nasHost.range(of: "^[A-Za-z0-9._-]+$", options: .regularExpression) != nil else {
            throw PipelineError.invalidConfiguration("The server name contains invalid characters.")
        }
        guard remoteRoot.range(of: "^/[A-Za-z0-9_./-]+$", options: .regularExpression) != nil else {
            throw PipelineError.invalidConfiguration("The remote Ragdoc directory is invalid.")
        }
    }
}

struct ConversionArtifact: Sendable {
    let sourceURL: URL
    let markdownURL: URL
    let remoteFilename: String
    let metadataURL: URL?
    let artifactBundleURL: URL?
    let artifactCount: Int
}

enum PipelineError: LocalizedError, Sendable {
    case invalidPDF
    case oversizedPDF(provider: String, limitMB: Int)
    case missingConverter(String)
    case missingCredential(String)
    case invalidConfiguration(String)
    case invalidRemoteFilename
    case missingOutput(String)
    case timedOut(command: String, minutes: Int)
    case cancelled
    case processFailed(command: String, details: String)
    case verificationFailed

    var errorDescription: String? {
        switch self {
        case .invalidPDF: "The selected file is not a readable PDF."
        case .oversizedPDF(let provider, let limit): "\(provider) limits files to \(limit) MB."
        case .missingConverter(let path): "PDF converter not found: \(path)"
        case .missingCredential(let message): message
        case .invalidConfiguration(let message): message
        case .invalidRemoteFilename: "The remote Markdown filename is invalid."
        case .missingOutput(let provider): "\(provider) produced no Markdown file."
        case .timedOut(let command, let minutes):
            "\(command) stopped after \(minutes) minutes. You can retry this PDF."
        case .cancelled: "Conversion canceled. You can retry this PDF."
        case .processFailed(let command, let details):
            "\(command) failed. \(Self.lastUsefulLine(in: details))"
        case .verificationFailed: "Ragdoc cannot find the document after indexing."
        }
    }

    var diagnosticDetails: String? {
        if case .processFailed(_, let details) = self { return details }
        return nil
    }

    private static func lastUsefulLine(in details: String) -> String {
        details.split(whereSeparator: \.isNewline)
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .last(where: { !$0.isEmpty }) ?? "Unknown error."
    }
}

struct ImportPipeline: Sendable {
    let configuration: PipelineConfiguration

    func duplicateFilename(for pdfURL: URL) async throws -> (fingerprint: String, filename: String?) {
        try configuration.validate()
        let fingerprint = try await sha256(of: pdfURL)
        let hashPrefix = String(fingerprint.prefix(12))
        if configuration.location == .local {
            let directory = URL(fileURLWithPath: configuration.remoteRoot).appendingPathComponent("articles_markdown")
            let candidates = try FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
                .filter { $0.lastPathComponent.hasSuffix("_\(hashPrefix).md") }
            for candidate in candidates {
                if try await indexedChunkCount(for: candidate.lastPathComponent) > 0 { return (fingerprint, candidate.lastPathComponent) }
            }
            return (fingerprint, nil)
        }
        let remoteDirectory = "\(configuration.remoteRoot)/articles_markdown"
        let command = "find '\(remoteDirectory)' -maxdepth 1 -type f -name '*_\(hashPrefix).md' -print -quit"
        let result = try await ProcessRunner.run(
            executable: "/usr/bin/ssh",
            arguments: sshArguments(command),
            label: "Duplicate checking"
        )
        let path = result.output.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !path.isEmpty else { return (fingerprint, nil) }
        let filename = URL(fileURLWithPath: path).lastPathComponent
        let count = try await indexedChunkCount(for: filename)
        return (fingerprint, count > 0 ? filename : nil)
    }

    func convert(_ pdfURL: URL, fingerprint: String, metadata: ImportMetadata? = nil) async throws -> ConversionArtifact {
        try configuration.validate()
        guard pdfURL.pathExtension.lowercased() == "pdf",
              FileManager.default.isReadableFile(atPath: pdfURL.path) else {
            throw PipelineError.invalidPDF
        }
        let converterKind = configuration.converterKind
        let values = try pdfURL.resourceValues(forKeys: [.fileSizeKey])
        if let maximumBytes = converterKind.maximumBytes,
           let size = values.fileSize, size > maximumBytes {
            throw PipelineError.oversizedPDF(
                provider: converterKind.commandLabel,
                limitMB: maximumBytes / 1024 / 1024
            )
        }
        guard FileManager.default.fileExists(atPath: configuration.converterPath) else {
            throw PipelineError.missingConverter(configuration.converterPath)
        }
        if converterKind == .mistral, !MistralCredentialStore.isConfigured {
            throw PipelineError.missingCredential(
                "Mistral key missing. Add it in Settings > Mistral OCR."
            )
        }
        if converterKind == .mineru {
            let token = FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(".mineru_token")
            guard FileManager.default.fileExists(atPath: token.path) else {
                throw PipelineError.missingCredential("MinerU token missing from ~/.mineru_token.")
            }
        }

        let hashPrefix = String(fingerprint.prefix(12))
        let outputName = FilenameSanitizer.outputName(for: pdfURL, hashPrefix: hashPrefix)
        let markdownURL = URL(fileURLWithPath: "/tmp", isDirectory: true)
            .appendingPathComponent(outputName)
            .appendingPathExtension("md")
        try? FileManager.default.removeItem(at: markdownURL)

        _ = try await ProcessRunner.run(
            executable: configuration.localConnection?.python ?? "/usr/bin/env",
            arguments: (configuration.location == .local ? [] : ["python3"]) + [configuration.converterPath, pdfURL.path, outputName],
            label: converterKind.commandLabel,
            timeout: converterKind.timeout
        )
        guard FileManager.default.fileExists(atPath: markdownURL.path),
              (try markdownURL.resourceValues(forKeys: [.fileSizeKey]).fileSize ?? 0) > 0 else {
            throw PipelineError.missingOutput(converterKind.commandLabel)
        }
        let artifactBundleURL = URL(fileURLWithPath: "/tmp", isDirectory: true)
            .appendingPathComponent("\(outputName).ragdoc-artifacts", isDirectory: true)
        let manifestURL = artifactBundleURL.appendingPathComponent("manifest.json")
        let artifactCount: Int
        if let data = try? Data(contentsOf: manifestURL),
           let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
           let artifacts = object["artifacts"] as? [[String: Any]] {
            artifactCount = artifacts.count
        } else {
            artifactCount = 0
        }
        let metadataURL = URL(fileURLWithPath: "/tmp", isDirectory: true)
            .appendingPathComponent(outputName)
            .appendingPathExtension("metadata.json")
        try writeMetadata(
            to: metadataURL,
            pdfURL: pdfURL,
            markdownURL: markdownURL,
            fingerprint: fingerprint,
            manifestURL: manifestURL,
            metadata: metadata,
            parser: converterKind.parser,
            parserVersion: converterKind.parserVersion
        )
        return ConversionArtifact(
            sourceURL: pdfURL,
            markdownURL: markdownURL,
            remoteFilename: "\(outputName).md",
            metadataURL: metadataURL,
            artifactBundleURL: FileManager.default.fileExists(atPath: manifestURL.path)
                ? artifactBundleURL : nil,
            artifactCount: artifactCount
        )
    }

    func transfer(_ artifact: ConversionArtifact) async throws {
        try configuration.validate()
        if configuration.location == .local {
            try LocalTransfer.save(artifact, root: URL(fileURLWithPath: configuration.remoteRoot))
            return
        }
        if let bundle = artifact.artifactBundleURL {
            let stem = URL(fileURLWithPath: artifact.remoteFilename).deletingPathExtension().lastPathComponent
            guard stem.range(of: "^[A-Za-z0-9_.-]+$", options: .regularExpression) != nil else {
                throw PipelineError.invalidRemoteFilename
            }
            let archive = FileManager.default.temporaryDirectory
                .appendingPathComponent("ragdrop-\(UUID().uuidString).tar.gz")
            defer { try? FileManager.default.removeItem(at: archive) }
            _ = try await ProcessRunner.run(
                executable: "/usr/bin/tar",
                arguments: ["-czf", archive.path, "-C", bundle.path, "."],
                label: "Preparing tables and figures"
            )
            let artifactsRoot = "\(configuration.remoteRoot)/ragdoc_artifacts"
            let finalPath = "\(artifactsRoot)/\(stem)"
            let temporaryPath = "\(artifactsRoot)/.ragdrop-\(UUID().uuidString)"
            let command = "mkdir -p '\(artifactsRoot)' '\(temporaryPath)' && tar -xzf - -C '\(temporaryPath)' && if [ -d '\(finalPath)' ]; then rm -rf '\(temporaryPath)'; else mv '\(temporaryPath)' '\(finalPath)'; fi"
            _ = try await ProcessRunner.run(
                executable: "/usr/bin/ssh",
                arguments: sshArguments(command),
                standardInput: archive,
                label: "Transferring tables and figures"
            )
        }
        let remoteDirectory = "\(configuration.remoteRoot)/articles_markdown"
        let remotePath = "\(remoteDirectory)/\(artifact.remoteFilename)"
        if let metadataURL = artifact.metadataURL {
            let sidecarName = URL(fileURLWithPath: artifact.remoteFilename)
                .deletingPathExtension().lastPathComponent + ".metadata.json"
            let remoteSidecar = "\(remoteDirectory)/\(sidecarName)"
            let temporarySidecar = "\(remoteDirectory)/.ragdrop-\(UUID().uuidString).metadata"
            _ = try await ProcessRunner.run(
                executable: "/usr/bin/ssh",
                arguments: sshArguments(
                    "mkdir -p '\(remoteDirectory)' && cat > '\(temporarySidecar)' && mv -f '\(temporarySidecar)' '\(remoteSidecar)'"
                ),
                standardInput: metadataURL,
                label: "Transferring metadata"
            )
        }
        let temporaryPath = "\(remoteDirectory)/.ragdrop-\(UUID().uuidString).upload"
        _ = try await ProcessRunner.run(
            executable: "/usr/bin/ssh",
            arguments: sshArguments(
                "mkdir -p '\(remoteDirectory)' && cat > '\(temporaryPath)' && mv -f '\(temporaryPath)' '\(remotePath)'"
            ),
            standardInput: artifact.markdownURL,
            label: "Transfer to the server"
        )
    }

    private func writeMetadata(
        to destination: URL,
        pdfURL: URL,
        markdownURL: URL,
        fingerprint: String,
        manifestURL: URL,
        metadata: ImportMetadata?,
        parser: String,
        parserVersion: String
    ) throws {
        let markdown = try Data(contentsOf: markdownURL)
        let contentHash = SHA256.hash(data: markdown).map { String(format: "%02x", $0) }.joined()
        var pageSpans: [[String: Any]] = []
        if let data = try? Data(contentsOf: manifestURL),
           let manifest = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
           let rawSpans = manifest["page_spans"] as? [[String: Any]] {
            for span in rawSpans {
                guard let page = span["page"] as? Int,
                      let start = span["start"] as? Int,
                      let end = span["end"] as? Int else { continue }
                pageSpans.append([
                    "page": page,
                    "start": start,
                    "end": end
                ])
            }
        }
        var sidecar: [String: Any] = [
            "version": metadata == nil ? "finder-pdf" : "zotero-pdf",
            "source_pdf": pdfURL.path,
            "parser": parser,
            "parser_version": parserVersion,
            "completeness": "not_assessed",
            "pdf_sha256": fingerprint,
            "parsed_pdf_sha256": fingerprint,
            "content_sha256": contentHash,
            "page_spans": pageSpans
        ]
        if let metadata {
            sidecar["title"] = metadata.title
            sidecar["authors"] = metadata.authors
            if let year = metadata.year { sidecar["year"] = year }
            if let doi = metadata.doi { sidecar["doi"] = doi }
            if let key = metadata.zoteroItemKey { sidecar["zotero_item_key"] = key }
            if let key = metadata.zoteroAttachmentKey { sidecar["zotero_attachment_key"] = key }
        }
        let data = try JSONSerialization.data(withJSONObject: sidecar, options: [.prettyPrinted, .sortedKeys])
        try data.write(to: destination, options: .atomic)
    }

    func indexAll(sources: [String]) async throws {
        try configuration.validate()
        guard !sources.isEmpty,
              sources.allSatisfy({ $0.range(of: "^[A-Za-z0-9_.-]+$", options: .regularExpression) != nil }) else {
            throw PipelineError.invalidRemoteFilename
        }
        if configuration.location == .local {
            _ = try await LocalEngine.current.call("index", sources: sources, connection: configuration.localConnection)
            return
        }
        let root = configuration.remoteRoot
        let sourceArguments = sources.map { "--source \(shellQuote($0))" }.joined(separator: " ")
        let command = [
            "cd '\(root)'",
            "set -a",
            ". ./.env",
            "set +a",
            "export CHROMA_DB_PATH='\(root)/chroma_db_new'",
            "export RAGDOC_LIBRARY_DIR='\(root)/ragdoc_library/ragdoc_contextualized_v1'",
            "export COLLECTION_NAME=ragdoc_contextualized_v1",
            "export RAGDOC_CHROMA_MODE=persistent",
            "export RAGDOC_EMBEDDING_MODEL=voyage-context-4",
            "./ragdoc-env-new/bin/python3 scripts/index_incremental.py \(sourceArguments)",
            "./ragdoc-env-new/bin/python3 scripts/index_artifacts.py \(sourceArguments)"
        ].joined(separator: " && ")
        _ = try await ProcessRunner.run(
            executable: "/usr/bin/ssh",
            arguments: sshArguments(command),
            label: "Ragdoc indexing"
        )
    }

    func verify(_ artifact: ConversionArtifact) async throws -> Int {
        try configuration.validate()
        let count = try await indexedChunkCount(for: artifact.remoteFilename)
        guard count > 0 else { throw PipelineError.verificationFailed }
        return count
    }

    private func indexedChunkCount(for source: String) async throws -> Int {
        if configuration.location == .local {
            struct Count: Decodable { let count: Int }
            let data = try await LocalEngine.current.call("count", sources: [source], connection: configuration.localConnection)
            return try JSONDecoder().decode(Count.self, from: data).count
        }
        let root = configuration.remoteRoot
        let python = "import chromadb,sys;c=chromadb.PersistentClient(path='\(root)/chroma_db_new').get_collection('ragdoc_contextualized_v1');print(len(c.get(where={'source':sys.argv[1]},include=[])['ids']))"
        let command = "cd '\(root)' && ./ragdoc-env-new/bin/python3 -c \(shellQuote(python)) \(shellQuote(source))"
        let result = try await ProcessRunner.run(
            executable: "/usr/bin/ssh",
            arguments: sshArguments(command),
            label: "Ragdoc verification"
        )
        return result.output
            .split(whereSeparator: \.isWhitespace)
            .compactMap({ Int($0) })
            .last ?? 0
    }

    private func shellQuote(_ value: String) -> String {
        "'" + value.replacingOccurrences(of: "'", with: "'\\''") + "'"
    }

    private func sshArguments(_ command: String) -> [String] {
        ["-o", "ControlMaster=no", "-o", "ControlPath=none", configuration.nasHost, command]
    }

    private func sha256(of fileURL: URL) async throws -> String {
        try await Task.detached(priority: .userInitiated) {
            let handle = try FileHandle(forReadingFrom: fileURL)
            defer { try? handle.close() }
            var hasher = SHA256()
            while let data = try handle.read(upToCount: 1024 * 1024), !data.isEmpty {
                hasher.update(data: data)
            }
            return hasher.finalize()
                .map { String(format: "%02x", $0) }
                .joined()
        }.value
    }


}

private extension String {
    var nonEmpty: String? {
        guard !trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            return nil
        }
        return self
    }
}
