import CryptoKit
import Foundation

struct PipelineConfiguration: Sendable {
    let converterPath: String
    let nasHost: String
    let remoteRoot: String

    static var defaultConverterPath: String {
        let home = FileManager.default.homeDirectoryForCurrentUser
        let bundled = Bundle.main.resourceURL?.appendingPathComponent("ragdrop_mineru_convert.py")
        let project = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("scripts/ragdrop_mineru_convert.py")
        let candidates = [
            bundled,
            project,
            home.appendingPathComponent(".codex/skills/mineru-pdf/mineru_convert.py"),
            home.appendingPathComponent(".Codex/skills/mineru-pdf/mineru_convert.py"),
            home.appendingPathComponent(".agents/skills/mineru-pdf/mineru_convert.py")
        ].compactMap { $0 }
        return candidates.first(where: { FileManager.default.fileExists(atPath: $0.path) })?.path
            ?? candidates[0].path
    }

    static func current() -> PipelineConfiguration {
        let defaults = UserDefaults.standard
        let savedConverter = defaults.string(forKey: "converterPath")?.nonEmpty
        let converter: String
        if savedConverter.map(isLegacyConverterPath) == true {
            converter = defaultConverterPath
            defaults.set(converter, forKey: "converterPath")
        } else {
            converter = savedConverter ?? defaultConverterPath
        }
        let host = defaults.string(forKey: "nasHost")?.nonEmpty ?? "rorqual"
        let root = defaults.string(forKey: "remoteRoot")?.nonEmpty
            ?? "/volume1/Services/mcp/ragdoc"
        return PipelineConfiguration(converterPath: converter, nasHost: host, remoteRoot: root)
    }

    static func isLegacyConverterPath(_ path: String) -> Bool {
        path.hasSuffix("/skills/mineru-pdf/mineru_convert.py")
    }

    func validate() throws {
        guard nasHost.range(of: "^[A-Za-z0-9._-]+$", options: .regularExpression) != nil else {
            throw PipelineError.invalidConfiguration("Le nom du NAS contient des caractères invalides.")
        }
        guard remoteRoot.range(of: "^/[A-Za-z0-9_./-]+$", options: .regularExpression) != nil else {
            throw PipelineError.invalidConfiguration("Le dossier Ragdoc distant est invalide.")
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
    case oversizedPDF
    case missingConverter(String)
    case missingToken
    case invalidConfiguration(String)
    case invalidRemoteFilename
    case missingOutput
    case processFailed(command: String, details: String)
    case verificationFailed

    var errorDescription: String? {
        switch self {
        case .invalidPDF: "Le fichier sélectionné n’est pas un PDF lisible."
        case .oversizedPDF: "MinerU limite les fichiers à 200 Mo."
        case .missingConverter(let path): "Convertisseur MinerU introuvable : \(path)"
        case .missingToken: "Jeton MinerU absent de ~/.mineru_token."
        case .invalidConfiguration(let message): message
        case .invalidRemoteFilename: "Le nom du Markdown distant est invalide."
        case .missingOutput: "MinerU n’a produit aucun fichier Markdown."
        case .processFailed(let command, let details): "\(command) a échoué. \(details)"
        case .verificationFailed: "Ragdoc ne retrouve pas le document après l’indexation."
        }
    }
}

private struct ProcessResult: Sendable {
    let output: String
    let error: String
}

struct ImportPipeline: Sendable {
    let configuration: PipelineConfiguration

    func duplicateFilename(for pdfURL: URL) async throws -> (fingerprint: String, filename: String?) {
        try configuration.validate()
        let fingerprint = try await sha256(of: pdfURL)
        let hashPrefix = String(fingerprint.prefix(12))
        let remoteDirectory = "\(configuration.remoteRoot)/articles_markdown"
        let command = "find '\(remoteDirectory)' -maxdepth 1 -type f -name '*_\(hashPrefix).md' -print -quit"
        let result = try await run(
            executable: "/usr/bin/ssh",
            arguments: sshArguments(command),
            label: "La recherche de doublons"
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
        let values = try pdfURL.resourceValues(forKeys: [.fileSizeKey])
        if let size = values.fileSize, size > 200 * 1024 * 1024 {
            throw PipelineError.oversizedPDF
        }
        guard FileManager.default.fileExists(atPath: configuration.converterPath) else {
            throw PipelineError.missingConverter(configuration.converterPath)
        }
        let token = FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(".mineru_token")
        guard FileManager.default.fileExists(atPath: token.path) else {
            throw PipelineError.missingToken
        }

        let hashPrefix = String(fingerprint.prefix(12))
        let outputName = FilenameSanitizer.outputName(for: pdfURL, hashPrefix: hashPrefix)
        let markdownURL = URL(fileURLWithPath: "/tmp", isDirectory: true)
            .appendingPathComponent(outputName)
            .appendingPathExtension("md")
        try? FileManager.default.removeItem(at: markdownURL)

        _ = try await run(
            executable: "/usr/bin/env",
            arguments: ["python3", configuration.converterPath, pdfURL.path, outputName],
            label: "MinerU"
        )
        guard FileManager.default.fileExists(atPath: markdownURL.path),
              (try markdownURL.resourceValues(forKeys: [.fileSizeKey]).fileSize ?? 0) > 0 else {
            throw PipelineError.missingOutput
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
            metadata: metadata
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
        if let bundle = artifact.artifactBundleURL {
            let stem = URL(fileURLWithPath: artifact.remoteFilename).deletingPathExtension().lastPathComponent
            guard stem.range(of: "^[A-Za-z0-9_.-]+$", options: .regularExpression) != nil else {
                throw PipelineError.invalidRemoteFilename
            }
            let archive = FileManager.default.temporaryDirectory
                .appendingPathComponent("ragdrop-\(UUID().uuidString).tar.gz")
            defer { try? FileManager.default.removeItem(at: archive) }
            _ = try await run(
                executable: "/usr/bin/tar",
                arguments: ["-czf", archive.path, "-C", bundle.path, "."],
                label: "La préparation des tableaux et figures"
            )
            let artifactsRoot = "\(configuration.remoteRoot)/ragdoc_artifacts"
            let finalPath = "\(artifactsRoot)/\(stem)"
            let temporaryPath = "\(artifactsRoot)/.ragdrop-\(UUID().uuidString)"
            let command = "mkdir -p '\(artifactsRoot)' '\(temporaryPath)' && tar -xzf - -C '\(temporaryPath)' && if [ -d '\(finalPath)' ]; then rm -rf '\(temporaryPath)'; else mv '\(temporaryPath)' '\(finalPath)'; fi"
            _ = try await run(
                executable: "/usr/bin/ssh",
                arguments: sshArguments(command),
                standardInput: archive,
                label: "Le transfert des tableaux et figures"
            )
        }
        let remoteDirectory = "\(configuration.remoteRoot)/articles_markdown"
        let remotePath = "\(remoteDirectory)/\(artifact.remoteFilename)"
        if let metadataURL = artifact.metadataURL {
            let sidecarName = URL(fileURLWithPath: artifact.remoteFilename)
                .deletingPathExtension().lastPathComponent + ".metadata.json"
            let remoteSidecar = "\(remoteDirectory)/\(sidecarName)"
            let temporarySidecar = "\(remoteDirectory)/.ragdrop-\(UUID().uuidString).metadata"
            _ = try await run(
                executable: "/usr/bin/ssh",
                arguments: sshArguments(
                    "mkdir -p '\(remoteDirectory)' && cat > '\(temporarySidecar)' && mv -f '\(temporarySidecar)' '\(remoteSidecar)'"
                ),
                standardInput: metadataURL,
                label: "Le transfert des métadonnées"
            )
        }
        let temporaryPath = "\(remoteDirectory)/.ragdrop-\(UUID().uuidString).upload"
        _ = try await run(
            executable: "/usr/bin/ssh",
            arguments: sshArguments(
                "mkdir -p '\(remoteDirectory)' && cat > '\(temporaryPath)' && mv -f '\(temporaryPath)' '\(remotePath)'"
            ),
            standardInput: artifact.markdownURL,
            label: "Le transfert vers le NAS"
        )
    }

    private func writeMetadata(
        to destination: URL,
        pdfURL: URL,
        markdownURL: URL,
        fingerprint: String,
        manifestURL: URL,
        metadata: ImportMetadata?
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
            "parser": "mineru",
            "parser_version": "api-v4-pipeline",
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
        _ = try await run(
            executable: "/usr/bin/ssh",
            arguments: sshArguments(command),
            label: "L’indexation Ragdoc"
        )
    }

    func verify(_ artifact: ConversionArtifact) async throws -> Int {
        try configuration.validate()
        let count = try await indexedChunkCount(for: artifact.remoteFilename)
        guard count > 0 else { throw PipelineError.verificationFailed }
        return count
    }

    private func indexedChunkCount(for source: String) async throws -> Int {
        let root = configuration.remoteRoot
        let python = "import chromadb,sys;c=chromadb.PersistentClient(path='\(root)/chroma_db_new').get_collection('ragdoc_contextualized_v1');print(len(c.get(where={'source':sys.argv[1]},include=[])['ids']))"
        let command = "cd '\(root)' && ./ragdoc-env-new/bin/python3 -c \(shellQuote(python)) \(shellQuote(source))"
        let result = try await run(
            executable: "/usr/bin/ssh",
            arguments: sshArguments(command),
            label: "La vérification Ragdoc"
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

    private func run(
        executable: String,
        arguments: [String],
        standardInput: URL? = nil,
        label: String
    ) async throws -> ProcessResult {
        try await Task.detached(priority: .userInitiated) {
            let fileManager = FileManager.default
            let temporary = fileManager.temporaryDirectory
            let outputURL = temporary.appendingPathComponent("ragdrop-\(UUID().uuidString).out")
            let errorURL = temporary.appendingPathComponent("ragdrop-\(UUID().uuidString).err")
            fileManager.createFile(atPath: outputURL.path, contents: nil)
            fileManager.createFile(atPath: errorURL.path, contents: nil)
            defer {
                try? fileManager.removeItem(at: outputURL)
                try? fileManager.removeItem(at: errorURL)
            }

            let outputHandle = try FileHandle(forWritingTo: outputURL)
            let errorHandle = try FileHandle(forWritingTo: errorURL)
            let inputHandle = try standardInput.map { try FileHandle(forReadingFrom: $0) }
            defer {
                try? outputHandle.close()
                try? errorHandle.close()
                try? inputHandle?.close()
            }

            let process = Process()
            process.executableURL = URL(fileURLWithPath: executable)
            process.arguments = arguments
            process.standardOutput = outputHandle
            process.standardError = errorHandle
            process.standardInput = inputHandle
            var environment = ProcessInfo.processInfo.environment
            environment["PATH"] = "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
            process.environment = environment

            try process.run()
            process.waitUntilExit()
            try? outputHandle.synchronize()
            try? errorHandle.synchronize()
            let output = String(decoding: (try? Data(contentsOf: outputURL)) ?? Data(), as: UTF8.self)
            let error = String(decoding: (try? Data(contentsOf: errorURL)) ?? Data(), as: UTF8.self)
            guard process.terminationStatus == 0 else {
                let details = String((error.isEmpty ? output : error).suffix(1_500))
                throw PipelineError.processFailed(command: label, details: details)
            }
            return ProcessResult(output: output, error: error)
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
