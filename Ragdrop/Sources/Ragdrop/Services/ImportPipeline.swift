import CryptoKit
import Darwin
import Foundation

struct PipelineConfiguration: Sendable {
    let converterPath: String
    let nasHost: String
    let remoteRoot: String

    enum ConverterKind: String, CaseIterable, Identifiable, Sendable {
        case mistral
        case mineru
        case custom

        var id: Self { self }
        var title: String {
            switch self {
            case .mistral: "Mistral OCR"
            case .mineru: "MinerU (secours)"
            case .custom: "Personnalisé"
            }
        }
        var commandLabel: String {
            switch self {
            case .mistral: "Mistral OCR"
            case .mineru: "MinerU"
            case .custom: "Le convertisseur PDF"
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
        let project = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("scripts/\(filename)")
        let candidates = ([bundled, project].compactMap { $0 }) + skillCandidates
        return candidates.first(where: { FileManager.default.fileExists(atPath: $0.path) })?.path
            ?? candidates[0].path
    }

    static func current() -> PipelineConfiguration {
        let defaults = UserDefaults.standard
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
        let host = defaults.string(forKey: "nasHost")?.nonEmpty ?? "rorqual"
        let root = defaults.string(forKey: "remoteRoot")?.nonEmpty
            ?? "/volume1/Services/mcp/ragdoc"
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
        case .invalidPDF: "Le fichier sélectionné n’est pas un PDF lisible."
        case .oversizedPDF(let provider, let limit): "\(provider) limite les fichiers à \(limit) Mo."
        case .missingConverter(let path): "Convertisseur PDF introuvable : \(path)"
        case .missingCredential(let message): message
        case .invalidConfiguration(let message): message
        case .invalidRemoteFilename: "Le nom du Markdown distant est invalide."
        case .missingOutput(let provider): "\(provider) n’a produit aucun fichier Markdown."
        case .timedOut(let command, let minutes):
            "\(command) a été arrêté après \(minutes) minutes. Vous pouvez relancer ce PDF."
        case .cancelled: "Conversion annulée. Vous pouvez relancer ce PDF."
        case .processFailed(let command, let details):
            "\(command) a échoué. \(Self.lastUsefulLine(in: details))"
        case .verificationFailed: "Ragdoc ne retrouve pas le document après l’indexation."
        }
    }

    var diagnosticDetails: String? {
        if case .processFailed(_, let details) = self { return details }
        return nil
    }

    private static func lastUsefulLine(in details: String) -> String {
        details.split(whereSeparator: \.isNewline)
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .last(where: { !$0.isEmpty }) ?? "Erreur inconnue."
    }
}

private struct ProcessResult: Sendable {
    let output: String
    let error: String
}

private final class ProcessController: @unchecked Sendable {
    private let lock = NSLock()
    private var process: Process?
    private var cancelled = false

    func attach(_ process: Process) {
        lock.lock()
        self.process = process
        let shouldCancel = cancelled
        lock.unlock()
        if shouldCancel, process.isRunning { process.terminate() }
    }

    func detach() {
        lock.lock()
        process = nil
        lock.unlock()
    }

    func cancel() {
        lock.lock()
        cancelled = true
        let running = process
        lock.unlock()
        if running?.isRunning == true { running?.terminate() }
    }

    var wasCancelled: Bool {
        lock.lock()
        defer { lock.unlock() }
        return cancelled
    }
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
                "Clé Mistral absente. Ajoutez-la dans Réglages > Mistral OCR."
            )
        }
        if converterKind == .mineru {
            let token = FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(".mineru_token")
            guard FileManager.default.fileExists(atPath: token.path) else {
                throw PipelineError.missingCredential("Jeton MinerU absent de ~/.mineru_token.")
            }
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
        label: String,
        timeout: TimeInterval? = nil
    ) async throws -> ProcessResult {
        let controller = ProcessController()
        return try await withTaskCancellationHandler {
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
                controller.attach(process)
                defer { controller.detach() }
                let deadline = timeout.map { Date().addingTimeInterval($0) }
                var didTimeOut = false
                while process.isRunning {
                    if controller.wasCancelled { break }
                    if let deadline, Date() >= deadline {
                        didTimeOut = true
                        process.terminate()
                        break
                    }
                    try? await Task.sleep(for: .milliseconds(100))
                }
                if process.isRunning {
                    let grace = Date().addingTimeInterval(3)
                    while process.isRunning, Date() < grace {
                        try? await Task.sleep(for: .milliseconds(50))
                    }
                }
                if process.isRunning {
                    let killResult = kill(process.processIdentifier, SIGKILL)
                    let killError = killResult == 0 ? nil : String(cString: strerror(errno))
                    let killGrace = Date().addingTimeInterval(3)
                    while process.isRunning, Date() < killGrace {
                        try? await Task.sleep(for: .milliseconds(50))
                    }
                    if process.isRunning {
                        let reason = killError.map { "SIGKILL a échoué : \($0)" }
                            ?? "Le processus est resté actif après SIGKILL."
                        throw PipelineError.processFailed(
                            command: label,
                            details: "\(reason) Vous pouvez relancer ce PDF."
                        )
                    }
                }
                try? outputHandle.synchronize()
                try? errorHandle.synchronize()
                let output = String(decoding: (try? Data(contentsOf: outputURL)) ?? Data(), as: UTF8.self)
                let error = String(decoding: (try? Data(contentsOf: errorURL)) ?? Data(), as: UTF8.self)
                if controller.wasCancelled { throw CancellationError() }
                if didTimeOut, let timeout {
                    throw PipelineError.timedOut(command: label, minutes: max(1, Int(timeout / 60)))
                }
                guard process.terminationStatus == 0 else {
                    let details = String((error.isEmpty ? output : error).suffix(8_000))
                    throw PipelineError.processFailed(command: label, details: details)
                }
                return ProcessResult(output: output, error: error)
            }.value
        } onCancel: {
            controller.cancel()
        }
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
