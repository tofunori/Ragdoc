import Foundation
import CryptoKit

enum LibraryLocation: String, CaseIterable, Sendable {
    case local, server

    static func resolve(defaults: UserDefaults = .standard, hasLegacyQueue: Bool? = nil) -> Self {
        if let raw = defaults.string(forKey: "libraryLocation"), let saved = Self(rawValue: raw) { return saved }
        let queue = FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent("Library/Application Support/Ragdrop/queue.json")
        // Missing preference on an existing installation must never redirect an import.
        if (hasLegacyQueue ?? FileManager.default.fileExists(atPath: queue.path))
            || defaults.object(forKey: "nasHost") != nil || defaults.object(forKey: "converterPath") != nil {
            return .server
        }
        return .local
    }
}

struct LocalConnection: Codable, Sendable {
    let version: Int
    let python: String
    let script: String
    let library: String
}

struct LocalLibraryStatus: Codable, Sendable {
    let documents: Int
    let chunks: Int
    let ready: Bool
    let needsRepair: Bool
    let issues: [String]
}

struct LocalEngine: Sendable {
    let support: URL
    let resources: URL
    static var current: Self {
        Self(support: FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent("Library/Application Support/Ragdrop"),
             resources: Bundle.main.resourceURL!)
    }
    var connectionURL: URL { support.appendingPathComponent("local-connection.json") }
    var recordedConnection: LocalConnection? {
        guard let data = try? Data(contentsOf: connectionURL),
              let value = try? JSONDecoder().decode(LocalConnection.self, from: data), value.version == 1 else { return nil }
        return value
    }
    var connection: LocalConnection? {
        guard let value = recordedConnection, FileManager.default.isExecutableFile(atPath: value.python),
              FileManager.default.fileExists(atPath: value.script) else { return nil }
        return value
    }
    var defaultLibrary: URL { support.appendingPathComponent("Library") }
    var extensionURL: URL { resources.appendingPathComponent("Ragdoc.mcpb") }

    static func cleanEnvironment() -> [String: String] {
        var env = ProcessInfo.processInfo.environment.filter { key, _ in
            !key.hasPrefix("RAGDOC_") && !key.hasPrefix("CHROMA_") && !key.hasPrefix("UV_")
                && !["VOYAGE_API_KEY", "COHERE_API_KEY", "MISTRAL_API_KEY", "PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "COLLECTION_NAME"].contains(key)
        }
        env["PYTHON_DOTENV_DISABLED"] = "1"
        env["ANONYMIZED_TELEMETRY"] = "False"
        return env
    }

    func prepare(library: URL, progress: @escaping @Sendable (String) async -> Void) async throws -> LocalConnection {
        let fm = FileManager.default
        let source = resources.appendingPathComponent("LocalEngine")
        let uv = resources.appendingPathComponent("uv")
        guard fm.isExecutableFile(atPath: uv.path), fm.fileExists(atPath: source.appendingPathComponent("uv.lock").path) else {
            throw PipelineError.invalidConfiguration("This build does not include the local engine. Download the full Ragdrop app.")
        }
        // An immutable source digest gives upgrades a separate environment. Existing engines remain usable.
        let identity = try String(contentsOf: source.appendingPathComponent("engine-id"), encoding: .utf8).trimmingCharacters(in: .whitespacesAndNewlines)
        guard identity.range(of: "^[a-f0-9]{64}$", options: .regularExpression) != nil else {
            throw PipelineError.invalidConfiguration("The bundled engine is incomplete. Download Ragdrop again.")
        }
        let engine = support.appendingPathComponent("Engines/\(identity)")
        try fm.createDirectory(at: engine.deletingLastPathComponent(), withIntermediateDirectories: true)
        if !fm.fileExists(atPath: engine.path) {
            let staging = engine.deletingLastPathComponent().appendingPathComponent(".staging-\(UUID().uuidString)")
            defer { try? fm.removeItem(at: staging) }
            try fm.copyItem(at: source, to: staging)
            try fm.moveItem(at: staging, to: engine)
        }
        var env = Self.cleanEnvironment()
        env["UV_PYTHON_INSTALL_DIR"] = support.appendingPathComponent("Python").path
        env["UV_CACHE_DIR"] = support.appendingPathComponent("Cache/uv").path
        env["UV_PYTHON_PREFERENCE"] = "only-managed"
        env["UV_PROJECT_ENVIRONMENT"] = engine.appendingPathComponent(".venv").path
        env["NLTK_DATA"] = engine.appendingPathComponent("nltk_data").path
        env["HF_HOME"] = engine.appendingPathComponent("model_cache").path
        await progress("Downloading the private Python runtime and engine…")
        _ = try await ProcessRunner.run(executable: uv.path,
            arguments: ["sync", "--locked", "--no-dev", "--extra", "desktop", "--no-install-project", "--python", "3.12", "--project", engine.path],
            label: "Engine installation", timeout: 1200, environment: env)
        let python = engine.appendingPathComponent(".venv/bin/python")
        await progress("Preparing language resources…")
        _ = try await ProcessRunner.run(executable: python.path, arguments: ["-c",
            "import nltk,sys; assert nltk.download('stopwords', download_dir=sys.argv[1], quiet=True); from chonkie import TokenChunker; TokenChunker(tokenizer='gpt2', chunk_size=512)", env["NLTK_DATA"]!],
            label: "Language resources", timeout: 600, environment: env)
        let value = LocalConnection(version: 1, python: python.path,
            script: engine.appendingPathComponent("scripts/ragdrop_local.py").path, library: library.path)
        await progress("Preparing your library and checking the engine…")
        _ = try await call("prepare", connection: value)
        let status: LocalLibraryStatus = try await decoded("status", connection: value)
        guard status.ready || (status.needsRepair && status.issues.isEmpty) else { throw PipelineError.invalidConfiguration("Engine check: \(status.issues.joined(separator: ", "))") }
        let data = try JSONEncoder().encode(value)
        try data.write(to: connectionURL, options: .atomic)
        try fm.setAttributes([.posixPermissions: 0o600], ofItemAtPath: connectionURL.path)
        return value
    }

    func call(_ action: String, sources: [String] = [], confirmRebuild: Bool = false, connection explicit: LocalConnection? = nil) async throws -> Data {
        guard let value = explicit ?? connection else {
            throw PipelineError.invalidConfiguration("Open Settings and prepare your local library first.")
        }
        let result = try await ProcessRunner.run(executable: value.python,
            arguments: [value.script, "--root", value.library, action] + sources.flatMap { ["--source", $0] } + (confirmRebuild ? ["--confirm-rebuild"] : []),
            label: "Local Ragdoc", timeout: ["index", "repair"].contains(action) ? 1800 : 120, environment: Self.cleanEnvironment())
        return Data(result.output.utf8)
    }
    func decoded<T: Decodable>(_ action: String, connection: LocalConnection? = nil) async throws -> T {
        try JSONDecoder().decode(T.self, from: await call(action, connection: connection))
    }
}
