import AppKit
import Foundation
import Observation

@MainActor
@Observable
final class LocalSetupStore {
    var library: URL
    var isPreparing = false
    var progress = ""
    var errorMessage: String?
    var status: LocalLibraryStatus?
    var isReady: Bool
    var mistralConfigured = false
    var voyageConfigured = false
    var credentialMessage: String?
    private let engine: LocalEngine
    private let defaults: UserDefaults
    let isPreview: Bool
    let allowsPersonalActions: Bool
    @ObservationIgnored private var task: Task<Void, Never>?

    init(engine: LocalEngine = .current, defaults: UserDefaults = .standard, isPreview: Bool = false, allowsPersonalActions: Bool = true) {
        self.engine = engine
        self.defaults = defaults
        self.isPreview = isPreview
        self.allowsPersonalActions = allowsPersonalActions
        library = engine.recordedConnection.map { URL(fileURLWithPath: $0.library) }
            ?? defaults.string(forKey: "localLibraryPath").map { URL(fileURLWithPath: $0) } ?? engine.defaultLibrary
        isReady = !isPreview && engine.connection != nil
    }
    func refreshKeys() {
        guard !isPreview, allowsPersonalActions else { return }
        mistralConfigured = MistralCredentialStore.isConfigured
        voyageConfigured = VoyageCredentialStore.isConfigured
    }
    func chooseFolder() {
        guard !isPreparing, !isReady, allowsPersonalActions else { return }
        let panel = NSOpenPanel()
        panel.canChooseFiles = false; panel.canChooseDirectories = true; panel.canCreateDirectories = true
        panel.allowsMultipleSelection = false; panel.prompt = "Use this folder"
        if panel.runModal() == .OK, let url = panel.url { library = url }
    }
    func prepare() {
        guard !isPreparing, !isPreview else { return }
        isPreparing = true; errorMessage = nil
        let selected = library
        task = Task { [self] in
            do {
                _ = try await engine.prepare(library: selected) { [weak self] text in
                    await MainActor.run { self?.progress = text }
                }
                status = try await engine.decoded("status")
                isReady = true
                defaults.set(selected.path, forKey: "localLibraryPath")
                progress = "Your local engine is ready."
            } catch is CancellationError {
                progress = "Setup stopped. You can retry; your articles are kept."
            } catch { errorMessage = error.localizedDescription }
            isPreparing = false
            task = nil
        }
    }
    func rebuildIndex() {
        guard isReady, !isPreparing, !isPreview, allowsPersonalActions else { return }
        isPreparing = true; errorMessage = nil
        progress = "Backing up and rebuilding the local index…"
        task = Task { [self] in
            do {
                _ = try await engine.call("repair", confirmRebuild: true)
                status = try await engine.decoded("status")
                progress = "Index rebuilt. The previous files are kept in the library’s Backups folder."
            } catch { errorMessage = error.localizedDescription }
            isPreparing = false; task = nil
        }
    }
    func cancel() { task?.cancel() }
    func check() async {
        guard !isPreparing, isReady, !isPreview else { return }
        do { status = try await engine.decoded("status"); errorMessage = nil }
        catch { errorMessage = error.localizedDescription }
    }
    func saveKeys(mistral: String, voyage: String) {
        guard !isPreview, allowsPersonalActions else { return }
        do {
            if !mistral.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty { try MistralCredentialStore.save(mistral) }
            if !voyage.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty { try VoyageCredentialStore.save(voyage) }
            refreshKeys()
            credentialMessage = "Saved in macOS Keychain. Service access is checked when you import or search."
        } catch { errorMessage = error.localizedDescription }
    }
    func connectClaude() {
        guard isReady, !isPreview, allowsPersonalActions else { return }
        if !NSWorkspace.shared.open(engine.extensionURL) {
            NSWorkspace.shared.activateFileViewerSelecting([engine.extensionURL])
            credentialMessage = "In Claude Desktop: Settings → Extensions → Advanced settings → Install Extension. Select Ragdoc.mcpb."
        }
    }
}
