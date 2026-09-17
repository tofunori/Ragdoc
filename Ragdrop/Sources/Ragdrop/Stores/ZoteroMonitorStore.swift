import Foundation
import Observation

struct ZoteroMonitorServices: Sendable {
    var localPDFs: @Sendable () async throws -> [ZoteroPDF]
    var indexedDocuments: @Sendable () async throws -> [HistoryDocument]
    static let live = ZoteroMonitorServices(
        localPDFs: { try await ZoteroLibraryService().fetchPDFs(includeFingerprints: false) },
        indexedDocuments: { try await RagdocHistoryService(configuration: .current()).fetchDocuments(includePDFIdentity: true) }
    )
}

@MainActor @Observable
final class ZoteroMonitorStore {
    private(set) var isEnabled: Bool
    private(set) var pendingDocuments: [ZoteroPDF] = []
    private(set) var isChecking = false
    private(set) var lastLocalCheck: Date?
    private(set) var lastRagdocCheck: Date?
    private(set) var statusText = "Surveillance désactivée."
    private(set) var errorMessage: String?
    var showingNewArticles = false
    let isIsolated: Bool
    @ObservationIgnored private let services: ZoteroMonitorServices
    @ObservationIgnored private let defaults: UserDefaults
    @ObservationIgnored private let ledgerURL: URL
    @ObservationIgnored private var ledger: ZoteroWatchLedger
    @ObservationIgnored private var task: Task<Void, Never>?
    @ObservationIgnored private var generation = UUID()
    @ObservationIgnored private let fingerprints = ZoteroFingerprintCache()
    @ObservationIgnored private weak var observedQueue: ImportStore?
    @ObservationIgnored private var cachedHistory: [HistoryDocument]?

    init(isIsolated: Bool = false, defaults: UserDefaults? = nil, ledgerURL: URL? = nil, services: ZoteroMonitorServices = .live) {
        self.services = services
        self.isIsolated = isIsolated
        let prefs = defaults ?? (isIsolated ? UserDefaults(suiteName: "com.tofunori.ragdrop.theme-demo")! : .standard)
        self.defaults = prefs
        self.isEnabled = isIsolated ? false : prefs.bool(forKey: "watchZotero")
        let path = ledgerURL ?? (isIsolated
            ? FileManager.default.temporaryDirectory.appendingPathComponent("ragdrop-watch-demo-\(UUID().uuidString).json")
            : FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
                .appendingPathComponent("Ragdrop/zotero-watch.json"))
        self.ledgerURL = path
        self.ledger = (try? Data(contentsOf: path)).flatMap { try? JSONDecoder().decode(ZoteroWatchLedger.self, from: $0) } ?? ZoteroWatchLedger()
        if isEnabled { statusText = "En attente de la première lecture de Zotero." }
    }

    func setEnabled(_ value: Bool) {
        guard value != isEnabled else { return }
        generation = UUID()
        task?.cancel()
        task = nil
        isEnabled = value
        defaults.set(value, forKey: "watchZotero")
        pendingDocuments = []
        errorMessage = nil
        // Each explicit activation establishes a new baseline, without announcing the backlog.
        ledger.hasBaseline = false
        ledger.pending = []
        cachedHistory = nil
        lastRagdocCheck = nil
        statusText = value ? "Création de l’état de référence au prochain contrôle…" : "Surveillance désactivée."
        persist()
    }

    func start(queue: ImportStore) {
        observedQueue = queue
        guard isEnabled, !isIsolated, task == nil else { return }
        task = Task { [weak self, weak queue] in
            while !Task.isCancelled {
                guard let self, let queue, self.isEnabled else { return }
                if self.isChecking {
                    do { try await Task.sleep(for: .seconds(2)) } catch { return }
                    continue
                }
                await self.check(queue: queue)
                do { try await Task.sleep(for: .seconds(300)) } catch { return }
            }
        }
    }

    func check(queue: ImportStore) async {
        guard isEnabled, !isIsolated, !isChecking else { return }
        if queue.isRunning { statusText = "Surveillance en attente de la fin du traitement en cours."; return }
        isChecking = true
        defer { isChecking = false }
        let epoch = generation
        do {
            // Read Zotero's local HTTP API, never its live SQLite file.
            let local = try await services.localPDFs()
            guard !Task.isCancelled, epoch == generation, isEnabled else { return }
            lastLocalCheck = .now
            if !ledger.hasBaseline {
                ledger.establishBaseline(local)
                pendingDocuments = []
                errorMessage = nil
                statusText = "État de référence créé : \(local.count) PDF locaux déjà présents. Les prochains ajouts seront signalés."
                persist()
                return
            }
            let candidates = ledger.candidates(in: local)
            guard !candidates.isEmpty else {
                pendingDocuments = []
                errorMessage = nil
                statusText = "Aucun nouveau PDF local à vérifier."
                return
            }
            let enriched = try await fingerprints.enrich(candidates)
            let queueHashes = try await fingerprints.hashes(for: queue.jobs)
            guard !Task.isCancelled, epoch == generation, isEnabled else { return }
            let hasUnseen = candidates.contains { !ledger.seenKeys.contains($0.attachmentKey) }
            let stale = lastRagdocCheck.map { Date().timeIntervalSince($0) >= 1800 } ?? true
            if cachedHistory == nil || hasUnseen || stale {
                let fresh = try await services.indexedDocuments()
                guard !Task.isCancelled, epoch == generation, isEnabled else { return }
                cachedHistory = fresh
                lastRagdocCheck = .now
            }
            ledger.reconcile(enriched, history: cachedHistory ?? [], jobs: queue.jobs, queueHashes: queueHashes)
            pendingDocuments = ledger.availableNotifications(in: local)
            errorMessage = nil
            statusText = pendingDocuments.isEmpty ? "Aucun nouveau PDF à intégrer après contrôle des doublons."
                : "\(pendingDocuments.count) PDF locaux à examiner."
            persist()
        } catch {
            guard !Task.isCancelled, epoch == generation, isEnabled else { return }
            pendingDocuments = []
            errorMessage = error.localizedDescription
            statusText = "Vérification différée. Aucune nouveauté n’est annoncée sans contrôle de Ragdoc."
            // Do not acknowledge unseen items on error: retry them after recovery.
        }
    }

    func checkNow() {
        guard let observedQueue else { return }
        Task { await check(queue: observedQueue) }
    }

    func reconcileQueue(_ jobs: [ImportJob]) {
        let keys = Set(ledger.pending.filter { ZoteroDeduplication.isQueued($0, jobs: jobs) }.map(\.attachmentKey))
        if !keys.isEmpty { dismiss(keys) }
        cachedHistory = nil
    }

    func dismiss(_ keys: Set<String>) {
        ledger.dismiss(keys)
        let remainingKeys = Set(ledger.pending.map(\.attachmentKey))
        pendingDocuments.removeAll { !remainingKeys.contains($0.attachmentKey) }
        persist()
    }

    func selectionStore(jobs: [ImportJob]) -> ZoteroImportStore {
        let selection = ZoteroImportStore(isIsolated: isIsolated, allowedKeys: Set(pendingDocuments.map(\.attachmentKey)), excludedJobs: jobs)
        selection.documents = pendingDocuments
        return selection
    }

    func simulate(_ documents: [ZoteroPDF]) {
        guard isIsolated else { return }
        isEnabled = true
        ledger.hasBaseline = true
        ledger.pending = documents
        pendingDocuments = documents
        lastLocalCheck = .now
        lastRagdocCheck = .now
        statusText = "Nouveauté simulée · aucun accès à Zotero ou au NAS."
    }

    private func persist() {
        do {
            try FileManager.default.createDirectory(at: ledgerURL.deletingLastPathComponent(), withIntermediateDirectories: true)
            try JSONEncoder().encode(ledger).write(to: ledgerURL, options: .atomic)
        } catch {
            errorMessage = "Impossible de conserver l’état de surveillance. \(error.localizedDescription)"
        }
    }
}
