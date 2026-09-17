import Foundation
import Testing
@testable import Ragdrop

private actor WatchFeed {
    var local: [ZoteroPDF] = []
    var history: [HistoryDocument] = []
    var fail = false
    var localReads = 0
    var remoteReads = 0
    func set(_ documents: [ZoteroPDF], fail: Bool = false, history: [HistoryDocument] = []) {
        local = documents; self.fail = fail; self.history = history
    }
    func readLocal() -> [ZoteroPDF] { localReads += 1; return local }
    func readHistory() throws -> [HistoryDocument] {
        remoteReads += 1
        if fail { throw HistoryServiceError.invalidResponse }
        return history
    }
    var counts: (Int, Int) { (localReads, remoteReads) }
}

struct ZoteroWatchTests {
    private func pdf(_ key: String, at url: URL, hash: String? = nil) -> ZoteroPDF {
        ZoteroPDF(attachmentKey: key, parentKey: nil, title: key, authorNames: [], year: nil, doi: nil,
                  fileURL: url, dateAdded: "2026-09-17", fingerprint: hash,
                  fingerprintPrefix: hash.map { String($0.prefix(12)) }, isIndexed: false)
    }

    @Test func baselineDoesNotAnnounceBacklogAndIgnoresSurviveDuplicateKeys() throws {
        let url = URL(fileURLWithPath: "/tmp/example.pdf")
        let old = pdf("OLD", at: url, hash: "old")
        let new = pdf("NEW", at: url, hash: "new")
        var ledger = ZoteroWatchLedger()
        ledger.establishBaseline([old])
        #expect(ledger.candidates(in: [old]).isEmpty)
        ledger.reconcile([new], history: [], jobs: [], queueHashes: [])
        #expect(ledger.pending.map(\.attachmentKey) == ["NEW"])
        ledger.dismiss(["NEW"])
        ledger.reconcile([pdf("COPY", at: url, hash: "new")], history: [], jobs: [], queueHashes: [])
        #expect(ledger.pending.isEmpty)
        let restored = try JSONDecoder().decode(ZoteroWatchLedger.self, from: JSONEncoder().encode(ledger))
        #expect(restored.dismissedFingerprints.contains("new"))
    }

    @Test func dedupUsesCanonicalIdentityLegacyNamesAndQueueHashes() {
        let doc = pdf("KEY123", at: URL(fileURLWithPath: "/tmp/new.pdf"), hash: String(repeating: "a", count: 64))
        var identity = HistoryDocument(source: "arbitrary.md", title: "Title", chunks: 1, indexedDate: nil, doi: nil)
        identity.pdfFingerprint = doc.fingerprint
        #expect(ZoteroDeduplication.isIndexed(doc, history: [identity]))
        identity.pdfFingerprint = nil
        identity.zoteroAttachmentKey = doc.attachmentKey
        #expect(ZoteroDeduplication.isIndexed(doc, history: [identity]))
        for source in ["paper_KEY123.md", "paper_aaaaaaaaaaaa.md"] {
            #expect(ZoteroDeduplication.isIndexed(doc, history: [.init(source: source, title: "Title", chunks: 2, indexedDate: nil, doi: nil)]))
        }
        var ledger = ZoteroWatchLedger()
        ledger.reconcile([doc], history: [], jobs: [], queueHashes: [doc.fingerprint!])
        #expect(ledger.pending.isEmpty)
        ledger.reconcile([doc, pdf("COPY", at: doc.fileURL, hash: doc.fingerprint)], history: [], jobs: [], queueHashes: [])
        #expect(ledger.availableNotifications(in: [doc]).count == 1)
    }

    @Test func pendingAliasesSurviveMissingRepresentativeAndDismissTogether() throws {
        let first = pdf("A", at: URL(fileURLWithPath: "/tmp/a.pdf"), hash: "same")
        let copy = pdf("B", at: URL(fileURLWithPath: "/tmp/b.pdf"), hash: "same")
        var ledger = ZoteroWatchLedger()
        ledger.establishBaseline([])
        ledger.reconcile([first, copy], history: [], jobs: [], queueHashes: [])
        #expect(ledger.availableNotifications(in: [first, copy]).count == 1)
        ledger = try JSONDecoder().decode(ZoteroWatchLedger.self, from: JSONEncoder().encode(ledger))
        let remaining = ledger.candidates(in: [copy])
        #expect(remaining.map(\.attachmentKey) == ["B"])
        ledger.reconcile(remaining, history: [], jobs: [], queueHashes: [])
        #expect(ledger.availableNotifications(in: [copy]).map(\.attachmentKey) == ["B"])
        ledger.dismiss(["B"])
        #expect(ledger.pending.isEmpty)
    }

    @MainActor @Test func monitoringRecoversFromNASFailureCachesReadsAndRespectsOff() async throws {
        let folder = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        let suite = "ragdrop-watch-test-\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: suite)!
        defer { defaults.removePersistentDomain(forName: suite); try? FileManager.default.removeItem(at: folder) }
        let feed = WatchFeed()
        let services = ZoteroMonitorServices(localPDFs: { await feed.readLocal() }, indexedDocuments: { try await feed.readHistory() })
        let monitor = ZoteroMonitorStore(defaults: defaults, ledgerURL: folder.appendingPathComponent("watch.json"), services: services)
        let queue = ImportStore(queueStoreURL: folder.appendingPathComponent("queue.json"), isIsolated: true)
        #expect(!monitor.isEnabled)
        await monitor.check(queue: queue)
        #expect(await feed.counts.0 == 0)
        monitor.setEnabled(true)
        await monitor.check(queue: queue) // Initial empty baseline, no NAS request.
        #expect(await feed.counts.1 == 0)
        let url = folder.appendingPathComponent("new.pdf")
        try Data("new PDF".utf8).write(to: url)
        let document = pdf("NEW", at: url)
        await feed.set([document], fail: true)
        await monitor.check(queue: queue)
        #expect(monitor.pendingDocuments.isEmpty && monitor.errorMessage != nil)
        await feed.set([document])
        await monitor.check(queue: queue)
        #expect(monitor.pendingDocuments.count == 1 && monitor.errorMessage == nil)
        let reads = await feed.counts.1
        await monitor.check(queue: queue)
        #expect(await feed.counts.1 == reads) // Pending alone uses the recent identity snapshot.
        let selected = monitor.selectionStore(jobs: [])
        #expect(selected.allowedKeys == ["NEW"])
        monitor.dismiss(["NEW"])
        await monitor.check(queue: queue)
        #expect(monitor.pendingDocuments.isEmpty)
        monitor.setEnabled(false)
        let before = await feed.counts.0
        await monitor.check(queue: queue)
        #expect(await feed.counts.0 == before)
        monitor.setEnabled(true)
        await monitor.check(queue: queue) // Re-enabling baselines the current library.
        #expect(monitor.pendingDocuments.isEmpty)
    }

    @MainActor @Test func notificationExcludesDifferentPathAlreadyInQueueAndPersistsAcknowledgement() async throws {
        let folder = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        let suite = "ragdrop-watch-test-\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: suite)!
        defer { defaults.removePersistentDomain(forName: suite); try? FileManager.default.removeItem(at: folder) }
        let feed = WatchFeed()
        let services = ZoteroMonitorServices(localPDFs: { await feed.readLocal() }, indexedDocuments: { try await feed.readHistory() })
        let monitor = ZoteroMonitorStore(defaults: defaults, ledgerURL: folder.appendingPathComponent("watch.json"), services: services)
        let queue = ImportStore(queueStoreURL: folder.appendingPathComponent("queue.json"), isIsolated: true)
        monitor.setEnabled(true)
        await monitor.check(queue: queue)
        let first = folder.appendingPathComponent("queued.pdf"), copy = folder.appendingPathComponent("zotero.pdf")
        try Data("identical bytes".utf8).write(to: first)
        try Data("identical bytes".utf8).write(to: copy)
        queue.jobs = [ImportJob(fileURL: first)] // No precomputed fingerprint.
        await feed.set([pdf("COPY", at: copy)])
        await monitor.check(queue: queue)
        #expect(monitor.pendingDocuments.isEmpty)
        let restored = ZoteroMonitorStore(defaults: defaults, ledgerURL: folder.appendingPathComponent("watch.json"), services: services)
        #expect(restored.isEnabled)
        queue.jobs = []
        await restored.check(queue: queue)
        #expect(restored.pendingDocuments.isEmpty)
    }

    @MainActor @Test func queuedAliasRemovesVisibleNotificationImmediately() {
        let path = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: path) }
        let monitor = ZoteroMonitorStore(isIsolated: true, ledgerURL: path)
        let first = pdf("A", at: URL(fileURLWithPath: "/tmp/a.pdf"), hash: "same")
        let copy = pdf("B", at: URL(fileURLWithPath: "/tmp/b.pdf"), hash: "same")
        monitor.simulate([first, copy])
        monitor.reconcileQueue([ImportJob(fileURL: copy.fileURL)])
        #expect(monitor.pendingDocuments.isEmpty)
    }

    @MainActor @Test func selectedCopiesAreAddedToQueueOnlyOnce() {
        let path = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: path) }
        let queue = ImportStore(queueStoreURL: path, isIsolated: true)
        let first = pdf("A", at: URL(fileURLWithPath: "/tmp/a.pdf"), hash: "same")
        let copy = pdf("B", at: URL(fileURLWithPath: "/tmp/b.pdf"), hash: "same")
        queue.addZoteroDocuments([first, copy])
        #expect(queue.jobs.count == 1)
        queue.addZoteroDocuments([copy])
        #expect(queue.jobs.count == 1)
    }

    @Test func fingerprintCacheInvalidatesWhenFileChanges() async throws {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: url) }
        try Data("one".utf8).write(to: url)
        let cache = ZoteroFingerprintCache()
        let original = try await cache.hash(url)
        #expect(try await cache.hash(url) == original)
        try Data("longer changed content".utf8).write(to: url)
        #expect(try await cache.hash(url) != original)
    }
}
