import Foundation
import Testing
@testable import Ragdrop

struct WorkspaceTests {
    @MainActor @Test func recentDocumentsAreOrderedByDateWithoutInventingMissingDates() {
        let history = HistoryStore(isIsolated: true)
        history.documents = [
            .init(source: "unknown", title: "Unknown", chunks: 2, indexedDate: nil, doi: nil),
            .init(source: "old", title: "Old", chunks: 3, indexedDate: "2024-01-01T12:00:00", doi: nil),
            .init(source: "latest", title: "Latest", chunks: 4, indexedDate: "2026-09-17T12:00:00", doi: nil),
            .init(source: "middle", title: "Middle", chunks: 5, indexedDate: "2025-01-01T12:00:00", doi: nil)
        ]
        #expect(history.recentDocuments.map(\.source) == ["latest", "middle", "old"])
        #expect(history.documents.first?.displayDate == "—")
    }

    @MainActor @Test func isolatedStoresCannotStartImportOrRefreshLiveServices() async throws {
        let folder = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: folder) }
        let imports = ImportStore(queueStoreURL: folder.appendingPathComponent("queue.json"), isIsolated: true)
        var job = ImportJob(fileURL: folder.appendingPathComponent("sample.pdf"))
        job.stage = .queued
        imports.jobs = [job]
        imports.start()
        #expect(!imports.isRunning)
        #expect(imports.jobs.first?.stage == .queued)
        imports.jobs[0].stage = .failed
        imports.retry(job.id)
        #expect(!imports.isRunning)
        #expect(imports.jobs.first?.stage == .failed)
        let history = HistoryStore(isIsolated: true)
        let status = RagdocStatusStore(isIsolated: true)
        let zotero = ZoteroImportStore(isIsolated: true)
        await history.refresh()
        await status.refresh()
        await zotero.load()
        #expect(history.lastUpdated == nil)
        #expect(status.lastChecked == nil)
        #expect(zotero.documents.isEmpty)
        #expect(history.errorMessage == nil && status.errorMessage == nil && zotero.errorMessage == nil)
    }

    @MainActor @Test func isolatedRejectionDoesNotDeleteFiles() throws {
        let folder = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: folder) }
        let artifact = folder.appendingPathComponent("sample.md")
        try "Preserve fixture".write(to: artifact, atomically: true, encoding: .utf8)
        let store = ImportStore(queueStoreURL: folder.appendingPathComponent("queue.json"), isIsolated: true)
        var job = ImportJob(fileURL: folder.appendingPathComponent("sample.pdf"))
        job.artifactURL = artifact
        job.stage = .awaitingReview
        store.jobs = [job]
        store.reject(job.id)
        #expect(store.jobs.first?.stage == .rejected)
        #expect(try String(contentsOf: artifact, encoding: .utf8) == "Preserve fixture")
    }
}
