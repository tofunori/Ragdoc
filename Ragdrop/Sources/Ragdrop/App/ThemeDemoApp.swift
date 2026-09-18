#if RAGDROP_THEME_DEMO
import SwiftUI

final class ThemeValidationDelegate: NSObject, NSApplicationDelegate {
    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.regular)
    }
}

@main
struct ThemeDemoApp: App {
    @NSApplicationDelegateAdaptor(ThemeValidationDelegate.self) private var appDelegate
    var body: some Scene {
        WindowGroup("Ragdrop — Light & dark · Demo") { ThemeDemoWorkspace() }
            .windowStyle(.hiddenTitleBar)
        .defaultSize(width: 1200, height: 900)
    }
}

private struct ThemeDemoWorkspace: View {
    @State private var store = ImportStore(
        queueStoreURL: FileManager.default.temporaryDirectory.appendingPathComponent("ragdrop-theme-\(UUID().uuidString)/queue.json"),
        isIsolated: true)
    @State private var history = HistoryStore(isIsolated: true)
    @State private var status = RagdocStatusStore(isIsolated: true)
    @State private var monitor = ZoteroMonitorStore(isIsolated: true)
    @State private var zotero = ZoteroImportStore(isIsolated: true)
    @State private var section = WorkspaceSection.home
    @State private var fixtures: [ImportJob] = []
    @State private var sampleHistory: [HistoryDocument] = []
    @State private var dataState = "Loaded"
    @State private var batchState = "None"
    @State private var showReview = false
    @State private var showZotero = false
    @State private var showError = false
    @State private var errorMessage: String?

    var body: some View {
        VStack(spacing: 0) {
            WorkspaceView(store: store, history: history, status: status, monitor: monitor, section: $section)
            HStack(spacing: 12) {
                Label("DEMO · synthetic data", systemImage: "testtube.2").font(.caption.weight(.semibold))
                Spacer()
                Picker("State", selection: $dataState) {
                    ForEach(["Loaded", "Empty", "Loading", "Failed"], id: \.self) { Text($0) }
                }.frame(width: 155)
                Picker("Batch", selection: $batchState) {
                    ForEach(["None", "Conversion", "Review", "Transfer", "Indexing", "Verification", "Error", "Duplicates", "Finished", "Mixed"], id: \.self) { Text($0) }
                }.frame(width: 160)
                Button("New in Zotero") {
                    if var document = zotero.documents.last {
                        document.fingerprint = String(repeating: "a", count: 64)
                        document.fingerprintPrefix = String(repeating: "a", count: 12)
                        monitor.simulate([document])
                    }
                }
                Button("Review") { showReview = true }.disabled(fixtures.isEmpty)
                Button("Zotero") { showZotero = true }
                Button("Error") { showError = true }.disabled(fixtures.isEmpty)
            }.padding(10).background(RagdropTheme.raised)
            if let errorMessage { Text(errorMessage).foregroundStyle(.orange) }
        }
        .ragdropSurface()
        .sheet(isPresented: $showReview) {
            if let job = fixtures.first {
                MarkdownPreviewView(job: job, onApprove: { showReview = false }, onReject: { showReview = false })
            }
        }
        .sheet(isPresented: $showZotero) { ZoteroImportView(store: zotero) { _ in } }
        .sheet(isPresented: $showError) {
            if var job = fixtures.first {
                let _ = { job.detail = "Conversion interrupted"; job.errorDetails = "Synthetic example: response timed out. No real processing was started." }()
                ErrorDetailView(job: job)
            }
        }
        .onChange(of: dataState) { _, _ in applyDataState() }
        .onChange(of: batchState) { _, _ in applyBatchState() }
        .task {
            do {
                fixtures = try ReviewDemoFixtures.make()
                sampleHistory = [
                    HistoryDocument(source: "demo-1.md", title: "Demo — Snow albedo and impurities", chunks: 14, indexedDate: "2026-09-17T14:30:00", doi: nil),
                    HistoryDocument(source: "demo-2.md", title: "Demo — Glacier measurement methods", chunks: 28, indexedDate: "2026-09-16T09:15:00", doi: nil),
                    HistoryDocument(source: "demo-3.md", title: "Demo — Radiation and energy balance", chunks: 19, indexedDate: "2026-09-15T16:00:00", doi: nil)
                ]
                zotero.documents = sampleHistory.enumerated().map { index, item in
                    ZoteroPDF(attachmentKey: "DEMO\(index)", parentKey: nil, title: item.title, authorNames: ["Demo author"], year: "2026", doi: nil,
                              fileURL: fixtures[0].fileURL, dateAdded: item.indexedDate ?? "", isIndexed: index == 0)
                }
                applyDataState()
            } catch { errorMessage = error.localizedDescription }
        }
    }

    private func applyBatchState() {
        guard var job = fixtures.first else { return }
        store.message = "Simulated state · no real processing."
        store.isRunning = false
        job.metadata = ImportMetadata(title: "Demo — Albedo, dust and black carbon", authors: ["Demo authors"], year: 2026, doi: nil, zoteroItemKey: nil, zoteroAttachmentKey: nil)
        job.converterName = "Mistral OCR"
        job.errorDetails = nil
        job.stageStartedAt = .now
        switch batchState {
        case "None": store.jobs = []; return
        case "Conversion": job.stage = .converting
        case "Review": job.stage = .awaitingReview
        case "Transfer": job.stage = .transferring
        case "Indexing": job.stage = .indexing
        case "Verification": job.stage = .verifying
        case "Error": job.stage = .readyForIndexing; job.errorDetails = "Indexing unavailable — simulated error."
        case "Duplicates": job.stage = .duplicate
        case "Finished": job.stage = .completed
        case "Mixed":
            job.stage = .converting
            var awaiting = ImportJob(fileURL: job.fileURL, metadata: job.metadata); awaiting.stage = .awaitingReview
            var failed = ImportJob(fileURL: job.fileURL, metadata: job.metadata); failed.stage = .failed; failed.errorDetails = "Simulated error"
            var completed = ImportJob(fileURL: job.fileURL, metadata: job.metadata); completed.stage = .completed
            store.jobs = [job, awaiting, failed, completed]; store.isRunning = true; return
        default: return
        }
        job.detail = "Demo · " + job.stage.title
        store.jobs = [job]
        store.isRunning = job.stage.isActive
    }

    private func applyDataState() {
        history.documents = dataState == "Loaded" ? sampleHistory : []
        history.lastUpdated = .now
        history.isLoading = dataState == "Loading"
        history.errorMessage = dataState == "Failed" ? "Server unavailable — demo example." : nil
        status.snapshot = dataState == "Loaded" ? RagdocStatusSnapshot(
            tools: ["semantic_search_hybrid", "search_by_source", "list_documents", "get_document_content", "get_indexation_status", "get_server_status"],
            searchOK: true, mcpError: nil, latencySeconds: 0.8, documents: 3, chunks: 61,
            revision: "demo-synthetic", writeState: "ready", repairing: false, models: ["demo-model": 3], expectedModel: "demo-model",
            serverModel: "demo-model", serverRevision: "demo-synthetic", lexicalReady: true, rerankingModel: "rerank-v4.0-pro", listenerCount: 1, testSource: "demo-1.md") : nil
        status.lastChecked = .now
        status.isChecking = dataState == "Loading"
        status.errorMessage = dataState == "Failed" ? "Connection unavailable — demo example." : nil
    }
}
#endif
