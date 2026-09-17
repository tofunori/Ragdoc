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
        WindowGroup("Ragdrop — Clair & sombre · Validation") { ThemeDemoWorkspace() }
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
    @State private var dataState = "Chargé"
    @State private var batchState = "Aucun"
    @State private var showReview = false
    @State private var showZotero = false
    @State private var showError = false
    @State private var errorMessage: String?

    var body: some View {
        VStack(spacing: 0) {
            WorkspaceView(store: store, history: history, status: status, monitor: monitor, section: $section)
            HStack(spacing: 12) {
                Label("DÉMONSTRATION · données synthétiques", systemImage: "testtube.2").font(.caption.weight(.semibold))
                Spacer()
                Picker("État", selection: $dataState) {
                    ForEach(["Chargé", "Vide", "Chargement", "Échec"], id: \.self) { Text($0) }
                }.frame(width: 155)
                Picker("Lot", selection: $batchState) {
                    ForEach(["Aucun", "Conversion", "Révision", "Transfert", "Indexation", "Contrôle", "Erreur", "Doublons", "Terminé", "Mixte"], id: \.self) { Text($0) }
                }.frame(width: 160)
                Button("Nouveauté Zotero") {
                    if var document = zotero.documents.last {
                        document.fingerprint = String(repeating: "a", count: 64)
                        document.fingerprintPrefix = String(repeating: "a", count: 12)
                        monitor.simulate([document])
                    }
                }
                Button("Révision") { showReview = true }.disabled(fixtures.isEmpty)
                Button("Zotero") { showZotero = true }
                Button("Erreur") { showError = true }.disabled(fixtures.isEmpty)
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
                let _ = { job.detail = "Conversion interrompue"; job.errorDetails = "Exemple synthétique : délai de réponse dépassé. Aucun traitement réel n’a été lancé." }()
                ErrorDetailView(job: job)
            }
        }
        .onChange(of: dataState) { _, _ in applyDataState() }
        .onChange(of: batchState) { _, _ in applyBatchState() }
        .task {
            do {
                fixtures = try ReviewDemoFixtures.make()
                sampleHistory = [
                    HistoryDocument(source: "demo-1.md", title: "Démonstration — Albédo et impuretés de la neige", chunks: 14, indexedDate: "2026-09-17T14:30:00", doi: nil),
                    HistoryDocument(source: "demo-2.md", title: "Démonstration — Méthodes de mesure sur glacier", chunks: 28, indexedDate: "2026-09-16T09:15:00", doi: nil),
                    HistoryDocument(source: "demo-3.md", title: "Démonstration — Rayonnement et bilan d’énergie", chunks: 19, indexedDate: "2026-09-15T16:00:00", doi: nil)
                ]
                zotero.documents = sampleHistory.enumerated().map { index, item in
                    ZoteroPDF(attachmentKey: "DEMO\(index)", parentKey: nil, title: item.title, authorNames: ["Auteur de démonstration"], year: "2026", doi: nil,
                              fileURL: fixtures[0].fileURL, dateAdded: item.indexedDate ?? "", isIndexed: index == 0)
                }
                applyDataState()
            } catch { errorMessage = error.localizedDescription }
        }
    }

    private func applyBatchState() {
        guard var job = fixtures.first else { return }
        store.message = "État simulé · aucun traitement réel."
        store.isRunning = false
        job.metadata = ImportMetadata(title: "Démonstration — Albédo, poussières et carbone noir", authors: ["Auteurs de démonstration"], year: 2026, doi: nil, zoteroItemKey: nil, zoteroAttachmentKey: nil)
        job.converterName = "Mistral OCR"
        job.errorDetails = nil
        job.stageStartedAt = .now
        switch batchState {
        case "Aucun": store.jobs = []; return
        case "Conversion": job.stage = .converting
        case "Révision": job.stage = .awaitingReview
        case "Transfert": job.stage = .transferring
        case "Indexation": job.stage = .indexing
        case "Contrôle": job.stage = .verifying
        case "Erreur": job.stage = .readyForIndexing; job.errorDetails = "Indexation indisponible — erreur simulée."
        case "Doublons": job.stage = .duplicate
        case "Terminé": job.stage = .completed
        case "Mixte":
            job.stage = .converting
            var awaiting = ImportJob(fileURL: job.fileURL, metadata: job.metadata); awaiting.stage = .awaitingReview
            var failed = ImportJob(fileURL: job.fileURL, metadata: job.metadata); failed.stage = .failed; failed.errorDetails = "Erreur simulée"
            var completed = ImportJob(fileURL: job.fileURL, metadata: job.metadata); completed.stage = .completed
            store.jobs = [job, awaiting, failed, completed]; store.isRunning = true; return
        default: return
        }
        job.detail = "Démonstration · " + job.stage.title
        store.jobs = [job]
        store.isRunning = job.stage.isActive
    }

    private func applyDataState() {
        history.documents = dataState == "Chargé" ? sampleHistory : []
        history.lastUpdated = .now
        history.isLoading = dataState == "Chargement"
        history.errorMessage = dataState == "Échec" ? "Serveur indisponible — exemple de validation." : nil
        status.snapshot = dataState == "Chargé" ? RagdocStatusSnapshot(
            tools: ["semantic_search_hybrid", "search_by_source", "list_documents", "get_document_content", "get_indexation_status", "get_server_status"],
            searchOK: true, mcpError: nil, latencySeconds: 0.8, documents: 3, chunks: 61,
            revision: "demo-synthetic", writeState: "ready", repairing: false, models: ["demo-model": 3], expectedModel: "demo-model",
            serverModel: "demo-model", serverRevision: "demo-synthetic", lexicalReady: true, rerankingModel: "rerank-v4.0-pro", listenerCount: 1, testSource: "demo-1.md") : nil
        status.lastChecked = .now
        status.isChecking = dataState == "Chargement"
        status.errorMessage = dataState == "Échec" ? "Connexion indisponible — exemple de validation." : nil
    }
}
#endif
