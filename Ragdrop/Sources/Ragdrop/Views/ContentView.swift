import SwiftUI
import UniformTypeIdentifiers

struct ContentView: View {
    @Bindable var store: ImportStore
    @Bindable var history: HistoryStore
    var reviewOnly = false
    let onLibrary: () -> Void
    @State private var isTargeted = false
    @State private var showingZotero = false
    @State private var previewJob: ImportJob?
    @State private var errorJob: ImportJob?
    @State private var showingQueue = false

    var body: some View {
        VStack(alignment: .leading, spacing: 20) {
            PageHeading(title: reviewOnly ? "À vérifier" : "Importation",
                        subtitle: reviewOnly ? "Comparez chaque extraction à son PDF original." : "Ajouter, vérifier, indexer.")
            if reviewOnly {
                queue
                footer
            } else if store.jobs.isEmpty {
                ScrollView {
                    VStack(spacing: 26) {
                        intake
                        RecentArticlesView(store: history, onLibrary: onLibrary)
                    }
                }
            } else {
                ScrollView {
                    VStack(alignment: .leading, spacing: 24) {
                        intake
                        BatchProgressView(jobs: store.jobs, message: store.message,
                                          onReview: { previewJob = $0 }, onError: { errorJob = $0 })
                        DisclosureGroup("Tous les articles (\(store.jobs.count))", isExpanded: $showingQueue) {
                            queue.frame(height: min(360, max(100, CGFloat(store.jobs.count) * 76)))
                                .padding(.top, 12)
                        }.font(.body).foregroundStyle(RagdropTheme.secondary)
                    }
                }
                footer
            }
        }
        .padding(RagdropTheme.pagePadding)
        .ragdropSurface()
        .fileImporter(
            isPresented: $store.showingFileImporter,
            allowedContentTypes: [.pdf],
            allowsMultipleSelection: true
        ) { result in
            if case .success(let urls) = result { store.addFiles(urls) }
        }
        .sheet(item: $previewJob) { job in
            let current = store.jobs.first(where: { $0.id == job.id }) ?? job
            MarkdownPreviewView(
                job: current,
                canDecide: !store.isRunning,
                remainingReviews: store.jobs.filter { $0.stage == .awaitingReview }.count,
                onNext: nextReview(after: job.id).map { next in { previewJob = next } },
                onApprove: {
                    guard !store.isRunning else { return }
                    let next = nextReview(after: job.id)
                    store.approve(job.id)
                    previewJob = next
                },
                onReject: {
                    guard !store.isRunning else { return }
                    let next = nextReview(after: job.id)
                    store.reject(job.id)
                    previewJob = next
                }
            ).id(job.id)

        }
        .sheet(item: $errorJob) { job in
            ErrorDetailView(job: job)
        }
        .sheet(isPresented: $showingZotero) {
            ZoteroImportView(isIsolated: store.isIsolated, excludedJobs: store.jobs) { documents in
                store.addZoteroDocuments(documents)
            }
        }
    }

    private func nextReview(after id: UUID) -> ImportJob? {
        let pending = store.jobs.filter { $0.stage == .awaitingReview && $0.id != id }
        guard let currentIndex = store.jobs.firstIndex(where: { $0.id == id }) else { return pending.first }
        return pending.first { candidate in
            (store.jobs.firstIndex(where: { $0.id == candidate.id }) ?? 0) > currentIndex
        } ?? pending.first
    }

    private var intake: some View {
        DropZoneView(isTargeted: isTargeted, compact: !store.jobs.isEmpty) {
            store.showingFileImporter = true
        } chooseZotero: { showingZotero = true }
        .dropDestination(for: URL.self) { urls, _ in
            store.addFiles(urls)
            return urls.contains { $0.pathExtension.lowercased() == "pdf" }
        } isTargeted: { isTargeted = $0 }
    }

    private var visibleJobs: [ImportJob] {
        reviewOnly ? store.jobs.filter { $0.stage == .awaitingReview } : store.jobs
    }

    @ViewBuilder
    private var queue: some View {
        if visibleJobs.isEmpty {
            ContentUnavailableView(
                reviewOnly ? "Aucune extraction à vérifier" : "Aucun PDF en attente",
                systemImage: "tray",
                description: Text("Vous pouvez ajouter un ou plusieurs articles à la fois.")
            )
            .frame(maxHeight: .infinity)
        } else {
            List {
                ForEach(visibleJobs) { job in
                    QueueRowView(
                        job: job,
                        onPreview: { previewJob = job },
                        onRetry: { store.retry(job.id) },
                        onShowError: { errorJob = job },
                        onRemove: { store.remove(job.id) },
                        canRemove: !store.isRunning
                    )
                }
                .onDelete { offsets in
                    let ids = offsets.compactMap { visibleJobs.indices.contains($0) ? visibleJobs[$0].id : nil }
                    for id in ids { store.remove(id) }
                }
            }
            .listStyle(.inset)
            .scrollContentBackground(.hidden)
            .background(RagdropTheme.panel)
            .clipShape(.rect(cornerRadius: 10))
        }
    }

    private var footer: some View {
        HStack(spacing: 12) {
            if store.isRunning {
                ProgressView()
                    .controlSize(.small)
            }
            if reviewOnly {
                Text(store.message).font(.callout).foregroundStyle(RagdropTheme.secondary).lineLimit(2)
            }
            Spacer()
            if store.canCancelConversion {
                Button("Annuler", role: .cancel, action: store.cancelConversion)
            }
            if store.jobs.contains(where: { [.completed, .duplicate, .rejected].contains($0.stage) }) && !store.isRunning {
                Button("Effacer les terminés", action: store.clearCompleted)
            }
            if store.jobs.contains(where: { $0.stage == .awaitingReview }) && !store.isRunning {
                Button("Tout approuver", action: store.approveAll)
            }
            Button(primaryButtonTitle) {
                store.start()
            }
            .buttonStyle(RagdropPrimaryButtonStyle())
            .disabled(store.isIsolated || store.isRunning || !store.jobs.contains {
                $0.stage == .queued || $0.stage == .failed || $0.stage == .readyForIndexing
            })
            .keyboardShortcut(.return, modifiers: .command)
        }
    }

    private var primaryButtonTitle: String {
        if store.isRunning { return "Traitement…" }
        if store.jobs.contains(where: { $0.stage == .readyForIndexing }) { return "Ajouter les approuvés" }
        let count = store.jobs.filter { $0.stage == .queued || $0.stage == .failed }.count
        return count > 0 ? "Lancer l’analyse de \(count) PDF" : "Analyser les PDF"
    }
}
