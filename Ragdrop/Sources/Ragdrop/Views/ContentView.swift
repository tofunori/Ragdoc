import SwiftUI
import UniformTypeIdentifiers

struct ContentView: View {
    @Bindable var store: ImportStore
    @State private var isTargeted = false
    @State private var showingImporter = false
    @State private var showingZotero = false
    @State private var previewJob: ImportJob?
    @State private var errorJob: ImportJob?

    var body: some View {
        VStack(spacing: 12) {
            header
            DropZoneView(isTargeted: isTargeted) {
                showingImporter = true
            } chooseZotero: {
                showingZotero = true
            }
            .dropDestination(for: URL.self) { urls, _ in
                store.addFiles(urls)
                return urls.contains { $0.pathExtension.lowercased() == "pdf" }
            } isTargeted: { isTargeted = $0 }

            if !store.jobs.isEmpty {
                BatchProgressView(jobs: store.jobs, message: store.message)
            }

            queue
            footer
        }
        .padding(18)
        .fileImporter(
            isPresented: $showingImporter,
            allowedContentTypes: [.pdf],
            allowsMultipleSelection: true
        ) { result in
            if case .success(let urls) = result { store.addFiles(urls) }
        }
        .sheet(item: $previewJob) { job in
            MarkdownPreviewView(
                job: job,
                onApprove: {
                    store.approve(job.id)
                    previewJob = nil
                },
                onReject: {
                    store.reject(job.id)
                    previewJob = nil
                }
            )
        }
        .sheet(item: $errorJob) { job in
            ErrorDetailView(job: job)
        }
        .sheet(isPresented: $showingZotero) {
            ZoteroImportView { documents in
                store.addZoteroDocuments(documents)
            }
        }
        .toolbar {
            ToolbarItemGroup {
                Button("Ajouter des PDF", systemImage: "plus") { showingImporter = true }
                    .keyboardShortcut("o", modifiers: .command)
                SettingsLink {
                    Label("Réglages", systemImage: "gearshape")
                }
            }
        }
    }

    private var header: some View {
        HStack(alignment: .firstTextBaseline) {
            VStack(alignment: .leading, spacing: 4) {
                Text("Ragdrop")
                    .font(.largeTitle.bold())
                Text("Des PDF Finder vers votre bibliothèque scientifique")
                    .foregroundStyle(.secondary)
            }
            Spacer()
            Label("Ragdoc sur le NAS", systemImage: "externaldrive.connected.to.line.below")
                .font(.caption.weight(.medium))
                .foregroundStyle(.secondary)
        }
    }

    @ViewBuilder
    private var queue: some View {
        if store.jobs.isEmpty {
            ContentUnavailableView(
                "Aucun PDF en attente",
                systemImage: "tray",
                description: Text("Vous pouvez ajouter un ou plusieurs articles à la fois.")
            )
            .frame(maxHeight: .infinity)
        } else {
            List {
                ForEach(store.jobs) { job in
                    QueueRowView(
                        job: job,
                        onPreview: { previewJob = job },
                        onRetry: { store.retry(job.id) },
                        onShowError: { errorJob = job },
                        onRemove: { store.remove(job.id) },
                        canRemove: !store.isRunning
                    )
                }
                .onDelete(perform: store.removeJobs)
            }
            .listStyle(.inset)
            .clipShape(.rect(cornerRadius: 10))
        }
    }

    private var footer: some View {
        HStack(spacing: 12) {
            if store.isRunning {
                ProgressView()
                    .controlSize(.small)
            }
            Text(store.message)
                .font(.callout)
                .foregroundStyle(.secondary)
                .lineLimit(2)
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
            .buttonStyle(.borderedProminent)
            .disabled(store.isRunning || !store.jobs.contains {
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
