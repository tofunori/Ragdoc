import SwiftUI

struct HistoryView: View {
    @Bindable var store: HistoryStore
    @State private var query = ""

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            header

            if let error = store.errorMessage {
                errorBanner(error)
            }

            Group {
                if store.isLoading && store.documents.isEmpty {
                    VStack(spacing: 12) {
                        ProgressView()
                        Text("Lecture de la bibliothèque Ragdoc…")
                            .foregroundStyle(.secondary)
                    }
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                } else if store.errorMessage != nil, store.documents.isEmpty {
                    ContentUnavailableView(
                        "Historique indisponible",
                        systemImage: "externaldrive.badge.exclamationmark",
                        description: Text("Vérifiez la connexion au NAS, puis actualisez.")
                    )
                } else {
                    documentTable
                }
            }
        }
        .padding(18)
        .searchable(text: $query, prompt: "Titre, source ou DOI")
        .task {
            if store.documents.isEmpty { await store.refresh() }
        }
        .toolbar {
            ToolbarItem {
                Button("Actualiser", systemImage: "arrow.clockwise") {
                    Task { await store.refresh() }
                }
                .disabled(store.isLoading)
            }
            ToolbarItem {
                SettingsLink {
                    Label("Réglages", systemImage: "gearshape")
                }
            }
        }
    }

    private func errorBanner(_ message: String) -> some View {
        Label(message, systemImage: "exclamationmark.triangle.fill")
            .font(.callout)
            .foregroundStyle(.red)
            .padding(.horizontal, 12)
            .padding(.vertical, 9)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(.red.opacity(0.09), in: RoundedRectangle(cornerRadius: 10))
    }

    private var header: some View {
        HStack(alignment: .firstTextBaseline) {
            VStack(alignment: .leading, spacing: 4) {
                Text("Historique")
                    .font(.largeTitle.bold())
                Text("\(filteredDocuments.count) documents sur \(store.documents.count) dans la base canonique")
                    .foregroundStyle(.secondary)
            }
            Spacer()
            if store.isLoading {
                ProgressView().controlSize(.small)
            } else if let lastUpdated = store.lastUpdated {
                Text("Actualisé à \(lastUpdated.formatted(date: .omitted, time: .shortened))")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
    }

    private var documentTable: some View {
        Table(filteredDocuments) {
            TableColumn("Article") { document in
                VStack(alignment: .leading, spacing: 2) {
                    Text(document.displayTitle)
                        .lineLimit(1)
                    Text(document.source)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                }
                .padding(.vertical, 3)
            }
            .width(min: 330, ideal: 480)

            TableColumn("Passages") { document in
                Text(document.chunks.formatted())
                    .monospacedDigit()
            }
            .width(min: 70, ideal: 85, max: 95)

            TableColumn("Indexé le") { document in
                Text(document.displayDate)
                    .foregroundStyle(.secondary)
                    .monospacedDigit()
            }
            .width(min: 125, ideal: 145, max: 155)
        }
        .tableStyle(.inset)
    }

    private var filteredDocuments: [HistoryDocument] {
        let needle = query.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !needle.isEmpty else { return store.documents }
        return store.documents.filter {
            $0.displayTitle.localizedCaseInsensitiveContains(needle)
                || $0.source.localizedCaseInsensitiveContains(needle)
                || ($0.doi?.localizedCaseInsensitiveContains(needle) ?? false)
        }
    }
}
